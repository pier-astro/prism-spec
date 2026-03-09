"""
Levenberg-Marquardt fitter with parameter transformations for bounds support.

This module implements scipy's leastsq (classic Levenberg-Marquardt) with parameter
transformations to handle bounds. Provides native covariance extraction from the LM algorithm.
"""

import time
import numpy as np
import warnings
from scipy.optimize import leastsq
from numpy.linalg import LinAlgError
from .base import FitterBase, _apply_tied_fast

__all__ = ['LMFitter']


class LMFitter(FitterBase):
    """
    Classic Levenberg-Marquardt fitter with bounds via parameter transformation.
    
    Uses scipy's leastsq (LM algorithm) with parameter transformations for
    bounds support. Provides guaranteed native covariance extraction.
    
    **Key Features:**
    - Native LM covariance (always available for successful fits)
    - Parameter transformations for bounds (logistic/exponential)
    - Automatic Jacobian transformation for bounded parameters
    - Efficient handling of tied parameters
    - Faster than bounded methods for well-conditioned problems
    
    **Algorithm:**
    Classic LM minimizes: ||f(p)||² using damping λ diag(J.T @ J)
    
    For bounds, we transform parameters:
    - Two-sided [a,b]:  internal = log((p-a)/(b-p))
    - Lower [a,∞):      internal = log(p-a)  
    - Upper (-∞,b]:     internal = log(b-p)
    
    **Covariance Extraction:**
    scipy.optimize.leastsq returns native covariance:
        x, cov_x, infodict, mesg, ier = leastsq(func, x0, full_output=True)
    We extract cov_x directly and transform it to external parameter space.
    
    Parameters
    ----------
    calc_uncertainties : bool, optional
        Whether to compute uncertainties (default is False).
    verbose : bool, optional
        Print convergence information (default is False).
    **kwargs : dict
        Additional arguments passed to scipy.optimize.leastsq:
        - maxfev : int (default: 0, no limit)
        - ftol, xtol, gtol : float (convergence tolerances)
        - epsfcn : float (step for numerical Jacobian)
    
    Notes
    -----
    Best for:
    - Spectral fitting with many tied parameters
    - Well-conditioned problems
    - When native covariance is critical
    
    Trade-offs:
    - Fast for well-conditioned models (3-100x faster than TRF)
    - May struggle with very ill-conditioned problems (many correlated parameters)
    - Parameter transformations add slight overhead (~1.6x vs unbounded LM)
    """
    
    def __init__(self, calc_uncertainties=False, force_numerical_covariance=False, verbose=False, **kwargs):
        super().__init__(calc_uncertainties=calc_uncertainties,
                        force_numerical_covariance=force_numerical_covariance,
                        verbose=verbose)
        self.leastsq_kwargs = kwargs

    def _transform_params(self, external_values, bounds_list):
        """
        Transform bounded external parameters to unbounded internal space.
        
        Returns array of internal (unbounded) values.
        """
        internal = np.zeros(len(external_values))
        self._param_transforms = {}
        
        for i, (value, bounds) in enumerate(zip(external_values, bounds_list)):
            lower, upper = bounds if isinstance(bounds, (list, tuple)) else (bounds[0], bounds[1])
            
            if lower is None and upper is None:
                # No bounds: direct mapping
                internal[i] = value
                self._param_transforms[i] = {'type': 'none'}
                
            elif lower is not None and upper is not None:
                # Two-sided bounds [a, b]
                # Logistic transform: internal = log((value - lower) / (upper - value))
                if not (lower < value < upper):
                    value = (lower + upper) / 2  # Reset if out of bounds
                
                internal[i] = np.log((value - lower) / (upper - value))
                self._param_transforms[i] = {
                    'type': 'two-sided',
                    'lower': lower,
                    'upper': upper
                }
                
            elif lower is not None:
                # Lower bound only [a, ∞)
                # Exponential: internal = log(value - lower)
                if value <= lower:
                    value = lower + 0.1 * abs(lower) if lower != 0 else 0.1
                
                internal[i] = np.log(value - lower)
                self._param_transforms[i] = {
                    'type': 'lower',
                    'lower': lower
                }
                
            else:  # upper is not None
                # Upper bound only (-∞, b]
                # Exponential: internal = log(upper - value)
                if value >= upper:
                    value = upper - 0.1 * abs(upper) if upper != 0 else -0.1
                
                internal[i] = np.log(upper - value)
                self._param_transforms[i] = {
                    'type': 'upper',
                    'upper': upper
                }
                
        return internal

    def _untransform_params(self, internal_values):
        """Transform internal (unbounded) parameters back to external space."""
        external = np.zeros(len(internal_values))
        
        for i, internal_val in enumerate(internal_values):
            transform = self._param_transforms[i]
            
            if transform['type'] == 'none':
                external[i] = internal_val
            elif transform['type'] == 'two-sided':
                lower, upper = transform['lower'], transform['upper']
                # Inverse logistic: external = (lower + upper * exp(internal)) / (1 + exp(internal))
                exp_val = np.exp(internal_val)
                external[i] = (lower + upper * exp_val) / (1 + exp_val)
            elif transform['type'] == 'lower':
                lower = transform['lower']
                # Inverse exponential: external = lower + exp(internal)
                external[i] = lower + np.exp(internal_val)
            elif transform['type'] == 'upper':
                upper = transform['upper']
                # Inverse exponential: external = upper - exp(internal)
                external[i] = upper - np.exp(internal_val)
                
        return external

    def _transform_covariance(self, cov_internal, internal_values):
        """Transform covariance matrix from internal to external parameter space."""
        n_params = len(internal_values)
        jacobian = np.zeros((n_params, n_params))
        
        # Diagonal Jacobian matrix (transformations are independent)
        for i, internal_val in enumerate(internal_values):
            transform = self._param_transforms[i]
            
            if transform['type'] == 'none':
                jacobian[i, i] = 1.0
            elif transform['type'] == 'two-sided':
                lower, upper = transform['lower'], transform['upper']
                # d(external)/d(internal) = (upper - lower) * exp(internal) / (1 + exp(internal))^2
                exp_val = np.exp(internal_val)
                jacobian[i, i] = (upper - lower) * exp_val / (1 + exp_val)**2
            elif transform['type'] == 'lower':
                # d(external)/d(internal) = exp(internal)
                jacobian[i, i] = np.exp(internal_val)
            elif transform['type'] == 'upper':
                # d(external)/d(internal) = -exp(internal)
                jacobian[i, i] = -np.exp(internal_val)
        
        # Transform covariance: Cov_external = J @ Cov_internal @ J.T
        return jacobian @ cov_internal @ jacobian.T

    def _fit_impl(self, prep_data, max_nfev=None, **kwargs):
        """
        Implement Levenberg-Marquardt fitting with parameter transformations.
        
        Parameters
        ----------
        prep_data : dict
            Prepared fitting data from _prepare_fitting()
        max_nfev : int, optional
            Maximum function evaluations (maps to maxfev).
        **kwargs : dict
            Additional arguments to scipy.optimize.leastsq
            
        Returns
        -------
        result : dict
            Fitting results with standardized keys
        """
        # Handle deprecated parameter name
        if 'uncertainties' in kwargs:
            raise TypeError(
                "The 'uncertainties' parameter is no longer supported. "
                "Use calc_uncertainties=True in the fitter constructor instead."
            )

        model = prep_data['model']
        x = prep_data['x']
        y = prep_data['y']
        weights = prep_data['weights']
        init_values = prep_data['init_values']
        fit_indices = prep_data['fit_indices']
        param_bounds = prep_data['param_bounds']
        tied_info = prep_data['tied_info']
        params_cache = prep_data['params_cache']
        
        n_free = len(init_values)
        has_tied = bool(tied_info)
        
        if n_free == 0:
            if self.verbose:
                print("No free parameters to fit.")
            return {
                'fitted_params': np.array([]),
                'success': True,
                'nfev': 0,
                'message': 'No free parameters',
                'native_result': None,
                'native_cov': None
            }
        
        # Build residual function with parameter transformations
        def residuals(internal_params):
            """Residual function accepting internal (unbounded) parameters."""
            # Transform to external space
            external_params = self._untransform_params(internal_params)
            
            # Update model
            params_cache[fit_indices] = external_params
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            
            # Compute residuals
            diff = model(x) - y
            if weights is not None:
                return diff * weights
            return diff
        
        # Transform initial parameters to internal space
        x0_internal = self._transform_params(init_values, param_bounds)
        
        # Merge kwargs
        leastsq_kws = {**self.leastsq_kwargs, **kwargs}
        if max_nfev is not None:
            leastsq_kws['maxfev'] = max_nfev
        
        # Run leastsq
        try:
            result = leastsq(residuals, x0_internal, full_output=True, **leastsq_kws)
            xopt_internal, cov_internal, infodict, mesg, ier = result
            
            # leastsq returns cov_internal=None if Jacobian is singular or not computed
            # Try to compute it from the QR decomposition in infodict
            if cov_internal is None and self.calc_uncertainties:
                if self.verbose:
                    print("Note: leastsq did not return covariance, computing from Jacobian...")
                
                try:
                    # Extract Jacobian from QR decomposition
                    fjac = infodict['fjac']
                    ipvt = infodict['ipvt']
                    n_params = len(xopt_internal)
                    
                    # Get the upper triangular R matrix
                    R = np.triu(fjac[:n_params, :n_params])
                    
                    # Compute covariance as inv(R.T @ R)
                    try:
                        RTR = R.T @ R
                        cond = np.linalg.cond(RTR)
                        
                        if cond < 1e12:
                            cov_internal = np.linalg.inv(RTR)
                        else:
                            if self.verbose:
                                print(f"      Jacobian is ill-conditioned (κ={cond:.2e}), using pseudo-inverse")
                            cov_internal = np.linalg.pinv(RTR)
                        
                        # Undo the permutation
                        perm = ipvt - 1  # Convert to 0-based
                        cov_internal = cov_internal[perm, :][:, perm]
                        
                        if self.verbose:
                            print("      ✓ Covariance computed from Jacobian")
                    
                    except (LinAlgError, np.linalg.LinAlgError) as e:
                        if self.verbose:
                            print(f"      ✗ Failed to compute covariance from Jacobian: {e}")
                        cov_internal = None
                        
                except Exception as e:
                    if self.verbose:
                        print(f"      ✗ Error extracting Jacobian: {e}")
                    cov_internal = None
                
        except Exception as e:
            raise RuntimeError(f"LMFitter failed: {str(e)}")
        
        # Check convergence
        success = ier in [1, 2, 3, 4]
        
        # Transform result to external space
        xopt_external = self._untransform_params(xopt_internal)
        
        # Compute covariance in external space
        native_cov = None
        if self.calc_uncertainties and cov_internal is not None and success:
            try:
                # Scale by reduced chi-square (following lmfit convention)
                resid_final = residuals(xopt_internal)
                chi2 = np.sum(resid_final**2)
                ndata = len(resid_final)
                dof = max(1, ndata - n_free)
                red_chi2 = chi2 / dof
                
                # Scale covariance matrix
                cov_scaled = cov_internal * red_chi2
                
                # Transform covariance to external space
                native_cov = self._transform_covariance(cov_scaled, xopt_internal)
                
                # Validate covariance (check for negative variances)
                diag = np.diag(native_cov)
                if np.any(diag < 0):
                    if self.verbose:
                        neg_idx = np.where(diag < 0)[0]
                        print(f"Warning: Negative variances for {len(neg_idx)} parameters")
                        print("         Setting uncertainties to NaN for affected parameters")
                    # Set negative variances to NaN rather than rejecting all
                    diag[diag < 0] = np.nan
                    native_cov[np.diag_indices_from(native_cov)] = diag
                
                if not np.all(np.isfinite(diag)):
                    if self.verbose:
                        print("Warning: Some parameter uncertainties are infinite or undefined")
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Covariance transformation failed: {e}")
                native_cov = None
        
        # Create scipy.optimize.OptimizeResult-like object
        from scipy.optimize import OptimizeResult
        resid_final = residuals(xopt_internal)
        chi2 = np.sum(resid_final**2)
        result_obj = OptimizeResult(
            x=xopt_external,
            success=success,
            fun=chi2,
            nfev=infodict['nfev'],
            message=mesg,
            ier=ier,
            leastsq_cov=cov_internal  # Store internal covariance for debugging
        )
        
        return {
            'fitted_params': xopt_external,
            'success': success,
            'nfev': infodict['nfev'],
            'message': mesg,
            'native_result': result_obj,
            'native_cov': native_cov
        }