"""
SciPy-based fitters for prism.modeling.fitting

This module provides wrappers around scipy.optimize.least_squares with
standardized interfaces and robust error handling.
"""

import warnings
import numpy as np
from scipy import optimize
from numpy.linalg import LinAlgError
from .base import FitterBase, _apply_tied_fast

__all__ = ['ScipyFitter', 'ScipyTRF', 'ScipyDogBox']


class ScipyFitter(FitterBase):
    """
    Wrapper around scipy.optimize.least_squares.
    
    **Covariance Extraction Strategy:**
    - Uses native Jacobian from scipy (result.jac) to compute covariance
    - Covariance = inv(J.T @ J) [standard least-squares approximation]
    - If Jacobian is ill-conditioned (κ > 10¹²), uses pseudo-inverse
    - Optional expensive numerical Hessian fallback (disabled by default)
    
    Supports bounds natively with 'trf' and 'dogbox' methods.
    
    Parameters
    ----------
    method : str
        'trf' (default) or 'dogbox'.
    calc_uncertainties : bool
        Whether to compute parameter uncertainties.
    force_numerical_covariance : bool
        If True, compute numerical Hessian when Jacobian fails (expensive).
    verbose : bool
        Print diagnostic information.
    """
    
    def __init__(self, method='trf', calc_uncertainties=False, force_numerical_covariance=False,
                 verbose=False, filter_non_finite=False):
        super().__init__(calc_uncertainties=calc_uncertainties,
                        force_numerical_covariance=force_numerical_covariance,
                        verbose=verbose,
                        filter_non_finite=filter_non_finite)
        if method not in ('trf', 'dogbox'):
            raise ValueError(f"method must be 'trf' or 'dogbox', got {method}")
        self.method = method

    def _build_jacobian(self, model, x, fit_indices, tied_info, params_cache, weights):
        """Build Jacobian function with proper error handling."""
        use_analytic = True
        
        if not hasattr(model, 'fit_deriv'):
            return '2-point'
        
        try:
            # Test if fit_deriv works and check for NaNs/Infs
            test_deriv = model.fit_deriv(x, *model.parameters)
            
            if np.any(~np.isfinite(test_deriv)):
                if self.verbose:
                    print("Warning: Model has invalid derivatives (NaN/Inf), using numeric Jacobian")
                use_analytic = False
            
            if not use_analytic:
                return '2-point'
            
            def jacobian(p):
                params_cache[fit_indices] = p
                model.parameters = params_cache
                if tied_info:
                    _apply_tied_fast(model, tied_info, params_cache)
                
                # Get full jacobian
                J_all = np.array(model.fit_deriv(x, *model.parameters))
                
                # Check for NaNs/infs
                if np.any(~np.isfinite(J_all)):
                    raise ValueError("Invalid derivatives encountered")
                
                # Select only free parameters
                # fit_deriv returns (n_params, n_points)
                # We need (n_points, n_free_params)
                J = J_all[fit_indices].T
                
                if weights is not None:
                    return J * weights[:, np.newaxis]
                return J
                
            return jacobian
            
        except Exception as e:
            if self.verbose:
                print(f"Note: Using numeric Jacobian due to: {type(e).__name__}")
            return '2-point'

    def _fit_impl(self, prep_data, max_nfev=None, **kwargs):
        """
        Implement SciPy least_squares fitting.
        
        Parameters
        ----------
        prep_data : dict
            Prepared fitting data from _prepare_fitting()
        max_nfev : int, optional
            Maximum function evaluations (replaces deprecated maxiter).
        **kwargs : dict
            Additional arguments to scipy.optimize.least_squares
            
        Returns
        -------
        result : dict
            Fitting results with standardized keys
        """
        model = prep_data['model']
        x = prep_data['x']
        y = prep_data['y']
        weights = prep_data['weights']
        init_values = prep_data['init_values']
        fit_indices = prep_data['fit_indices']
        param_bounds = prep_data['param_bounds']
        tied_info = prep_data['tied_info']
        params_cache = prep_data['params_cache']
        
        has_tied = bool(tied_info)
        
        # Build residual function
        def residual(p):
            params_cache[fit_indices] = p
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            diff = model(x) - y
            if weights is not None:
                return diff * weights
            return diff
        
        # Build Jacobian
        jac = self._build_jacobian(model, x, fit_indices, tied_info, params_cache, weights)
        
        # Prepare bounds
        min_vals = param_bounds[:, 0]
        max_vals = param_bounds[:, 1]
        bounds = (min_vals, max_vals)
        
        # Prepare kwargs
        fit_kwargs = {
            'method': self.method,
            'bounds': bounds,
            'jac': jac,
            **kwargs
        }
        if max_nfev is not None:
            fit_kwargs['max_nfev'] = max_nfev
        
        try:
            result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
        except (LinAlgError, ValueError) as e:
            # Fallback strategy: retry with TRF and numeric Jacobian
            if self.method == 'dogbox':
                if self.verbose:
                    print(f"Warning: DogBox failed ({type(e).__name__}), retrying with TRF...")
                fit_kwargs.update({'method': 'trf', 'jac': '2-point'})
                result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
            else:
                # For TRF, just use numeric Jacobian
                if self.verbose:
                    print(f"Warning: Optimization failed ({type(e).__name__}), using numeric Jacobian...")
                fit_kwargs['jac'] = '2-point'
                result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
        
        # Compute covariance if requested
        native_cov = None
        if self.calc_uncertainties:
            try:
                J = result.jac
                if J is None or J.shape[0] == 0:
                    if self.verbose:
                        print("⚠ No Jacobian available for covariance calculation.")
                else:
                    # Compute Hessian approximation: H = J.T @ J
                    try:
                        H = np.dot(J.T, J)
                        cond = np.linalg.cond(H)
                        if cond > 1e12:
                            if self.verbose:
                                print(f"⚠ Jacobian is ill-conditioned (κ={cond:.2e}), using pseudo-inverse")
                        native_cov = np.linalg.pinv(H)
                    except LinAlgError as e:
                        if self.verbose:
                            print(f"⚠ Covariance calculation failed: {e}")
                        native_cov = None
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Failed to calculate covariance: {e}")
        
        return {
            'fitted_params': result.x,
            'success': result.success,
            'nfev': result.nfev,
            'message': getattr(result, 'message', ''),
            'native_result': result,
            'native_cov': native_cov
        }


class ScipyTRF(ScipyFitter):
    """
    Trust Region Reflective via scipy.optimize.least_squares (supports bounds, default).
    
    **Key Parameters (pass in __call__):**
    - max_nfev : int - Maximum function evaluations (default: 100×n_params)
    - ftol : float - Tolerance for cost function change (default: 1e-8)
    - xtol : float - Tolerance for parameter change (default: 1e-8)
    - gtol : float - Tolerance for gradient norm (default: 1e-8)
    - verbose : int - Scipy verbosity level {0, 1, 2} (default: 0)
    
    Example:
        fitter = ScipyTRF(calc_uncertainties=True, verbose=True)
        result = fitter(model, x, y, max_nfev=10000, ftol=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='trf', **kwargs)


class ScipyDogBox(ScipyFitter):
    """
    Dogleg with rectangular trust regions via scipy.optimize.least_squares (supports bounds).
    
    **Key Parameters (pass in __call__):**
    - max_nfev : int - Maximum function evaluations (default: 100×n_params)
    - ftol : float - Tolerance for cost function change (default: 1e-8)
    - xtol : float - Tolerance for parameter change (default: 1e-8)
    - gtol : float - Tolerance for gradient norm (default: 1e-8)
    - verbose : int - Scipy verbosity level {0, 1, 2} (default: 0)
    
    Example:
        fitter = ScipyDogBox(calc_uncertainties=True)
        result = fitter(model, x, y, max_nfev=5000, xtol=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='dogbox', **kwargs)