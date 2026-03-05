
import time
import warnings
import numpy as np
from scipy import optimize
from numpy.linalg import LinAlgError
from astropy.modeling.fitting import model_to_fit_params
from .fitting import FantasyFitter, _get_tied_info, _apply_tied_fast

__all__ = ['ScipyFitter', 'ScipyTRF', 'ScipyDogBox']

class ScipyFitter(FantasyFitter):
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
    
    def __init__(self, method='trf', calc_uncertainties=False, force_numerical_covariance=False, verbose=False):
        super().__init__(calc_uncertainties=calc_uncertainties, 
                        force_numerical_covariance=force_numerical_covariance, 
                        verbose=verbose)
        if method not in ('trf', 'dogbox'):
            raise ValueError(f"method must be 'trf' or 'dogbox', got {method}")
        self.method = method
        
    def __call__(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None, inplace=False, max_nfev=None, **kwargs):
        """Fit model to data.
        
        Parameters
        ----------
        model : astropy.modeling.Model
            Model to fit.
        x, y : array_like
            Data to fit.
        z : array_like, optional
            For 2D models (not yet implemented).
        yerr : array_like, optional
            Data uncertainties.
        statistic : str
            'chi2' or 'poisson'.
        weights : array_like, optional
            Explicit weights (overrides yerr).
        inplace : bool
            Modify model in-place.
        max_nfev : int, optional
            Maximum function evaluations.
        **kwargs : dict
            Additional arguments to scipy.optimize.least_squares.
        
        Returns
        -------
        fitted_model : astropy.modeling.Model
            Model with fitted parameters and uncertainties (if calc_uncertainties=True).
        """
        if z is not None:
            raise NotImplementedError("2D fitting not yet implemented for ScipyFitter")
        if not inplace:
            model = model.copy()
            
        # Compute weights if not explicitly provided
        if weights is None and yerr is not None:
            weights = self._compute_weights(y, yerr, statistic=statistic)
        
        # Get free parameters
        init_values, fit_indices, _ = model_to_fit_params(model)
        
        # Get bounds
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                 b[1] if b[1] is not None else np.inf) 
                                for b in bounds_list])
        min_vals = all_bounds[fit_indices, 0]
        max_vals = all_bounds[fit_indices, 1]
        
        # Cache for tied parameters
        tied_info = _get_tied_info(model)
        has_tied = bool(tied_info)
        params_cache = model.parameters.copy()
        
        def residual(p):
            params_cache[fit_indices] = p
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            diff = model(x) - y
            if weights is not None:
                return diff * weights
            return diff
        
        # Jacobian (if available)
        jac = '2-point'
        use_analytic = True
        
        if hasattr(model, 'fit_deriv'):
            try:
                # Test if fit_deriv works and check for NaNs/Infs
                test_deriv = model.fit_deriv(x, *model.parameters)
                
                if np.any(~np.isfinite(test_deriv)):
                    if self.verbose:
                        print("Warning: Model has invalid derivatives (NaN/Inf), using numeric Jacobian")
                    use_analytic = False
                
                if use_analytic:
                    def jacobian(p):
                        params_cache[fit_indices] = p
                        model.parameters = params_cache
                        if has_tied:
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
                        
                    jac = jacobian
            except Exception as e:
                # Fallback to numeric if analytic derivative fails
                if self.verbose:
                    print(f"Note: Using numeric Jacobian due to: {type(e).__name__}")
                use_analytic = False
        
        bounds = (min_vals, max_vals)
        
        t0 = time.perf_counter()
        try:
            result = optimize.least_squares(
                residual, np.array(init_values),
                jac=jac,
                bounds=bounds,
                method=self.method,
                max_nfev=max_nfev,
                **kwargs
            )
        except (LinAlgError, ValueError) as e:
            # SVD/numerical failure - retry with TRF if we're using DogBox
            if self.method == 'dogbox':
                if self.verbose:
                    print(f"Warning: DogBox failed ({type(e).__name__}), retrying with TRF...")
                result = optimize.least_squares(
                    residual, np.array(init_values),
                    jac='2-point',  # Force numeric Jacobian
                    bounds=bounds,
                    method='trf',
                    max_nfev=max_nfev,
                    **kwargs
                )
            else:
                # For TRF, just re-raise or use fallback
                if self.verbose:
                    print(f"Warning: Optimization failed ({type(e).__name__}), using numeric Jacobian...")
                result = optimize.least_squares(
                    residual, np.array(init_values),
                    jac='2-point',  # Force numeric Jacobian
                    bounds=bounds,
                    method=self.method,
                    max_nfev=max_nfev,
                    **kwargs
                )
        
        self.fit_info['time'] = time.perf_counter() - t0
        self.fit_info['nfev'] = result.nfev
        self.fit_info['result'] = result
        self.fit_info['success'] = result.success
        
        # Update model
        params_cache[fit_indices] = result.x
        model.parameters = params_cache
        if has_tied:
            _apply_tied_fast(model, tied_info, params_cache)
            
        # Calculate covariance matrix if requested
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
                        cov = np.linalg.pinv(H)
                        self.fit_info['param_cov'] = cov
                    except LinAlgError as e:
                        # Jacobian-based covariance failed
                        if self.force_numerical_covariance:
                            if self.verbose:
                                print("⚠ WARNING: Jacobian-based covariance failed!")
                                print("⚠ Computing expensive numerical Hessian (O(n²) evaluations)...")
                            cov = self.estimate_covariance(model, x, y, weights=weights)
                            self.fit_info['param_cov'] = cov
                        else:
                            if self.verbose:
                                print(f"⚠ Covariance calculation failed: {e}")
                                print("  Set force_numerical_covariance=True for expensive numerical fallback")
                            self.fit_info['param_cov'] = None
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Failed to calculate covariance: {e}")
        
        if self.verbose:
            status = "✓ converged" if result.success else "✗ FAILED"
            print(f"ScipyFitter({self.method}): {self.fit_info['time']*1000:.1f}ms, nfev={result.nfev}, {status}")
        
        return model


class ScipyTRF(ScipyFitter):
    """
    Trust Region Reflective (supports bounds, default).
    
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
    Dogleg with rectangular trust regions (supports bounds).
    
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
