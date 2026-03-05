
import time
import warnings
import numpy as np
from astropy.modeling.fitting import model_to_fit_params
from .fitting import FantasyFitter, _get_tied_info, _apply_tied_fast

# Try to import Sherpa
try:
    import sherpa.optmethods
    import sherpa.stats
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

__all__ = ['SherpaFitter', 'SherpaLevMar', 'SherpaNelderMead', 'SherpaMonCar', 'HAS_SHERPA']

class SherpaFitter(FantasyFitter):
    """
    Wrapper around Sherpa optimizers with C++-backed statistics.
    
    Uses Sherpa's robust optimization algorithms (Levenberg-Marquardt, Nelder-Mead, Monte Carlo)
    with Sherpa's C++-backed statistic functions for performance.
    
    **Statistics (statistic parameter):**
    - 'chi2' (default): Uses Sherpa's Chi2DataVar (with yerr) or LeastSq (no yerr)
      → Leverages C++ backend for fast computation
    - 'poisson': Not yet implemented (raises NotImplementedError)
      → Cash/CStat statistics require different data handling
    
    **Internal Sherpa Stats Selection:**
    - No yerr provided → sherpa.stats.LeastSq() (uniform weighting)
    - yerr provided → sherpa.stats.Chi2DataVar() (variance from data)
    
    **Covariance Extraction Strategy:**
    - Levenberg-Marquardt: Extracts native covariance from Sherpa/MINPACK (fast)
    - Nelder-Mead & Monte Carlo: Do NOT provide native covariance
      Falls back to expensive numerical Hessian estimation (not recommended)
    - Recommendation: Use SherpaLevMar for uncertainties, others for optimization only
    
    **Stored MINPACK Parameters:**
    - nfev : Number of function evaluations
    - ier (stored as 'info' in result): MINPACK termination flag
    - covar : Native covariance matrix from MINPACK (LevMar only)
    - num_parallel_map : Sherpa-specific parallelization info
    
    Note: ~5-10x slower than scipy/astropy due to numerical derivatives.
    
    Parameters
    ----------
    method : str
        'levmar' (default), 'neldermead', 'moncar'
    calc_uncertainties : bool
        Calculate parameter uncertainties (default: False).
    """
    
    def __init__(self, method='levmar', calc_uncertainties=False, force_numerical_covariance=False, verbose=False):
        if not HAS_SHERPA:
            raise ImportError("Sherpa is not installed. Install with: pip install sherpa")
        super().__init__(calc_uncertainties=calc_uncertainties, 
                         force_numerical_covariance=force_numerical_covariance, 
                         verbose=verbose)
        self.method = method.lower()
        
        if self.method == 'levmar':
            self.opt = sherpa.optmethods.LevMar()
        elif self.method == 'neldermead':
            self.opt = sherpa.optmethods.NelderMead()
        elif self.method == 'moncar':
            self.opt = sherpa.optmethods.MonCar()
        else:
            raise ValueError(f"Unknown method: {method}. Options: levmar, neldermead, moncar")
    
    def __call__(self, model, x, y, z=None, yerr=None, weights=None, statistic='chi2', inplace=False, **kwargs):
        """
        Fit model to data using Sherpa optimizers.
        
        Parameters
        ----------
        yerr : array-like, optional
            Uncertainties on y data. If provided with statistic='chi2',
            uses Chi2DataVar. If None with statistic='chi2', uses LeastSq.
        weights : array-like, optional
            NOT SUPPORTED. Use yerr instead. Raises error if provided.
        statistic : str, optional
            'chi2' (default) or 'poisson'. Currently only 'chi2' is implemented.
            'poisson' raises NotImplementedError.
        
        Notes
        -----
        Internally uses Sherpa's C++-backed statistics:
        - statistic='chi2' + yerr=None → sherpa.stats.LeastSq()
        - statistic='chi2' + yerr → sherpa.stats.Chi2DataVar()
        - statistic='poisson' → NotImplementedError (Cash/CStat need special handling)
        """
        # Filter out deprecated parameters
        if 'uncertainties' in kwargs:
            raise TypeError(
                "The 'uncertainties' parameter is no longer supported. "
                "Use calc_uncertainties=True in the fitter constructor instead."
            )
        
        if weights is not None:
            raise TypeError(
                "The 'weights' parameter is not supported. Use 'yerr' instead "
                "to enable Chi2DataVar statistic."
            )
        
        # Validate statistic parameter
        statistic = statistic.lower()
        if statistic not in ['chi2', 'poisson']:
            raise ValueError(f"Unknown statistic: {statistic}. Options: 'chi2', 'poisson'")
        
        if statistic == 'poisson':
            raise NotImplementedError(
                "Poisson statistics (Cash/CStat) are not yet supported. "
                "These require specialized data handling in Sherpa. Use statistic='chi2' instead."
            )
        
        if z is not None:
            warnings.warn("2D fitting not fully supported in SherpaFitter")
        if not inplace:
            model = model.copy()
        
        # Select Sherpa statistic based on statistic parameter and yerr
        # For chi2: use LeastSq (no yerr) or Chi2DataVar (with yerr)
        if statistic == 'chi2':
            if yerr is None:
                sherpa_stat = sherpa.stats.LeastSq()
            else:
                sherpa_stat = sherpa.stats.Chi2DataVar()
        # poisson already handled above with NotImplementedError
            
        init_values, fit_indices, _ = model_to_fit_params(model)
        
        # Bounds
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                 b[1] if b[1] is not None else np.inf) 
                                for b in bounds_list])
        min_vals = all_bounds[fit_indices, 0]
        max_vals = all_bounds[fit_indices, 1]
        
        # Handle infinities
        huge = np.finfo(float).max
        min_vals = np.where(np.isinf(min_vals), -huge, min_vals)
        max_vals = np.where(np.isinf(max_vals), huge, max_vals)
        
        # Cache for tied parameters
        tied_info = _get_tied_info(model)
        has_tied = bool(tied_info)
        params_cache = model.parameters.copy()
        
        # Compute statistical errors using Sherpa's methods
        # This gives us the proper weighting for the fit
        if statistic == 'chi2':
            if yerr is None:
                # LeastSq: uniform weighting (staterr = 1.0)
                staterr = sherpa_stat.calc_staterror(y)  # Returns ones_like(y)
            else:
                # Chi2DataVar: use provided yerr directly
                staterr = yerr
            # Compute weights for chi-square: w = 1 / sigma^2
            weights_sq = 1.0 / (staterr ** 2)
        
        def statfunc(p):
            """Objective function using Sherpa-style weighting"""
            params_cache[fit_indices] = p
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            
            # Compute residuals
            resid = (model(x) - y)
            
            # Apply Sherpa-style chi-square weighting
            # Chi2 = sum((data - model)^2 / sigma^2)
            weighted_resid = resid * np.sqrt(weights_sq)
            stat = np.sum(weighted_resid**2)
            
            return stat, weighted_resid
        
        t0 = time.perf_counter()
        sherpa_result = self.opt.fit(statfunc, np.array(init_values), min_vals, max_vals)
        elapsed = time.perf_counter() - t0
        
        # Update model
        params_cache[fit_indices] = sherpa_result[1]
        model.parameters = params_cache
        if has_tied:
            _apply_tied_fast(model, tied_info, params_cache)
        
        # Wrap Sherpa result in scipy.optimize.OptimizeResult-like structure
        from scipy.optimize import OptimizeResult
        sherpa_info = sherpa_result[4] if len(sherpa_result) > 4 and isinstance(sherpa_result[4], dict) else {}
        nfev = sherpa_info.get('nfev', None)
        
        result = OptimizeResult(
            x=sherpa_result[1],
            success=sherpa_result[0],
            fun=sherpa_result[2],
            nfev=nfev,
            message=sherpa_result[3] if len(sherpa_result) > 3 else '',
            # Store all MINPACK info from Sherpa
            info=sherpa_info.get('info'),  # MINPACK info flag (same as scipy)
            num_parallel_map=sherpa_info.get('num_parallel_map'),  # Sherpa-specific
            covar=sherpa_info.get('covar')  # Native covariance from MINPACK
        )
        
        # Store fit info with standard keys matching scipy where applicable
        self.fit_info['time'] = elapsed
        self.fit_info['result'] = result
        self.fit_info['success'] = result.success
        self.fit_info['nfev'] = result.nfev
        self.fit_info['ier'] = result.info  # MINPACK info flag (scipy uses 'ier')
        
        # Calculate covariance matrix if requested
        if self.calc_uncertainties:
            try:
                # Try to extract covariance from Sherpa optimizer (native)
                cov = result.covar  # Get from OptimizeResult
                
                # If native covariance available, use it
                if cov is not None:
                    # Sherpa returns covariance for free parameters only (matches fit_indices)
                    # Store directly in fit_info
                    self.fit_info['param_cov'] = cov
                elif self.force_numerical_covariance:
                    # Fallback to numerical estimation (expensive, only if forced)
                    if self.verbose:
                        print("⚠️  WARNING: Sherpa optimizer did not provide native covariance matrix!")
                        print("⚠️  FALLING BACK to expensive numerical Hessian estimation (O(n²) evaluations).")
                        print("⚠️  Only Levenberg-Marquardt provides native covariance. Use SherpaLevMar for efficiency.")
                    cov_matrix = self.estimate_covariance(model, x, y, weights=weights)
                    self.fit_info['param_cov'] = cov_matrix
                else:
                    if self.verbose:
                        print("⚠️  WARNING: Sherpa optimizer did not provide native covariance.")
                        print("⚠️  Set force_numerical_covariance=True to enable expensive fallback.")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to calculate covariance: {e}")
        
        if self.verbose:
            status = "✓ converged" if self.fit_info['success'] else "✗ FAILED"
            print(f"SherpaFitter({self.method}): {self.fit_info['time']*1000:.1f}ms, nfev={self.fit_info['nfev']}, {status}")
        
        return model


class SherpaLevMar(SherpaFitter):
    """
    Sherpa Levenberg-Marquardt optimizer (fast, local).
    
    **Key Parameters (pass in __call__):**
    - ftol : float - Tolerance for cost function change (default: 1.19e-7)
    - xtol : float - Tolerance for parameter change (default: 1.19e-7)
    - gtol : float - Tolerance for gradient norm (default: 1.19e-7)
    - maxfev : int - Maximum function evaluations (default: 1000×n_params)
    - epsfcn : float - Step size for numerical derivatives (default: 1.19e-7)
    - factor : float - Initial step bound (default: 100.0)
    - verbose : int - Verbosity level {0, 1, 2} (default: 0)
    
    Example:
        fitter = SherpaLevMar(calc_uncertainties=True, verbose=True)
        result = fitter(model, x, y, maxfev=10000, ftol=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='levmar', **kwargs)

class SherpaNelderMead(SherpaFitter):
    """
    Sherpa Nelder-Mead simplex optimizer (derivative-free, robust).
    
    **Key Parameters (pass in __call__):**
    - ftol : float - Tolerance for cost function change (default: 1.19e-4)
    - xtol : float - Tolerance for parameter change (default: 1.19e-4)
    - maxfev : int - Maximum function evaluations (default: 1000×n_params)
    - step : list[float] - Initial simplex step sizes (default: 0.4×params)
    - finalsimplex : int - Simplex behavior at convergence {0, 1, 2} (default: 1)
    - iquad : int - Use quadratic interpolation {0, 1} (default: 1)
    - verbose : int - Verbosity level {0, 1, 2} (default: 0)
    
    Example:
        fitter = SherpaNelderMead(verbose=True)
        result = fitter(model, x, y, maxfev=50000, ftol=1e-6)
    """
    def __init__(self, **kwargs):
        super().__init__(method='neldermead', **kwargs)

class SherpaMonCar(SherpaFitter):
    """
    Sherpa Monte Carlo optimizer (global search, stochastic).
    
    **Key Parameters (pass in __call__):**
    - ftol : float - Tolerance for cost function change (default: 1.19e-4)
    - maxfev : int - Maximum function evaluations (default: 10000)
    - verbose : int - Verbosity level {0, 1, 2} (default: 0)
    - numcores : int - Number of CPU cores for parallel eval (default: 1)
    - seed : int - Random seed for reproducibility (default: varies)
    - population_size : int - Size of random population (default: None)
    - xprob : float - Crossover probability [0,1] (default: 0.9)
    - weighting_factor : float - Differential weight [0,2] (default: 0.8)
    
    Example:
        fitter = SherpaMonCar(verbose=True)
        result = fitter(model, x, y, maxfev=100000, numcores=4, seed=42)
    """
    def __init__(self, **kwargs):
        super().__init__(method='moncar', **kwargs)
