"""
Sherpa-based fitters for prism.modeling.fitting

This module provides wrappers around Sherpa optimizers with C++-backed statistics
and standardized interfaces.
"""

import numpy as np
from .base import FitterBase, _apply_tied_fast

# Try to import Sherpa
try:
    import sherpa.optmethods
    import sherpa.stats
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

__all__ = ['SherpaFitter', 'SherpaLM', 'SherpaSimplex', 'SherpaMonCar', 'HAS_SHERPA']


class SherpaFitter(FitterBase):
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
    - Recommendation: Use SherpaLM for uncertainties, others for optimization only
    
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
    
    def __init__(self, method='levmar', calc_uncertainties=False,
                 force_numerical_covariance=False, verbose=False,
                 filter_non_finite=False):
        if not HAS_SHERPA:
            raise ImportError("Sherpa is not installed. Install with: pip install sherpa")
        
        super().__init__(calc_uncertainties=calc_uncertainties, 
                         force_numerical_covariance=force_numerical_covariance, 
                         verbose=verbose,
                         filter_non_finite=filter_non_finite)
        self.method = method.lower()
        
        if self.method == 'levmar':
            self.opt = sherpa.optmethods.LevMar()
        elif self.method == 'neldermead':
            self.opt = sherpa.optmethods.NelderMead()
        elif self.method == 'moncar':
            self.opt = sherpa.optmethods.MonCar()
        else:
            raise ValueError(f"Unknown method: {method}. Options: levmar, neldermead, moncar")

    def _fit_impl(self, prep_data, statistic='chi2', **kwargs):
        """
        Implement Sherpa fitting using the prepared data.
        
        Parameters
        ----------
        prep_data : dict
            Prepared fitting data from _prepare_fitting()
        statistic : str, optional
            'chi2' (default) or 'poisson'. Currently only 'chi2' is implemented.
        **kwargs : dict
            Additional arguments to Sherpa optimizers
            
        Returns
        -------
        result : dict
            Fitting results with standardized keys
        """
        # Validate deprecated parameters in __call__
        if 'uncertainties' in kwargs:
            raise TypeError(
                "The 'uncertainties' parameter is no longer supported. "
                "Use calc_uncertainties=True in the fitter constructor instead."
            )
        
        if 'weights' in kwargs and kwargs['weights'] is not None:
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
        
        model = prep_data['model']
        x = prep_data['x']
        y = prep_data['y']
        init_values = prep_data['init_values']
        fit_indices = prep_data['fit_indices']
        param_bounds = prep_data['param_bounds']
        tied_info = prep_data['tied_info']
        params_cache = prep_data['params_cache']
        
        # Get yerr from the original weights computation
        # We need to reconstruct yerr for Sherpa statistic selection
        yerr = None  # Will be extracted from prep_data if available
        weights = prep_data['weights']
        if weights is not None:
            # Back-calculate yerr from weights (weights = 1/yerr for chi2)
            with np.errstate(divide='ignore'):
                yerr = np.where(weights > 0, 1.0 / weights, np.inf)
        
        has_tied = bool(tied_info)
        
        # Select Sherpa statistic
        if statistic == 'chi2':
            if yerr is None:
                sherpa_stat = sherpa.stats.LeastSq()
            else:
                sherpa_stat = sherpa.stats.Chi2DataVar()
        
        # Prepare bounds
        min_vals = param_bounds[:, 0]  
        max_vals = param_bounds[:, 1]
        
        # Handle infinities
        huge = np.finfo(float).max
        min_vals = np.where(np.isinf(min_vals), -huge, min_vals)
        max_vals = np.where(np.isinf(max_vals), huge, max_vals)
        
        # Compute statistical errors using Sherpa's methods
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
        
        # Run Sherpa optimization
        # Filter kwargs to remove unsupported parameters
        sherpa_kwargs = {k: v for k, v in kwargs.items() if k not in ['maxfev', 'max_nfev']}
        sherpa_result = self.opt.fit(statfunc, np.array(init_values), min_vals, max_vals, **sherpa_kwargs)
        
        # Extract results
        success = sherpa_result[0]
        fitted_params = sherpa_result[1]
        fun = sherpa_result[2]
        message = sherpa_result[3] if len(sherpa_result) > 3 else ''
        
        # Extract detailed info
        sherpa_info = sherpa_result[4] if len(sherpa_result) > 4 and isinstance(sherpa_result[4], dict) else {}
        nfev = sherpa_info.get('nfev', None)
        native_cov = sherpa_info.get('covar')
        
        # Create scipy.optimize.OptimizeResult-like structure for compatibility
        from scipy.optimize import OptimizeResult
        result = OptimizeResult(
            x=fitted_params,
            success=success,
            fun=fun,
            nfev=nfev,
            message=message,
            # Store all MINPACK info from Sherpa
            info=sherpa_info.get('info'),  # MINPACK info flag (same as scipy)
            num_parallel_map=sherpa_info.get('num_parallel_map'),  # Sherpa-specific
            covar=native_cov  # Native covariance from MINPACK
        )
        
        return {
            'fitted_params': fitted_params,
            'success': success,
            'nfev': nfev,
            'message': message,
            'native_result': result,
            'native_cov': native_cov
        }


class SherpaLM(SherpaFitter):
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
        fitter = SherpaLM(calc_uncertainties=True, verbose=True)
        result = fitter(model, x, y, maxfev=10000, ftol=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='levmar', **kwargs)


class SherpaSimplex(SherpaFitter):
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
        fitter = SherpaSimplex(verbose=True)
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