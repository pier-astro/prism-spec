"""
Sherpa-based fitters for prism.modeling.fitting
"""

import numpy as np
from astropy.modeling.fitting import Fitter, model_to_fit_params
from .utils import _get_tied_info, _apply_tied_fast

# Try to import Sherpa
try:
    import sherpa.optmethods
    import sherpa.stats
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

__all__ = ['SherpaFitter', 'SherpaLM', 'SherpaSimplex', 'SherpaMonCar', 'HAS_SHERPA']


class SherpaFitter(Fitter):
    """Wrapper around Sherpa optimizers with Sherpa statistics."""
    
    def __init__(self, method='levmar', calc_uncertainties=False, verbose=False):
        if not HAS_SHERPA:
            raise ImportError("Sherpa is not installed. Install with: pip install sherpa")
        
        self.method = method.lower()
        self.calc_uncertainties = calc_uncertainties
        self.verbose = verbose
        
        if self.method == 'levmar':
            self.opt = sherpa.optmethods.LevMar()
        elif self.method == 'neldermead':
            self.opt = sherpa.optmethods.NelderMead()
        elif self.method == 'moncar':
            self.opt = sherpa.optmethods.MonCar()
        else:
            raise ValueError(f"Unknown method: {method}. Options: levmar, neldermead, moncar")

    def __call__(self, model, x, y, z=None, weights=None, statistic='chi2', yerr=None, **kwargs):
        if 'uncertainties' in kwargs:
            raise TypeError("The 'uncertainties' parameter is no longer supported.")
        
        # When called natively through the dynamic patcher, `weights` is present.
        # But for Sherpa, we need yerr to do chi2 fitting via Chi2DataVar.
        # So we reverse-engineer yerr if it's missing but weights are given.
        if statistic == 'poisson':
            raise NotImplementedError("Poisson statistics not supported in Sherpa wrappers yet.")
        
        model = model.copy()
        init_values, fit_indices, _ = model_to_fit_params(model)
        
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                b[1] if b[1] is not None else np.inf)
                               for b in bounds_list])
        param_bounds = all_bounds[fit_indices]

        tied_info = _get_tied_info(model)
        params_cache = model.parameters.copy()
        has_tied = bool(tied_info)
        
        # Reverse engineer yerr
        if yerr is None and weights is not None:
            with np.errstate(divide='ignore'):
                yerr = np.where(weights > 0, 1.0 / weights, np.inf)
        
        if yerr is None:
            sherpa_stat = sherpa.stats.LeastSq()
        else:
            sherpa_stat = sherpa.stats.Chi2DataVar()
        
        min_vals = param_bounds[:, 0]  
        max_vals = param_bounds[:, 1]
        
        huge = np.finfo(float).max
        min_vals = np.where(np.isinf(min_vals), -huge, min_vals)
        max_vals = np.where(np.isinf(max_vals), huge, max_vals)
        
        if yerr is None:
            staterr = sherpa_stat.calc_staterror(y)
        else:
            staterr = yerr
        weights_sq = 1.0 / (staterr ** 2)
        
        def statfunc(p):
            params_cache[fit_indices] = p
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            
            resid = (model(x) - y)
            weighted_resid = resid * np.sqrt(weights_sq)
            stat = np.sum(weighted_resid**2)
            return stat, weighted_resid
        
        sherpa_kwargs = {k: v for k, v in kwargs.items() if k not in ['maxfev', 'max_nfev']}
        sherpa_result = self.opt.fit(statfunc, np.array(init_values), min_vals, max_vals, **sherpa_kwargs)
        
        success = sherpa_result[0]
        fitted_params = sherpa_result[1]
        message = sherpa_result[3] if len(sherpa_result) > 3 else ''
        
        sherpa_info = sherpa_result[4] if len(sherpa_result) > 4 and isinstance(sherpa_result[4], dict) else {}
        nfev = sherpa_info.get('nfev', None)
        native_cov = sherpa_info.get('covar')
        
        self.fit_info = {
            'success': success,
            'nfev': nfev,
            'message': message,
            'param_cov': native_cov
        }
        
        params_cache[fit_indices] = fitted_params
        model.parameters = params_cache
        if tied_info:
            _apply_tied_fast(model, tied_info, params_cache)
            
        return model


class SherpaLM(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='levmar', **kwargs)


class SherpaSimplex(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='neldermead', **kwargs)


class SherpaMonCar(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='moncar', **kwargs)