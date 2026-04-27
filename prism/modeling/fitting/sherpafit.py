"""
Sherpa-based fitters for prism.modeling.fitting
"""

import numpy as np
from astropy.modeling.fitting import Fitter, model_to_fit_params
from .extension import (
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    get_max_evaluations,
    validate_symmetric_yerr,
)
from .multifit import MultiFitMixin
from .utils import _get_tied_info, _apply_tied_fast

# Try to import Sherpa
try:
    import sherpa.optmethods
    import sherpa.stats
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

__all__ = ['SherpaFitter', 'SherpaLM', 'SherpaSimplex', 'SherpaMonCar', 'HAS_SHERPA']


class SherpaFitter(MultiFitMixin, Fitter):
    """Wrapper around Sherpa optimizers with Sherpa statistics."""

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)
    
    def __init__(self, method='levmar', calc_uncertainties=False, verbose=False, **kwargs):
        if not HAS_SHERPA:
            raise ImportError("Sherpa is not installed. Install with: pip install sherpa")
        
        self.method = method.lower()
        self.calc_uncertainties = calc_uncertainties
        self.verbose = verbose
        self.fit_kwargs = kwargs
        
        if self.method == 'levmar':
            self.opt = sherpa.optmethods.LevMar()
        elif self.method == 'neldermead':
            self.opt = sherpa.optmethods.NelderMead()
        elif self.method == 'moncar':
            self.opt = sherpa.optmethods.MonCar()
        else:
            raise ValueError(f"Unknown method: {method}. Options: levmar, neldermead, moncar")

    @property
    def max_evaluations(self):
        value = self.fit_kwargs.get('maxfev', self.fit_kwargs.get('max_nfev'))
        if value is not None:
            return value
        return get_max_evaluations()

    @max_evaluations.setter
    def max_evaluations(self, value):
        coerced = _coerce_max_evaluations(value)
        if coerced is None:
            self.fit_kwargs.pop('maxfev', None)
            self.fit_kwargs.pop('max_nfev', None)
        else:
            self.fit_kwargs['maxfev'] = coerced

    def __call__(self, model, x, y, z=None, weights=None, statistic='chi2', yerr=None,
                 inplace=True, **kwargs):
        if 'uncertainties' in kwargs:
            raise TypeError("The 'uncertainties' parameter is no longer supported.")
        
        # When called natively through the dynamic patcher, `weights` is present.
        # But for Sherpa, we need yerr to do chi2 fitting via Chi2DataVar.
        # So we reverse-engineer yerr if it's missing but weights are given.
        if statistic == 'poisson':
            raise NotImplementedError("Poisson statistics not supported in Sherpa wrappers yet.")
        if yerr is not None:
            yerr = validate_symmetric_yerr(yerr)
        
        fit_model = model if inplace else model.copy()
        init_values, fit_indices, _ = model_to_fit_params(fit_model)
        
        bounds_list = [getattr(fit_model, n).bounds for n in fit_model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                b[1] if b[1] is not None else np.inf)
                               for b in bounds_list])
        param_bounds = all_bounds[fit_indices]

        tied_info = _get_tied_info(fit_model)
        params_cache = fit_model.parameters.copy()
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
            fit_model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(fit_model, tied_info, params_cache)
            
            resid = (fit_model(x) - y)
            weighted_resid = resid * np.sqrt(weights_sq)
            stat = np.sum(weighted_resid**2)
            return stat, weighted_resid
        
        # Merge call-time kwargs with init-time fit_kwargs
        combined_kwargs = self.fit_kwargs.copy()
        combined_kwargs.update(kwargs)
        
        sherpa_kwargs = {k: v for k, v in combined_kwargs.items() 
                         if k not in ['maxfev', 'max_nfev', 'inplace']}
        sherpa_result = self.opt.fit(statfunc, np.array(init_values), min_vals, max_vals, **sherpa_kwargs)
        
        success = sherpa_result[0]
        fitted_params = sherpa_result[1]
        stat_val = sherpa_result[2] if len(sherpa_result) > 2 else np.nan
        message = sherpa_result[3] if len(sherpa_result) > 3 else ''
        
        sherpa_info = sherpa_result[4] if len(sherpa_result) > 4 and isinstance(sherpa_result[4], dict) else {}
        nfev = sherpa_info.get('nfev', None)
        native_cov = sherpa_info.get('covar')
        
        n_data = len(y)
        n_free = len(init_values)
        stat_name = 'chi2' if yerr is not None else 'leastsq'
        
        self.fit_info = {
            'success': success,
            'nfev': nfev,
            'message': message,
            'param_cov': native_cov,
            'stat': stat_val,
            'cost': stat_val,
            'statmethod': stat_name,
            'nfree': n_free,
            'ndata': n_data,
            'dof': n_data - n_free,
        }
        # Merge Sherpa optimizer info directly into fit_info
        for k, v in sherpa_info.items():
            if k in self.fit_info:
                self.fit_info[k + '_sherpa'] = v
            else:
                self.fit_info[k] = v
        
        params_cache[fit_indices] = fitted_params
        fit_model.parameters = params_cache
        if tied_info:
            _apply_tied_fast(fit_model, tied_info, params_cache)
            
        return fit_model


class SherpaLM(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='levmar', **kwargs)


class SherpaSimplex(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='neldermead', **kwargs)


class SherpaMonCar(SherpaFitter):
    def __init__(self, **kwargs):
        super().__init__(method='moncar', **kwargs)