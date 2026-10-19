"""
Sherpa-based fitters for prism.modeling.fitting
"""

import numpy as np
from .extension import _validate_spectral_inputs
from astropy.modeling.fitting import Fitter
from .extension import (
    _attach_covariance,
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    _resolve_max_evaluations,
    get_max_evaluations,
)
from .multifit import MultiFitMixin
from .problem import _FitProblem
from .data import _prepare_fit_data

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
        """
        Parameters
        ----------
        method : {'levmar', 'neldermead', 'moncar'}, optional
            The Sherpa optimization method to use. Default is ``'levmar'``.
        calc_uncertainties : bool, optional
            Compute parameter uncertainties after fitting. Default is ``False``.
        verbose : bool, optional
            Enable verbose output from the Sherpa optimizer. Default is ``False``.
        **kwargs
            Additional arguments passed to the specific Sherpa optimizer.
        """
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
        """int or None: Maximum number of model evaluations during the fit."""
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
        """Perform the fit.

        Parameters
        ----------
        model : astropy.modeling.core.Model
            The model to fit to the data.
        x, y : array-like
            Independent and dependent data arrays.
        z : array-like, optional
            Secondary independent data array for 2D models. Default is ``None``.
        weights : array-like, optional
            Residual weights (1 / yerr). Explicit weights take precedence over `yerr`.
        statistic : str, optional
            The fit statistic to minimize. Sherpa wrappers typically use a chi-square
            equivalent when errors/weights are provided, or least-squares otherwise.
            Poisson is not yet supported. Default is ``'chi2'``.
        yerr : array-like, optional
            1-sigma uncertainties on `y`. Default is ``None``.
        inplace : bool, optional
            Update the model in place. Default is ``True``.
        **kwargs
            Extra optimization arguments to merge with those provided at initialization.

        Returns
        -------
        astropy.modeling.core.Model
            The fitted model. If `inplace=False`, a new copy is returned.
        """
        _validate_spectral_inputs(model, x, y, z, yerr, weights, statistic=statistic)
        if 'uncertainties' in kwargs:
            raise TypeError("The 'uncertainties' parameter is no longer supported.")
        
        x, y, yerr, weights, valid = _prepare_fit_data(
            x, y, yerr, weights, kwargs.pop('mask', None),
            filter_non_finite=kwargs.pop('filter_non_finite', False))
        weighted = weights is not None or yerr is not None
        # Explicit residual weights take precedence, including rejection masks.
        # Preserve the previous Sherpa arithmetic on ordinary valid samples.
        with np.errstate(divide='ignore', invalid='ignore'):
            if weights is not None:
                staterr = np.where(valid, 1.0 / weights, np.inf)
            elif yerr is not None:
                staterr = np.where(valid, yerr, np.inf)
            else:
                staterr = sherpa.stats.LeastSq().calc_staterror(y)
            residual_weights = np.sqrt(1.0 / (staterr ** 2))

        fit_model = model if inplace else model.copy()
        problem = _FitProblem(fit_model, x, y, residual_weights, valid)
        parameters = problem.parameters
        init_values = parameters.values
        param_bounds = parameters.bounds
        huge = np.finfo(float).max
        min_vals = np.where(np.isinf(param_bounds[:, 0]), -huge, param_bounds[:, 0])
        max_vals = np.where(np.isinf(param_bounds[:, 1]), huge, param_bounds[:, 1])

        def statfunc(p):
            weighted_resid = problem.residual(p)
            return np.sum(weighted_resid**2), weighted_resid

        # Merge call-time kwargs with init-time fit_kwargs
        combined_kwargs = self.fit_kwargs.copy()
        budget_aliases = ('max_evaluations', 'max_nfev', 'maxiter', 'maxfev', 'maxfun')
        if any(name in kwargs for name in budget_aliases):
            for name in budget_aliases:
                combined_kwargs.pop(name, None)
        combined_kwargs.update(kwargs)
        
        max_evaluations = _resolve_max_evaluations(self, combined_kwargs)
        combined_kwargs.pop('inplace', None)
        if max_evaluations is not None:
            combined_kwargs['maxfev'] = max_evaluations
        combined_kwargs.setdefault('verbose', int(self.verbose))
        unknown = combined_kwargs.keys() - self.opt.config.keys()
        if unknown:
            raise TypeError(f"Unsupported Sherpa options: {', '.join(sorted(unknown))}")

        # Sherpa accepts optimizer settings through config, not fit keywords.
        # Call-time overrides must not change subsequent fits on this instance.
        original_config = self.opt.config.copy()
        try:
            self.opt.config.update(combined_kwargs)
            sherpa_result = self.opt.fit(
                statfunc, np.array(init_values), min_vals, max_vals)
        finally:
            self.opt.config.update(original_config)

        success = sherpa_result[0]
        fitted_params = sherpa_result[1]
        stat_val = sherpa_result[2] if len(sherpa_result) > 2 else np.nan
        message = sherpa_result[3] if len(sherpa_result) > 3 else ''
        
        sherpa_info = sherpa_result[4] if len(sherpa_result) > 4 and isinstance(sherpa_result[4], dict) else {}
        nfev = sherpa_info.get('nfev', None)
        native_cov = sherpa_info.get('covar')

        # Some Sherpa versions map MINPACK budget exhaustion to success.
        # Reaching an explicit budget is conservatively reported as incomplete.
        if max_evaluations is not None and nfev is not None and nfev >= max_evaluations:
            success = False
            message = f"Evaluation budget reached (maxfev={max_evaluations})."
            native_cov = None
        
        n_data = problem.ndata
        n_free = len(init_values)
        stat_name = 'chi2' if weighted else 'leastsq'
        
        self.fit_info = {
            'success': success,
            'nfev': nfev,
            'message': message,
            'param_cov': native_cov,
            'stat': stat_val,
            'cost': 0.5 * stat_val,
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
        
        problem.finish(fitted_params)

        return _attach_covariance(fit_model, native_cov)


class SherpaLM(SherpaFitter):
    """Levenberg-Marquardt optimizer using the Sherpa backend.

    An efficient gradient-based local optimizer. Excellent for well-behaved,
    smooth functions when good initial parameter guesses are available.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Optimization parameters passed to `sherpa.optmethods.LevMar`.
        """
        super().__init__(method='levmar', **kwargs)


class SherpaSimplex(SherpaFitter):
    """Nelder-Mead (Simplex) optimizer using the Sherpa backend.

    A gradient-free local optimizer that is robust to noisy objective functions,
    though it typically converges slower than Levenberg-Marquardt.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Optimization parameters passed to `sherpa.optmethods.NelderMead`.
        """
        super().__init__(method='neldermead', **kwargs)


class SherpaMonCar(SherpaFitter):
    """Monte Carlo global optimizer using the Sherpa backend.

    A global optimization method that randomly samples the parameter space.
    Best for complex likelihood surfaces with multiple local minima.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Optimization parameters passed to `sherpa.optmethods.MonCar`.
        """
        super().__init__(method='moncar', **kwargs)