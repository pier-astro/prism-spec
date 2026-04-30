"""Prism wrappers around Astropy fitters.

These classes keep Astropy fitter names inside the Prism namespace while
adding Prism features like ``multifit()``, ``yerr``/``statistic`` support,
normalized diagnostics, and unified ``max_evaluations`` control, without
mutating ``astropy.modeling.fitting`` globally.
"""

from __future__ import annotations

import astropy.modeling.fitting as ast_fit
import numpy as np

from .extension import (
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    _wrap_fitter_call,
    get_max_evaluations,
    set_max_evaluations,
)
from .multifit import MultiFitMixin, _MULTIFIT_STAT_KEYS

__all__ = [
    'TRFLSQFitter',
    'DogBoxLSQFitter',
    'LevMarLSQFitter',
    'LinearLSQFitter',
    'SimplexLSQFitter',
    'SLSQPLSQFitter',
    'get_max_evaluations',
    'set_max_evaluations',
]


class _PrismAstropyFitterMixin(MultiFitMixin):
    """Shared Prism behavior layered on top of native Astropy fitter classes."""

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)

    @property
    def calc_uncertainties(self):
        return getattr(self, '_prism_calc_unc', getattr(self, '_calc_uncertainties', False))

    @calc_uncertainties.setter
    def calc_uncertainties(self, value):
        self._prism_calc_unc = value

    @property
    def verbose(self):
        return getattr(self, '_prism_verbose', False)

    @verbose.setter
    def verbose(self, value):
        self._prism_verbose = value

    @property
    def max_evaluations(self):
        value = getattr(self, '_prism_max_evaluations', None)
        if value is not None:
            return value
        return get_max_evaluations()

    @max_evaluations.setter
    def max_evaluations(self, value):
        self._prism_max_evaluations = _coerce_max_evaluations(value)

    def _multifit_entry_from_fit_info(self):
        fit_info = getattr(self, 'fit_info', {})
        entry = {
            'success': fit_info.get('success', False),
            'nfev': fit_info.get('nfev', np.nan),
            'message': fit_info.get('message', ''),
        }
        for name in _MULTIFIT_STAT_KEYS:
            entry[name] = fit_info.get(name, np.nan)
        return entry

    def _apply_scalar_multifit_overrides(self, model, initpars=None, bounds=None):
        if initpars:
            for name, value in initpars.items():
                getattr(model, name).value = value
        if bounds:
            for name, pair in bounds.items():
                getattr(model, name).bounds = pair
        return model


class TRFLSQFitter(_PrismAstropyFitterMixin, ast_fit.TRFLSQFitter):
    """Trust Region Reflective bounding box optimizer.

    A robust Scipy-based optimizer highly recommended for models with strict 
    parameter bounds. Enhanced in Prism to support error propagation, dynamic 
    evaluation limits, and parallel 2D/3D fitting via ``multifit()``.

    Attributes
    ----------
    calc_uncertainties : bool
        If True, computes the covariance matrix and 1-sigma standard deviations 
        for the fitted parameters, available via the ``.covariance`` and ``.std`` 
        properties. Default is ``False``.
    max_evaluations : int or None
        Maximum number of model evaluations during the optimization. If ``None``, 
        defaults to the global limit specified by ``get_max_evaluations()``.
    verbose : bool
        If True, prints optimization status messages. Default is ``False``.

    Examples
    --------
    >>> from prism.modeling.fitting.astropyfitters import TRFLSQFitter
    >>> from astropy.modeling.models import Gaussian1D
    >>> import numpy as np
    >>> 
    >>> fitter = TRFLSQFitter()
    >>> fitter.calc_uncertainties = True
    >>> fitter.max_evaluations = 2000
    >>> 
    >>> model = Gaussian1D(amplitude=10, mean=5000, stddev=5)
    >>> model.mean.bounds = (4950, 5050)
    >>> 
    >>> # Fake data
    >>> x = np.linspace(4900, 5100, 100)
    >>> y = model(x) + np.random.normal(0, 0.5, 100)
    >>> yerr = np.full_like(x, 0.5)
    >>> 
    >>> best_fit = fitter(model, x, y, yerr=yerr)
    >>> print("Param Errors:", fitter.std)
    """
    __call__ = _wrap_fitter_call(ast_fit.TRFLSQFitter.__call__)


class DogBoxLSQFitter(_PrismAstropyFitterMixin, ast_fit.DogBoxLSQFitter):
    """Dogleg algorithm with rectangular trust regions.
    
    A Scipy-based optimizer supporting parameter bounds.
    """
    __call__ = _wrap_fitter_call(ast_fit.DogBoxLSQFitter.__call__)


class LevMarLSQFitter(_PrismAstropyFitterMixin, ast_fit.LevMarLSQFitter):
    """Levenberg-Marquardt optimizer (Astropy/Scipy wrapper).

    Fast local gradient-based optimizer. Does not formally support parameter bounds 
    in its native scipy.optimize.leastsq implementation.
    """
    __call__ = _wrap_fitter_call(ast_fit.LevMarLSQFitter.__call__)


class LinearLSQFitter(_PrismAstropyFitterMixin, ast_fit.LinearLSQFitter):
    """Exact linear least-squares fitter for linear models."""
    __call__ = _wrap_fitter_call(ast_fit.LinearLSQFitter.__call__)


class SimplexLSQFitter(_PrismAstropyFitterMixin, ast_fit.SimplexLSQFitter):
    """Nelder-Mead (Simplex) local optimizer (Astropy/Scipy wrapper)."""
    __call__ = _wrap_fitter_call(ast_fit.SimplexLSQFitter.__call__)


class SLSQPLSQFitter(_PrismAstropyFitterMixin, ast_fit.SLSQPLSQFitter):
    """Sequential Least Squares Programming (SLSQP) optimizer."""
    __call__ = _wrap_fitter_call(ast_fit.SLSQPLSQFitter.__call__)