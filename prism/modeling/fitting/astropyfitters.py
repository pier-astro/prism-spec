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
    __call__ = _wrap_fitter_call(ast_fit.TRFLSQFitter.__call__)


class DogBoxLSQFitter(_PrismAstropyFitterMixin, ast_fit.DogBoxLSQFitter):
    __call__ = _wrap_fitter_call(ast_fit.DogBoxLSQFitter.__call__)


class LevMarLSQFitter(_PrismAstropyFitterMixin, ast_fit.LevMarLSQFitter):
    __call__ = _wrap_fitter_call(ast_fit.LevMarLSQFitter.__call__)


class LinearLSQFitter(_PrismAstropyFitterMixin, ast_fit.LinearLSQFitter):
    __call__ = _wrap_fitter_call(ast_fit.LinearLSQFitter.__call__)


class SimplexLSQFitter(_PrismAstropyFitterMixin, ast_fit.SimplexLSQFitter):
    __call__ = _wrap_fitter_call(ast_fit.SimplexLSQFitter.__call__)


class SLSQPLSQFitter(_PrismAstropyFitterMixin, ast_fit.SLSQPLSQFitter):
    __call__ = _wrap_fitter_call(ast_fit.SLSQPLSQFitter.__call__)