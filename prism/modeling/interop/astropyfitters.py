"""Prism wrappers around Astropy fitters.

These classes keep Astropy fitter names inside the Prism namespace while
adding Prism features like ``multifit()``, ``yerr``/``statistic`` support,
normalized diagnostics, and unified ``max_evaluations`` control, without
mutating ``astropy.modeling.fitting`` globally.
"""

from __future__ import annotations

import astropy.modeling.fitting as ast_fit
import numpy as np

from ..fitting.extension import (
    _attach_covariance,
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    _wrap_fitter_call,
    get_max_evaluations,
    set_max_evaluations,
)
from ..fitting.multifit import MultiFitMixin
from ..fitting.utils import (
    _free_parameter_indices,
    _reduce_tied_analytic_jacobian,
)

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

    @staticmethod
    def _prepare_links(model, x, y, z):
        """Retain numerical-unit conventions while Astropy strips quantities."""
        from astropy import units as u
        from ..composition import Model
        if isinstance(model, Model):
            raise TypeError('Use TRF, Sherpa, or SeparableTRF for a native Prism Model; Astropy fitters accept Astropy models.')
        from ..parameters import _Expression
        from ..fitting.utils import _get_tied_info
        ties = _get_tied_info(model)
        if not any(isinstance(expression, _Expression) for expression, _, _ in ties):
            return
        coordinates = [x] if z is None else [x, y]
        inputs = {name: getattr(value, 'unit', u.dimensionless_unscaled)
                  for name, value in zip(model.inputs, coordinates)}
        outputs = {model.outputs[0]: getattr(y if z is None else z, 'unit', u.dimensionless_unscaled)}
        units = model._parameter_units_for_data_units(inputs, outputs)
        for name, unit in units.items():
            getattr(model, name)._prism_evaluation_unit = unit

    @staticmethod
    def _restore_link_units(model):
        """Restore dimensional parameter units retained for declared links."""
        from astropy import units as u
        for name in model.param_names:
            parameter = getattr(model, name)
            unit = getattr(parameter, '_prism_evaluation_unit', None)
            if parameter.unit is None and unit is not None and unit != u.dimensionless_unscaled:
                setattr(model, name, parameter.value * unit)

    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z=None, fit_param_indices=None):
        """Apply linked-parameter chain rules only inside Prism's adapters."""
        if not model.has_tied:
            return ast_fit.TRFLSQFitter._wrap_deriv(
                params, model, weights, x, y, z=z,
                fit_param_indices=fit_param_indices)

        ast_fit.fitter_to_model_params(model, params)
        if z is None:
            full = model.fit_deriv(x, *model.parameters)
        else:
            full = [np.ravel(item) for item in model.fit_deriv(x, y, *model.parameters)]
        corrected = _reduce_tied_analytic_jacobian(model, full)
        weighted = corrected if weights is None else np.ravel(weights) * corrected
        indices = _free_parameter_indices(model) if fit_param_indices is None else fit_param_indices
        residues = weighted[np.asarray(indices, dtype=int)]
        if not model.col_fit_deriv:
            residues = residues.T
        return [np.ravel(item) for item in residues]


def _attach_linear_lsq_covariance(fitter, model):
    cov = getattr(getattr(model, 'cov_matrix', None), 'cov_matrix', None)
    cov = None if cov is None else np.asarray(cov, dtype=float)
    fitter.fit_info['param_cov'] = cov
    return _attach_covariance(model, cov)


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
    >>> from prism.modeling.interop.astropyfitters import TRFLSQFitter
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
    """Exact linear least-squares fitter for linear models.

    Prism enables covariance estimation by default for this wrapper because
    Astropy's linear solver can derive it directly from the design matrix with
    little additional overhead. Prism also exposes that covariance even when
    ``calc_uncertainties=False`` was requested, because it is a direct product
    of the linear solve rather than an extra expensive estimation step. The
    resulting covariance is exposed through
    ``fit_info['param_cov']``, ``fitter.covariance``, and per-parameter
    ``.std`` attributes, matching Prism's non-linear fitter wrappers.
    """

    _wrapped_call = _wrap_fitter_call(ast_fit.LinearLSQFitter.__call__)

    def __init__(self, calc_uncertainties=True, **kwargs):
        ast_fit.LinearLSQFitter.__init__(self, calc_uncertainties=True, **kwargs)
        self.calc_uncertainties = calc_uncertainties

    def __call__(self, model, x, y, z=None, weights=None, max_nfev=None,
                 yerr=None, statistic='chi2', **kwargs):
        fitted = self._wrapped_call(
            model, x, y, z=z, weights=weights, yerr=yerr, statistic=statistic, **kwargs
        )
        _attach_linear_lsq_covariance(self, fitted)
        return fitted


class SimplexLSQFitter(_PrismAstropyFitterMixin, ast_fit.SimplexLSQFitter):
    """Nelder-Mead (Simplex) local optimizer (Astropy/Scipy wrapper)."""
    __call__ = _wrap_fitter_call(ast_fit.SimplexLSQFitter.__call__)


class SLSQPLSQFitter(_PrismAstropyFitterMixin, ast_fit.SLSQPLSQFitter):
    """Sequential Least Squares Programming (SLSQP) optimizer."""
    __call__ = _wrap_fitter_call(ast_fit.SLSQPLSQFitter.__call__)
