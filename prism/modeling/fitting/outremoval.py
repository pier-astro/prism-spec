import numpy as np
from astropy.stats import sigma_clip
from astropy.modeling.fitting import Fitter

from .extension import _fitter_covariance, _fitter_stdevs, _validate_native_inputs
from .data import _prepare_fit_data
from .multifit import MultiFitMixin
from .single import _SingleFitMixin
from .data import _resolve_fit_inputs

__all__ = ['FittingWithOutlierRemoval']

class FittingWithOutlierRemoval(_SingleFitMixin, MultiFitMixin, Fitter):
    """
    Iterative outlier-rejection wrapper around any Prism fitter.

    Parameters
    ----------
    fitter : astropy.modeling.fitting.Fitter
        Wrapped fitter called at each iteration.
    outlier_func : callable, optional
        Function returning a mask or masked array from residuals. Default is
        :func:`astropy.stats.sigma_clip`.
    niter : int, optional
        Number of rejection iterations after the initial fit. Default is ``3``.
    **outlier_kwargs
        Additional keyword arguments passed to ``outlier_func``.

    Notes
    -----
    The wrapper alternates between fitting and residual clipping. Newly flagged
    outliers are assigned zero weight in subsequent iterations, so the wrapped
    fitter sees a standard weighted least-squares problem at every step. Supplied
    errors become residual weights before clipping, so they cannot reintroduce
    rejected samples. The outlier function receives masked raw residuals and
    may return a masked array or a boolean rejection mask (True means rejected).
    Exceptions from the inner fitter propagate.
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)

    def __init__(self, fitter, outlier_func=sigma_clip, niter=3, **outlier_kwargs):
        self._fitter = fitter
        self.outlier_func = outlier_func
        self.niter = niter
        self.outlier_kwargs = outlier_kwargs
        self.outlier_mask = None

    @property
    def _estimate_covariance(self):
        return getattr(self._fitter, '_estimate_covariance', getattr(self._fitter, 'calc_uncertainties', False))


    @property
    def verbose(self):
        return getattr(self._fitter, 'verbose', False)

    @verbose.setter
    def verbose(self, value):
        self._fitter.verbose = value

    @property
    def max_evaluations(self):
        return getattr(self._fitter, 'max_evaluations', None)

    @max_evaluations.setter
    def max_evaluations(self, value):
        self._fitter.max_evaluations = value

    def _multifit_spawn_fitter(self):
        # Override to ensure the nested fitter is safely cloned for parallel workers
        import copy
        worker_fitter = copy.copy(self)
        if hasattr(self._fitter, '_multifit_spawn_fitter'):
            worker_fitter._fitter = self._fitter._multifit_spawn_fitter()
        else:
            worker_fitter._fitter = copy.copy(self._fitter)
        return worker_fitter

    def _fit_single(self, model, x, y=None, z=None, weights=None, **kwargs):
        yerr = kwargs.pop('yerr', None)
        model, x, y, yerr, weights, mask = _resolve_fit_inputs(
            model, x, y, z, yerr, weights, kwargs.pop('mask', None))
        _validate_native_inputs(model, x, y, z, yerr, weights)
        x, y, yerr, weights, valid = _prepare_fit_data(
            x, y, yerr, weights, mask,
            filter_non_finite=kwargs.pop('filter_non_finite', False))
        if self.niter < 0:
            raise ValueError('niter must be nonnegative.')
        if weights is not None:
            weights = np.array(weights, dtype=float, copy=True)
        elif yerr is not None:
            weights = np.zeros_like(y, dtype=float)
            np.divide(1., np.asarray(yerr), out=weights, where=valid)
        else:
            weights = np.ones_like(y, dtype=float)
        weights[~valid] = 0.
        rejected = ~valid

        for iteration in range(self.niter + 1):
            # Pass one effective weighting, so errors cannot undo rejection.
            fit = getattr(self._fitter, '_fit_single', self._fitter)
            best_model = fit(model, x, y, z=z, weights=weights.copy(), **kwargs)
            self.fit_info = getattr(self._fitter, 'fit_info', {})
            if not self.fit_info.get('success', False) or iteration == self.niter:
                break
            coordinates = np.isfinite(x)
            if np.all(coordinates):
                residuals = np.ma.array(y - best_model(x), mask=rejected)
            else:
                residuals = np.ma.masked_all(y.shape)
                residuals[coordinates] = y[coordinates] - best_model(x[coordinates])
                residuals.mask |= rejected
            clipped = self.outlier_func(residuals, **self.outlier_kwargs)
            if np.ma.isMaskedArray(clipped):
                current_mask = np.ma.getmaskarray(clipped)
            else:
                current_mask = np.asarray(clipped)
                if current_mask.dtype != bool:
                    raise ValueError('outlier_func must return a masked array or boolean mask.')
            if current_mask.shape != y.shape:
                raise ValueError('Outlier mask must have the same shape as y.')
            current_mask = rejected | current_mask
            if np.array_equal(current_mask, rejected):
                break
            rejected = current_mask
            weights[rejected] = 0.

        self.outlier_mask = rejected
        self._fit_mask = ~rejected
        return best_model
