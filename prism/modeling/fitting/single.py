"""Single-observation fitting, mutation policy and result construction."""
import copy

from .data import Dataset
from .results import FitResult, _clear_model_estimates


class _SingleFitMixin:
    def __call__(self, model, x, y=None, z=None, weights=None, yerr=None,
                 statistic='chi2', inplace=False, **kwargs):
        """Fit one model and return a detached FitResult.

        Parameters
        ----------
        model : Model or astropy.modeling.Model
            Source model. Explicit pipes remain part of this model.
        x, y : array-like or Dataset
            Coordinates and observations, or a Dataset with y omitted. Set
            errors, weights, mask and response on the Dataset when using it.
        yerr, weights, mask : array-like, optional
            Standard deviations, residual multipliers and inclusion mask.
            Explicit weights take precedence over yerr. mask is a keyword.
        statistic : {'chi2', 'leastsq'}, optional
            Weighted or unweighted least squares; Poisson is not supported.
        inplace : bool, optional
            If True, update parameter values in the supplied model and any
            fitted Dataset response. Default False copies their graph together.
            Uncertainties and diagnostics belong to the returned result.
        **kwargs
            Optimizer controls, including filter_non_finite and evaluation budget.

        Returns
        -------
        FitResult
            Snapshot with source model, Dataset, observable predict(x=None),
            fitted parameter estimates, covariance and fit diagnostics.
        """
        if z is not None:
            if isinstance(x, Dataset):
                raise TypeError('A Dataset replaces coordinate and observed-value arguments.')
            raise NotImplementedError('Numerical fitters are one-dimensional; use an Astropy spatial adapter for z.')
        mask = kwargs.pop('mask', None)
        if isinstance(x, Dataset):
            if y is not None or yerr is not None or weights is not None or mask is not None:
                raise TypeError('Set y, yerr, weights and mask on the Dataset when fitting a Dataset.')
            data = x
        else:
            data = Dataset(x, y, yerr=yerr, weights=weights, mask=mask)
        source = model
        if not inplace:
            # Copy the entire parameter graph once, retaining cross-links between
            # source and response. Data arrays are only copied in the result.
            source, response = copy.deepcopy((model, data.response))
            data = copy.copy(data)
            data.response = response
        fitted = self._fit_single(source, data, statistic=statistic,
                                  inplace=True, **kwargs)
        from ..composition import Model
        if isinstance(source, Model) and fitted is not source:
            source._evaluation_parent = fitted
        # Store the selection actually fitted, including explicit filtering and
        # zero weights. Reusing a result must not silently change its sample set.
        data = copy.copy(data)
        data.mask = self._fit_mask.copy()
        result = FitResult({'data': (source, data)}, fitted, self.fit_info,
            {'data': {key: self.fit_info[key] for key in ('stat', 'ndata')}},
            _single=True)
        # In-place fitting updates values, never leaves estimates from this or
        # an earlier fit attached to the caller's model/response objects.
        _clear_model_estimates(fitted)
        return result
