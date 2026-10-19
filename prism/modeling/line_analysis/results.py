"""Scalar and spatial containers for line measurements and their units."""
import numpy as np
import pandas as pd
import astropy.units as u
from ..metrics import Metric

_METRIC_NAMES = [
    'flux', 'peak_value', 'peak_position',
    'fwhm', 'hw_blue', 'hw_red', 'asymmetry',
    'moment1', 'moment2', 'sigma',
    'skewness', 'kurtosis',
    'fw10m', 'fw20m', 'fw80m', 'bisector_span',
]


class _BoundsProxy:
    __slots__ = ('min', 'max')
    def __init__(self, min_val=None, max_val=None):
        self.min = min_val
        self.max = max_val

_NO_BOUNDS = _BoundsProxy()
_POSITIVE = _BoundsProxy(min_val=0.0)

_METRIC_BOUNDS = {
    'fwhm': _POSITIVE, 'hw_blue': _POSITIVE, 'hw_red': _POSITIVE,
    'sigma': _POSITIVE, 'moment2': _POSITIVE,
    'fw10m': _POSITIVE, 'fw20m': _POSITIVE, 'fw80m': _POSITIVE,
}


def _normalize_unit(unit):
    if unit in (None, ''):
        return None
    return u.Unit(unit)


def _metric_units(axis_unit=None, output_unit=None):
    axis_unit = _normalize_unit(axis_unit)
    output_unit = _normalize_unit(output_unit)
    flux_unit = None if output_unit is None or axis_unit is None else output_unit * axis_unit
    return {
        'flux': flux_unit,
        'peak_value': output_unit,
        'peak_position': axis_unit,
        'fwhm': axis_unit,
        'hw_blue': axis_unit,
        'hw_red': axis_unit,
        'asymmetry': None,
        'moment1': axis_unit,
        'moment2': None if axis_unit is None else axis_unit ** 2,
        'sigma': axis_unit,
        'skewness': None,
        'kurtosis': None,
        'fw10m': axis_unit,
        'fw20m': axis_unit,
        'fw80m': axis_unit,
        'bisector_span': axis_unit,
    }


def _as_metric(value, unit=None) -> Metric:
    unit = _normalize_unit(unit)
    if isinstance(value, Metric):
        if value.unit is None or unit is None or value.unit == unit:
            return Metric(
                value=float(value.value),
                std=None if value.std is None else float(value.std),
                lolim=None if value.lolim is None else float(value.lolim),
                uplim=None if value.uplim is None else float(value.uplim),
                unit=value.unit if unit is None else unit)
        scale = (1.0 * value.unit).to_value(unit)
        return Metric(
            value=float(value.value) * scale,
            std=None if value.std is None else (
                float(value.std) * scale if np.isfinite(value.std) else float('nan')
            ),
            lolim=None if value.lolim is None else (
                float(value.lolim) * scale if np.isfinite(value.lolim) else float('nan')
            ),
            uplim=None if value.uplim is None else (
                float(value.uplim) * scale if np.isfinite(value.uplim) else float('nan')
            ),
            unit=unit,
        )
    if isinstance(value, u.Quantity):
        target_unit = value.unit if unit is None else unit
        return Metric(value=float(value.to_value(target_unit)), unit=target_unit)
    return Metric(value=float(value) if value is not None else float('nan'), unit=unit)

# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------

class LineResult:
    """Scalar measurements for a selected line profile.

    Parameters
    ----------
    selector : str
        User-facing line selector used to build the measurement.
    mode : str
        Measurement mode label stored for provenance.
    wave_min : float
        Lower bound of the sampled spectral window.
    wave_max : float
        Upper bound of the sampled spectral window.
    n_grid : int
        Number of grid samples used to evaluate the profile.
    metrics : dict
        Mapping from metric name to scalar or ``Metric`` value.
    metric_units : dict, optional
        Explicit units for each metric. Default is ``None``.
    axis_unit : str or astropy.units.Unit, optional
        Unit of the spectral axis. Default is ``None``.

    Returns
    -------
    LineResult
        Container whose public attributes expose scalar ``Metric`` objects for
        flux, widths, centroid, and higher-order line-shape summaries.

    Notes
    -----
    Prism evaluates the selected line on a 1-D grid, then derives geometric and
    moment-based summaries from the sampled profile. Flux is integrated with the
    trapezoidal rule, width metrics are measured from fractional-height crossing
    points, and central moments are computed directly from the sampled line shape.

    Examples
    --------
    >>> selected = select_line(model, 'Hb4861')
    >>> result = measure_line(selected, x=wave)
    >>> result.flux.value
    >>> result.fwhm.value
    """

    def __init__(self, selector, mode, wave_min, wave_max, n_grid, metrics,
                 metric_units=None, axis_unit=None):
        self.selector = str(selector)
        self.mode = str(mode)
        self.wave_min = float(wave_min)
        self.wave_max = float(wave_max)
        self.n_grid = int(n_grid)
        self.axis_unit = _normalize_unit(axis_unit)
        self.metric_units = {
            name: _normalize_unit(unit)
            for name, unit in (metric_units or {}).items()
        }
        for name in _METRIC_NAMES:
            setattr(
                self, name,
                _as_metric(metrics.get(name, float('nan')),
                           unit=self.metric_units.get(name)))

    def to_frame(self, metrics=None) -> pd.DataFrame:
        """Return the stored metrics as a tidy dataframe.

        Parameters
        ----------
        metrics : sequence of str, optional
            Subset of metric names to include. Default is all available public
            metrics.

        Returns
        -------
        pandas.DataFrame
            Table indexed by metric name with ``value``, ``std``, ``lolim``,
            ``uplim``, and ``unit`` columns.
        """
        names = list(metrics) if metrics is not None else _METRIC_NAMES
        rows = {}
        for name in names:
            m = getattr(self, name, Metric(value=float('nan')))
            rows[name] = {
                'value': m.value, 'std': m.std,
                'lolim': m.lolim, 'uplim': m.uplim,
                'unit': m.unit}
        return pd.DataFrame(rows).T

    def __repr__(self):
        return (
            f"LineResult(selector={self.selector!r}, "
            f"flux={self.flux.value:.4g}, fwhm={self.fwhm.value:.4g})")


class MultiLineMeasurements:
    """Array-valued line measurements over a spatial grid.

    Parameters
    ----------
    selector : str
        User-facing line selector used for the measurement.
    mode : str
        Measurement mode label stored for provenance.
    shape : tuple of int
        Spatial shape of the batched result.
    metric_units : dict, optional
        Explicit units for each metric. Default is ``None``.
    axis_unit : str or astropy.units.Unit, optional
        Unit of the spectral axis. Default is ``None``.
    **metric_arrays
        Metric maps keyed by public metric name.

    Notes
    -----
    Each metric is stored as an ``ndarray`` with the same spatial shape as the
    parent ``MultiFitResult``. Use :meth:`get_measurement` to recover the scalar
    ``LineResult`` view for one spaxel.
    """

    def __init__(self, selector, mode, shape, metric_units=None, axis_unit=None,
                 **metric_arrays):
        self.selector = str(selector)
        self.mode = str(mode)
        self.shape = tuple(shape)
        self.axis_unit = _normalize_unit(axis_unit)
        self.metric_units = {
            name: _normalize_unit(unit)
            for name, unit in (metric_units or {}).items()
        }
        for name in _METRIC_NAMES:
            setattr(self, name, np.asarray(
                metric_arrays.get(name, np.full(shape, np.nan)), dtype=float))
        self.wave_min = np.asarray(
            metric_arrays.get('wave_min', np.full(shape, np.nan)), dtype=float)
        self.wave_max = np.asarray(
            metric_arrays.get('wave_max', np.full(shape, np.nan)), dtype=float)
        self.n_grid = np.asarray(
            metric_arrays.get('n_grid', np.zeros(shape)), dtype=int)

    def get_measurement(self, index) -> LineResult:
        """Return the scalar measurement for one spatial element.

        Parameters
        ----------
        index : int or tuple of int
            Flat or multi-index selecting one spaxel.

        Returns
        -------
        LineResult
            Scalar view into the batched measurement maps.
        """
        idx = (tuple(index) if isinstance(index, tuple)
               else np.unravel_index(int(index), self.shape))
        metrics = {name: float(getattr(self, name)[idx])
                   for name in _METRIC_NAMES}
        return LineResult(
            selector=self.selector, mode=self.mode,
            wave_min=float(self.wave_min[idx]),
            wave_max=float(self.wave_max[idx]),
            n_grid=int(self.n_grid[idx]), metrics=metrics,
            metric_units=self.metric_units, axis_unit=self.axis_unit)

