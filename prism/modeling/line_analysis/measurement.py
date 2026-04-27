"""Line measurement and sampling helpers for prism.modeling.line_analysis."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.modeling.fitting import model_to_fit_params

from ..models.components import get_components
from ..models.lines import Metric
from ..fitting.uncertainty.resample import extract_limits

from .selection import (
    _SelectionEntry, _SelectionParameter,
    _evaluate_component_profile, _evaluate_linegroup_template,
    SelectedLineProfile, SelectedLineCollection,
    _linegroup_template_index,
    _is_multifit_like, select_line,
)

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
                value=float(value.value), std=float(value.std),
                lolim=float(value.lolim), uplim=float(value.uplim),
                unit=value.unit if unit is None else unit)
        scale = (1.0 * value.unit).to_value(unit)
        return Metric(
            value=float(value.value) * scale,
            std=float(value.std) * scale if np.isfinite(value.std) else float('nan'),
            lolim=float(value.lolim) * scale if np.isfinite(value.lolim) else float('nan'),
            uplim=float(value.uplim) * scale if np.isfinite(value.uplim) else float('nan'),
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

# ---------------------------------------------------------------------------
# Profile metrics
# ---------------------------------------------------------------------------

def _crossings_at_fraction(x, y, fraction):
    if x.size < 3:
        return float('nan'), float('nan')
    peak_index = int(np.argmax(y))
    peak_value = float(y[peak_index])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        return float('nan'), float('nan')
    level = fraction * peak_value

    left_x = float('nan')
    for i in range(peak_index, 0, -1):
        if y[i - 1] < level:
            denom = y[i] - y[i - 1]
            if denom != 0.0:
                left_x = float(
                    x[i - 1] + (level - y[i - 1]) / denom * (x[i] - x[i - 1]))
            break

    right_x = float('nan')
    for i in range(peak_index, x.size - 1):
        if y[i + 1] < level:
            denom = y[i + 1] - y[i]
            if denom != 0.0:
                right_x = float(
                    x[i] + (level - y[i]) / denom * (x[i + 1] - x[i]))
            break

    return left_x, right_x


def _compute_profile_metrics(x_arr, y_arr) -> dict:
    flux = float(_trapezoid(y_arr, x_arr))
    peak_index = int(np.argmax(y_arr))
    peak_value = float(y_arr[peak_index])
    peak_position = float(x_arr[peak_index])

    moment1 = moment2 = sigma = skewness = kurtosis = float('nan')
    if np.isfinite(flux) and flux != 0.0:
        moment1 = float(_trapezoid(x_arr * y_arr, x_arr) / flux)
        residuals = x_arr - moment1
        moment2 = float(_trapezoid(residuals ** 2 * y_arr, x_arr) / flux)
        if moment2 >= 0.0:
            sigma = float(np.sqrt(moment2))
            if sigma > 0.0:
                skewness = float(
                    _trapezoid(residuals ** 3 * y_arr, x_arr)
                    / (flux * sigma ** 3))
                kurtosis = float(
                    _trapezoid(residuals ** 4 * y_arr, x_arr)
                    / (flux * sigma ** 4)) - 3.0

    fwhm_l, fwhm_r = _crossings_at_fraction(x_arr, y_arr, 0.5)
    fwhm = (float(fwhm_r - fwhm_l)
            if (np.isfinite(fwhm_l) and np.isfinite(fwhm_r)) else float('nan'))
    hw_blue = (float(peak_position - fwhm_l)
               if np.isfinite(fwhm_l) else float('nan'))
    hw_red = (float(fwhm_r - peak_position)
              if np.isfinite(fwhm_r) else float('nan'))
    asymmetry = (float((hw_red - hw_blue) / fwhm)
                 if (np.isfinite(fwhm) and fwhm > 0.0) else float('nan'))

    fw10m_l, fw10m_r = _crossings_at_fraction(x_arr, y_arr, 0.10)
    fw20m_l, fw20m_r = _crossings_at_fraction(x_arr, y_arr, 0.20)
    fw80m_l, fw80m_r = _crossings_at_fraction(x_arr, y_arr, 0.80)
    fw10m = (float(fw10m_r - fw10m_l)
             if (np.isfinite(fw10m_l) and np.isfinite(fw10m_r)) else float('nan'))
    fw20m = (float(fw20m_r - fw20m_l)
             if (np.isfinite(fw20m_l) and np.isfinite(fw20m_r)) else float('nan'))
    fw80m = (float(fw80m_r - fw80m_l)
             if (np.isfinite(fw80m_l) and np.isfinite(fw80m_r)) else float('nan'))

    bisector_20 = (0.5 * (fw20m_l + fw20m_r)
                   if (np.isfinite(fw20m_l) and np.isfinite(fw20m_r))
                   else float('nan'))
    bisector_80 = (0.5 * (fw80m_l + fw80m_r)
                   if (np.isfinite(fw80m_l) and np.isfinite(fw80m_r))
                   else float('nan'))
    bisector_span = (float(bisector_80 - bisector_20)
                     if (np.isfinite(bisector_20) and np.isfinite(bisector_80))
                     else float('nan'))

    return {
        'flux': flux, 'peak_value': peak_value, 'peak_position': peak_position,
        'fwhm': fwhm, 'hw_blue': hw_blue, 'hw_red': hw_red,
        'asymmetry': asymmetry,
        'moment1': moment1, 'moment2': moment2, 'sigma': sigma,
        'skewness': skewness, 'kurtosis': kurtosis,
        'fw10m': fw10m, 'fw20m': fw20m, 'fw80m': fw80m,
        'bisector_span': bisector_span,
    }


def _compute_ew_numerical(x_arr, y_line, y_continuum) -> float:
    safe = np.where(y_continuum > 0.0, y_continuum, np.nan)
    return float(_trapezoid(y_line / safe, x_arr))

# ---------------------------------------------------------------------------
# Selection resolution (inline to avoid circular import)
# ---------------------------------------------------------------------------

def _resolve_selection(model_or_selection, selector=None,
                       components=None, additive=True, index=None):
    if isinstance(model_or_selection, (SelectedLineProfile, SelectedLineCollection)):
        return model_or_selection
    return select_line(
        model_or_selection, selector=selector,
        components=components, additive=additive, index=index)

# ---------------------------------------------------------------------------
# measure_line
# ---------------------------------------------------------------------------

def measure_line(model_or_selection, selector=None,
                 components=None, additive=True, x=None, window=None,
                 num=4096, index=None) -> LineResult:
    """Measure geometric and moment-based properties of a selected line.

    Parameters
    ----------
    model_or_selection : astropy.modeling.Model or SelectedLineProfile or SelectedLineCollection
        Source model or precomputed line selection.
    selector : str, optional
        Line tag resolved by :func:`select_line`. Default is ``None`` when a
        ``SelectedLineProfile`` is passed directly.
    components : sequence of str, optional
        Optional component-name filter applied during selection. Default is
        ``None``.
    additive : bool, optional
        If ``True`` (default), select within the additive decomposition of the
        model.
    x : array-like or astropy.units.Quantity, optional
        Explicit spectral grid used for the measurement. Default is ``None``.
    window : tuple of float, optional
        Spectral interval used to build an internal grid when ``x`` is omitted.
        Default is ``None``, in which case Prism infers a window from the line
        centers and widths.
    num : int, optional
        Number of samples for the internally generated grid. Default is ``4096``.
    index : int or tuple of int, optional
        Spatial index used when ``model_or_selection`` is a ``MultiFitResult``.
        Default is ``None``.

    Returns
    -------
    LineResult or MultiLineMeasurements
        Scalar measurement for a single profile, or spatial maps for a selected
        line collection.

    Notes
    -----
    The selected profile is evaluated on a 1-D grid and summarized numerically.
    Flux is integrated with the trapezoidal rule. Widths such as FWHM and FW10M
    come from linear interpolation at fixed fractions of the peak height. The
    first four central moments are then used to derive centroid, variance,
    skewness, and kurtosis in a way that remains agnostic to the analytic model
    family once the profile has been selected.

    Examples
    --------
    >>> selected = select_line(model, 'Hb4861')
    >>> result = measure_line(selected, x=wave)
    >>> result.peak_position.value
    >>> result.fwhm.value
    """
    selection = _resolve_selection(
        model_or_selection, selector=selector,
        components=components, additive=additive, index=index)

    if isinstance(selection, SelectedLineCollection):
        arrays = {name: np.full(selection.shape, np.nan, dtype=float)
                  for name in _METRIC_NAMES}
        wave_min = np.full(selection.shape, np.nan, dtype=float)
        wave_max = np.full(selection.shape, np.nan, dtype=float)
        n_grid = np.zeros(selection.shape, dtype=int)
        metric_units = None
        axis_unit = None
        for fi in range(selection.source_result.n_spaxels):
            si = np.unravel_index(fi, selection.shape)
            m = selection.get_profile(fi).measure(x=x, window=window, num=num)
            for name in _METRIC_NAMES:
                arrays[name][si] = getattr(m, name).value
            wave_min[si] = m.wave_min
            wave_max[si] = m.wave_max
            n_grid[si] = m.n_grid
            if metric_units is None:
                metric_units = m.metric_units
                axis_unit = m.axis_unit
        return MultiLineMeasurements(
            selector=selection.selector, mode='auto',
            shape=selection.shape,
            metric_units=metric_units, axis_unit=axis_unit,
            wave_min=wave_min, wave_max=wave_max, n_grid=n_grid, **arrays)

    if x is None:
        if window is None:
            window = selection.infer_window()
        x_arr = np.linspace(
            float(window[0]), float(window[1]), int(num), dtype=float)
    else:
        x_arr = selection._coerce_domain_axis(x)
        if x_arr.ndim != 1 or x_arr.size < 3:
            raise ValueError(
                "x must be a 1-D spectral grid with at least three samples.")

    y_arr = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    raw_metrics = _compute_profile_metrics(x_arr, y_arr)
    units = _metric_units(selection.axis_unit, selection.output_unit)
    metrics = {
        name: Metric(value=float(value), unit=units.get(name))
        for name, value in raw_metrics.items()
    }
    return LineResult(
        selector=selection.selector, mode='auto',
        wave_min=float(x_arr[0]), wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size), metrics=metrics,
        metric_units=units, axis_unit=selection.axis_unit)

# ---------------------------------------------------------------------------
# Parameter sampling utilities
# ---------------------------------------------------------------------------

def _fit_param_names(model):
    _, fit_indices, _ = model_to_fit_params(model)
    names = []
    cumulative_idx = 0
    for pname in model.param_names:
        param = getattr(model, pname)
        for offset in range(param.size):
            if cumulative_idx in fit_indices:
                names.append(
                    pname if param.size == 1 else f"{pname}[{offset}]")
            cumulative_idx += 1
    return names


def _selection_parameter_specs(selection):
    components = get_components(selection.source_model,
                                additive=selection.additive)
    specs = []
    seen = set()

    for entry in selection.entries:
        component = components[entry.component_key]
        if entry.template_name is None:
            relevant = list(component.param_names)
        else:
            idx, _ = _linegroup_template_index(component, entry.template_name)
            relevant = ([component._param_names_list[idx]]
                        + list(component._shared_params.keys()))

        for param_name in relevant:
            param = getattr(component, param_name)
            key = (entry.component_key, param_name)
            if key in seen:
                continue

            pvalue = float(np.asarray(param.value, dtype=float))
            std_raw = getattr(param, 'std', None)
            std_val = (float(np.asarray(std_raw, dtype=float))
                       if std_raw is not None else float('nan'))
            lolim_raw = getattr(param, 'lolim', None)
            uplim_raw = getattr(param, 'uplim', None)
            lolim_val = (float(np.asarray(lolim_raw, dtype=float))
                         if lolim_raw is not None else float('nan'))
            uplim_val = (float(np.asarray(uplim_raw, dtype=float))
                         if uplim_raw is not None else float('nan'))

            has_std = np.isfinite(std_val) and std_val > 0.0
            has_limits = (np.isfinite(lolim_val) and np.isfinite(uplim_val)
                          and uplim_val > lolim_val)
            if not (has_std or has_limits):
                continue

            raw_bounds = getattr(param, 'bounds', (None, None))
            specs.append(_SelectionParameter(
                component_key=entry.component_key,
                param_name=param_name, value=pvalue,
                std=std_val, lolim=lolim_val, uplim=uplim_val,
                bounds=tuple(raw_bounds) if raw_bounds is not None else (None, None)))
            seen.add(key)
    return specs


def _non_line_parameter_specs(model, selection):
    all_comps = get_components(model, additive=False)
    line_keys = {entry.component_key for entry in selection.entries}
    specs = []
    seen = set()

    for key, name in zip(all_comps.indices, all_comps.names):
        if name in line_keys:
            continue
        component = all_comps[key]
        if not hasattr(component, 'param_names'):
            continue
        for param_name in component.param_names:
            param = getattr(component, param_name)
            if getattr(param, 'fixed', False):
                continue
            uid = (name, param_name)
            if uid in seen:
                continue
            pvalue = float(np.asarray(param.value, dtype=float))
            std_raw = getattr(param, 'std', None)
            std_val = (float(np.asarray(std_raw, dtype=float))
                       if std_raw is not None else float('nan'))
            lolim_raw = getattr(param, 'lolim', None)
            uplim_raw = getattr(param, 'uplim', None)
            lolim_val = (float(np.asarray(lolim_raw, dtype=float))
                         if lolim_raw is not None else float('nan'))
            uplim_val = (float(np.asarray(uplim_raw, dtype=float))
                         if uplim_raw is not None else float('nan'))
            has_std = np.isfinite(std_val) and std_val > 0.0
            has_limits = (np.isfinite(lolim_val) and np.isfinite(uplim_val)
                          and uplim_val > lolim_val)
            if not (has_std or has_limits):
                continue
            raw_bounds = getattr(param, 'bounds', (None, None))
            specs.append(_SelectionParameter(
                component_key=name, param_name=param_name, value=pvalue,
                std=std_val, lolim=lolim_val, uplim=uplim_val,
                bounds=tuple(raw_bounds) if raw_bounds is not None else (None, None)))
            seen.add(uid)
    return specs


def _leaf_component_suffix_map(model):
    leaves = get_components(model, additive=False)
    if len(leaves.names) == 1:
        return {leaves.names[0]: None}
    return {name: index for index, name in enumerate(leaves.names)}


def _selection_parent_name_map(selection, specs):
    model = selection.source_model
    free_names = set(_fit_param_names(model))
    leaves = get_components(model, additive=False)
    single = len(leaves.names) == 1
    resolved = {}

    for spec in specs:
        # For a compound model, astropy names free parameters as
        # "<param>_<submodel_index>". spec.component_key IS the integer
        # submodel index.  For a single-model (no suffix), try bare name.
        if single:
            candidates = [spec.param_name]
        else:
            candidates = [f"{spec.param_name}_{spec.component_key}",
                          spec.param_name]

        for candidate in candidates:
            if candidate in free_names:
                resolved[(spec.component_key, spec.param_name)] = candidate
                break
    return resolved


def _clip_draws_to_bounds(specs, draws):
    for j, spec in enumerate(specs):
        lo, hi = spec.bounds
        if lo is not None:
            draws[:, j] = np.maximum(draws[:, j], lo)
        if hi is not None:
            draws[:, j] = np.minimum(draws[:, j], hi)


def _draw_selection_samples(selection, n_samples, method='auto',
                            distribution='auto', random_state=None):
    specs = _selection_parameter_specs(selection)
    if not specs:
        raise ValueError(
            "No uncertainty information (std, lolim, or uplim) is attached "
            "to the selected line components.")

    method = str(method).strip().lower()
    distribution = str(distribution).strip().lower()
    if method not in {'auto', 'covariance', 'limits', 'std'}:
        raise ValueError("method must be: 'auto', 'covariance', 'limits', 'std'.")
    if distribution not in {'auto', 'uniform', 'gaussian'}:
        raise ValueError("distribution must be: 'auto', 'uniform', 'gaussian'.")
    if method == 'limits' and distribution == 'gaussian':
        raise ValueError("method='limits' + distribution='gaussian' not supported.")

    rng = np.random.default_rng(random_state)
    cov = getattr(selection.source_model, '_param_cov', None)
    parent_name_map = _selection_parent_name_map(selection, specs)

    enriched = [
        _SelectionParameter(
            component_key=s.component_key, param_name=s.param_name,
            value=s.value, std=s.std, lolim=s.lolim, uplim=s.uplim,
            bounds=s.bounds,
            parent_name=parent_name_map.get((s.component_key, s.param_name)))
        for s in specs]

    # Strategy 1: multivariate Normal from covariance
    if method in ('auto', 'covariance') and cov is not None:
        free_names = _fit_param_names(selection.source_model)
        cov_specs = [s for s in enriched
                     if s.parent_name is not None and s.parent_name in free_names]
        if cov_specs:
            indices = [free_names.index(s.parent_name) for s in cov_specs]
            mean = np.asarray([s.value for s in cov_specs], dtype=float)
            subcov = np.asarray(cov, dtype=float)[np.ix_(indices, indices)]
            if np.all(np.isfinite(subcov)):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            'covariance is not symmetric positive-semidefinite',
                            RuntimeWarning)
                        draws = rng.multivariate_normal(
                            mean, subcov, size=int(n_samples))
                    _clip_draws_to_bounds(cov_specs, draws)
                    return cov_specs, draws
                except np.linalg.LinAlgError:
                    if method == 'covariance':
                        raise
    if method == 'covariance':
        raise ValueError(
            "method='covariance' but no usable covariance matrix available.")

    # Strategy 2+3: per-parameter draws
    draws = np.empty((int(n_samples), len(enriched)), dtype=float)
    for j, spec in enumerate(enriched):
        lo, up = spec.lolim, spec.uplim
        has_limits = np.isfinite(lo) and np.isfinite(up) and up > lo
        has_std = np.isfinite(spec.std) and spec.std > 0.0

        _sqrt3 = np.sqrt(3.0)
        drawn = False
        if (method in ('auto', 'limits') and has_limits
                and distribution in ('auto', 'uniform')):
            draws[:, j] = rng.uniform(lo, up, size=int(n_samples))
            drawn = True
        if (not drawn and method in ('auto', 'std') and has_std
                and distribution in ('auto', 'uniform')):
            # Uniform(value ± √3·σ) → Var = σ², matching Gaussian variance
            half = _sqrt3 * spec.std
            draws[:, j] = rng.uniform(
                spec.value - half, spec.value + half, size=int(n_samples))
            drawn = True
        if (not drawn and method in ('auto', 'std') and has_std
                and distribution in ('auto', 'gaussian')):
            draws[:, j] = rng.normal(spec.value, spec.std, size=int(n_samples))
            drawn = True
        if not drawn:
            if method == 'auto':
                draws[:, j] = spec.value
            else:
                raise ValueError(
                    f"Parameter '{spec.param_name}' of '{spec.component_key}' "
                    f"lacks info for method='{method}'.")

    _clip_draws_to_bounds(enriched, draws)
    return enriched, draws

# ---------------------------------------------------------------------------
# sample_line_measurements
# ---------------------------------------------------------------------------

def sample_line_measurements(model_or_selection, selector=None,
                             components=None, additive=True, x=None,
                             window=None, num=4096, n_samples=256,
                             confidence=68, method='auto', distribution='auto',
                             random_state=None, return_samples=False,
                             index=None) -> LineResult:
    """Propagate parameter uncertainties into sampled line measurements.

    Parameters
    ----------
    model_or_selection : astropy.modeling.Model or SelectedLineProfile
        Source model or precomputed line selection.
    selector : str, optional
        Line tag resolved by :func:`select_line`. Default is ``None`` when a
        ``SelectedLineProfile`` is passed directly.
    components : sequence of str, optional
        Optional component-name filter applied during selection. Default is
        ``None``.
    additive : bool, optional
        If ``True`` (default), select within the additive decomposition of the
        model.
    x : array-like or astropy.units.Quantity, optional
        Explicit spectral grid used for the measurement. Default is ``None``.
    window : tuple of float, optional
        Spectral interval used to build an internal grid when ``x`` is omitted.
        Default is ``None``.
    num : int, optional
        Number of samples for the internally generated grid. Default is ``4096``.
    n_samples : int, optional
        Number of Monte Carlo draws. Default is ``256``.
    confidence : float, optional
        Central confidence interval reported through ``lolim`` and ``uplim``.
        Default is ``68``.
    method : {'auto', 'covariance', 'limits', 'std'}, optional
        Sampling source. ``'covariance'`` uses the fitted covariance matrix,
        ``'limits'`` uses attached lower/upper limits, and ``'std'`` uses stored
        standard deviations. Default is ``'auto'``.
    distribution : {'auto', 'uniform', 'gaussian'}, optional
        Distribution used for per-parameter draws. Default is ``'auto'``.
    random_state : int or numpy.random.Generator, optional
        Seed or generator used for reproducible draws. Default is ``None``.
    return_samples : bool, optional
        If ``True``, also return the raw sampled metric table. Default is
        ``False``.
    index : int or tuple of int, optional
        Spatial index used when ``model_or_selection`` is a ``MultiFitResult``.
        Default is ``None``.

    Returns
    -------
    LineResult or tuple
        Sampled summary metrics. When ``return_samples=True``, returns
        ``(LineResult, pandas.DataFrame)``.

    Notes
    -----
    This routine resamples model parameters, re-evaluates the selected line for
    each draw, and measures every sampled profile with the same numerical logic
    used by :func:`measure_line`. If a covariance matrix is attached, Prism draws
    a multivariate normal sample so parameter correlations are preserved. When no
    covariance matrix is available, independent draws fall back to attached
    symmetric ``std`` values or asymmetric ``lolim``/``uplim`` intervals.

    Examples
    --------
    >>> selected = select_line(fitted_model, 'Hb4861')
    >>> summary = sample_line_measurements(selected, x=wave, n_samples=128)
    >>> summary.flux.std
    """
    selection = _resolve_selection(
        model_or_selection, selector=selector,
        components=components, additive=additive, index=index)

    if isinstance(selection, SelectedLineCollection):
        raise ValueError(
            "sample_line_measurements does not support full MultiFitResult "
            "batches. Pass index= or call collection[index].sample(...).")

    if x is None:
        if window is None:
            window = selection.infer_window()
        x_arr = np.linspace(
            float(window[0]), float(window[1]), int(num), dtype=float)
    else:
        x_arr = selection._coerce_domain_axis(x)

    model = selection.source_model
    specs, draws = _draw_selection_samples(
        selection, n_samples=n_samples, method=method,
        distribution=distribution, random_state=random_state)

    y_nominal = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    nominal_metrics = _compute_profile_metrics(x_arr, y_nominal)
    metric_units = _metric_units(selection.axis_unit, selection.output_unit)

    working_model = model.copy()
    working_comps = get_components(working_model, additive=selection.additive)
    eval_plan = [(entry, working_comps[entry.component_key])
                 for entry in selection.entries]

    sample_records = []
    for draw in draws:
        for spec, value in zip(specs, draw):
            getattr(working_comps[spec.component_key],
                    spec.param_name).value = value
        y_sampled = np.zeros_like(x_arr, dtype=float)
        for entry, comp in eval_plan:
            if entry.template_name is None:
                vals, _, _ = _evaluate_component_profile(comp, x_arr)
            else:
                vals, _, _, _ = _evaluate_linegroup_template(
                    comp, entry.template_name, x_arr,
                    position=entry.template_position)
            y_sampled += selection._coerce_output_array(vals, name='line profile')
        sample_records.append(_compute_profile_metrics(x_arr, y_sampled))

    alpha = (100.0 - float(confidence)) / 2.0
    sampled_metrics = {}
    for name in _METRIC_NAMES:
        nom = nominal_metrics.get(name, float('nan'))
        values = np.array(
            [rec.get(name, float('nan')) for rec in sample_records], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            sampled_metrics[name] = Metric(value=float(nom), unit=metric_units.get(name))
        else:
            bp = _METRIC_BOUNDS.get(name, _NO_BOUNDS)
            lo, up = extract_limits(bp, values, alpha, 100.0 - alpha)
            sampled_metrics[name] = Metric(
                value=float(nom),
                std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                lolim=float(lo), uplim=float(up),
                unit=metric_units.get(name))

    if np.any(~np.isfinite(
            np.array([r.get('fwhm', float('nan')) for r in sample_records]))):
        warnings.warn(
            "Some sampled profiles have undefined FWHM.",
            RuntimeWarning)

    result = LineResult(
        selector=selection.selector, mode='auto',
        wave_min=float(x_arr[0]), wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size), metrics=sampled_metrics,
        metric_units=metric_units, axis_unit=selection.axis_unit)
    if return_samples:
        return result, pd.DataFrame(sample_records)
    return result
