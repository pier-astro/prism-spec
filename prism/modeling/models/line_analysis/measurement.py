"""
prism.modeling.models.line_analysis.measurement — line measurement and sampling.

Provides ``LineResult``, ``MultiLineMeasurements``, ``measure_line``,
``sample_line_measurements``.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from astropy.modeling.fitting import model_to_fit_params

from ..components import get_components
from ..lines import Metric
from ...fitting.uncertainty.resample import extract_limits

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


def _as_metric(value) -> Metric:
    if isinstance(value, Metric):
        return value
    return Metric(value=float(value) if value is not None else float('nan'))

# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------

class LineResult:
    """Measurement results for a selected emission line."""

    def __init__(self, selector, mode, wave_min, wave_max, n_grid, metrics):
        self.selector = str(selector)
        self.mode = str(mode)
        self.wave_min = float(wave_min)
        self.wave_max = float(wave_max)
        self.n_grid = int(n_grid)
        for name in _METRIC_NAMES:
            setattr(self, name, _as_metric(metrics.get(name, float('nan'))))

    def to_frame(self, metrics=None) -> pd.DataFrame:
        names = list(metrics) if metrics is not None else _METRIC_NAMES
        rows = {}
        for name in names:
            m = getattr(self, name, Metric(value=float('nan')))
            rows[name] = {
                'value': m.value, 'std': m.std,
                'lolim': m.lolim, 'uplim': m.uplim}
        return pd.DataFrame(rows).T

    def __repr__(self):
        return (
            f"LineResult(selector={self.selector!r}, "
            f"flux={self.flux.value:.4g}, fwhm={self.fwhm.value:.4g})")


class MultiLineMeasurements:
    """Batch measurement results over a spatial grid."""

    def __init__(self, selector, mode, shape, **metric_arrays):
        self.selector = str(selector)
        self.mode = str(mode)
        self.shape = tuple(shape)
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
        idx = (tuple(index) if isinstance(index, tuple)
               else np.unravel_index(int(index), self.shape))
        metrics = {name: float(getattr(self, name)[idx])
                   for name in _METRIC_NAMES}
        return LineResult(
            selector=self.selector, mode=self.mode,
            wave_min=float(self.wave_min[idx]),
            wave_max=float(self.wave_max[idx]),
            n_grid=int(self.n_grid[idx]), metrics=metrics)

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

def _resolve_selection(model_or_selection, selector=None, mode='auto',
                       components=None, additive=True, index=None):
    if isinstance(model_or_selection, (SelectedLineProfile, SelectedLineCollection)):
        return model_or_selection
    return select_line(
        model_or_selection, selector=selector, mode=mode,
        components=components, additive=additive, index=index)

# ---------------------------------------------------------------------------
# measure_line
# ---------------------------------------------------------------------------

def measure_line(model_or_selection, selector=None, mode='auto',
                 components=None, additive=True, x=None, window=None,
                 num=4096, index=None) -> LineResult:
    selection = _resolve_selection(
        model_or_selection, selector=selector, mode=mode,
        components=components, additive=additive, index=index)

    if isinstance(selection, SelectedLineCollection):
        arrays = {name: np.full(selection.shape, np.nan, dtype=float)
                  for name in _METRIC_NAMES}
        wave_min = np.full(selection.shape, np.nan, dtype=float)
        wave_max = np.full(selection.shape, np.nan, dtype=float)
        n_grid = np.zeros(selection.shape, dtype=int)
        for fi in range(selection.source_result.n_spaxels):
            si = np.unravel_index(fi, selection.shape)
            m = selection.get_profile(fi).measure(x=x, window=window, num=num)
            for name in _METRIC_NAMES:
                arrays[name][si] = getattr(m, name).value
            wave_min[si] = m.wave_min
            wave_max[si] = m.wave_max
            n_grid[si] = m.n_grid
        return MultiLineMeasurements(
            selector=selection.selector, mode=selection.mode,
            shape=selection.shape,
            wave_min=wave_min, wave_max=wave_max, n_grid=n_grid, **arrays)

    if x is None:
        if window is None:
            window = selection.infer_window()
        x_arr = np.linspace(
            float(window[0]), float(window[1]), int(num), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 1 or x_arr.size < 3:
            raise ValueError(
                "x must be a 1-D wavelength grid with at least three samples.")

    y_arr = np.asarray(selection.evaluate(x_arr), dtype=float)
    metrics = _compute_profile_metrics(x_arr, y_arr)
    return LineResult(
        selector=selection.selector, mode=selection.mode,
        wave_min=float(x_arr[0]), wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size), metrics=metrics)

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
    suffix_map = _leaf_component_suffix_map(model)
    resolved = {}

    for spec in specs:
        suffix = suffix_map.get(spec.component_key)
        candidates = []
        if suffix is None:
            candidates.append(spec.param_name)
        candidates.append(f"{spec.param_name}_{suffix}")

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
    if method == 'std' and distribution == 'uniform':
        raise ValueError("method='std' + distribution='uniform' not supported.")

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

        drawn = False
        if (method in ('auto', 'limits') and has_limits
                and distribution in ('auto', 'uniform')):
            draws[:, j] = rng.uniform(lo, up, size=int(n_samples))
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

def sample_line_measurements(model_or_selection, selector=None, mode='auto',
                             components=None, additive=True, x=None,
                             window=None, num=4096, n_samples=256,
                             confidence=68, method='auto', distribution='auto',
                             random_state=None, return_samples=False,
                             index=None) -> LineResult:
    selection = _resolve_selection(
        model_or_selection, selector=selector, mode=mode,
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
        x_arr = np.asarray(x, dtype=float)

    model = selection.source_model
    specs, draws = _draw_selection_samples(
        selection, n_samples=n_samples, method=method,
        distribution=distribution, random_state=random_state)

    y_nominal = np.asarray(selection.evaluate(x_arr), dtype=float)
    nominal_metrics = _compute_profile_metrics(x_arr, y_nominal)

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
                    comp, entry.template_name, x_arr)
            y_sampled += vals
        sample_records.append(_compute_profile_metrics(x_arr, y_sampled))

    alpha = (100.0 - float(confidence)) / 2.0
    sampled_metrics = {}
    for name in _METRIC_NAMES:
        nom = nominal_metrics.get(name, float('nan'))
        values = np.array(
            [rec.get(name, float('nan')) for rec in sample_records], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            sampled_metrics[name] = Metric(value=float(nom))
        else:
            bp = _METRIC_BOUNDS.get(name, _NO_BOUNDS)
            lo, up = extract_limits(bp, values, alpha, 100.0 - alpha)
            sampled_metrics[name] = Metric(
                value=float(nom),
                std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                lolim=float(lo), uplim=float(up))

    if np.any(~np.isfinite(
            np.array([r.get('fwhm', float('nan')) for r in sample_records]))):
        warnings.warn(
            "Some sampled profiles have undefined FWHM.",
            RuntimeWarning)

    result = LineResult(
        selector=selection.selector, mode=selection.mode,
        wave_min=float(x_arr[0]), wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size), metrics=sampled_metrics)
    if return_samples:
        return result, pd.DataFrame(sample_records)
    return result
