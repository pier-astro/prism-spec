from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from astropy.modeling.fitting import model_to_fit_params

from .components import get_components
from ..fitting.uncertainty.resample import extract_limits
from .lines import (
    GaussianLine,
    GaussianLines,
    LineGroupBase,
    LorentzianLine,
    LorentzianLines,
    VoigtLine,
    VoigtLines,
    sigma2fwhm,
)


try:
    _trapezoid = np.trapezoid          # numpy ≥ 2.0
except AttributeError:
    _trapezoid = np.trapz              # numpy < 2.0

_SINGLE_LINE_TYPES = (GaussianLine, LorentzianLine, VoigtLine)
_GROUP_LINE_TYPES = (GaussianLines, LorentzianLines, VoigtLines)


def _selector_token(value) -> str:
    return str(value).strip().lower()


def _approx_voigt_fwhm(fwhm_g, fwhm_l):
    fwhm_g = np.asarray(fwhm_g, dtype=float)
    fwhm_l = np.asarray(fwhm_l, dtype=float)
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)


def _profile_width_from_args(model, args):
    if isinstance(model, (GaussianLine, GaussianLines)):
        return float(args[2] * sigma2fwhm)
    if isinstance(model, (LorentzianLine, LorentzianLines)):
        fwhm_g = float(args[2] * sigma2fwhm)
        fwhm_l = float(2.0 * args[3])
        return float(_approx_voigt_fwhm(fwhm_g, fwhm_l))
    if isinstance(model, (VoigtLine, VoigtLines)):
        fwhm_g = float(args[2] * sigma2fwhm)
        fwhm_l = float(2.0 * args[3])
        return float(_approx_voigt_fwhm(fwhm_g, fwhm_l))
    raise TypeError(f"Unsupported line model type: {type(model).__name__}")


def _linegroup_template_index(model, template_name):
    selector = _selector_token(template_name)
    for index, candidate in enumerate(model._templates):
        if _selector_token(candidate) == selector:
            return index, str(candidate)
    raise KeyError(f"Template '{template_name}' not found in component '{getattr(model, 'name', type(model).__name__)}'.")


def _evaluate_linegroup_template(model, template_name, x):
    idx, resolved = _linegroup_template_index(model, template_name)
    amplitude = getattr(model, model._param_names_list[idx]).value
    shared_values = [getattr(model, pname).value for pname in model._shared_params.keys()]

    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    total = np.zeros_like(x_arr, dtype=float)
    centers = []
    widths = []

    positions = model._tmpl_positions[idx]
    weights = model._tmpl_weights[idx]
    for pos, weight in zip(positions, weights):
        args = model._single_profile_args(pos, amplitude, weight, *shared_values)
        total += model._profile_func(x_arr, *args)
        centers.append(float(args[1]))
        widths.append(_profile_width_from_args(model, args))

    return total, np.asarray(centers, dtype=float), np.asarray(widths, dtype=float), resolved


def _evaluate_single_line(model, x):
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    values = np.asarray(model(x_arr), dtype=float)

    if isinstance(model, GaussianLine):
        center = model.position.value * (1.0 + model.redshift.value) * np.exp(model.offset.value / 299792.458)
        fwhm = model.fwhm.value / 299792.458 * center
    elif isinstance(model, LorentzianLine):
        center = model.position.value * (1.0 + model.redshift.value) * np.exp(model.offset.value / 299792.458)
        fwhm = model.fwhm.value / 299792.458 * center
    elif isinstance(model, VoigtLine):
        center = model.position.value * (1.0 + model.redshift.value) * np.exp(model.offset.value / 299792.458)
        fwhm_g = model.fwhm_G.value / 299792.458 * center
        fwhm_l = model.fwhm_L.value / 299792.458 * center
        fwhm = float(_approx_voigt_fwhm(fwhm_g, fwhm_l))
    else:
        raise TypeError(f"Unsupported line model type: {type(model).__name__}")

    return values, np.asarray([center], dtype=float), np.asarray([fwhm], dtype=float)


def _evaluate_component_profile(model, x):
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if isinstance(model, LineGroupBase):
        values = np.asarray(model(x_arr), dtype=float)
        centers = []
        widths = []
        shared_values = [getattr(model, pname).value for pname in model._shared_params.keys()]
        for idx in range(model._n_templates):
            amplitude = getattr(model, model._param_names_list[idx]).value
            positions = model._tmpl_positions[idx]
            template_weights = model._tmpl_weights[idx]
            for pos, weight in zip(positions, template_weights):
                args = model._single_profile_args(pos, amplitude, weight, *shared_values)
                centers.append(float(args[1]))
                widths.append(_profile_width_from_args(model, args))
        return values, np.asarray(centers, dtype=float), np.asarray(widths, dtype=float)

    if isinstance(model, _SINGLE_LINE_TYPES):
        return _evaluate_single_line(model, x_arr)

    raise TypeError(f"Unsupported component type for line analysis: {type(model).__name__}")


@dataclass(frozen=True)
class _SelectionEntry:
    component_key: str | int
    component_name: str
    template_name: str | None = None


@dataclass(frozen=True)
class _SelectionParameter:
    component_key: str | int
    param_name: str
    value: float
    std: float = float('nan')
    lolim: float = float('nan')
    uplim: float = float('nan')
    bounds: tuple = (None, None)
    parent_name: str | None = None


@dataclass(frozen=True)
class Metric:
    """A single measured quantity with optional uncertainty bounds.

    Attributes
    ----------
    value  : nominal (best-fit or integrated) measurement.
    std    : standard deviation from sampling (NaN when unavailable).
    lolim  : lower percentile limit from sampling (NaN when unavailable).
    uplim  : upper percentile limit from sampling (NaN when unavailable).
    """
    value: float
    std: float = float('nan')
    lolim: float = float('nan')
    uplim: float = float('nan')

    def __repr__(self) -> str:
        if not np.isfinite(self.std):
            return f"{self.value:.4g}"
        return f"{self.value:.4g} \u00b1 {self.std:.4g}  [{self.lolim:.4g}, {self.uplim:.4g}]"


def _as_metric(value) -> Metric:
    if isinstance(value, Metric):
        return value
    return Metric(value=float(value) if value is not None else float('nan'))


_METRIC_NAMES = [
    'flux', 'peak_value', 'peak_position',
    'fwhm', 'hw_blue', 'hw_red', 'asymmetry',
    'moment1', 'moment2', 'sigma',
    'skewness', 'kurtosis',
    'fw10m', 'fw20m', 'fw80m', 'bisector_span',
]


class _BoundsProxy:
    """Lightweight proxy with .min/.max for extract_limits compatibility."""
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


class LineResult:
    """Measurement results for a selected emission line.

    Each metric attribute is a :class:`Metric` with ``.value``, ``.std``,
    ``.lolim``, and ``.uplim``.  Call :meth:`to_frame` for a tidy overview.

    Metrics: flux, peak_value, peak_position, fwhm, hw_blue, hw_red,
    asymmetry, moment1, moment2, sigma, skewness, kurtosis,
    fw10m, fw20m, fw80m, bisector_span.
    """

    def __init__(self, selector: str, mode: str, wave_min: float, wave_max: float,
                 n_grid: int, metrics: dict):
        self.selector = str(selector)
        self.mode = str(mode)
        self.wave_min = float(wave_min)
        self.wave_max = float(wave_max)
        self.n_grid = int(n_grid)
        for name in _METRIC_NAMES:
            setattr(self, name, _as_metric(metrics.get(name, float('nan'))))

    def to_frame(self, metrics=None) -> pd.DataFrame:
        """Return a DataFrame with columns value/std/lolim/uplim, one row per metric."""
        names = list(metrics) if metrics is not None else _METRIC_NAMES
        rows = {}
        for name in names:
            m = getattr(self, name, Metric(value=float('nan')))
            rows[name] = {'value': m.value, 'std': m.std, 'lolim': m.lolim, 'uplim': m.uplim}
        return pd.DataFrame(rows).T

    def __repr__(self) -> str:
        return (
            f"LineResult(selector={self.selector!r}, "
            f"flux={self.flux.value:.4g}, fwhm={self.fwhm.value:.4g})"
        )


class MultiLineMeasurements:
    """Batch measurement results over a spatial grid (from a MultiFitResult).

    Each metric attribute is a shaped numpy array.
    Use :meth:`get_measurement` to extract a :class:`LineResult` for one spaxel.
    """

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
        idx = tuple(index) if isinstance(index, tuple) else np.unravel_index(int(index), self.shape)
        metrics = {name: float(getattr(self, name)[idx]) for name in _METRIC_NAMES}
        return LineResult(
            selector=self.selector,
            mode=self.mode,
            wave_min=float(self.wave_min[idx]),
            wave_max=float(self.wave_max[idx]),
            n_grid=int(self.n_grid[idx]),
            metrics=metrics,
        )


class SelectedLineCollection:
    def __init__(self, result, selector, mode, components=None, additive=True):
        self.source_result = result
        self.selector = str(selector)
        self.mode = str(mode)
        self.components = components
        self.additive = bool(additive)

    def __repr__(self):
        return (
            f"SelectedLineCollection(selector={self.selector!r}, mode={self.mode!r}, "
            f"shape={self.source_result.shape})"
        )

    @property
    def shape(self):
        return self.source_result.shape

    def get_profile(self, index):
        model = self.source_result.get_model(index)
        return select_line(
            model,
            selector=self.selector,
            mode=self.mode,
            components=self.components,
            additive=self.additive,
        )

    def __getitem__(self, index):
        return self.get_profile(index)

    def evaluate(self, x=None, spectral_axis=None):
        eval_x = self.source_result._x if x is None else np.asarray(x, dtype=float)
        if eval_x is None:
            raise ValueError("Provide x= or construct the MultiFitResult with x= to enable batch line evaluation.")

        out = np.empty(self.shape + (eval_x.size,), dtype=float)
        for flat_index in range(self.source_result.n_spaxels):
            spatial_index = np.unravel_index(flat_index, self.shape)
            out[spatial_index] = self.get_profile(flat_index)(eval_x)

        target_axis = self.source_result.spectral_axis if spectral_axis is None else spectral_axis
        if target_axis < 0:
            target_axis += out.ndim
        if target_axis != out.ndim - 1:
            out = np.moveaxis(out, -1, target_axis)
        return out

    def __call__(self, x=None, spectral_axis=None):
        return self.evaluate(x=x, spectral_axis=spectral_axis)

    def measure(self, x=None, window=None, num=4096):
        return measure_line(self, x=x, window=window, num=num)


def _is_multifit_like(obj) -> bool:
    return (
        hasattr(obj, 'n_spaxels')
        and hasattr(obj, 'get_model')
        and hasattr(obj, 'shape')
    )


class SelectedLineProfile:
    def __init__(self, source_model, selector, mode, entries, additive=True):
        self.source_model = source_model
        self.selector = str(selector)
        self.mode = mode
        self.entries = tuple(entries)
        self.additive = bool(additive)

    def __repr__(self):
        return (
            f"SelectedLineProfile(selector={self.selector!r}, mode={self.mode!r}, "
            f"components={len(self.entries)})"
        )

    @property
    def component_names(self):
        return [entry.component_name for entry in self.entries]

    def copy_for_model(self, model):
        return SelectedLineProfile(
            source_model=model,
            selector=self.selector,
            mode=self.mode,
            entries=self.entries,
            additive=self.additive,
        )

    def _iter_evaluated_entries(self, x):
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                values, centers, widths = _evaluate_component_profile(component, x)
                yield values, centers, widths
                continue

            values, centers, widths, _ = _evaluate_linegroup_template(component, entry.template_name, x)
            yield values, centers, widths

    def infer_window(self, padding=6.0, min_width=2.0):
        all_centers = []
        all_widths = []
        components = get_components(self.source_model, additive=self.additive)

        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                _, centers, widths = _evaluate_component_profile(component, np.asarray([0.0]))
            else:
                _, centers, widths, _ = _evaluate_linegroup_template(component, entry.template_name, np.asarray([0.0]))
            all_centers.extend(np.atleast_1d(centers).tolist())
            all_widths.extend(np.atleast_1d(widths).tolist())

        centers = np.asarray(all_centers, dtype=float)
        widths = np.asarray(all_widths, dtype=float)
        widths = np.where(np.isfinite(widths) & (widths > 0.0), widths, min_width)
        if centers.size == 0:
            raise ValueError("Cannot infer a wavelength window for an empty line selection.")
        return (
            float(np.min(centers - padding * widths)),
            float(np.max(centers + padding * widths)),
        )

    def evaluate(self, x):
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        total = np.zeros_like(x_arr, dtype=float)
        for values, _, _ in self._iter_evaluated_entries(x_arr):
            total += values
        if np.ndim(x) == 0:
            return float(total[0])
        return total

    def __call__(self, x):
        return self.evaluate(x)

    @property
    def position(self) -> float:
        """Rest-frame wavelength of the selected line (Å).

        Computed as the mean rest position across all selected components.
        A warning is issued if components disagree beyond 0.1% of the mean.
        """
        components = get_components(self.source_model, additive=self.additive)
        positions = []
        labels = []
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                pos_param = getattr(component, 'position', None)
                if pos_param is not None:
                    p = float(pos_param.value)
                    positions.append(p)
                    labels.append(f"{entry.component_name}: {p:.4f} Å")
            else:
                idx, _ = _linegroup_template_index(component, entry.template_name)
                for p in component._tmpl_positions[idx]:
                    positions.append(float(p))
                    labels.append(f"{entry.component_name}/{entry.template_name}: {float(p):.4f} Å")

        if not positions:
            raise ValueError(
                f"Cannot determine rest position for selector '{self.selector}': "
                "no position information found in the selected components."
            )

        positions = np.asarray(positions, dtype=float)
        mean_pos = float(np.mean(positions))
        spread = float(np.max(positions) - np.min(positions))
        tol = 0.001 * mean_pos  # 0.1% of mean position

        if spread > tol:
            warnings.warn(
                f"Rest positions of the '{self.selector}' selected components differ by "
                f"{spread:.4f} Å (tolerance {tol:.4f} Å):\n"
                + "\n".join(f"  {lbl}" for lbl in labels),
                UserWarning,
                stacklevel=2,
            )

        return mean_pos

    def evaluate_components(self, x=None, window=None, num=4096) -> dict:
        """Evaluate each selected component separately.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from component name to flux array sampled on *x*.
            When a component name appears more than once (multiple templates
            from the same model), the key is disambiguated as
            ``"component/template"``.
        """
        if x is None:
            if window is None:
                window = self.infer_window()
            x_arr = np.linspace(float(window[0]), float(window[1]), int(num), dtype=float)
        else:
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        components = get_components(self.source_model, additive=self.additive)
        result = {}
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                values, _, _ = _evaluate_component_profile(component, x_arr)
                key = entry.component_name
            else:
                values, _, _, _ = _evaluate_linegroup_template(component, entry.template_name, x_arr)
                key = entry.component_name
            if key in result:
                key = f"{entry.component_name}/{entry.template_name or '*'}"
            result[key] = values

        return result

    def measure(self, x=None, window=None, num=4096) -> LineResult:
        return measure_line(self, x=x, window=window, num=num)

    def sample(self, n_samples=256, confidence=68, x=None, window=None, num=4096,
               method='auto', distribution='auto',
               random_state=None, return_samples=False) -> LineResult:
        return sample_line_measurements(
            self, n_samples=n_samples, confidence=confidence,
            x=x, window=window, num=num,
            method=method, distribution=distribution,
            random_state=random_state, return_samples=return_samples,
        )


def _component_name(component, fallback):
    name = getattr(component, 'name', None)
    return str(name) if name else str(fallback)


def _normalize_explicit_components(model, selector, components, additive=True):
    if not components:
        raise ValueError("Explicit selection requires a non-empty components= sequence.")
    available = get_components(model, additive=additive)
    entries = []
    seen = set()

    for component in components:
        if isinstance(component, (str, int)):
            key = component
        else:
            name = getattr(component, 'name', None)
            if not name:
                raise ValueError("Explicit component models must have a unique .name attribute.")
            key = name
        resolved = available[key]
        component_name = _component_name(resolved, key)
        token = (str(key), component_name)
        if token in seen:
            continue

        if isinstance(resolved, LineGroupBase):
            try:
                _, template_name = _linegroup_template_index(resolved, selector)
            except KeyError as exc:
                raise KeyError(
                    f"Explicit component '{component_name}' is a line-group model; selector '{selector}' must match one of its template tags."
                ) from exc
            entries.append(
                _SelectionEntry(
                    component_key=key,
                    component_name=component_name,
                    template_name=template_name,
                )
            )
        else:
            entries.append(_SelectionEntry(component_key=key, component_name=component_name))
        seen.add(token)

    return entries


def _resolve_physical_entries(model, selector, additive=True):
    selector_token = _selector_token(selector)
    components = get_components(model, additive=additive)
    entries = []

    for key, name in zip(components.indices, components.names):
        component = components[key]
        if not isinstance(component, LineGroupBase):
            continue
        for template_name in component._templates:
            if _selector_token(template_name) == selector_token:
                entries.append(
                    _SelectionEntry(
                        component_key=name,
                        component_name=name,
                        template_name=str(template_name),
                    )
                )
                break

    return entries


def _resolve_component_entries(model, selector, additive=True):
    selector_token = _selector_token(selector)
    components = get_components(model, additive=additive)
    entries = []

    for key, name in zip(components.indices, components.names):
        if _selector_token(name) != selector_token:
            continue
        entries.append(_SelectionEntry(component_key=name, component_name=name))

    return entries


def select_line(model, selector, mode='auto', components=None, additive=True, index=None):
    if selector is None or str(selector).strip() == '':
        raise ValueError("A line tag or name is required.")

    if _is_multifit_like(model):
        if index is not None:
            return select_line(
                model.get_model(index),
                selector=selector,
                mode=mode,
                components=components,
                additive=additive,
            )
        return SelectedLineCollection(
            result=model,
            selector=selector,
            mode=mode,
            components=components,
            additive=additive,
        )

    mode = str(mode).strip().lower()
    if mode not in {'auto', 'physical', 'component', 'explicit'}:
        raise ValueError("mode must be one of: 'auto', 'physical', 'component', 'explicit'.")

    if mode == 'explicit':
        entries = _normalize_explicit_components(model, selector, components, additive=additive)
    elif mode == 'physical':
        entries = _resolve_physical_entries(model, selector, additive=additive)
    elif mode == 'component':
        entries = _resolve_component_entries(model, selector, additive=additive)
    else:
        physical_entries = _resolve_physical_entries(model, selector, additive=additive)
        component_entries = _resolve_component_entries(model, selector, additive=additive)
        if physical_entries and component_entries:
            raise ValueError(
                f"Selector '{selector}' matches both a physical line tag and a component name; pass mode='physical' or mode='component'."
            )
        entries = physical_entries or component_entries

    if not entries:
        raise KeyError(f"No line selection found for '{selector}' in mode '{mode}'.")

    return SelectedLineProfile(
        source_model=model,
        selector=str(selector),
        mode=mode,
        entries=entries,
        additive=additive,
    )


def _resolve_selection(model_or_selection, selector=None, mode='auto', components=None, additive=True, index=None):
    if isinstance(model_or_selection, (SelectedLineProfile, SelectedLineCollection)):
        return model_or_selection
    return select_line(
        model_or_selection,
        selector=selector,
        mode=mode,
        components=components,
        additive=additive,
        index=index,
    )


def _crossings_at_fraction(x, y, fraction):
    """Return (left_x, right_x) where the profile crosses fraction*peak on each side of the peak."""
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
                left_x = float(x[i - 1] + (level - y[i - 1]) / denom * (x[i] - x[i - 1]))
            break

    right_x = float('nan')
    for i in range(peak_index, x.size - 1):
        if y[i + 1] < level:
            denom = y[i + 1] - y[i]
            if denom != 0.0:
                right_x = float(x[i] + (level - y[i]) / denom * (x[i + 1] - x[i]))
            break

    return left_x, right_x


def _compute_profile_metrics(x_arr, y_arr) -> dict:
    """Compute all line profile metrics from a (wavelength, flux) pair of 1-D arrays."""
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
                    _trapezoid(residuals ** 3 * y_arr, x_arr) / (flux * sigma ** 3))
                kurtosis = float(
                    _trapezoid(residuals ** 4 * y_arr, x_arr) / (flux * sigma ** 4)) - 3.0

    fwhm_l, fwhm_r = _crossings_at_fraction(x_arr, y_arr, 0.5)
    fwhm = float(fwhm_r - fwhm_l) if (np.isfinite(fwhm_l) and np.isfinite(fwhm_r)) else float('nan')
    hw_blue = float(peak_position - fwhm_l) if np.isfinite(fwhm_l) else float('nan')
    hw_red = float(fwhm_r - peak_position) if np.isfinite(fwhm_r) else float('nan')
    asymmetry = (
        float((hw_red - hw_blue) / fwhm)
        if (np.isfinite(fwhm) and fwhm > 0.0)
        else float('nan')
    )

    fw10m_l, fw10m_r = _crossings_at_fraction(x_arr, y_arr, 0.10)
    fw20m_l, fw20m_r = _crossings_at_fraction(x_arr, y_arr, 0.20)
    fw80m_l, fw80m_r = _crossings_at_fraction(x_arr, y_arr, 0.80)
    fw10m = float(fw10m_r - fw10m_l) if (np.isfinite(fw10m_l) and np.isfinite(fw10m_r)) else float('nan')
    fw20m = float(fw20m_r - fw20m_l) if (np.isfinite(fw20m_l) and np.isfinite(fw20m_r)) else float('nan')
    fw80m = float(fw80m_r - fw80m_l) if (np.isfinite(fw80m_l) and np.isfinite(fw80m_r)) else float('nan')

    bisector_20 = (0.5 * (fw20m_l + fw20m_r)
                   if (np.isfinite(fw20m_l) and np.isfinite(fw20m_r)) else float('nan'))
    bisector_80 = (0.5 * (fw80m_l + fw80m_r)
                   if (np.isfinite(fw80m_l) and np.isfinite(fw80m_r)) else float('nan'))
    bisector_span = (
        float(bisector_80 - bisector_20)
        if (np.isfinite(bisector_20) and np.isfinite(bisector_80))
        else float('nan')
    )

    return {
        'flux': flux,
        'peak_value': peak_value,
        'peak_position': peak_position,
        'fwhm': fwhm,
        'hw_blue': hw_blue,
        'hw_red': hw_red,
        'asymmetry': asymmetry,
        'moment1': moment1,
        'moment2': moment2,
        'sigma': sigma,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'fw10m': fw10m,
        'fw20m': fw20m,
        'fw80m': fw80m,
        'bisector_span': bisector_span,
    }


def measure_line(model_or_selection, selector=None, mode='auto', components=None,
                 additive=True, x=None, window=None, num=4096, index=None) -> LineResult:
    selection = _resolve_selection(
        model_or_selection,
        selector=selector,
        mode=mode,
        components=components,
        additive=additive,
        index=index,
    )

    if isinstance(selection, SelectedLineCollection):
        arrays = {name: np.full(selection.shape, np.nan, dtype=float) for name in _METRIC_NAMES}
        wave_min = np.full(selection.shape, np.nan, dtype=float)
        wave_max = np.full(selection.shape, np.nan, dtype=float)
        n_grid = np.zeros(selection.shape, dtype=int)
        for flat_index in range(selection.source_result.n_spaxels):
            spatial_index = np.unravel_index(flat_index, selection.shape)
            m = selection.get_profile(flat_index).measure(x=x, window=window, num=num)
            for name in _METRIC_NAMES:
                arrays[name][spatial_index] = getattr(m, name).value
            wave_min[spatial_index] = m.wave_min
            wave_max[spatial_index] = m.wave_max
            n_grid[spatial_index] = m.n_grid
        return MultiLineMeasurements(
            selector=selection.selector,
            mode=selection.mode,
            shape=selection.shape,
            wave_min=wave_min,
            wave_max=wave_max,
            n_grid=n_grid,
            **arrays,
        )

    if x is None:
        if window is None:
            window = selection.infer_window()
        x_arr = np.linspace(float(window[0]), float(window[1]), int(num), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 1 or x_arr.size < 3:
            raise ValueError("x must be a one-dimensional wavelength grid with at least three samples.")

    y_arr = np.asarray(selection.evaluate(x_arr), dtype=float)
    metrics = _compute_profile_metrics(x_arr, y_arr)
    return LineResult(
        selector=selection.selector,
        mode=selection.mode,
        wave_min=float(x_arr[0]),
        wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size),
        metrics=metrics,
    )


def _fit_param_names(model):
    _, fit_indices, _ = model_to_fit_params(model)
    names = []
    cumulative_idx = 0
    for pname in model.param_names:
        param = getattr(model, pname)
        for offset in range(param.size):
            if cumulative_idx in fit_indices:
                names.append(pname if param.size == 1 else f"{pname}[{offset}]")
            cumulative_idx += 1
    return names


def _selection_parameter_specs(selection):
    components = get_components(selection.source_model, additive=selection.additive)
    specs = []
    seen = set()

    for entry in selection.entries:
        component = components[entry.component_key]
        if entry.template_name is None:
            relevant_names = list(component.param_names)
        else:
            idx, _ = _linegroup_template_index(component, entry.template_name)
            relevant_names = [component._param_names_list[idx]] + list(component._shared_params.keys())

        for param_name in relevant_names:
            param = getattr(component, param_name)
            key = (entry.component_key, param_name)
            if key in seen:
                continue

            pvalue = float(np.asarray(param.value, dtype=float))

            std_raw = getattr(param, 'std', None)
            std_val = (
                float(np.asarray(std_raw, dtype=float))
                if std_raw is not None else float('nan')
            )
            lolim_raw = getattr(param, 'lolim', None)
            uplim_raw = getattr(param, 'uplim', None)
            lolim_val = (
                float(np.asarray(lolim_raw, dtype=float))
                if lolim_raw is not None else float('nan')
            )
            uplim_val = (
                float(np.asarray(uplim_raw, dtype=float))
                if uplim_raw is not None else float('nan')
            )

            has_std = np.isfinite(std_val) and std_val > 0.0
            has_limits = (
                np.isfinite(lolim_val) and np.isfinite(uplim_val) and uplim_val > lolim_val
            )
            if not (has_std or has_limits):
                continue

            raw_bounds = getattr(param, 'bounds', (None, None))
            bounds = tuple(raw_bounds) if raw_bounds is not None else (None, None)

            specs.append(
                _SelectionParameter(
                    component_key=entry.component_key,
                    param_name=param_name,
                    value=pvalue,
                    std=std_val,
                    lolim=lolim_val,
                    uplim=uplim_val,
                    bounds=bounds,
                )
            )
            seen.add(key)

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


def _apply_sampled_values(model, sampled_values):
    for name, value in sampled_values.items():
        if '[' in name:
            base_name, index_str = name.split('[', 1)
            idx = int(index_str.rstrip(']'))
            param = getattr(model, base_name)
            updated = np.array(param.value, copy=True)
            updated.flat[idx] = value
            param.value = updated
            continue
        getattr(model, name).value = value
    return model


def _clip_draws_to_bounds(specs, draws):
    """Clip parameter draws in-place to parameter bounds."""
    for j, spec in enumerate(specs):
        lo, hi = spec.bounds
        if lo is not None:
            draws[:, j] = np.maximum(draws[:, j], lo)
        if hi is not None:
            draws[:, j] = np.minimum(draws[:, j], hi)


def _draw_selection_samples(selection, n_samples, method='auto', distribution='auto',
                            random_state=None):
    specs = _selection_parameter_specs(selection)
    if not specs:
        raise ValueError(
            "No uncertainty information (std, lolim, or uplim) is attached to the selected "
            "line components. Run covar_attach or resample_attach on the fitted model first."
        )

    method = str(method).strip().lower()
    distribution = str(distribution).strip().lower()
    if method not in {'auto', 'covariance', 'limits', 'std'}:
        raise ValueError("method must be one of: 'auto', 'covariance', 'limits', 'std'.")
    if distribution not in {'auto', 'uniform', 'gaussian'}:
        raise ValueError("distribution must be one of: 'auto', 'uniform', 'gaussian'.")
    if method == 'limits' and distribution == 'gaussian':
        raise ValueError(
            "method='limits' with distribution='gaussian' is not supported "
            "(asymmetric Gaussian is not straightforward)."
        )
    if method == 'std' and distribution == 'uniform':
        raise ValueError(
            "method='std' with distribution='uniform' is not supported "
            "(std alone cannot define a uniform range)."
        )

    rng = np.random.default_rng(random_state)
    cov = getattr(selection.source_model, '_param_cov', None)
    parent_name_map = _selection_parent_name_map(selection, specs)

    enriched = [
        _SelectionParameter(
            component_key=spec.component_key,
            param_name=spec.param_name,
            value=spec.value,
            std=spec.std,
            lolim=spec.lolim,
            uplim=spec.uplim,
            bounds=spec.bounds,
            parent_name=parent_name_map.get((spec.component_key, spec.param_name)),
        )
        for spec in specs
    ]

    # Strategy 1: multivariate Normal from full parameter covariance (best for correlated params)
    if method in ('auto', 'covariance') and cov is not None:
        free_names = _fit_param_names(selection.source_model)
        cov_specs = [s for s in enriched if s.parent_name is not None and s.parent_name in free_names]
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
                            RuntimeWarning,
                        )
                        draws = rng.multivariate_normal(mean, subcov, size=int(n_samples))
                    _clip_draws_to_bounds(cov_specs, draws)
                    return cov_specs, draws
                except np.linalg.LinAlgError:
                    if method == 'covariance':
                        raise
    if method == 'covariance':
        raise ValueError(
            "method='covariance' requested but no usable covariance matrix is available "
            "for the selected parameters."
        )

    # Strategy 2 & 3: per-parameter draws
    draws = np.empty((int(n_samples), len(enriched)), dtype=float)
    for j, spec in enumerate(enriched):
        lo, up = spec.lolim, spec.uplim
        has_limits = np.isfinite(lo) and np.isfinite(up) and up > lo
        has_std = np.isfinite(spec.std) and spec.std > 0.0

        drawn = False
        if method in ('auto', 'limits') and has_limits and distribution in ('auto', 'uniform'):
            draws[:, j] = rng.uniform(lo, up, size=int(n_samples))
            drawn = True
        if not drawn and method in ('auto', 'std') and has_std and distribution in ('auto', 'gaussian'):
            draws[:, j] = rng.normal(spec.value, spec.std, size=int(n_samples))
            drawn = True
        if not drawn:
            if method == 'auto':
                draws[:, j] = spec.value  # no uncertainty → constant
            else:
                raise ValueError(
                    f"Parameter '{spec.param_name}' of component '{spec.component_key}' "
                    f"lacks the uncertainty information required by method='{method}'."
                )

    _clip_draws_to_bounds(enriched, draws)
    return enriched, draws


def _apply_selection_draw(sampled_model, selection, specs, draw):
    components = get_components(sampled_model, additive=selection.additive)
    for spec, value in zip(specs, draw):
        getattr(components[spec.component_key], spec.param_name).value = value
    return sampled_model


def sample_line_measurements(model_or_selection, selector=None, mode='auto', components=None,
                             additive=True, x=None, window=None, num=4096,
                             n_samples=256, confidence=68,
                             method='auto', distribution='auto',
                             random_state=None,
                             return_samples=False, index=None) -> LineResult:
    selection = _resolve_selection(
        model_or_selection,
        selector=selector,
        mode=mode,
        components=components,
        additive=additive,
        index=index,
    )
    if isinstance(selection, SelectedLineCollection):
        raise ValueError(
            "sample_line_measurements does not support full MultiFitResult batches. "
            "Pass index=... to sample one spaxel, or call collection[index].sample(...)."
        )

    if x is None:
        if window is None:
            window = selection.infer_window()
        x_arr = np.linspace(float(window[0]), float(window[1]), int(num), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float)

    model = selection.source_model
    specs, draws = _draw_selection_samples(
        selection, n_samples=n_samples,
        method=method, distribution=distribution,
        random_state=random_state,
    )

    y_nominal = np.asarray(selection.evaluate(x_arr), dtype=float)
    nominal_metrics = _compute_profile_metrics(x_arr, y_nominal)

    # --- Optimized sampling loop ---
    # Single working copy: avoids per-draw model.copy() + get_components overhead.
    working_model = model.copy()
    working_components = get_components(working_model, additive=selection.additive)
    eval_plan = [
        (entry, working_components[entry.component_key])
        for entry in selection.entries
    ]

    sample_records = []
    for draw in draws:
        for spec, value in zip(specs, draw):
            getattr(working_components[spec.component_key], spec.param_name).value = value
        y_sampled = np.zeros_like(x_arr, dtype=float)
        for entry, comp in eval_plan:
            if entry.template_name is None:
                vals, _, _ = _evaluate_component_profile(comp, x_arr)
            else:
                vals, _, _, _ = _evaluate_linegroup_template(comp, entry.template_name, x_arr)
            y_sampled += vals
        sample_records.append(_compute_profile_metrics(x_arr, y_sampled))

    # --- Aggregate with boundary-aware limits (via extract_limits) ---
    alpha = (100.0 - float(confidence)) / 2.0
    sampled_metrics = {}
    for name in _METRIC_NAMES:
        nominal_value = nominal_metrics.get(name, float('nan'))
        values = np.array([rec.get(name, float('nan')) for rec in sample_records], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            sampled_metrics[name] = Metric(value=float(nominal_value))
        else:
            bounds_proxy = _METRIC_BOUNDS.get(name, _NO_BOUNDS)
            lo, up = extract_limits(bounds_proxy, values, alpha, 100.0 - alpha)
            sampled_metrics[name] = Metric(
                value=float(nominal_value),
                std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                lolim=float(lo),
                uplim=float(up),
            )

    if np.any(~np.isfinite(
            np.array([rec.get('fwhm', float('nan')) for rec in sample_records]))):
        warnings.warn(
            "Some sampled profiles have undefined FWHM because the half-maximum "
            "contour is not single-valued.",
            RuntimeWarning,
        )

    result = LineResult(
        selector=selection.selector,
        mode=selection.mode,
        wave_min=float(x_arr[0]),
        wave_max=float(x_arr[-1]),
        n_grid=int(x_arr.size),
        metrics=sampled_metrics,
    )
    if return_samples:
        return result, pd.DataFrame(sample_records)
    return result


__all__ = [
    'LineResult',
    'Metric',
    'MultiLineMeasurements',
    'SelectedLineProfile',
    'SelectedLineCollection',
    'measure_line',
    'sample_line_measurements',
    'select_line',
]