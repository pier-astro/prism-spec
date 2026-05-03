"""
prism.modeling.models.line_analysis.selection — line selection and evaluation.

Provides ``select_line``, ``SelectedLineProfile``, ``SelectedLineCollection``.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import astropy.units as u

from ..models.components import get_components
from ..models.lines import (
    GaussianLine, GaussianLines,
    LineGroupBase,
    LorentzianLine, LorentzianLines,
    Metric,
    VoigtLine, VoigtLines,
    sigma2fwhm,
)
from ..models.lines import profiles

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

_SINGLE_LINE_TYPES = (GaussianLine, LorentzianLine, VoigtLine)
_GROUP_LINE_TYPES = (GaussianLines, LorentzianLines, VoigtLines)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _selector_token(value) -> str:
    return str(value).strip().lower()


def _selector_position_value(position, position_unit=None) -> float | None:
    """Convert position to native model unit.
    
    If position_unit is not provided, position is returned as-is (assumed native).
    If provided, performs unit conversion to the target position_unit.
    """
    if position is None:
        return None
    if position_unit is None:
        return float(position)
    if isinstance(position, u.Quantity):
        target_unit = u.Unit(position_unit) if not isinstance(position_unit, u.Unit) else position_unit
        return float(position.to_value(target_unit, equivalencies=u.spectral()))
    return float(position)


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
    raise KeyError(
        f"Template '{template_name}' not found in component "
        f"'{getattr(model, 'name', type(model).__name__)}'.")


def _resolve_linegroup_member(model, template_name, position=None):
    idx, resolved = _linegroup_template_index(model, template_name)
    positions = np.asarray(model._tmpl_positions[idx], dtype=float)
    weights = np.asarray(model._tmpl_weights[idx], dtype=float)

    if positions.size == 0:
        raise ValueError(
            f"Template '{template_name}' in component "
            f"'{getattr(model, 'name', type(model).__name__)}' has no lines.")

    selector_position = _selector_position_value(position)
    pos_unit = str(model._position_unit) if hasattr(model, '_position_unit') else "native units"
    if selector_position is not None:
        matches = np.flatnonzero(np.isclose(positions, selector_position, rtol=0.0, atol=1e-3))
        if matches.size == 0:
            available = ', '.join(f"{value:.4f}" for value in positions)
            raise ValueError(
                f"No line at {selector_position:.4f} {pos_unit} found for template "
                f"'{template_name}' in component '{getattr(model, 'name', type(model).__name__)}'. "
                f"Available positions: {available} {pos_unit}")
        if matches.size > 1:
            raise ValueError(
                f"Template '{template_name}' in component "
                f"'{getattr(model, 'name', type(model).__name__)}' has multiple "
                f"lines matching {selector_position:.4f} {pos_unit}.")
        member_idx = int(matches[0])
    else:
        anchors = np.flatnonzero(np.isclose(weights, 1.0, rtol=0.0, atol=1e-12))
        if anchors.size == 1:
            member_idx = int(anchors[0])
        elif anchors.size > 1:
            available = ', '.join(f"{positions[i]:.4f}" for i in anchors)
            raise ValueError(
                f"Ambiguous selector '{template_name}' in component "
                f"'{getattr(model, 'name', type(model).__name__)}': multiple lines "
                f"have weight=1 ({available} {pos_unit}). Pass position=... to disambiguate.")
        elif positions.size == 1:
            member_idx = 0
        else:
            available = ', '.join(
                f"{pos:.4f} (w={weight:g})" for pos, weight in zip(positions, weights))
            raise ValueError(
                f"Ambiguous selector '{template_name}' in component "
                f"'{getattr(model, 'name', type(model).__name__)}': no unique weight=1 line. "
                f"Pass position=... to disambiguate. Candidates: {available} {pos_unit}")

    return idx, resolved, member_idx, float(positions[member_idx]), float(weights[member_idx])


def _evaluate_linegroup_template(model, template_name, x, position=None):
    """Evaluate one resolved line within a ``LineGroupBase`` component."""
    idx, resolved, member_idx, member_pos, member_weight = _resolve_linegroup_member(
        model, template_name, position=position)
    amplitude_param = getattr(model, model._param_names_list[idx])
    amplitude = model._parameter_payload(amplitude_param)
    amplitude_unit = getattr(amplitude_param, 'unit', None)
    shared_values = [getattr(model, pname).value
                     for pname in model._shared_params.keys()]

    x_native = model._coerce_domain_axis(x)
    is_scalar = np.ndim(x) == 0
    x_native = np.atleast_1d(x_native)
    total = np.zeros_like(x_native, dtype=float)
    centers = []
    widths = []

    args = model._single_profile_args(
        member_pos, amplitude, member_weight, *shared_values)
    total += model._profile_func(x_native, *args)
    centers.append(float(args[1]))
    widths.append(_profile_width_from_args(model, args))

    values = total[0] if is_scalar else total
    return (values, np.asarray(centers, dtype=float),
            np.asarray(widths, dtype=float), resolved)


def _evaluate_single_line(model, x):
    values = model(x)
    center = profiles.observed_center(
        model.position.value, model.offset.value, model.redshift.value)

    if isinstance(model, (GaussianLine, LorentzianLine)):
        fwhm = model.fwhm.value / profiles.C_KMS * center
    elif isinstance(model, VoigtLine):
        fwhm_g = model.fwhm_G.value / profiles.C_KMS * center
        fwhm_l = model.fwhm_L.value / profiles.C_KMS * center
        fwhm = float(_approx_voigt_fwhm(fwhm_g, fwhm_l))
    else:
        raise TypeError(f"Unsupported line model type: {type(model).__name__}")

    return values, np.asarray([center], dtype=float), np.asarray([fwhm], dtype=float)


def _evaluate_component_profile(model, x):
    """Evaluate a full component profile, returning values, centers, widths."""
    if isinstance(model, LineGroupBase):
        values = model(x)
        centers = []
        widths = []
        shared_values = [getattr(model, pname).value
                         for pname in model._shared_params.keys()]
        for idx in range(model._n_templates):
            amplitude_param = getattr(model, model._param_names_list[idx])
            amplitude = model._parameter_payload(amplitude_param)
            amplitude_unit = getattr(amplitude_param, 'unit', None)
            for pos, weight in zip(model._tmpl_positions[idx],
                                   model._tmpl_weights[idx]):
                args = model._single_profile_args(
                    pos, amplitude, weight, *shared_values)
                centers.append(float(args[1]))
                widths.append(_profile_width_from_args(model, args))
        return (values, np.asarray(centers, dtype=float),
                np.asarray(widths, dtype=float))

    if isinstance(model, _SINGLE_LINE_TYPES):
        return _evaluate_single_line(model, x)

    raise TypeError(
        f"Unsupported component type for line analysis: {type(model).__name__}")


def _is_multifit_like(obj) -> bool:
    return (hasattr(obj, 'n_spaxels')
            and hasattr(obj, 'get_model')
            and hasattr(obj, 'shape'))


def _component_name(component, fallback):
    name = getattr(component, 'name', None)
    return str(name) if name else str(fallback)


def _normalize_unit(unit):
    if unit in (None, ''):
        return None
    return u.Unit(unit)


def _merge_output_units(units):
    resolved = None
    saw_unitless = False
    saw_unitful = False
    for unit in units:
        unit = _normalize_unit(unit)
        if unit is None:
            saw_unitless = True
            continue
        saw_unitful = True
        if resolved is None:
            resolved = unit
            continue
        try:
            (1.0 * unit).to(resolved)
        except Exception as exc:
            raise ValueError(
                "Selected line combines components with incompatible output units.") from exc
    if saw_unitless and saw_unitful:
        raise ValueError(
            "Selected line cannot mix unitless and quantity-valued components.")
    return resolved

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SelectionEntry:
    component_key: str | int
    component_name: str
    template_name: str | None = None
    template_position: float | None = None


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

# ---------------------------------------------------------------------------
# SelectedLineProfile
# ---------------------------------------------------------------------------

class SelectedLineProfile:
    """
    A selected emission-line profile extracted from a composite model.

    This object acts like a standalone 1D model and can be evaluated on a 
    wavelength grid. It also provides methods to inspect its components separately,
    measure its properties, or sample over uncertainties via Monte Carlo.

    Parameters
    ----------
    source_model : astropy.modeling.Model
        The original compound model. 
    selector : str
        The line tag or name of the component(s) corresponding to this profile.
    entries : sequence of _SelectionEntry
        Information linking the selected components from the source model to this profile.
    additive : bool
        Whether this line selection restricted its component search to the additive block.
    """

    def __init__(self, source_model, selector, entries, additive=True):
        self.source_model = source_model
        self.selector = str(selector)
        self.entries = tuple(entries)
        self.additive = bool(additive)
        # Cache the position unit from the first component to avoid silent assumptions.
        # All components must have the same domain_family (checked in domain_family property),
        # so they all share the same position unit.
        components = get_components(source_model, additive=additive)
        self._cached_position_unit = None
        if self.entries:
            first_component = components[self.entries[0].component_key]
            self._cached_position_unit = getattr(first_component, '_position_unit', None)

    def __repr__(self):
        return (
            f"SelectedLineProfile(selector={self.selector!r}, "
            f"components={len(self.entries)})")

    @property
    def component_names(self):
        return [entry.component_name for entry in self.entries]

    def copy_for_model(self, model):
        return SelectedLineProfile(
            source_model=model, selector=self.selector,
            entries=self.entries, additive=self.additive)

    def _iter_components(self):
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            yield components[entry.component_key]

    @property
    def domain_family(self):
        domains = {getattr(component, 'domain_family', 'wavelength')
                   for component in self._iter_components()}
        if len(domains) != 1:
            raise ValueError("Selected line combines components from different domains.")
        return domains.pop()

    @property
    def position_unit(self):
        """The spectral unit of the line positions stored in the model.
        
        This is the native unit in which positions are parameterized,
        read directly from the line models' _position_unit attribute.
        """
        if self._cached_position_unit is None:
            raise ValueError(
                "Cannot determine position unit: no spectral unit found on the "
                "selected components. This should not happen for valid line models.")
        return self._cached_position_unit

    @property
    def axis_unit(self):
        """Alias for position_unit for backward compatibility."""
        return self.position_unit

    @property
    def output_unit(self):
        return _merge_output_units(
            getattr(component, '_model_output_unit', lambda: None)()
            for component in self._iter_components())

    @property
    def flux_unit(self):
        return None if self.output_unit is None else self.output_unit * self.axis_unit

    @property
    def eqw_unit(self):
        return None if self.output_unit is None else self.axis_unit

    def _coerce_domain_axis(self, x):
        if isinstance(x, u.Quantity):
            values = x.to_value(self.axis_unit, equivalencies=u.spectral())
        else:
            values = x
        return np.atleast_1d(np.asarray(values, dtype=float))

    def _coerce_output_array(self, values, name='value'):
        output_unit = self.output_unit
        if isinstance(values, u.Quantity):
            if output_unit is None:
                raise ValueError(
                    f"{name} was provided as a Quantity, but this selection has no output unit.")
            return np.asarray(values.to_value(output_unit), dtype=float)
        return np.asarray(values, dtype=float)

    def _coerce_output_scalar(self, value, name='value'):
        output_unit = self.output_unit
        if isinstance(value, u.Quantity):
            if output_unit is None:
                raise ValueError(
                    f"{name} was provided as a Quantity, but this selection has no output unit.")
            return float(np.asarray(value.to_value(output_unit), dtype=float).reshape(-1)[0])
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    def _restore_output(self, values, is_scalar=False):
        out = np.asarray(values, dtype=float)
        if self.output_unit is not None:
            out = out * self.output_unit
        return out[0] if is_scalar else out

    def _probe_axis_value(self, component, entry):
        """Get the probe position in the model's native position unit.
        
        For single-line components, position comes from the parameter value.
        For template groups, position comes from the template entry.
        Both are already in the model's _position_unit, so no conversion needed.
        """
        if entry.template_name is None:
            position = float(getattr(component.position, 'value', component.position))
        else:
            position = float(entry.template_position)
        return np.asarray([float(position)], dtype=float)

    def _iter_evaluated_entries(self, x):
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                yield _evaluate_component_profile(component, x)
            else:
                values, centers, widths, _ = _evaluate_linegroup_template(
                    component, entry.template_name, x,
                    position=entry.template_position)
                yield values, centers, widths

    def infer_window(self, padding=6.0, min_width=2.0):
        all_centers, all_widths = [], []
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            component = components[entry.component_key]
            probe_x = self._probe_axis_value(component, entry)
            if entry.template_name is None:
                _, centers, widths = _evaluate_component_profile(
                    component, probe_x)
            else:
                _, centers, widths, _ = _evaluate_linegroup_template(
                    component, entry.template_name, probe_x)
            all_centers.extend(np.atleast_1d(centers).tolist())
            all_widths.extend(np.atleast_1d(widths).tolist())

        centers = np.asarray(all_centers, dtype=float)
        widths = np.asarray(all_widths, dtype=float)
        widths = np.where(
            np.isfinite(widths) & (widths > 0.0), widths, min_width)
        if centers.size == 0:
            raise ValueError(
                "Cannot infer a spectral window for an empty line selection.")
        window = np.array([
            float(np.min(centers - padding * widths)),
            float(np.max(centers + padding * widths)),
        ], dtype=float)
        # Window is already in the model's native position_unit; return as-is.
        return float(window[0]), float(window[1])

    def evaluate(self, x):
        is_scalar = np.ndim(x) == 0
        x_arr = self._coerce_domain_axis(x)
        total = np.zeros_like(x_arr, dtype=float)
        for values, _, _ in self._iter_evaluated_entries(x_arr):
            total += self._coerce_output_array(values, name='line profile')
        return self._restore_output(total, is_scalar=is_scalar)

    def __call__(self, x):
        return self.evaluate(x)

    @property
    def position(self) -> float:
        """Rest-frame position of the selected line in the model's native unit.
        
        Returns the mean position of all selected components, expressed in
        position_unit (e.g., Å for wavelength models or eV for energy models).
        Issues a warning if positions differ significantly (>0.1% spread).
        """
        components = get_components(self.source_model, additive=self.additive)
        positions, labels = [], []
        unit_str = str(self.position_unit)
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                pos_param = getattr(component, 'position', None)
                if pos_param is not None:
                    p = float(pos_param.value)
                    positions.append(p)
                    labels.append(f"{entry.component_name}: {p:.4f} {unit_str}")
            else:
                p = float(entry.template_position)
                positions.append(p)
                labels.append(
                    f"{entry.component_name}/{entry.template_name}: "
                    f"{p:.4f} {unit_str}")

        if not positions:
            raise ValueError(
                f"Cannot determine rest position for selector "
                f"'{self.selector}'.")

        positions = np.asarray(positions, dtype=float)
        mean_pos = float(np.mean(positions))
        spread = float(np.max(positions) - np.min(positions))
        tol = 0.001 * mean_pos
        if spread > tol:
            warnings.warn(
                f"Rest positions of the '{self.selector}' selected components "
                f"differ by {spread:.4f} {unit_str} (tolerance {tol:.4f} {unit_str}):\n"
                + "\n".join(f"  {lbl}" for lbl in labels),
                UserWarning, stacklevel=2)
        return mean_pos

    def evaluate_components(self, x=None, window=None, num=4096) -> dict:
        """Evaluate each selected component separately."""
        if x is None:
            if window is None:
                window = self.infer_window()
            x_arr = np.linspace(
                float(window[0]), float(window[1]), int(num), dtype=float)
        else:
            x_arr = self._coerce_domain_axis(x)

        components = get_components(self.source_model, additive=self.additive)
        result = {}
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                values, _, _ = _evaluate_component_profile(component, x_arr)
                key = entry.component_name
            else:
                values, _, _, _ = _evaluate_linegroup_template(
                    component, entry.template_name, x_arr,
                    position=entry.template_position)
                key = entry.component_name
            if key in result:
                key = f"{entry.component_name}/{entry.template_name or '*'}"
            result[key] = self._restore_output(
                self._coerce_output_array(values, name='component profile'))
        return result

    def measure(self, x=None, window=None, num=4096):
        from .measurement import measure_line
        return measure_line(self, x=x, window=window, num=num)

    def sample(self, n_samples=256, confidence=68, x=None, window=None,
               num=4096, method='auto', distribution='auto',
               random_state=None, return_samples=False):
        from .measurement import sample_line_measurements
        return sample_line_measurements(
            self, n_samples=n_samples, confidence=confidence,
            x=x, window=window, num=num,
            method=method, distribution=distribution,
            random_state=random_state, return_samples=return_samples)

    def eqw(self, continuum=None, x=None, window=None, num=2048) -> Metric:
        """Equivalent width of the selected line profile.

        Parameters
        ----------
        continuum : None | float | callable, optional
        x : array-like, optional
        window : (float, float), optional
        num : int

        Returns
        -------
        Metric
        """
        if x is None:
            if window is None:
                window = self.infer_window()
            x_arr = np.linspace(
                float(window[0]), float(window[1]), int(num), dtype=float)
        else:
            x_arr = self._coerce_domain_axis(x)

        y_line = self._coerce_output_array(self.evaluate(x_arr), name='line profile')
        flux = float(_trapezoid(y_line, x_arr))

        if continuum is None:
            _wb = window if window is not None else self.infer_window()
            sigma_est = (_wb[1] - _wb[0]) / (12.0 * 2.3548)
            centroid = float(self.position)
            # Work in the model's native position unit; no conversion needed.
            x_cont_edges = np.array([centroid - sigma_est, centroid + sigma_est], dtype=float)
            x_cont = np.linspace(float(np.min(x_cont_edges)), float(np.max(x_cont_edges)), 17, dtype=float)
            cont_vals = (
                self._coerce_output_array(self.source_model(x_cont), name='continuum model')
                - self._coerce_output_array(self.evaluate(x_cont), name='line profile'))
            _fin = cont_vals[np.isfinite(cont_vals)]
            cont_c = float(np.median(_fin)) if _fin.size > 0 else float('nan')
            ew_val = (flux / cont_c) if (
                np.isfinite(cont_c) and cont_c > 0.0) else float('nan')
            result = Metric(value=ew_val, unit=self.eqw_unit)
        elif callable(continuum):
            y_cont = self._coerce_output_array(continuum(x_arr), name='continuum')
            safe = np.where(y_cont > 0.0, y_cont, np.nan)
            ew_val = float(_trapezoid(y_line / safe, x_arr))
            result = Metric(value=ew_val, unit=self.eqw_unit)
        else:
            cont_val = self._coerce_output_scalar(continuum, name='continuum')
            ew_val = (flux / cont_val) if (
                np.isfinite(cont_val) and cont_val != 0.0) else float('nan')
            result = Metric(value=ew_val, unit=self.eqw_unit)

        self.ew = result
        return result

    def sample_eqw(self, continuum=None, n_samples=256, confidence=68,
                   x=None, window=None, num=2048,
                   method='auto', distribution='auto',
                   perturb_continuum=False, random_state=None) -> Metric:
        """Equivalent width with Monte Carlo uncertainty propagation."""
        from .measurement import (
            _draw_selection_samples, _clip_draws_to_bounds,
            _non_line_parameter_specs, _compute_ew_numerical,
        )
        from ..fitting.uncertainty.resample import extract_limits

        if x is None:
            if window is None:
                window = self.infer_window()
            x_arr = np.linspace(
                float(window[0]), float(window[1]), int(num), dtype=float)
        else:
            x_arr = self._coerce_domain_axis(x)

        # Nominal EW
        y_line_nom = self._coerce_output_array(self.evaluate(x_arr), name='line profile')
        flux_nom = float(_trapezoid(y_line_nom, x_arr))
        _wb = window if window is not None else self.infer_window()

        if continuum is None:
            sigma_est = (_wb[1] - _wb[0]) / (12.0 * 2.3548)
            centroid = float(self.position)
            # Work in the model's native position unit; no conversion needed.
            x_cont_edges = np.array([centroid - sigma_est, centroid + sigma_est], dtype=float)
            x_cont = np.linspace(float(np.min(x_cont_edges)), float(np.max(x_cont_edges)), 17, dtype=float)
            _cvals = (
                self._coerce_output_array(self.source_model(x_cont), name='continuum model')
                - self._coerce_output_array(self.evaluate(x_cont), name='line profile'))
            _fin = _cvals[np.isfinite(_cvals)]
            _cc = float(np.median(_fin)) if _fin.size > 0 else float('nan')
            ew_nominal = (flux_nom / _cc) if (
                np.isfinite(_cc) and _cc > 0.0) else float('nan')
            y_cont_nom = None
        elif callable(continuum):
            y_cont_nom = self._coerce_output_array(continuum(x_arr), name='continuum')
            ew_nominal = _compute_ew_numerical(x_arr, y_line_nom, y_cont_nom)
            x_cont = None
        else:
            _cv = self._coerce_output_scalar(continuum, name='continuum')
            ew_nominal = (flux_nom / _cv) if (
                np.isfinite(_cv) and _cv != 0.0) else float('nan')
            y_cont_nom = None
            x_cont = None

        # Draw line parameter samples
        if isinstance(random_state, np.random.Generator):
            line_rng = random_state
            _cont_rng = np.random.default_rng()
        else:
            _ss = np.random.SeedSequence(random_state)
            _lc, _cc_seed = _ss.spawn(2)
            line_rng = int(_lc.generate_state(1)[0] & 0x7FFFFFFF)
            _cont_rng = np.random.default_rng(_cc_seed)

        specs, draws = _draw_selection_samples(
            self, n_samples=n_samples,
            method=method, distribution=distribution,
            random_state=line_rng)

        # Optional continuum perturbation
        cont_specs = []
        cont_draws = None
        if perturb_continuum and continuum is None:
            cont_specs = _non_line_parameter_specs(self.source_model, self)
            if cont_specs:
                cont_draws = np.empty((int(n_samples), len(cont_specs)),
                                      dtype=float)
                for j, spec in enumerate(cont_specs):
                    lo, up = spec.lolim, spec.uplim
                    has_lim = (np.isfinite(lo) and np.isfinite(up) and up > lo)
                    has_std = np.isfinite(spec.std) and spec.std > 0.0
                    if has_lim:
                        cont_draws[:, j] = _cont_rng.uniform(
                            lo, up, size=int(n_samples))
                    elif has_std:
                        cont_draws[:, j] = _cont_rng.normal(
                            spec.value, spec.std, size=int(n_samples))
                    else:
                        cont_draws[:, j] = spec.value
                _clip_draws_to_bounds(cont_specs, cont_draws)

        # Sampling loop
        working_model = self.source_model.copy()
        working_comps = get_components(working_model, additive=self.additive)
        all_comps_flat = get_components(working_model, additive=False)
        eval_plan = [(entry, working_comps[entry.component_key])
                     for entry in self.entries]

        ew_samples = np.empty(int(n_samples), dtype=float)
        for i, draw in enumerate(draws):
            for spec, value in zip(specs, draw):
                getattr(working_comps[spec.component_key],
                        spec.param_name).value = value
            if cont_draws is not None:
                for spec, value in zip(cont_specs, cont_draws[i]):
                    comp = all_comps_flat.get(spec.component_key, None)
                    if comp is not None:
                        getattr(comp, spec.param_name).value = value

            y_line = np.zeros_like(x_arr, dtype=float)
            for entry, comp in eval_plan:
                if entry.template_name is None:
                    vals, _, _ = _evaluate_component_profile(comp, x_arr)
                else:
                    vals, _, _, _ = _evaluate_linegroup_template(
                        comp, entry.template_name, x_arr)
                y_line += vals

            flux_draw = float(_trapezoid(y_line, x_arr))

            if continuum is None:
                y_lxc = np.zeros(len(x_cont), dtype=float)
                for entry, comp in eval_plan:
                    if entry.template_name is None:
                        vc, _, _ = _evaluate_component_profile(comp, x_cont)
                    else:
                        vc, _, _, _ = _evaluate_linegroup_template(
                            comp, entry.template_name, x_cont)
                    y_lxc += self._coerce_output_array(vc, name='line profile').ravel()[:len(x_cont)]
                cvd = self._coerce_output_array(working_model(x_cont), name='continuum model') - y_lxc
                _fd = cvd[np.isfinite(cvd)]
                cc = float(np.median(_fd)) if _fd.size > 0 else float('nan')
                ew_samples[i] = (flux_draw / cc) if (
                    np.isfinite(cc) and cc > 0.0) else float('nan')
            elif callable(continuum):
                ew_samples[i] = _compute_ew_numerical(
                    x_arr, y_line, y_cont_nom)
            else:
                ew_samples[i] = (flux_draw / _cv) if (
                    np.isfinite(_cv) and _cv != 0.0) else float('nan')

        # Aggregate
        finite = ew_samples[np.isfinite(ew_samples)]
        if finite.size == 0:
            result = Metric(value=float(ew_nominal), unit=self.eqw_unit)
            self.ew = result
            return result

        alpha = (100.0 - float(confidence)) / 2.0
        lo, up = extract_limits(
            _BoundsProxy(), ew_samples, alpha, 100.0 - alpha)
        result = Metric(
            value=float(ew_nominal),
            std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            lolim=float(lo), uplim=float(up),
            unit=self.eqw_unit)
        self.ew = result
        return result


class _BoundsProxy:
    __slots__ = ('min', 'max')
    def __init__(self, min_val=None, max_val=None):
        self.min = min_val
        self.max = max_val

# ---------------------------------------------------------------------------
# SelectedLineCollection (multi-spaxel)
# ---------------------------------------------------------------------------

class SelectedLineCollection:
    """
    A batch of selected emission-line profiles over a MultiFitResult.
    
    Provides spatial evaluations, mappings, and EW computations for multi-spaxel
    datasets where the same line setup is tracked.
    """

    def __init__(self, result, selector, components=None, additive=True,
                 position=None):
        self.source_result = result
        self.selector = str(selector)
        self.components = components
        self.additive = bool(additive)
        self.position = position

    def __repr__(self):
        return (
            f"SelectedLineCollection(selector={self.selector!r}, "
            f"shape={self.source_result.shape})")

    @property
    def shape(self):
        return self.source_result.shape

    @property
    def axis_unit(self):
        return self.get_profile(0).axis_unit

    @property
    def output_unit(self):
        return self.get_profile(0).output_unit

    @property
    def eqw_unit(self):
        return self.get_profile(0).eqw_unit

    def get_profile(self, index):
        model = self.source_result.get_model(index)
        return select_line(
            model, selector=self.selector,
            components=self.components, additive=self.additive,
            position=self.position)

    def __getitem__(self, index):
        return self.get_profile(index)

    def evaluate(self, x=None, spectral_axis=None):
        profile0 = self.get_profile(0)
        eval_x = self.source_result._x if x is None else x
        if eval_x is None:
            raise ValueError(
                "Provide x= or construct the MultiFitResult with x=.")
        eval_x = profile0._coerce_domain_axis(eval_x)
        out = np.empty(self.shape + (eval_x.size,), dtype=float)
        for fi in range(self.source_result.n_spaxels):
            si = np.unravel_index(fi, self.shape)
            profile = self.get_profile(fi)
            out[si] = profile._coerce_output_array(profile(eval_x), name='line profile')
        target = (self.source_result.spectral_axis
                  if spectral_axis is None else spectral_axis)
        if target < 0:
            target += out.ndim
        if target != out.ndim - 1:
            out = np.moveaxis(out, -1, target)
        if self.output_unit is not None:
            return out * self.output_unit
        return out

    def __call__(self, x=None, spectral_axis=None):
        return self.evaluate(x=x, spectral_axis=spectral_axis)

    def measure(self, x=None, window=None, num=4096):
        from .measurement import measure_line
        return measure_line(self, x=x, window=window, num=num)

    def eqw(self, continuum=None, x=None, window=None, num=2048) -> np.ndarray:
        """EW for every spaxel; returns a shaped numpy array."""
        out = np.full(self.shape, np.nan, dtype=float)
        for fi in range(self.source_result.n_spaxels):
            si = np.unravel_index(fi, self.shape)
            prof = self.get_profile(fi)
            m = prof.eqw(continuum=continuum, x=x, window=window, num=num)
            out[si] = m.value
        if self.eqw_unit is not None:
            return out * self.eqw_unit
        return out

# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------

def select_line(model, selector, components=None, additive=True, index=None,
                position=None):
    """
    Select an emission or absorption line profile from a composite model.

    This function isolates the components of a model that represent a specific physical 
    line (e.g., 'Hb4861'). It searches across the model's components and returns a 
    ``SelectedLineProfile`` (or ``SelectedLineCollection`` for multi-fit models) that 
    can be independently evaluated, measured, and sampled.

    Parameters
    ----------
    model : astropy.modeling.Model or MultiFitResult
        The composite model or multi-fit result containing the line components.
        Linear-operator pipe inputs are not supported; pass ``model.left``
        instead.
    selector : str
        The tag identifying the physics line to extract (e.g., 'OIII5007'). 
        The search is case-insensitive.
    components : sequence of str or astropy.modeling.Model, optional
        If provided, the search is restricted to this explicit subset of components.
        For `LineGroup` components in this subset, the function extracts the 
        sub-profile matching `selector`. For single-line components in this subset, 
        they are included unconditionally as part of the selected line.
    additive : bool, default True
        If True (the default), extract components from the additive block of the model,
        ignoring multiplicative components like extinction. 
        Ignored if `components` is explicitly passed.
    index : tuple or int, optional
        Used only if `model` is a `MultiFitResult`. If provided, it extracts the 
        selected line from the specific 1D model at that spatial index. 
        If None, a batch selection (`SelectedLineCollection`) is returned instead.
    position : float or Quantity, optional
        Rest-frame position (in the model's native position_unit) used to 
        disambiguate template groups where multiple physical lines share the 
        same selector. When omitted, grouped selectors default to the unique 
        sub-line with weight=1.

    Returns
    -------
    SelectedLineProfile or SelectedLineCollection
        An object representing the isolated line profile(s).
    """
    if selector is None or str(selector).strip() == '':
        raise ValueError("A line tag or name is required as 'selector'.")

    from ..operators.matop import is_linear_operator_pipe

    if is_linear_operator_pipe(model):
        raise ValueError(
            "line_analysis does not accept linear-operator pipe models. "
            "Pass model.left, or intrinsic components from "
            "get_components(model, additive=True, deconvolve=True)."
        )

    if _is_multifit_like(model):
        if index is not None:
            return select_line(
                model.get_model(index), selector=selector,
                components=components, additive=additive,
                position=position)
        return SelectedLineCollection(
            result=model, selector=selector,
            components=components, additive=additive,
            position=position)

    selector_token = _selector_token(selector)
    entries = []

    if components is not None:
        if not components:
            raise ValueError("Explicit selection requires a non-empty `components` sequence.")
        # Only search within the specified subset
        available = get_components(model, additive=False)  # Pull from all since they asked explicitly
        seen = set()
        
        for comp_obj in components:
            key = comp_obj if isinstance(comp_obj, (str, int)) else getattr(comp_obj, 'name', None)
            if key is None:
                raise ValueError("Explicit component models must have a unique '.name'.")
            
            resolved = available[key]
            comp_name = _component_name(resolved, key)
            if (str(key), comp_name) in seen:
                continue

            if isinstance(resolved, LineGroupBase):
                try:
                    _, template_name, _, template_position, _ = _resolve_linegroup_member(
                        resolved, selector, position=position)
                    entries.append(_SelectionEntry(
                        component_key=key, component_name=comp_name,
                        template_name=template_name,
                        template_position=template_position))
                except KeyError:
                    # LineGroup does not have the selector; skip.
                    pass
            else:
                # Single line: include unconditionally since they passed it in `components`
                entries.append(_SelectionEntry(
                    component_key=key, component_name=comp_name))
            seen.add((str(key), comp_name))
            
        if not entries:
            raise KeyError(f"No line selection found for '{selector}' in the specified components.")
            
    else:
        # Default behavior: search natively across LineGroups (and matching component names as fallback)
        available = get_components(model, additive=additive)
        for key, name in zip(available.indices, available.names):
            comp = available[key]
            if isinstance(comp, LineGroupBase):
                for tmpl_name in comp._templates:
                    if _selector_token(tmpl_name) == selector_token:
                        _, _, _, template_position, _ = _resolve_linegroup_member(
                            comp, tmpl_name, position=position)
                        entries.append(_SelectionEntry(
                            component_key=key, component_name=name,
                            template_name=str(tmpl_name),
                            template_position=template_position))
                        break
            else:
                # Fallback for single components purely matching the name
                if _selector_token(name) == selector_token:
                    entries.append(_SelectionEntry(
                        component_key=key, component_name=name))

        if not entries:
            raise KeyError(f"No line selection found for '{selector}'.")

    return SelectedLineProfile(
        source_model=model, selector=str(selector),
        entries=entries, additive=additive)
