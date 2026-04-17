"""
prism.modeling.models.line_analysis.selection — line selection and evaluation.

Provides ``select_line``, ``SelectedLineProfile``, ``SelectedLineCollection``.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

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


def _evaluate_linegroup_template(model, template_name, x):
    """Evaluate a single template within a ``LineGroupBase`` component."""
    idx, resolved = _linegroup_template_index(model, template_name)
    amplitude = getattr(model, model._param_names_list[idx]).value
    shared_values = [getattr(model, pname).value
                     for pname in model._shared_params.keys()]

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

    return (total, np.asarray(centers, dtype=float),
            np.asarray(widths, dtype=float), resolved)


def _evaluate_single_line(model, x):
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    values = np.asarray(model(x_arr), dtype=float)
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
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if isinstance(model, LineGroupBase):
        values = np.asarray(model(x_arr), dtype=float)
        centers = []
        widths = []
        shared_values = [getattr(model, pname).value
                         for pname in model._shared_params.keys()]
        for idx in range(model._n_templates):
            amplitude = getattr(model, model._param_names_list[idx]).value
            for pos, weight in zip(model._tmpl_positions[idx],
                                   model._tmpl_weights[idx]):
                args = model._single_profile_args(
                    pos, amplitude, weight, *shared_values)
                centers.append(float(args[1]))
                widths.append(_profile_width_from_args(model, args))
        return (values, np.asarray(centers, dtype=float),
                np.asarray(widths, dtype=float))

    if isinstance(model, _SINGLE_LINE_TYPES):
        return _evaluate_single_line(model, x_arr)

    raise TypeError(
        f"Unsupported component type for line analysis: {type(model).__name__}")


def _is_multifit_like(obj) -> bool:
    return (hasattr(obj, 'n_spaxels')
            and hasattr(obj, 'get_model')
            and hasattr(obj, 'shape'))


def _component_name(component, fallback):
    name = getattr(component, 'name', None)
    return str(name) if name else str(fallback)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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

    def _iter_evaluated_entries(self, x):
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                yield _evaluate_component_profile(component, x)
            else:
                values, centers, widths, _ = _evaluate_linegroup_template(
                    component, entry.template_name, x)
                yield values, centers, widths

    def infer_window(self, padding=6.0, min_width=2.0):
        all_centers, all_widths = [], []
        components = get_components(self.source_model, additive=self.additive)
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                _, centers, widths = _evaluate_component_profile(
                    component, np.asarray([0.0]))
            else:
                _, centers, widths, _ = _evaluate_linegroup_template(
                    component, entry.template_name, np.asarray([0.0]))
            all_centers.extend(np.atleast_1d(centers).tolist())
            all_widths.extend(np.atleast_1d(widths).tolist())

        centers = np.asarray(all_centers, dtype=float)
        widths = np.asarray(all_widths, dtype=float)
        widths = np.where(
            np.isfinite(widths) & (widths > 0.0), widths, min_width)
        if centers.size == 0:
            raise ValueError(
                "Cannot infer a wavelength window for an empty line selection.")
        return (float(np.min(centers - padding * widths)),
                float(np.max(centers + padding * widths)))

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
        """Rest-frame wavelength of the selected line (Å)."""
        components = get_components(self.source_model, additive=self.additive)
        positions, labels = [], []
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                pos_param = getattr(component, 'position', None)
                if pos_param is not None:
                    p = float(pos_param.value)
                    positions.append(p)
                    labels.append(f"{entry.component_name}: {p:.4f} Å")
            else:
                idx, _ = _linegroup_template_index(
                    component, entry.template_name)
                for p in component._tmpl_positions[idx]:
                    positions.append(float(p))
                    labels.append(
                        f"{entry.component_name}/{entry.template_name}: "
                        f"{float(p):.4f} Å")

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
                f"differ by {spread:.4f} Å (tolerance {tol:.4f} Å):\n"
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
            x_arr = np.atleast_1d(np.asarray(x, dtype=float))

        components = get_components(self.source_model, additive=self.additive)
        result = {}
        for entry in self.entries:
            component = components[entry.component_key]
            if entry.template_name is None:
                values, _, _ = _evaluate_component_profile(component, x_arr)
                key = entry.component_name
            else:
                values, _, _, _ = _evaluate_linegroup_template(
                    component, entry.template_name, x_arr)
                key = entry.component_name
            if key in result:
                key = f"{entry.component_name}/{entry.template_name or '*'}"
            result[key] = values
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
            x_arr = np.asarray(x, dtype=float)

        y_line = np.asarray(self.evaluate(x_arr), dtype=float)
        flux = float(_trapezoid(y_line, x_arr))

        if continuum is None:
            _wb = window if window is not None else self.infer_window()
            sigma_est = (_wb[1] - _wb[0]) / (12.0 * 2.3548)
            centroid = float(self.position)
            x_cont = np.linspace(
                centroid - sigma_est, centroid + sigma_est, 17, dtype=float)
            cont_vals = (np.asarray(self.source_model(x_cont), dtype=float)
                         - np.asarray(self.evaluate(x_cont), dtype=float))
            _fin = cont_vals[np.isfinite(cont_vals)]
            cont_c = float(np.median(_fin)) if _fin.size > 0 else float('nan')
            ew_val = (flux / cont_c) if (
                np.isfinite(cont_c) and cont_c > 0.0) else float('nan')
            result = Metric(value=ew_val)
        elif callable(continuum):
            y_cont = np.asarray(continuum(x_arr), dtype=float)
            safe = np.where(y_cont > 0.0, y_cont, np.nan)
            ew_val = float(_trapezoid(y_line / safe, x_arr))
            result = Metric(value=ew_val)
        else:
            cont_val = float(continuum)
            ew_val = (flux / cont_val) if (
                np.isfinite(cont_val) and cont_val != 0.0) else float('nan')
            result = Metric(value=ew_val)

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
            x_arr = np.asarray(x, dtype=float)

        # Nominal EW
        y_line_nom = np.asarray(self.evaluate(x_arr), dtype=float)
        flux_nom = float(_trapezoid(y_line_nom, x_arr))
        _wb = window if window is not None else self.infer_window()

        if continuum is None:
            sigma_est = (_wb[1] - _wb[0]) / (12.0 * 2.3548)
            centroid = float(self.position)
            x_cont = np.linspace(
                centroid - sigma_est, centroid + sigma_est, 17, dtype=float)
            _cvals = (np.asarray(self.source_model(x_cont), dtype=float)
                      - np.asarray(self.evaluate(x_cont), dtype=float))
            _fin = _cvals[np.isfinite(_cvals)]
            _cc = float(np.median(_fin)) if _fin.size > 0 else float('nan')
            ew_nominal = (flux_nom / _cc) if (
                np.isfinite(_cc) and _cc > 0.0) else float('nan')
            y_cont_nom = None
        elif callable(continuum):
            y_cont_nom = np.asarray(continuum(x_arr), dtype=float)
            ew_nominal = _compute_ew_numerical(x_arr, y_line_nom, y_cont_nom)
            x_cont = None
        else:
            _cv = float(continuum)
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
                    y_lxc += np.asarray(vc, dtype=float).ravel()[:len(x_cont)]
                cvd = (np.asarray(working_model(x_cont), dtype=float) - y_lxc)
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
            result = Metric(value=float(ew_nominal))
            self.ew = result
            return result

        alpha = (100.0 - float(confidence)) / 2.0
        lo, up = extract_limits(
            _BoundsProxy(), ew_samples, alpha, 100.0 - alpha)
        result = Metric(
            value=float(ew_nominal),
            std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            lolim=float(lo), uplim=float(up))
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

    def __init__(self, result, selector, components=None, additive=True):
        self.source_result = result
        self.selector = str(selector)
        self.components = components
        self.additive = bool(additive)

    def __repr__(self):
        return (
            f"SelectedLineCollection(selector={self.selector!r}, "
            f"shape={self.source_result.shape})")

    @property
    def shape(self):
        return self.source_result.shape

    def get_profile(self, index):
        model = self.source_result.get_model(index)
        return select_line(
            model, selector=self.selector,
            components=self.components, additive=self.additive)

    def __getitem__(self, index):
        return self.get_profile(index)

    def evaluate(self, x=None, spectral_axis=None):
        eval_x = self.source_result._x if x is None else np.asarray(
            x, dtype=float)
        if eval_x is None:
            raise ValueError(
                "Provide x= or construct the MultiFitResult with x=.")
        out = np.empty(self.shape + (eval_x.size,), dtype=float)
        for fi in range(self.source_result.n_spaxels):
            si = np.unravel_index(fi, self.shape)
            out[si] = self.get_profile(fi)(eval_x)
        target = (self.source_result.spectral_axis
                  if spectral_axis is None else spectral_axis)
        if target < 0:
            target += out.ndim
        if target != out.ndim - 1:
            out = np.moveaxis(out, -1, target)
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
        return out

# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------

def select_line(model, selector, components=None, additive=True, index=None):
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
        Convolved ``LinearOperatorCompoundModel`` inputs are not supported;
        pass ``model.source_model`` instead.
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

    Returns
    -------
    SelectedLineProfile or SelectedLineCollection
        An object representing the isolated line profile(s).
    """
    if selector is None or str(selector).strip() == '':
        raise ValueError("A line tag or name is required as 'selector'.")

    from ..operators.convolved import LinearOperatorCompoundModel

    if isinstance(model, LinearOperatorCompoundModel):
        raise ValueError(
            "line_analysis does not accept convolved models. "
            "Pass model.source_model, or intrinsic components from "
            "get_components(model, additive=True, deconvolve=True)."
        )

    if _is_multifit_like(model):
        if index is not None:
            return select_line(
                model.get_model(index), selector=selector,
                components=components, additive=additive)
        return SelectedLineCollection(
            result=model, selector=selector,
            components=components, additive=additive)

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
                    _, template_name = _linegroup_template_index(resolved, selector)
                    entries.append(_SelectionEntry(
                        component_key=key, component_name=comp_name,
                        template_name=template_name))
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
                        entries.append(_SelectionEntry(
                            component_key=key, component_name=name,
                            template_name=str(tmpl_name)))
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
