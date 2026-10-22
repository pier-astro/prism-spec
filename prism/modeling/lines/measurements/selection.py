"""Resolve explicit components once, then evaluate their selected transitions."""
import copy
from dataclasses import dataclass
import warnings

import numpy as np
import astropy.units as u
from astropy.modeling import Model
from ...composition import Model as NativeModel, _is_compound

from ...models.components import get_components
from ...metrics import Metric

from ...component import _Function
from ..component import _LineGroup
from ._members import _LineMember, _native_line, _position_unit, _group_data




@dataclass(frozen=True)
class _SelectionEntry:
    component_key: int
    component_name: str
    template_name: str | None = None
    template_position: float | None = None
    amplitude_name: str = 'amplitude'
    weight: float = 1.0
    template_index: int | None = None


def _resolve_linegroup_member(model, selector, position):
    """Resolve a template and one member; ambiguity requires user input."""
    names, all_positions, all_weights = _group_data(model)
    unit = _position_unit(model)
    if selector in names:
        index = names.index(selector)
    else:
        matches = [i for i, name in enumerate(names) if name.casefold() == selector.casefold()]
        if not matches:
            raise KeyError(f'Template {selector!r} not found in {model.name!r}.')
        if len(matches) != 1:
            raise ValueError(f'Ambiguous template {selector!r}; use its exact name.')
        index = matches[0]
    positions, weights = all_positions[index], all_weights[index]
    if position is not None:
        if isinstance(position, u.Quantity):
            position = position.to_value(unit, equivalencies=u.spectral())
        requested = np.atleast_1d(np.asarray(position, dtype=float))
        if requested.ndim != 1 or not requested.size or not np.isfinite(requested).all():
            raise ValueError('Pass one or more finite rest positions.')
        members = []
        for value in requested:
            matches = np.flatnonzero(np.isclose(positions, value, rtol=0., atol=1e-3))
            if len(matches) != 1:
                raise ValueError(f'Template {selector!r} has no unique matching member at {value} {unit}.')
            if int(matches[0]) not in members:
                members.append(int(matches[0]))
        return index, members
    elif len(positions) == 1:
        matches = [0]
    else:
        matches = np.flatnonzero(np.isclose(weights, 1., rtol=0., atol=1e-12))
    if len(matches) != 1:
        raise ValueError(
            f'Template {selector!r} has no unique matching member; pass position= in '
            f'{unit}. Available positions: {list(positions)}.')
    return index, [int(matches[0])]


class SelectedLineProfile:
    """A physical transition selected from explicit source-model components.

    Parameters
    ----------
    source_model : astropy.modeling.Model
        Parent model retained for linked-parameter and covariance propagation.
    selector : str
        Transition tag or label.
    entries : sequence of _SelectionEntry
        Resolved leaf-component indices and member data, prepared by select_line.
    additive : bool, optional
        Retained for existing calls. Entries always refer to intrinsic leaves;
        algebraic decomposition and instrumental corrections are not inferred.
    """

    def __init__(self, source_model, selector, entries, additive=False):
        self.source_model = source_model
        self.selector = str(selector)
        self.entries = tuple(entries)
        self.additive = False
        leaves = get_components(source_model, additive=False)
        self._members = tuple(_LineMember(entry, leaves[entry.component_key]) for entry in self.entries)
        if not self._members:
            raise ValueError('A line selection requires at least one component.')
        self.position_unit  # Validate native coordinate units before evaluation.
        self.output_unit

    @property
    def _parameter_model(self):
        """Complete fitted parameter context, while profiles remain intrinsic."""
        return getattr(self.source_model, '_evaluation_parent', self.source_model)

    def __repr__(self):
        return f'SelectedLineProfile(selector={self.selector!r}, components={len(self.entries)})'

    @property
    def component_names(self):
        return list(dict.fromkeys(entry.component_name for entry in self.entries))

    def copy_for_model(self, model):
        return SelectedLineProfile(model, self.selector, self.entries)

    @property
    def position_unit(self):
        units = [member.position_unit for member in self._members]
        if any(unit != units[0] for unit in units[1:]):
            raise ValueError('Selected components must use the same native spectral unit.')
        return units[0]

    @property
    def axis_unit(self):
        return self.position_unit

    @property
    def domain_family(self):
        self.position_unit
        return 'wavelength' if self.position_unit.is_equivalent(u.m) else 'linear'

    @property
    def output_unit(self):
        units = [member.output_unit for member in self._members]
        if any(unit is None for unit in units):
            if any(unit is not None for unit in units):
                raise ValueError('Selected components cannot mix unitless and quantity amplitudes.')
            return None
        if any(not unit.is_equivalent(units[0]) for unit in units[1:]):
            raise ValueError('Selected components have incompatible output units.')
        return units[0]

    @property
    def flux_unit(self):
        unit = self.output_unit
        return None if unit is None else unit*self.axis_unit

    @property
    def eqw_unit(self):
        return None if self.output_unit is None else self.axis_unit

    def _coerce_domain_axis(self, x):
        unit = self.axis_unit
        if isinstance(x, u.Quantity):
            x = x.to_value(unit, equivalencies=u.spectral())
        return np.atleast_1d(np.asarray(x, dtype=float))

    def _coerce_output_array(self, values, name='value'):
        if isinstance(values, u.Quantity):
            unit = self.output_unit
            if unit is None:
                raise ValueError(f'{name} carries units but the selected amplitudes do not.')
            values = values.to_value(unit)
        return np.asarray(values, dtype=float)

    def _coerce_output_scalar(self, value, name='value'):
        values = self._coerce_output_array(value, name)
        if values.size != 1:
            raise ValueError(f'{name} must be a scalar.')
        return float(values.reshape(-1)[0])

    def _restore_output(self, values, is_scalar=False):
        unit = self.output_unit
        if unit is not None:
            values = values*unit
        return values[0] if is_scalar else values

    def infer_window(self, padding=6.0):
        """Enclose each selected center by padding times its effective FWHM."""
        self.position_unit
        from ...fitting.utils import apply_tied
        apply_tied(self._parameter_model)
        lower, upper = [], []
        for member in self._members:
            center, width = member.center_width()
            lower.append(center-padding*width)
            upper.append(center+padding*width)
        return float(min(lower)), float(max(upper))

    def evaluate(self, x):
        from ...fitting.utils import apply_tied
        apply_tied(self._parameter_model)
        x_arr = self._coerce_domain_axis(x)
        total = np.zeros_like(x_arr, dtype=float)
        for member in self._members:
            values = member.evaluate(x_arr)
            total += self._coerce_output_array(values, name='line profile')
        return self._restore_output(total, is_scalar=np.ndim(x) == 0)

    def __call__(self, x):
        return self.evaluate(x)

    @property
    def position(self):
        """Mean rest position, in the common native coordinate unit."""
        self.position_unit
        positions = [member.position for member in self._members]
        mean = float(np.mean(positions))
        if np.ptp(positions) > .001*abs(mean):
            warnings.warn('Selected rest positions differ by more than 0.1%; choose the reference explicitly.',
                          UserWarning, stacklevel=2)
        return mean

    def evaluate_components(self, x=None, window=None, num=4096):
        """Evaluate the contributing profiles on one shared spectral grid."""
        from .measurement import _measurement_grid
        from ...fitting.utils import apply_tied
        apply_tied(self._parameter_model)
        x_arr = (_measurement_grid(self, None, window, num) if x is None
                 else self._coerce_domain_axis(x))
        result = {}
        for member in self._members:
            values = member.evaluate(x_arr)
            values = self._restore_output(self._coerce_output_array(values))
            name = member.entry.component_name
            result[name] = result[name] + values if name in result else values
        return result

    def flux(self):
        """Analytic total intrinsic flux; propagate the retained fit covariance.

        Returns a Metric, with std absent when no covariance is available. A
        bootstrap selection propagates the actual successful refits instead.
        This is distinct from measure's numerical finite-window integral.
        """
        from .measurement import _analytic_flux
        return _analytic_flux(self)

    def measure(self, x=None, window=None, num=4096):
        from .measurement import measure_line
        return measure_line(self, x=x, window=window, num=num)

    def sample(self, n_samples=256, confidence=68, x=None, window=None,
               num=4096, method='auto', distribution='auto',
               random_state=None, return_samples=False):
        from .sampling import sample_line_measurements
        return sample_line_measurements(
            self, n_samples=n_samples, confidence=confidence,
            x=x, window=window, num=num,
            method=method, distribution=distribution,
            random_state=random_state, return_samples=return_samples)

    def eqw(self, continuum=None, x=None, window=None, num=2048) -> Metric:
        """Measure equivalent width against an explicit or inferred continuum."""
        from .measurement import _measure_eqw
        result = _measure_eqw(self, continuum, x, window, num)
        self.ew = result
        return result

    def sample_eqw(self, continuum=None, n_samples=256, confidence=68,
                   x=None, window=None, num=2048,
                   method='auto', distribution='auto',
                   perturb_continuum=False, random_state=None) -> Metric:
        """Propagate parameter uncertainties into the equivalent width."""
        from .sampling import _sample_eqw
        result = _sample_eqw(
            self, continuum, n_samples, confidence, x, window, num,
            method, distribution, perturb_continuum, random_state)
        self.ew = result
        return result


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
        return self._reference_profile().axis_unit

    @property
    def output_unit(self):
        return self._reference_profile().output_unit

    @property
    def eqw_unit(self):
        return self._reference_profile().eqw_unit

    def _reference_profile(self):
        return select_line(self.source_result._template_model, self.selector,
                           components=self.components, additive=self.additive,
                           position=self.position)

    def _has_values(self, index):
        flat = self.source_result._normalize_spaxel_index(index)
        return np.all(np.isfinite(self.source_result._parameter_vector(flat)))

    def get_profile(self, index):
        if not self._has_values(index):
            raise ValueError('No finite fitted model is available for this spectrum.')
        model = self.source_result.get_model(index)
        return select_line(
            model, selector=self.selector,
            components=self.components, additive=self.additive,
            position=self.position)

    def __getitem__(self, index):
        return self.get_profile(index)

    def evaluate(self, x=None, axis=None):
        profile0 = self._reference_profile()
        eval_x = self.source_result._x if x is None else x
        if eval_x is None:
            raise ValueError(
                "Provide x= or construct the MultiFitResult with x=.")
        eval_x = profile0._coerce_domain_axis(eval_x)
        out = np.full(self.shape + (eval_x.size,), np.nan, dtype=float)
        for fi in range(self.source_result.n_spaxels):
            if not self._has_values(fi):
                continue
            si = np.unravel_index(fi, self.shape)
            profile = self.get_profile(fi)
            out[si] = profile._coerce_output_array(profile(eval_x), name='line profile')
        target = (self.source_result.axis
                  if axis is None else axis)
        if target < 0:
            target += out.ndim
        if target != out.ndim - 1:
            out = np.moveaxis(out, -1, target)
        if self.output_unit is not None:
            return out * self.output_unit
        return out

    def __call__(self, x=None, axis=None):
        return self.evaluate(x=x, axis=axis)

    def measure(self, x=None, window=None, num=4096):
        from .measurement import measure_line
        return measure_line(self, x=x, window=window, num=num)

    def eqw(self, continuum=None, x=None, window=None, num=2048) -> np.ndarray:
        """EW for every spaxel; returns a shaped numpy array."""
        out = np.full(self.shape, np.nan, dtype=float)
        for fi in range(self.source_result.n_spaxels):
            if not self._has_values(fi):
                continue
            si = np.unravel_index(fi, self.shape)
            prof = self.get_profile(fi)
            m = prof.eqw(continuum=continuum, x=x, window=window, num=num)
            out[si] = m.value
        if self.eqw_unit is not None:
            return out * self.eqw_unit
        return out


def select_line(model, selector, components=None, additive=True, index=None, position=None):
    """Extract a transition from explicit components, retaining its parent model.

    Parameters
    ----------
    model : Model, sequence of Model, FitResult, observation view, or MultiFitResult
        Parent model for linked/correlated components. A list/tuple of independent
        line components is also accepted; it does not recover a former parent's
        covariance. Pass the parent for linked parameters or correlated errors.
        A single FitResult or a joint observation result[label] retains the
        complete fitted parameter graph and covariance for sampling.
    selector : str
        Template tag, or label for an explicitly selected single-line component.
    components : str, int, Model, or sequence, optional
        Explicit leaves of the parent, identified by name, index or object
        identity. Without this argument, a small convenience search matches the
        tag in a single group or a simple additive model. For cubes use names or
        indices, which remain meaningful for each reconstructed model.
    additive : bool, optional
        Retained for existing calls. Selection extracts intrinsic leaves; it
        never distributes multiplication, subtraction or instrumental operators.
    index : int or tuple, optional
        Spatial index for a MultiFitResult. Omit to obtain a collection.
    position : float, Quantity, or sequence, optional
        Rest coordinates identifying members of a multi-transition template.
        A sequence sums those members, retaining their shared covariance.
        Bare values use the component's native spectral unit. Otherwise the
        unique member with weight=1 is chosen (or the sole template member).

    Returns
    -------
    SelectedLineProfile or SelectedLineCollection
        Selected transition with measure, sample, EW and evaluation tools.

    Notes
    -----
    Component choices are resolved once. Parameter values remain live; rebuild
    the selection after replacing components or changing the line-table layout.
    Use get_components and profile_metrics for whole-template band integration.
    Response pipes require explicit components=; the full fitted parent is
    retained for constraints and covariance, while profiles remain intrinsic.
    Equivalent widths then require an explicit intrinsic continuum.
    """
    from ...fitting.multifit import MultiFitResult
    from ...fitting.results import FitResult, _DatasetFitView
    from ...fitting.uncertainty.resample import BootstrapResult

    bootstrap = model if isinstance(model, BootstrapResult) else None
    if bootstrap is not None:
        model = bootstrap.model
    from ...operators.matop import is_linear_operator_pipe

    if isinstance(model, (FitResult, _DatasetFitView)):
        if isinstance(model, FitResult):
            if not model._single:
                raise TypeError('Select a joint observation by label for line measurements.')
            parent, label = model, 'data'
        else:
            parent, label = model._parent, model._label
        parent._require_fitted()
        # Copy source and global graph together: external drivers and the full
        # fitted covariance remain aligned with the selected physical leaves.
        graph, model = copy.deepcopy((parent._graph, parent._associations[label][0]))
        model._evaluation_parent = graph
    if isinstance(model, _Function):
        model = NativeModel(component=model)
    if selector is None or not str(selector).strip():
        raise ValueError('A line tag or name is required as selector.')
    selector = str(selector).strip()
    if isinstance(model, MultiFitResult):
        if components is not None:
            choices = [components] if isinstance(components, (str, int, np.integer, Model, NativeModel, _Function)) else list(components)
            if any(not isinstance(choice, (str, int, np.integer)) for choice in choices):
                raise TypeError('For cube selections, identify components by name or index.')
            components = choices
        if index is not None:
            return select_line(model.get_model(index), selector, components, additive, position=position)
        return SelectedLineCollection(model, selector, components, additive, position)

    if isinstance(model, (list, tuple)):
        if components is not None or not model:
            raise ValueError('Pass either a nonempty component list or a parent with components=.')
        if any(not _native_line(component) for component in model):
            raise TypeError('Standalone lists must contain line models or groups.')
        if any(component.has_tied for component in model):
            raise ValueError('Linked components require the original parent model with components=.')
        components = [NativeModel(component=item) if isinstance(item, _Function) else item for item in model]
        model = components[0]
        if any(isinstance(component, NativeModel) for component in components):
            model = NativeModel.from_astropy(model)
        for component in components[1:]:
            model = model+component

    if not isinstance(model, (Model, NativeModel)):
        raise TypeError('Pass a line model, parent model, component list, or MultiFitResult.')
    if is_linear_operator_pipe(model) and components is None:
        raise ValueError('For a response pipe, select intrinsic components= explicitly to retain parent covariance.')
    available = get_components(model, additive=False)
    explicit = components is not None or (
        isinstance(model, NativeModel) and model.op is None and _native_line(model) and not isinstance(model.component, _LineGroup))
    if components is None:
        # Automatic discovery deliberately covers sums, not arbitrary algebra.
        pending = [model]
        while pending:
            component = pending.pop()
            if _is_compound(component):
                if component.op != '+':
                    raise ValueError('For non-additive models, pass intrinsic components= explicitly.')
                pending.extend([component.left, component.right])
        choices = available.indices
    else:
        choices = [components] if isinstance(components, (str, int, np.integer, Model, NativeModel, _Function)) else list(components)
        if not choices:
            raise ValueError('Explicit selection requires at least one component.')

    entries, seen = [], set()
    for choice in choices:
        if isinstance(choice, (Model, NativeModel, _Function)):
            leaf = choice.component if isinstance(choice, NativeModel) and choice.op is None else choice
            matches = [key for key in available.indices if available[key] is leaf]
            if len(matches) != 1:
                raise ValueError('A component object must occur exactly once in the supplied parent; otherwise use an index.')
            key = matches[0]
        else:
            component = available[choice]
            key = next(key for key in available.indices if available[key] is component)
        if key in seen:
            continue
        seen.add(key)
        component = available[key]
        name = available.names[key]
        if isinstance(component, _LineGroup):
            try:
                template, members = _resolve_linegroup_member(component, selector, position)
            except KeyError:
                continue
            names, positions, weights = _group_data(component)
            scale_index = component._groups[template][2][component._scale_slot]
            amplitude_name = component.param_names[scale_index]
            for member in members:
                entries.append(_SelectionEntry(key, name, str(names[template]),
                                               float(positions[template][member]), amplitude_name,
                                               float(weights[template][member]), template))
        elif _native_line(component):
            if explicit or name.casefold() == selector.casefold():
                entries.append(_SelectionEntry(key, name))
        elif explicit:
            raise TypeError(f'{name!r} is not a line model. Use profile_metrics for arbitrary profiles.')
    if not entries:
        raise KeyError(f'No line selection found for {selector!r} in the chosen components.')
    selection = SelectedLineProfile(model, selector, entries)
    selection._bootstrap_result = bootstrap
    return selection
