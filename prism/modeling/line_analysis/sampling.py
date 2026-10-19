"""Optional parameter propagation for selected profiles and equivalent widths."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from ..fitting.utils import _ParameterMap, _free_parameter_indices, _apply_tied_fast

from ..models.components import get_components
from ..metrics import Metric
from ..fitting.uncertainty.resample import extract_limits
from .selection import SelectedLineCollection
from .results import LineResult, _METRIC_NAMES, _METRIC_BOUNDS, _NO_BOUNDS, _metric_units
from .numerical import _compute_profile_metrics, _compute_ew_numerical
from .measurement import _resolve_selection, _measurement_grid, _continuum_grid, _continuum_level


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


def _fit_param_names(model):
    """Scalar free parameters, in the covariance/fitter ordering."""
    if any(getattr(model, name).size != 1 for name in model.param_names):
        raise ValueError('Line parameter sampling requires scalar model parameters.')
    indices = _free_parameter_indices(model)
    return [model.param_names[i] for i in indices]


def _selection_parameter_specs(selection, include_continuum=False):
    """Map selected parameters to parent-model free parameters once.

    With arbitrary callable links, dependencies cannot be inferred reliably.
    Include all parent free parameters in that case; missing uncertainties stay
    at their nominal values. Fixed and derived parameters are never drawn.
    """
    model = selection.source_model
    leaves = get_components(model, additive=False)
    free_names = set(_fit_param_names(model))
    relevant = set()
    for entry, component in selection._components:
        if entry.template_name is None:
            names = component.param_names
        else:
            names = [entry.amplitude_name, *component._shared_params]
        relevant.update((id(component), name) for name in names)

    specs = []
    for index, component in zip(leaves.indices, leaves):
        for name in component.param_names:
            parent_name = name if len(leaves) == 1 else f'{name}_{index}'
            if parent_name not in free_names:
                continue
            if not (include_continuum or model.has_tied or (id(component), name) in relevant):
                continue
            param = getattr(model, parent_name)
            def scalar_attribute(attribute):
                value = getattr(param, attribute, None)
                if value is None:
                    value = getattr(getattr(component, name), attribute, None)
                return float(value) if value is not None else float('nan')
            specs.append(_SelectionParameter(
                component_key=index, param_name=name, parent_name=parent_name,
                value=float(param.value), std=scalar_attribute('std'),
                lolim=scalar_attribute('lolim'), uplim=scalar_attribute('uplim'),
                bounds=param.bounds))
    return specs


def _clip_draws_to_bounds(specs, draws):
    """Preserve the existing clipping policy, not a truncated posterior."""
    for j, spec in enumerate(specs):
        lo, hi = spec.bounds
        if lo is not None:
            draws[:, j] = np.maximum(draws[:, j], lo)
        if hi is not None:
            draws[:, j] = np.minimum(draws[:, j], hi)


def _draw_selection_samples(selection, n_samples, method='auto',
                            distribution='auto', random_state=None,
                            include_continuum=False):
    if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 1:
        raise ValueError('n_samples must be a positive integer.')
    method = str(method).strip().lower()
    distribution = str(distribution).strip().lower()
    if method not in {'auto', 'covariance', 'limits', 'std'}:
        raise ValueError("method must be: 'auto', 'covariance', 'limits', 'std'.")
    if distribution not in {'auto', 'uniform', 'gaussian'}:
        raise ValueError("distribution must be: 'auto', 'uniform', 'gaussian'.")
    if method == 'limits' and distribution == 'gaussian':
        raise ValueError("method='limits' + distribution='gaussian' not supported.")
    specs = _selection_parameter_specs(selection, include_continuum)
    rng = np.random.default_rng(random_state)
    cov = getattr(selection.source_model, '_param_cov', None)

    if method in ('auto', 'covariance') and cov is not None:
        free_names = _fit_param_names(selection.source_model)
        cov = np.asarray(cov, dtype=float)
        try:
            # Cube results expand covariance to all parameters, padding fixed
            # rows with NaN. Their explicit fit indices identify the free block.
            fit_indices = getattr(selection.source_model, '_fit_indices', None)
            if fit_indices is not None:
                expected = [selection.source_model.param_names.index(name) for name in free_names]
                if list(fit_indices) != expected:
                    raise ValueError('Stored covariance indices disagree with current free parameters.')
                if cov.shape == (len(selection.source_model.param_names),)*2:
                    cov = cov[np.ix_(expected, expected)]
            if cov.shape != (len(free_names), len(free_names)) or not specs:
                raise ValueError('Covariance shape must match the parent free-parameter ordering.')
            indices = [free_names.index(s.parent_name) for s in specs]
            subcov = cov[np.ix_(indices, indices)]
            if not np.all(np.isfinite(subcov)) or not np.allclose(subcov, subcov.T, rtol=1e-10, atol=0.):
                raise ValueError('Covariance must be finite and symmetric.')
            eigenvalues = np.linalg.eigvalsh(subcov)
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
            if np.min(eigenvalues) < -1e-10 * scale:
                raise ValueError('Covariance must be positive semidefinite.')
            mean = np.asarray([s.value for s in specs])
            draws = rng.multivariate_normal(mean, subcov, size=int(n_samples))
            _clip_draws_to_bounds(specs, draws)
            return specs, draws
        except (ValueError, np.linalg.LinAlgError) as error:
            if method == 'covariance':
                raise ValueError(f'No usable covariance matrix: {error}') from error
            warnings.warn(f'Ignoring unusable covariance: {error}', RuntimeWarning)
    if method == 'covariance':
        raise ValueError("method='covariance' but no usable covariance matrix available.")

    # Independent parameter perturbations: intervals, or std with a chosen shape.
    # Keep legacy auto/uniform behavior explicit rather than inferring a posterior.
    sampled_specs = []
    columns = []
    for spec in specs:
        lo, up = spec.lolim, spec.uplim
        has_limits = np.isfinite(lo) and np.isfinite(up) and up > lo
        has_std = np.isfinite(spec.std) and spec.std > 0.0
        if method in ('auto', 'limits') and has_limits and distribution in ('auto', 'uniform'):
            values = rng.uniform(lo, up, size=int(n_samples))
        elif method in ('auto', 'std') and has_std:
            if distribution in ('auto', 'uniform'):
                half = np.sqrt(3.0) * spec.std
                values = rng.uniform(spec.value-half, spec.value+half, size=int(n_samples))
            else:
                values = rng.normal(spec.value, spec.std, size=int(n_samples))
        else:
            continue
        sampled_specs.append(spec)
        columns.append(values)
    if not sampled_specs:
        raise ValueError('No uncertainty information usable for the selected sampling method.')
    draws = np.column_stack(columns)
    _clip_draws_to_bounds(sampled_specs, draws)
    return sampled_specs, draws


def _iter_sampled_profiles(selection, specs, draws, x):
    """Update one model copy, refresh links, then evaluate selected members."""
    working = selection.copy_for_model(selection.source_model.copy())
    model = working.source_model
    targets = [getattr(model, spec.parent_name) for spec in specs]
    has_tied = model.has_tied
    if has_tied:
        free_names = _fit_param_names(model)
        parameters = _ParameterMap(model)
        params_cache = parameters.initial.copy()
        columns = [free_names.index(s.parent_name) for s in specs]

    for draw in draws:
        if has_tied:
            params_cache[parameters.free[columns]] = draw
            model.parameters = params_cache
            _apply_tied_fast(model, parameters.tied, params_cache)
        else:
            for param, value in zip(targets, draw):
                param.value = value
        values = working._coerce_output_array(working.evaluate(x), name='line profile')
        yield working, values


def _summarize(value, samples, confidence, unit=None, bounds=_NO_BOUNDS):
    finite = samples[np.isfinite(samples)]
    if not finite.size:
        return Metric(value=float(value), unit=unit)
    alpha = (100.0-float(confidence))/2.0
    lo, up = extract_limits(bounds, finite, alpha, 100.-alpha)
    return Metric(value=float(value), std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.,
                  lolim=float(lo), uplim=float(up), unit=unit)


def _validate_confidence(confidence):
    if not np.isfinite(confidence) or not 0. < confidence < 100.:
        raise ValueError('confidence must be between 0 and 100.')


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
    components : str, int, Model, or sequence, optional
        Explicit parent-model leaves, identified by name, index or identity.
        Default is ``None`` for simple automatic discovery.
    additive : bool, optional
        Retained for existing calls. Selection extracts intrinsic leaves;
        use explicit components for non-additive models.
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
    _validate_confidence(confidence)
    selection = _resolve_selection(model_or_selection, selector, components, additive, index)
    if isinstance(selection, SelectedLineCollection):
        raise ValueError('Sampling a cube requires index= or collection[index].sample(...).')
    x_arr = _measurement_grid(selection, x, window, num)
    y_nominal = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    nominal = _compute_profile_metrics(x_arr, y_nominal)
    units = _metric_units(selection.axis_unit, selection.output_unit)
    specs, draws = _draw_selection_samples(
        selection, n_samples, method, distribution, random_state)

    # Every realization uses the same numerical measurement core.
    records = []
    for _, values in _iter_sampled_profiles(selection, specs, draws, x_arr):
        records.append(_compute_profile_metrics(x_arr, values))
    samples = pd.DataFrame(records, columns=_METRIC_NAMES)
    metrics = {name: _summarize(nominal[name], samples[name].to_numpy(), confidence,
                              units[name], _METRIC_BOUNDS.get(name, _NO_BOUNDS))
               for name in _METRIC_NAMES}
    if np.any(~np.isfinite(samples['fwhm'])):
        warnings.warn('Some sampled profiles have undefined FWHM.', RuntimeWarning)
    result = LineResult(selection.selector, 'auto', x_arr[0], x_arr[-1], x_arr.size,
                        metrics, metric_units=units, axis_unit=selection.axis_unit)
    return (result, samples) if return_samples else result


def _sample_eqw(selection, continuum, n_samples, confidence, x, window, num,
                method, distribution, perturb_continuum, random_state):
    _validate_confidence(confidence)
    x_arr = _measurement_grid(selection, x, window, num)
    y_nominal = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    x_cont = _continuum_grid(selection, window) if continuum is None else None
    if continuum is None:
        y_cont = _continuum_level(selection, x_cont)
    elif callable(continuum):
        y_cont = selection._coerce_output_array(continuum(x_arr), name='continuum')
    else:
        y_cont = selection._coerce_output_scalar(continuum, name='continuum')
    nominal = _compute_ew_numerical(x_arr, y_nominal, y_cont)
    specs, draws = _draw_selection_samples(
        selection, n_samples, method, distribution, random_state,
        include_continuum=perturb_continuum and continuum is None)
    values = np.empty(int(n_samples), dtype=float)
    for i, (working, y_line) in enumerate(_iter_sampled_profiles(selection, specs, draws, x_arr)):
        local_continuum = y_cont
        if continuum is None:
            # The inferred continuum follows changes in the parent model.
            local_continuum = _continuum_level(working, x_cont)
        values[i] = _compute_ew_numerical(x_arr, y_line, local_continuum)
    return _summarize(nominal, values, confidence, selection.eqw_unit)
