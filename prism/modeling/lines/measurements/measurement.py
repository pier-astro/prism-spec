"""Model-selection adapters for deterministic line measurements."""
import numpy as np
import astropy.units as u
from ...metrics import Metric
from .selection import SelectedLineProfile, SelectedLineCollection, select_line
from .results import LineResult, MultiLineMeasurements, _METRIC_NAMES, _metric_units
from .numerical import _compute_profile_metrics, _compute_ew_numerical

def _measurement_grid(selection, x, window, num):
    """Normalize spectral units and orient a valid integration grid."""
    if x is None:
        if window is None:
            window = selection.infer_window()
        if any(isinstance(bound, u.Quantity) for bound in window):
            window = u.Quantity(window)
        window = selection._coerce_domain_axis(window)
        if window.size != 2:
            raise ValueError("window must contain exactly two spectral coordinates.")
        x_arr = np.linspace(float(window[0]), float(window[1]), int(num))
    else:
        x_arr = selection._coerce_domain_axis(x)
    if x_arr.ndim != 1 or x_arr.size < 3 or not np.all(np.isfinite(x_arr)):
        raise ValueError('x must be a finite 1-D spectral grid with at least three samples.')
    if np.all(np.diff(x_arr) < 0):
        x_arr = x_arr[::-1]
    elif not np.all(np.diff(x_arr) > 0):
        raise ValueError('x must be strictly monotonic with no repeated coordinates.')
    return x_arr


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
        reference = selection._reference_profile()
        metric_units = _metric_units(reference.axis_unit, reference.output_unit)
        axis_unit = reference.axis_unit
        for fi in range(selection.source_result.n_spaxels):
            if not selection._has_values(fi):
                continue
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

    x_arr = _measurement_grid(selection, x, window, num)

    y_arr = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    return _profile_result(x_arr, y_arr, selection.axis_unit, selection.output_unit,
                           selector=selection.selector, mode='auto')


def _profile_result(x_arr, y_arr, axis_unit, output_unit, selector, mode):
    raw_metrics = _compute_profile_metrics(x_arr, y_arr)
    units = _metric_units(axis_unit, output_unit)
    return LineResult(
        selector=selector, mode=mode, wave_min=x_arr[0], wave_max=x_arr[-1],
        n_grid=x_arr.size, metrics=raw_metrics, metric_units=units, axis_unit=axis_unit)


def profile_metrics(x, y):
    """Measure a sampled line profile independently of a model or fitter.

    Parameters
    ----------
    x : array-like or astropy.units.Quantity
        Finite, strictly monotonic spectral coordinates, with at least three
        samples. Descending coordinates are accepted and oriented internally.
    y : array-like or astropy.units.Quantity
        Finite line-only spectral density at each coordinate. Subtract the
        continuum before calling this function. The shape must match ``x``.

    Returns
    -------
    LineResult
        Finite-window flux, peak, crossing widths and moments. Input Quantity
        units are preserved; plain arrays do not imply physical units.

    Notes
    -----
    This function performs no model selection, continuum inference, uncertainty
    propagation, or instrumental correction. Widths are in the input coordinate
    unit. Moments depend on the chosen finite window; crossing widths refer to
    the nearest crossings about the highest positive peak. Negative-only
    profiles have signed flux and moments but undefined crossing widths.

    Examples
    --------
    >>> metrics = profile_metrics(wave, line_flux)
    >>> metrics.flux.value
    >>> metrics.fwhm.value
    """
    axis_unit = x.unit if isinstance(x, u.Quantity) else None
    output_unit = y.unit if isinstance(y, u.Quantity) else None
    x_arr = np.asarray(x.value if axis_unit is not None else x, dtype=float)
    y_arr = np.asarray(y.value if output_unit is not None else y, dtype=float)
    if x_arr.ndim != 1 or x_arr.size < 3 or y_arr.shape != x_arr.shape:
        raise ValueError('x and y must be matching 1-D arrays with at least three samples.')
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError('x and y must contain only finite samples.')
    if np.all(np.diff(x_arr) < 0):
        x_arr, y_arr = x_arr[::-1], y_arr[::-1]
    elif not np.all(np.diff(x_arr) > 0):
        raise ValueError('x must be strictly monotonic with no repeated coordinates.')
    return _profile_result(x_arr, y_arr, axis_unit, output_unit, 'profile', 'profile')



def _continuum_grid(selection, window):
    """Legacy local continuum probe; an explicit continuum avoids this policy."""
    if window is None:
        window = selection.infer_window()
    if any(isinstance(bound, u.Quantity) for bound in window):
        window = u.Quantity(window)
    bounds = selection._coerce_domain_axis(window)
    sigma_est = abs(bounds[1]-bounds[0]) / (12.0*2.3548)
    centroid = float(selection.position)
    return np.linspace(centroid-sigma_est, centroid+sigma_est, 17)


def _continuum_level(selection, x):
    from ...operators.matop import is_linear_operator_pipe
    if is_linear_operator_pipe(selection.source_model):
        raise ValueError('Pass an explicit intrinsic continuum for equivalent widths from response-pipe selections.')
    model = selection._coerce_output_array(selection.source_model(x), name='continuum model')
    line = selection._coerce_output_array(selection.evaluate(x), name='line profile')
    continuum = model-line
    finite = continuum[np.isfinite(continuum)]
    level = float(np.median(finite)) if finite.size else float('nan')
    return level if level > 0. else float('nan')


def _eqw_continuum(selection, continuum):
    """Resolve a named intrinsic continuum in the same live parameter graph."""
    if isinstance(continuum, str):
        from ...models.components import get_components
        component = get_components(selection.source_model, additive=False)[continuum]
        if component.n_inputs != 1:
            raise ValueError('An EW continuum must be a one-input spectral component.')
        return component
    return continuum


def _eqw_value(selection, continuum, x_arr, x_cont):
    line = selection._coerce_output_array(selection.evaluate(x_arr), name='line profile')
    if continuum is None:
        values = _continuum_level(selection, x_cont)
    elif callable(continuum):
        values = selection._coerce_output_array(continuum(x_arr), name='continuum')
    else:
        values = selection._coerce_output_scalar(continuum, name='continuum')
    return _compute_ew_numerical(x_arr, line, values)


def _measure_eqw(selection, continuum, x, window, num):
    from ...fitting.utils import _ParameterMap, apply_tied
    x_arr = _measurement_grid(selection, x, window, num)
    x_cont = _continuum_grid(selection, window) if continuum is None else None
    resolved = _eqw_continuum(selection, continuum)
    nominal = _eqw_value(selection, resolved, x_arr, x_cont)
    unit = selection.eqw_unit
    if not np.isfinite(nominal):
        return Metric(nominal, unit=unit)
    bootstrap = getattr(selection, '_bootstrap_result', None)
    if bootstrap is not None:
        from .sampling import _bootstrap_models
        draws = []
        sampled_continuum = None
        for working in _bootstrap_models(selection):
            if sampled_continuum is None:
                sampled_continuum = _eqw_continuum(working, continuum)
            draws.append(_eqw_value(working, sampled_continuum, x_arr, x_cont))
        draws = np.asarray(draws); draws = draws[np.isfinite(draws)]
        if draws.size < 2:
            return Metric(nominal, unit=unit)
        alpha = (100-bootstrap._confidence)/2
        lower, upper = np.percentile(draws, [alpha, 100-alpha])
        return Metric(nominal, float(np.std(draws, ddof=1)), float(lower), float(upper), unit)
    model = selection._parameter_model
    if getattr(model, '_param_cov', None) is None:
        return Metric(nominal, unit=unit)
    if any(getattr(model, name).size != 1 for name in model.param_names):
        raise ValueError('EW covariance propagation requires scalar model parameters.')
    mapping = _ParameterMap(model)
    covariance = _covariance_matrix(model, mapping.free)
    original = mapping.initial
    gradient = np.zeros(len(mapping.free))
    relevant = {id(getattr(member.component, name)) for member in selection._members
                for name in member.parameter_names}
    if isinstance(continuum, str):
        relevant.update(id(getattr(resolved, name)) for name in resolved.param_names)
    # Local finite differences of EW itself retain line/continuum correlations.
    # Opaque links require all drivers; otherwise unrelated fit parameters skip.
    try:
        for column, index in enumerate(mapping.free):
            parameter = getattr(model, model.param_names[index])
            if continuum is not None and not model.has_tied and id(parameter) not in relevant:
                continue
            value = original[index]
            scale = abs(value) if value != 0 else np.sqrt(covariance[column, column])
            if scale == 0:
                continue
            step = max(scale*np.cbrt(np.finfo(float).eps), np.finfo(float).tiny)
            lower, upper = mapping.bounds[column]
            up = min(value+step, upper); down = max(value-step, lower)
            if up == down:
                continue
            values = original.copy(); values[index] = up
            model.parameters = values; apply_tied(model)
            above = _eqw_value(selection, resolved, x_arr, x_cont)
            values[index] = down
            model.parameters = values; apply_tied(model)
            below = _eqw_value(selection, resolved, x_arr, x_cont)
            gradient[column] = (above-below)/(up-down)
    finally:
        model.parameters = original
        apply_tied(model)
    if not np.isfinite(gradient).all():
        raise ValueError('EW derivatives are non-finite on the selected grid.')
    return Metric(nominal, _covariance_std(model, gradient, mapping.free), unit=unit)


def _analytic_flux(selection):
    """Total intrinsic flux with the full fit-graph delta-method uncertainty."""
    from ...fitting.utils import apply_tied, _reduce_tied_analytic_jacobian, _free_parameter_indices
    model = selection._parameter_model
    apply_tied(model)
    rows = {id(getattr(model, name)): i for i, name in enumerate(model.param_names)}
    gradient = np.zeros((len(model.param_names), 1))
    total = 0.
    for member in selection._members:
        value, derivatives = member.integral()
        scale = (1. if member.output_unit is None else
                 (1.*member.output_unit).to_value(selection.output_unit))
        total += value*scale
        for parameter, derivative in derivatives:
            gradient[rows[id(parameter)], 0] += derivative*scale
    free = _free_parameter_indices(model)
    gradient = _reduce_tied_analytic_jacobian(model, gradient)[free, 0]
    std = _covariance_std(model, gradient, free)
    bootstrap = getattr(selection, '_bootstrap_result', None)
    if bootstrap is not None:
        from .sampling import _bootstrap_models
        draws = []
        for working in _bootstrap_models(selection):
            draws.append(sum(member.integral()[0]*(1. if member.output_unit is None else
                (1.*member.output_unit).to_value(selection.output_unit)) for member in working._members))
        if len(draws) < 2:
            return Metric(total, unit=selection.flux_unit)
        alpha = (100-bootstrap._confidence)/2
        lower, upper = np.percentile(draws, [alpha, 100-alpha])
        return Metric(total, float(np.std(draws, ddof=1)), float(lower), float(upper), selection.flux_unit)
    return Metric(total, std, unit=selection.flux_unit)


def _covariance_matrix(model, free):
    """Validate covariance and select its canonical free-parameter block."""
    covariance = getattr(model, '_param_cov', None)
    if covariance is None:
        return None
    fit_indices = getattr(model, '_fit_indices', None)
    if fit_indices is not None and list(fit_indices) != list(free):
        raise ValueError('Stored covariance indices disagree with current free parameters.')
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (len(model.param_names),)*2:
        covariance = covariance[np.ix_(free, free)]
    if covariance.shape != (len(free),)*2 or not np.isfinite(covariance).all():
        raise ValueError('Covariance must be finite and match the fitted free-parameter order.')
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.):
        raise ValueError('Covariance must be symmetric.')
    if np.any(np.diag(covariance) < 0):
        raise ValueError('Covariance diagonal must be nonnegative.')
    return covariance


def _covariance_std(model, gradient, free):
    """Contract a free-parameter gradient with the validated full covariance."""
    covariance = _covariance_matrix(model, free)
    if covariance is None:
        return None
    variance = float(gradient @ covariance @ gradient)
    tolerance = 1e-10*max(float(np.sum(abs(np.outer(gradient, gradient)*covariance))), np.finfo(float).tiny)
    if variance < -tolerance:
        raise ValueError('Covariance gives a negative propagated variance.')
    return float(np.sqrt(max(0., variance)))
