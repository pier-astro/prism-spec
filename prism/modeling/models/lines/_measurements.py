"""Analytic line measurements shared by single-line and template adapters.

Flux derivatives use local parameter units. Covariance is the local free-
parameter matrix in Astropy order; without it, attached errors are treated as
independent. Parameter-limit envelopes are not joint confidence intervals.
These helpers do not replace the finite-window measurements in line_analysis.
"""
import numpy as np
from astropy import units as u
from astropy.modeling.fitting import model_to_fit_params

from ...metrics import Metric


def _has_param_std(param) -> bool:
    return hasattr(param, 'std') and param.std is not None and np.isfinite(param.std)


def _get_param_limits(param):
    has_lo = hasattr(param, 'lolim') and param.lolim is not None and np.isfinite(param.lolim)
    has_hi = hasattr(param, 'uplim') and param.uplim is not None and np.isfinite(param.uplim)
    lo = param.lolim if has_lo else param.value
    hi = param.uplim if has_hi else param.value
    return lo, hi, has_lo, has_hi


def _evaluate_at(model, evaluate, overrides):
    """Evaluate a temporary parameter state, restoring it even on failure."""
    original = {name: getattr(model, name).value for name in overrides}
    try:
        for name, value in overrides.items():
            getattr(model, name).value = value
        return evaluate()
    finally:
        for name, value in original.items():
            getattr(model, name).value = value


def _flux(model, evaluate, relevant, governing):
    """Return one integrated flux and its local error/limit treatment.

    ``relevant`` names the parameters that affect this flux. ``governing``
    names the amplitude followed by widths whose stored limits define an
    envelope. A covariance includes every local free parameter; irrelevant
    gradient entries remain zero. Continuum and cross-component errors are
    handled outside this analytic measurement.
    """
    raw = evaluate()
    unit = raw.unit if isinstance(raw, u.Quantity) else None
    value = float(raw.value) if unit is not None else float(raw)

    def numeric():
        result = evaluate()
        return float(result.to_value(unit)) if unit is not None else float(result)

    covariance = getattr(model, '_param_cov', None)
    if covariance is not None:
        _, indices, _ = model_to_fit_params(model)
        names = [model.param_names[i] for i in indices]
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(names), len(names)):
            raise ValueError('Flux covariance must match the local free parameters.')
    else:
        names = [name for name in relevant if _has_param_std(getattr(model, name))]

    gradient = np.zeros(len(names))
    for i, name in enumerate(names):
        if name not in relevant:
            continue
        original = getattr(model, name).value
        delta = abs(original)*1e-6 if original != 0 else 1e-6
        if original == 0 and name in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
            delta /= model._to_kms
        gradient[i] = (_evaluate_at(model, numeric, {name: original+delta})-value)/delta

    if covariance is not None:
        variance = float(gradient @ covariance @ gradient)
    else:
        errors = np.array([getattr(model, name).std for name in names])
        variance = float(np.sum((gradient*errors)**2)) if names else np.nan
    std = float(np.sqrt(variance)) if variance >= 0 else np.nan

    amplitude = getattr(model, governing[0])
    limits = []
    for side in ('lolim', 'uplim'):
        available = _get_param_limits(amplitude)[2 if side == 'lolim' else 3]
        if not available:
            limits.append(np.nan)
            continue
        overrides = {}
        for name in governing:
            param = getattr(model, name)
            lo, hi, has_lo, has_hi = _get_param_limits(param)
            if (side == 'lolim' and has_lo) or (side == 'uplim' and has_hi):
                overrides[name] = lo if side == 'lolim' else hi
        limits.append(float(_evaluate_at(model, numeric, overrides)))
    return Metric(value=value, std=std, lolim=limits[0], uplim=limits[1], unit=unit)


def _constant_eqw(flux, continuum, axis_unit):
    """Flux divided by a fixed continuum; continuum uncertainty is omitted."""
    unit = axis_unit if flux.unit is not None else None
    if isinstance(continuum, u.Quantity):
        continuum = continuum.to_value(flux.unit/axis_unit) if flux.unit is not None else continuum.value
    continuum = float(continuum)
    if not np.isfinite(continuum) or continuum == 0:
        return Metric(value=np.nan, std=np.nan, lolim=np.nan, uplim=np.nan, unit=unit)
    return Metric(value=flux.value/continuum, std=flux.std/abs(continuum),
                  lolim=flux.lolim/continuum, uplim=flux.uplim/continuum, unit=unit)


def _integrated_eqw(x, line, continuum, axis_unit, amplitude_unit=None):
    """Integrate line/continuum on a monotonic native grid; no error estimate."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('EW integration requires a finite 1-D grid with at least two samples.')
    differences = np.diff(x)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError('EW integration requires a strictly monotonic spectral grid.')
    values = continuum(x) if callable(continuum) else continuum
    if isinstance(values, u.Quantity):
        values = values.to_value(amplitude_unit) if amplitude_unit is not None else values.value
    values = np.asarray(values, dtype=float)
    if differences[0] < 0:
        x, line = x[::-1], line[::-1]
        if values.ndim:
            values = values[::-1]
    safe = np.where(np.abs(values) > 0, values, np.nan)
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return Metric(value=float(integrate(line/safe, x)),
                  std=np.nan, lolim=np.nan, uplim=np.nan,
                  unit=axis_unit if amplitude_unit is not None else None)
