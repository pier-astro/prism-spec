"""Utility functions for prism.modeling.fitting."""

import warnings

import numpy as np
from astropy.modeling import CompoundModel

_TIE_DERIV_REL_STEP = 1e-7
_TIE_DERIV_ABS_STEP = 1e-10
_TIE_DERIV_ZERO_TOL = 1e-12
_TIED_ANALYTIC_JACOBIAN_MESSAGE = (
    "\nPrism detected tied parameters together with analytic derivatives available.\n"
    "If analytic derivatives are being used, tie sensitivities are being estimated\n"
    "numerically to reduce the Jacobian to the free parameters.\n"
    "This is usually reliable for smooth ties, but can be inaccurate near non-smooth\n"
    "links or active bounds. If the fit looks suspicious, retry with a numeric Jacobian\n"
    "in the fitter settings, e.g. prism.modeling.fitting.ScipyTRF(jac='2-point')"
)

def tie(submodel_name, func):
    """
    Create a tie function for parameter constraints in compound models.

    This is the recommended way to tie parameters in prism models.

    Parameters
    ----------
    submodel_name : str
        Name of the submodel containing the parameter to reference.
    func : callable
        Function taking submodel, returns tied value.

    Returns
    -------
    callable
        Tie function for use with model.param.tied = ...
    """
    def _tie(model):
        if isinstance(model, CompoundModel) or hasattr(model, 'submodel_names'):
            if submodel_name not in model.submodel_names:
                raise KeyError(
                    f"Submodel '{submodel_name}' not found. "
                    f"Available: {model.submodel_names}"
                )
            sub = model[submodel_name]
        else:
            sub = model
        try:
            return func(sub)
        except AttributeError as e:
            params = list(getattr(sub, 'param_names', []))
            raise AttributeError(
                f"Tie error: {e}. Available params in '{submodel_name}': {params}"
            ) from e
    return _tie


def apply_tied(model):
    """Apply all tied parameter constraints to a model."""
    if not model.has_tied:
        return model

    param_metrics = model._param_metrics
    parameters = model.parameters.copy()

    for name in model.param_names:
        tied_func = model.tied[name]
        if tied_func:
            value = tied_func(model)
            slice_ = param_metrics[name]["slice"]
            parameters[slice_] = value

    model.parameters = parameters
    return model


def _get_tied_info(model):
    """Extract and cache tied parameter information for fast evaluation."""
    tied_info = []
    if not model.has_tied:
        return tied_info

    param_metrics = model._param_metrics
    tied_dict = model.tied

    for name in model.param_names:
        tied_func = tied_dict[name]
        if tied_func:
            slice_ = param_metrics[name]["slice"]
            tied_info.append((tied_func, slice_))

    return tied_info


def _apply_tied_fast(model, tied_info, parameters_cache):
    """Apply tied constraints using pre-cached info (fast path)."""
    if not tied_info:
        return
    for tied_func, slice_ in tied_info:
        value = tied_func(model)
        parameters_cache[slice_] = value
    model.parameters = parameters_cache


def _prime_tied_analytic_jacobian_warning(model):
    """Reset the per-fit warning state for tied analytic Jacobians."""
    model._prism_warn_tied_analytic_jacobian = True
    model._prism_tied_analytic_jacobian_warned = False


def _warn_tied_analytic_jacobian(model, *, stacklevel=3):
    """Warn once per fit when tie sensitivities are estimated numerically."""
    if not getattr(model, '_prism_warn_tied_analytic_jacobian', False):
        return
    if getattr(model, '_prism_tied_analytic_jacobian_warned', False):
        return
    warnings.warn(
        _TIED_ANALYTIC_JACOBIAN_MESSAGE,
        UserWarning,
        stacklevel=stacklevel,
    )
    model._prism_tied_analytic_jacobian_warned = True


def _coerce_tied_value_array(value, size):
    """Convert a tied callable result to a flat float array of known size."""
    if hasattr(value, 'value'):
        value = value.value
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1 and size == 1:
        return array
    if array.size != size:
        raise ValueError(
            f"Tied parameter returned {array.size} values, expected {size}."
        )
    return array


def _parameter_index_array(model, name):
    """Return the flattened parameter indices corresponding to a model parameter."""
    return np.arange(model.parameters.size)[model._param_metrics[name]['slice']].reshape(-1)


def _free_parameter_indices(model):
    """Return flattened indices for free parameters only."""
    free = []
    for name in model.param_names:
        if model.fixed[name] or model.tied[name]:
            continue
        free.extend(_parameter_index_array(model, name).tolist())
    return np.asarray(free, dtype=int)


def _analytic_jacobian_parameter_major(model, jacobian):
    """Normalize an analytic Jacobian to ``(n_params, n_data)`` layout."""
    derivs = np.asanyarray(jacobian)
    if not model.col_fit_deriv:
        derivs = np.moveaxis(derivs, -1, 0)
    return derivs.reshape((derivs.shape[0], -1))


def _reduce_tied_analytic_jacobian(model, jacobian, *, warn=False, stacklevel=3):
    """Apply the tie-function chain rule to a full analytic Jacobian."""
    full = _analytic_jacobian_parameter_major(model, jacobian).astype(float, copy=True)
    if not model.has_tied:
        return full

    if warn:
        _warn_tied_analytic_jacobian(model, stacklevel=stacklevel)

    original = model.parameters.copy()
    tied_rows = []
    free_rows = _free_parameter_indices(model)

    for name in model.param_names:
        tied_func = model.tied[name]
        if not tied_func:
            continue
        tied_rows.append((name, _parameter_index_array(model, name), tied_func))

    try:
        base_tied_values = {
            name: _coerce_tied_value_array(tied_func(model), len(indices))
            for name, indices, tied_func in tied_rows
        }

        for free_index in free_rows:
            p0 = float(original[free_index])
            eps = max(abs(p0) * _TIE_DERIV_REL_STEP, _TIE_DERIV_ABS_STEP)
            perturbed = original.copy()
            perturbed[free_index] = p0 + eps
            model.parameters = perturbed
            apply_tied(model)

            for name, tied_indices, tied_func in tied_rows:
                delta = (
                    _coerce_tied_value_array(tied_func(model), len(tied_indices))
                    - base_tied_values[name]
                ) / eps
                if np.any(np.abs(delta) > _TIE_DERIV_ZERO_TOL):
                    full[free_index] += np.sum(full[tied_indices] * delta[:, np.newaxis], axis=0)
    finally:
        model.parameters = original
        apply_tied(model)

    for _, tied_indices, _ in tied_rows:
        full[tied_indices] = 0.0

    return full
