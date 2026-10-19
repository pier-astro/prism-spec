"""Utility functions for prism.modeling.fitting."""

import warnings

import numpy as np

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

def _parameter_slices(model):
    """Map public parameter names/sizes to the flattened model vector."""
    slices = {}
    cursor = 0
    for name in model.param_names:
        size = getattr(model, name).size
        slices[name] = slice(cursor, cursor + size)
        cursor += size
    return slices


class _ParameterMap:
    """One flattened parameter order for optimization, bounds and constraints."""

    def __init__(self, model):
        self.initial = model.parameters.copy()
        self.slices = _parameter_slices(model)
        free = []
        bounds = []
        self.names = []
        self.tied = []
        for name, section in self.slices.items():
            param = getattr(model, name)
            self.names.extend([name] * param.size)
            if param.tied:
                self.tied.append((param.tied, section, param))
            if param.fixed or param.tied:
                continue
            free.extend(range(section.start, section.stop))
            lower, upper = param.bounds
            bounds.extend([(lower if lower is not None else -np.inf,
                            upper if upper is not None else np.inf)] * param.size)
        from ..parameters import _ordered_ties
        self.tied = _ordered_ties(model, self.tied)
        self.free = np.asarray(free, dtype=int)
        self.values = self.initial[self.free]
        self.bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)


def apply_tied(model):
    """Update links after ordering their declared dependencies.

    Declarative expressions expose dependencies and reject missing references
    or cycles. Opaque callables retain their original parameter order unless a
    declarative expression explicitly depends on their target.
    """
    if model.has_tied:
        _apply_tied_fast(model, _get_tied_info(model), model.parameters.copy())
    return model


def _get_tied_info(model):
    """Prepare callable links using public parameter names and sizes."""
    if not model.has_tied:
        return []
    from ..parameters import _ordered_ties
    items = [(model.tied[name], section, getattr(model, name))
             for name, section in _parameter_slices(model).items() if model.tied[name]]
    return _ordered_ties(model, items)


def _apply_tied_fast(model, tied_info, parameters_cache):
    """Apply prepared links, making each new value visible to subsequent links."""
    for tied_func, section, parameter in tied_info:
        size = section.stop - section.start
        values = _coerce_tied_value_array(tied_func(model), size)
        parameters_cache[section] = values
        parameter.value = values.reshape(parameter.shape)


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
    if array.size == 1:
        return np.full(size, array.item())
    if array.size != size:
        raise ValueError(
            f"Tied parameter returned {array.size} values, expected {size}."
        )
    return array


def _parameter_index_array(model, name):
    """Return flattened indices using public parameter names and sizes."""
    section = _parameter_slices(model)[name]
    return np.arange(section.start, section.stop)


def _free_parameter_indices(model):
    """Return flattened free indices in the canonical model parameter order."""
    indices = []
    for name, section in _parameter_slices(model).items():
        if model.fixed[name] or model.tied[name]:
            continue
        indices.extend(range(section.start, section.stop))
    return np.asarray(indices, dtype=int)


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
    tied_info = _get_tied_info(model)
    tied_rows = [np.arange(section.start, section.stop)
                 for _, section, _ in tied_info]
    free_rows = _free_parameter_indices(model)

    try:
        for free_index in free_rows:
            p0 = float(original[free_index])
            eps = max(abs(p0) * _TIE_DERIV_REL_STEP, _TIE_DERIV_ABS_STEP)
            perturbed = original.copy()
            perturbed[free_index] = p0 + eps
            model.parameters = perturbed
            _apply_tied_fast(model, tied_info, perturbed)
            constrained = model.parameters.copy()

            for indices in tied_rows:
                delta = (constrained[indices] - original[indices]) / eps
                if np.any(np.abs(delta) > _TIE_DERIV_ZERO_TOL):
                    full[free_index] += np.sum(full[indices] * delta[:, np.newaxis], axis=0)
    finally:
        model.parameters = original
        _apply_tied_fast(model, tied_info, original.copy())

    for indices in tied_rows:
        full[indices] = 0.0

    return full
