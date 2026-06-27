"""
Separable least-squares utilities and fitters.

``SeparableTRF`` solves designated linear parameters exactly inside each outer
Trust Region Reflective iteration, so only the remaining non-linear parameters
are optimized numerically.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from scipy import optimize
from astropy.modeling import CompoundModel
from astropy.modeling.fitting import Fitter, model_to_fit_params

from .extension import (
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    get_max_evaluations,
    validate_symmetric_yerr,
)
from .multifit import MultiFitMixin, _MULTIFIT_STAT_KEYS
from .utils import (
    _apply_tied_fast,
    _get_tied_info,
    _prime_tied_analytic_jacobian_warning,
    _reduce_tied_analytic_jacobian,
)

__all__ = ['SeparableTRF', 'inspect_separable']


def _slice_to_indices(model, name):
    indices = np.arange(model.parameters.size)[model._param_metrics[name]['slice']]
    if indices.size != 1:
        raise ValueError(
            f"SeparableTRF currently supports scalar parameters only; '{name}' has size {indices.size}."
        )
    return int(indices[0])


def _leaf_linear_names(model):
    names = getattr(model, '_separable_linear_params', None)
    if names is None:
        if getattr(model, 'linear', False):
            return tuple(model.param_names)
        return ()
    if names == 'all':
        return tuple(model.param_names)
    return tuple(names)


def _auto_linear_param_names(model):
    if isinstance(model, CompoundModel):
        full_names = []
        for idx in range(model.n_submodels):
            submodel = model[idx]
            for local_name in _leaf_linear_names(submodel):
                full_names.append(f'{local_name}_{idx}')
        return tuple(full_names)
    return _leaf_linear_names(model)


def _resolve_linear_param_names(model, linear_params):
    if linear_params in (None, 'auto'):
        names = _auto_linear_param_names(model)
    elif isinstance(linear_params, str):
        names = (linear_params,)
    else:
        names = tuple(linear_params)

    invalid = [name for name in names if name not in model.param_names]
    if invalid:
        raise ValueError(
            f"Unknown linear parameter(s): {invalid}. Available parameters: {list(model.param_names)}"
        )
    return names


def _parameter_bounds(model, name):
    lower, upper = getattr(model, name).bounds
    lower = -np.inf if lower is None else float(lower)
    upper = np.inf if upper is None else float(upper)
    return lower, upper


def _compute_numeric_jacobian(residual_func, p0):
    p0 = np.asarray(p0, dtype=float)
    r0 = np.asarray(residual_func(p0), dtype=float)
    jac = np.empty((r0.size, p0.size), dtype=float)
    for idx, value in enumerate(p0):
        step = max(abs(value) * 1e-7, 1e-9)
        trial = p0.copy()
        trial[idx] = value + step
        jac[:, idx] = (np.asarray(residual_func(trial), dtype=float) - r0) / step
    return jac


def _analytic_weighted_jacobian(model, x, weights, row_indices):
    full = model.fit_deriv(x, *model.parameters)
    reduced = _reduce_tied_analytic_jacobian(
        model,
        full,
        warn=True,
        stacklevel=4,
    )
    rows = np.asarray(reduced, dtype=float)[np.asarray(row_indices, dtype=int)]
    weighted = rows if weights is None else rows * np.asarray(weights, dtype=float)
    return weighted.T


def _project_from_linear_span(weighted_design, weighted_jacobian):
    if weighted_design.size == 0:
        return weighted_jacobian
    q, _ = np.linalg.qr(weighted_design, mode='reduced')
    return weighted_jacobian - q @ (q.T @ weighted_jacobian)


def inspect_separable(model, linear_params='auto'):
    """
    Inspect Prism's linear/non-linear partition for a model.

    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    linear_params : {'auto'} or iterable of str, optional
        Requested linear-parameter declaration. ``'auto'`` (default) uses
        Prism model metadata and Astropy's ``model.linear`` flag on simple
        submodels.

    Returns
    -------
    info : dict
        Dictionary summarizing the partition with keys:

        - ``linear_params``: free parameters solved in the inner linear step
        - ``nonlinear_params``: free parameters optimized by outer TRF
        - ``free_params``: all free parameters
        - ``fixed_params``: fixed parameters
        - ``tied_params``: tied parameters
        - ``candidate_linear_params``: all declared linear candidates
        - ``supported``: whether at least one free linear parameter exists
        - ``message``: short human-readable summary

    Notes
    -----
    Tied parameters are only safe inside the linear block when the tie preserves
    linear superposition. A ratio tie like ``amp2 = 0.33 * amp1`` remains
    linear; a non-linear tie like ``amp2 = amp1**2`` does not.
    """
    candidate_linear = _resolve_linear_param_names(model, linear_params)
    _, fit_indices, _ = model_to_fit_params(model)
    fit_indices = set(np.asarray(fit_indices, dtype=int).tolist())

    free_params = []
    fixed_params = []
    tied_params = []
    for name in model.param_names:
        flat_index = _slice_to_indices(model, name)
        if getattr(model, name).tied:
            tied_params.append(name)
        elif getattr(model, name).fixed:
            fixed_params.append(name)
        elif flat_index in fit_indices:
            free_params.append(name)

    linear_free = [name for name in free_params if name in candidate_linear]
    nonlinear_free = [name for name in free_params if name not in candidate_linear]
    supported = bool(linear_free)
    message = (
        f"{len(linear_free)} linear and {len(nonlinear_free)} non-linear free parameters."
        if supported
        else "No free linear parameters detected; use a generic non-linear fitter or pass linear_params explicitly."
    )
    return {
        'linear_params': tuple(linear_free),
        'nonlinear_params': tuple(nonlinear_free),
        'free_params': tuple(free_params),
        'fixed_params': tuple(fixed_params),
        'tied_params': tuple(tied_params),
        'candidate_linear_params': tuple(candidate_linear),
        'supported': supported,
        'message': message,
    }


def _linear_probe_values(n_linear):
    if n_linear == 0:
        return np.empty(0, dtype=float)
    values = np.linspace(0.25, 1.25, n_linear, dtype=float)
    values += 0.17
    return values


def _validate_linear_superposition(model, x, linear_names, base, design, tied_info):
    if len(linear_names) == 0:
        return

    params0 = model.parameters.copy()
    linear_indices = np.asarray([_slice_to_indices(model, name) for name in linear_names], dtype=int)
    coeffs = _linear_probe_values(len(linear_names))
    probe = params0.copy()
    probe[linear_indices] = coeffs
    model.parameters = probe
    if tied_info:
        tied_probe = model.parameters.copy()
        _apply_tied_fast(model, tied_info, tied_probe)

    actual = np.asarray(model(x), dtype=float)
    predicted = base + design @ coeffs
    model.parameters = params0
    if tied_info:
        restored = model.parameters.copy()
        _apply_tied_fast(model, tied_info, restored)

    scale = max(1.0, float(np.max(np.abs(actual))))
    if np.max(np.abs(actual - predicted)) > 1e-8 * scale:
        raise ValueError(
            "Declared linear parameters do not preserve linear superposition under the current ties. "
            "Use affine ties such as amp2 = 0.33 * amp1, not non-linear ties such as amp2 = amp1**2."
        )


class SeparableTRF(MultiFitMixin, Fitter):
    """
    Trust Region Reflective fitter with exact inner linear solves.

    Parameters
    ----------
    linear_params : {'auto'} or iterable of str, optional
        Names of the parameters to solve linearly. ``'auto'`` (default) uses
        Prism model metadata and Astropy's native ``model.linear`` flag on
        simple submodels.
    calc_uncertainties : bool, optional
        If ``True``, estimate a covariance matrix for all free parameters from
        a finite-difference Jacobian of the final full model. Default is
        ``False``. Purely linear fits still expose their exact covariance even
        when this is ``False`` because it is available directly from the inner
        linear solve.
    verbose : bool, optional
        Print basic diagnostics. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.optimize.least_squares`` for the outer non-linear problem.

    Notes
    -----
    The current implementation intentionally keeps the scope narrow:

    - only 1-D fitting is supported
    - linear parameters must be scalar
    - tied parameters are supported only when they preserve linear
      superposition of the chosen linear block

    The linear and non-linear steps are nested, not run as two disconnected
    fits. For each outer trial point in the non-linear parameters, the fitter
    rebuilds the linear design matrix and solves the linear parameters exactly.
    The residual returned to TRF therefore already includes the best linear
    coefficients for that trial shape.

    The inner linear solve respects parameter bounds via
    ``scipy.optimize.lsq_linear`` when any linear bound is finite.

    When the inner linear block is unconstrained and the model exposes
    analytic derivatives, ``SeparableTRF`` automatically uses a projected
    analytic outer Jacobian. Bounded linear blocks fall back to numeric outer
    derivatives.

    A tied parameter remains safe inside the linear block only if the tie is
    affine in the free linear parameters. For example:

    - safe: ``amp2 = 0.33 * amp1``
    - not separable: ``amp2 = amp1**2``
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)

    def __init__(self, linear_params='auto', calc_uncertainties=False, verbose=False, **kwargs):
        self.linear_params = linear_params
        self.calc_uncertainties = calc_uncertainties
        self.verbose = verbose
        self.fit_kwargs = kwargs

    @property
    def max_evaluations(self):
        value = self.fit_kwargs.get('max_nfev')
        if value is not None:
            return value
        return get_max_evaluations()

    @max_evaluations.setter
    def max_evaluations(self, value):
        coerced = _coerce_max_evaluations(value)
        if coerced is None:
            self.fit_kwargs.pop('max_nfev', None)
        else:
            self.fit_kwargs['max_nfev'] = coerced

    def _multifit_entry_from_fit_info(self):
        entry = {
            'success': self.fit_info.get('success', False),
            'nfev': self.fit_info.get('nfev', np.nan),
            'message': self.fit_info.get('message', ''),
        }
        for name in _MULTIFIT_STAT_KEYS:
            entry[name] = self.fit_info.get(name, np.nan)
        return entry

    def _partition_parameters(self, model, linear_param_names):
        _, fit_indices, _ = model_to_fit_params(model)
        fit_indices = np.asarray(fit_indices, dtype=int)
        free_names = []
        for name in model.param_names:
            flat_index = _slice_to_indices(model, name)
            if flat_index in fit_indices:
                free_names.append(name)

        linear_free_names = tuple(name for name in free_names if name in linear_param_names)
        nonlinear_free_names = tuple(name for name in free_names if name not in linear_param_names)

        linear_indices = np.asarray([_slice_to_indices(model, name) for name in linear_free_names], dtype=int)
        nonlinear_indices = np.asarray([_slice_to_indices(model, name) for name in nonlinear_free_names], dtype=int)
        return fit_indices, linear_free_names, nonlinear_free_names, linear_indices, nonlinear_indices

    def _set_parameters(self, model, params, tied_info):
        params = np.asarray(params, dtype=float)
        model.parameters = params
        if tied_info:
            params = model.parameters.copy()
            _apply_tied_fast(model, tied_info, params)
        return model.parameters.copy()

    def _build_linear_system(self, model, x, linear_names, tied_info):
        params0 = model.parameters.copy()
        trial_params = params0.copy()
        linear_indices = np.asarray([_slice_to_indices(model, name) for name in linear_names], dtype=int)

        for idx in linear_indices:
            trial_params[idx] = 0.0

        self._set_parameters(model, trial_params, tied_info)
        base = np.asarray(model(x), dtype=float)

        design = np.empty((base.size, len(linear_names)), dtype=float)
        lower = np.empty(len(linear_names), dtype=float)
        upper = np.empty(len(linear_names), dtype=float)

        try:
            for col, name in enumerate(linear_names):
                idx = linear_indices[col]
                one_params = trial_params.copy()
                one_params[idx] = 1.0
                self._set_parameters(model, one_params, tied_info)
                design[:, col] = np.asarray(model(x), dtype=float) - base
                lower[col], upper[col] = _parameter_bounds(model, name)
        finally:
            self._set_parameters(model, params0, tied_info)

        return base, design, lower, upper

    def _solve_linear_block(self, base, design, y, weights, lower, upper):
        rhs = np.asarray(y, dtype=float) - base
        if weights is None:
            weighted_design = design
            weighted_rhs = rhs
        else:
            w = np.asarray(weights, dtype=float)
            weighted_design = design * w[:, np.newaxis]
            weighted_rhs = rhs * w

        constrained = np.isfinite(lower).any() or np.isfinite(upper).any()
        if constrained:
            result = optimize.lsq_linear(
                weighted_design,
                weighted_rhs,
                bounds=(lower, upper),
                method='trf',
            )
            coeffs = result.x
            success = bool(result.success)
            message = result.message
        else:
            coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)
            success = True
            message = 'Linear least-squares solve converged.'

        model_y = base + design @ coeffs
        residual = model_y - y
        if weights is not None:
            residual = residual * weights
        return coeffs, np.asarray(residual, dtype=float), success, message

    def __call__(self, model, x, y, z=None, weights=None, max_nfev=None,
                 yerr=None, statistic='chi2', inplace=True, linear_params=None, **kwargs):
        if z is not None:
            raise NotImplementedError("SeparableTRF currently supports only 1-D models.")
        if yerr is not None:
            yerr = validate_symmetric_yerr(yerr)
        if yerr is not None and weights is None:
            weights = 1.0 / np.asarray(yerr)

        fit_model = model if inplace else model.copy()
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
        tied_info = _get_tied_info(fit_model)

        chosen_linear = self.linear_params if linear_params is None else linear_params
        linear_param_names = _resolve_linear_param_names(fit_model, chosen_linear)
        fit_indices, linear_free_names, nonlinear_free_names, linear_indices, nonlinear_indices = (
            self._partition_parameters(fit_model, linear_param_names)
        )

        if fit_indices.size == 0:
            self.fit_info = {
                'success': True,
                'nfev': 0,
                'message': 'No free parameters',
                'param_cov': None,
                'cost': 0.0,
                'stat': 0.0,
                'statmethod': statistic if (weights is not None or yerr is not None) else 'leastsq',
                'ndata': len(y),
                'nparam': len(fit_model.param_names),
                'nfree': 0,
                'dof': len(y),
                'optimality': 0.0,
                'status': 0,
                'n_linear': 0,
                'n_nonlinear': 0,
            }
            return fit_model

        if len(linear_free_names) == 0:
            raise ValueError(
                "SeparableTRF found no free linear parameters. "
                "Pass linear_params=... explicitly or use a generic non-linear fitter."
            )

        outer_template = fit_model.parameters.copy()
        nonlinear_bounds = np.array([_parameter_bounds(fit_model, name) for name in nonlinear_free_names], dtype=float)
        nonlinear_init = outer_template[nonlinear_indices].copy()

        def apply_nonlinear(values):
            params = outer_template.copy()
            if nonlinear_indices.size:
                params[nonlinear_indices] = values
            self._set_parameters(fit_model, params, tied_info)

        def solve_current_linear():
            base, design, lower, upper = self._build_linear_system(
                fit_model, x, linear_free_names, tied_info
            )
            _validate_linear_superposition(fit_model, x, linear_free_names, base, design, tied_info)
            coeffs, residual, success, message = self._solve_linear_block(
                base, design, y, weights, lower, upper
            )
            params = fit_model.parameters.copy()
            params[linear_indices] = coeffs
            self._set_parameters(fit_model, params, tied_info)
            return {
                'coeffs': coeffs,
                'residual': residual,
                'success': success,
                'message': message,
                'base': base,
                'design': design,
                'lower': lower,
                'upper': upper,
            }

        cache = {}

        def cached_state(values):
            values = np.asarray(values, dtype=float)
            cached = cache.get('x')
            if cached is None or cached.shape != values.shape or not np.array_equal(cached, values):
                apply_nonlinear(values)
                state = solve_current_linear()
                cache['x'] = values.copy()
                cache['state'] = state
            return cache['state']

        def residual_nonlinear(values):
            state = cached_state(values)
            return state['residual']

        merged_kwargs = self.fit_kwargs.copy()
        merged_kwargs.update(kwargs)
        requested_jac = merged_kwargs.pop('jac', 'auto')
        if callable(requested_jac) or requested_jac not in {'auto', 'analytic', '2-point', '3-point', 'cs'}:
            raise ValueError(
                "SeparableTRF currently supports jac='auto', jac='analytic', "
                "or the numeric outer Jacobians '2-point', '3-point', and 'cs'."
            )
        outer_kwargs = {k: v for k, v in merged_kwargs.items() if k not in ['inplace', 'yerr', 'linear_params']}
        if max_nfev is not None:
            outer_kwargs['max_nfev'] = max_nfev

        linear_bounds_finite = bool(np.isfinite([_parameter_bounds(fit_model, name) for name in linear_free_names]).any())
        analytic_available = callable(getattr(fit_model, 'fit_deriv', None))
        use_analytic_outer = (
            nonlinear_indices.size > 0
            and analytic_available
            and not linear_bounds_finite
            and requested_jac in {'auto', 'analytic'}
        )
        if requested_jac == 'analytic' and not use_analytic_outer:
            raise ValueError(
                "jac='analytic' requires analytic model derivatives and an unconstrained linear block."
            )

        if use_analytic_outer:
            _prime_tied_analytic_jacobian_warning(fit_model)

            def jac_nonlinear(values):
                state = cached_state(values)
                weighted_design = state['design'] if weights is None else state['design'] * weights[:, np.newaxis]
                weighted_nonlinear = _analytic_weighted_jacobian(
                    fit_model, x, weights, nonlinear_indices
                )
                return _project_from_linear_span(weighted_design, weighted_nonlinear)

            outer_jac = jac_nonlinear
            outer_jac_label = 'analytic'
        else:
            outer_jac = '2-point' if requested_jac == 'auto' else requested_jac
            outer_jac_label = str(outer_jac)

        outer_kwargs.update({'method': 'trf', 'jac': outer_jac})

        if nonlinear_indices.size == 0:
            apply_nonlinear(np.empty(0, dtype=float))
            state = solve_current_linear()
            cost = 0.5 * float(np.dot(state['residual'], state['residual']))
            result_success = bool(state['success'])
            result_message = state['message']
            result_nfev = 0
            result_status = 0
            result_optimality = 0.0
        else:
            try:
                result = optimize.least_squares(
                    residual_nonlinear,
                    nonlinear_init,
                    bounds=(nonlinear_bounds[:, 0], nonlinear_bounds[:, 1]),
                    **outer_kwargs,
                )
            except (LinAlgError, ValueError) as exc:
                raise RuntimeError(f"SeparableTRF failed: {exc}") from exc

            apply_nonlinear(result.x)
            state = solve_current_linear()
            cost = float(result.cost)
            result_success = bool(result.success and state['success'])
            result_message = getattr(result, 'message', state['message'])
            result_nfev = int(result.nfev)
            result_status = int(getattr(result, 'status', 0))
            result_optimality = float(getattr(result, 'optimality', np.nan))

        param_cov = None
        if fit_indices.size and (self.calc_uncertainties or len(nonlinear_free_names) == 0):
            final_free = fit_model.parameters[fit_indices].copy()
            base_params = fit_model.parameters.copy()
            try:
                if len(nonlinear_free_names) == 0:
                    base, design, lower, upper = self._build_linear_system(
                        fit_model, x, linear_free_names, tied_info
                    )
                    _validate_linear_superposition(
                        fit_model, x, linear_free_names, base, design, tied_info
                    )
                    weighted_design = design if weights is None else design * weights[:, np.newaxis]
                    hess = weighted_design.T @ weighted_design
                    resid = np.asarray(fit_model(x), dtype=float) - y
                    if weights is not None:
                        resid = resid * weights
                    dof = max(1, len(resid) - weighted_design.shape[1])
                    scale = float(np.dot(resid, resid)) / dof
                    param_cov = np.linalg.pinv(hess) * scale
                else:
                    def residual_full(values):
                        trial = base_params.copy()
                        trial[fit_indices] = values
                        self._set_parameters(fit_model, trial, tied_info)
                        diff = np.asarray(fit_model(x), dtype=float) - y
                        if weights is not None:
                            diff = diff * weights
                        return diff

                    jac_full = _compute_numeric_jacobian(residual_full, final_free)
                    hess = jac_full.T @ jac_full
                    param_cov = np.linalg.pinv(hess)
            except Exception:
                if self.verbose:
                    print("Warning: covariance estimation failed for SeparableTRF.")
                param_cov = None
            finally:
                self._set_parameters(fit_model, base_params, tied_info)

        n_data = len(y)
        n_free = int(fit_indices.size)
        self.fit_info = {
            'success': result_success,
            'nfev': result_nfev,
            'message': result_message,
            'param_cov': param_cov,
            'cost': cost,
            'stat': 2.0 * cost,
            'statmethod': statistic if (weights is not None or yerr is not None) else 'leastsq',
            'ndata': n_data,
            'nparam': len(fit_model.param_names),
            'nfree': n_free,
            'dof': n_data - n_free,
            'optimality': result_optimality,
            'status': result_status,
            'n_linear': len(linear_free_names),
            'n_nonlinear': len(nonlinear_free_names),
            'outer_jacobian': 'none' if nonlinear_indices.size == 0 else outer_jac_label,
        }

        if param_cov is not None:
            std = np.sqrt(np.maximum(np.diag(param_cov), 0.0))
            for free_index, sigma in zip(fit_indices, std):
                getattr(fit_model, fit_model.param_names[int(free_index)]).std = sigma
            fit_model._param_cov = np.asarray(param_cov)

        return fit_model
