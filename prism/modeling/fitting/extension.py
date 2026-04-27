"""
Compatibility helpers for Prism fitter wrappers and narrow Astropy patches.

This module no longer patches Astropy fitter classes broadly. It provides:

- shared call-wrapping logic used by Prism-owned Astropy-style fitters
- shared fit-info and covariance helpers for those wrappers
- the narrow tied-analytic-Jacobian compatibility patch Prism still needs
    for Astropy nonlinear fitters when operator-enhanced compound models are
    fit with analytic derivatives
"""

import inspect
import warnings
import astropy.modeling.fitting as ast_fit
import numpy as np

from .utils import (
    _free_parameter_indices,
    _prime_tied_analytic_jacobian_warning,
    _reduce_tied_analytic_jacobian,
)


_DEFAULT_MAX_EVALUATIONS = None


def validate_symmetric_yerr(yerr, *, context='fitters'):
    """Return a symmetric uncertainty array or raise on asymmetric input."""
    if yerr is None:
        return None
    array = np.asarray(yerr)
    if array.ndim != 1:
        raise ValueError(
            f"Asymmetric yerr is not supported by direct {context}. "
            "Use symmetric yerr or bootstrap(..., noise_dist='uniform') for asymmetric resampling."
        )
    return array


def get_max_evaluations():
    """Return the Prism global default max evaluations for wrapper fitters."""
    return _DEFAULT_MAX_EVALUATIONS


def set_max_evaluations(value):
    """Set the Prism global default max evaluations for wrapper fitters."""
    global _DEFAULT_MAX_EVALUATIONS
    _DEFAULT_MAX_EVALUATIONS = _coerce_max_evaluations(value)


def _coerce_max_evaluations(value):
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"max_evaluations must be a positive integer or None, got {value!r}.")
    if ivalue < 1:
        raise ValueError(f"max_evaluations must be >= 1, got {ivalue}.")
    return ivalue


def _resolve_max_evaluations(self, call_kwargs):
    aliases = ('max_evaluations', 'max_nfev', 'maxiter', 'maxfev', 'maxfun')
    explicit = {}
    for name in aliases:
        if name in call_kwargs:
            val = call_kwargs.pop(name)
            if val is not None:
                explicit[name] = val

    if explicit:
        unique = {_coerce_max_evaluations(v) for v in explicit.values()}
        if len(unique) > 1:
            raise ValueError(
                "Conflicting max evaluation controls were passed: "
                + ", ".join(f"{k}={v}" for k, v in explicit.items())
            )
        return next(iter(unique))

    return _coerce_max_evaluations(getattr(self, 'max_evaluations', None))

def compute_weights(y, yerr, statistic):
    if yerr is None:
        return None
    yerr = validate_symmetric_yerr(yerr)
    stat = str(statistic).lower()
    if stat == 'chi2':
        return 1.0 / yerr
    elif stat == 'poisson':
        return 1.0 / np.sqrt(np.maximum(np.asarray(y), 1e-10))
    else:
        warnings.warn(f"Warning: Unknown statistic '{statistic}'. Using uniform weights.")
        return None

def _wrap_fitter_call(original_call):
    """
    Wraps the native Fitter.__call__ to intercept specific kwargs.
    """
    try:
        kw_signature = inspect.signature(original_call)
        params = kw_signature.parameters
    except (ValueError, TypeError):
        # Fallback for builtins or weird objects
        params = {}

    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepts_weights = 'weights' in params or accepts_var_kwargs
    accepts_filter = 'filter_non_finite' in params
    accepts_verbose = 'verbose' in params
    accepts_inplace = 'inplace' in params
    accepts_yerr = 'yerr' in params
    accepts_statistic = 'statistic' in params
    accepts_maxiter = 'maxiter' in params
    accepts_max_nfev = 'max_nfev' in params
    accepts_maxfev = 'maxfev' in params
    accepts_maxfun = 'maxfun' in params

    def wrapped_call(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None,
                     filter_non_finite=False, verbose=False, inplace=True, **kwargs):
        if yerr is not None:
            yerr = validate_symmetric_yerr(yerr)
        
        # 1. Handle non-finite
        if filter_non_finite:
            finite = np.isfinite(y)
            if weights is not None:
                finite &= np.isfinite(np.asarray(weights))
            elif yerr is not None:
                finite &= np.isfinite(np.asarray(yerr))
            
            x = np.asarray(x)[finite]
            y = np.asarray(y)[finite]
            if z is not None:
                z = np.asarray(z)[finite]
            if weights is not None:
                weights = np.asarray(weights)[finite]
            if yerr is not None:
                yerr = np.asarray(yerr)[finite]
        else:
            # Check inputs
            if not np.all(np.isfinite(y)):
                raise ValueError("Non-finite values detected in the data. Set filter_non_finite=True.")
            if weights is not None and not np.all(np.isfinite(np.asarray(weights))):
                raise ValueError("Non-finite values detected in the weights. Set filter_non_finite=True.")
            if yerr is not None and not np.all(np.isfinite(np.asarray(yerr))):
                raise ValueError("Non-finite values detected in the uncertainties. Set filter_non_finite=True.")

        # 2. Compute weights
        if weights is None and yerr is not None:
            weights = compute_weights(y, yerr, statistic)

        if not inplace:
            eval_model = model.copy()
        else:
            eval_model = model

        _prime_tied_analytic_jacobian_warning(eval_model)

        # 3. Build call kwargs
        call_kwargs = kwargs.copy()
        max_evaluations = _resolve_max_evaluations(self, call_kwargs)

        if accepts_weights and weights is not None:
            call_kwargs['weights'] = weights
        if accepts_filter:
            call_kwargs['filter_non_finite'] = filter_non_finite
        if accepts_verbose:
            call_kwargs['verbose'] = verbose
        if accepts_inplace:
            call_kwargs['inplace'] = inplace
        if accepts_yerr and yerr is not None:
            call_kwargs['yerr'] = yerr
        if accepts_statistic:
            call_kwargs['statistic'] = statistic

        if max_evaluations is not None:
            if accepts_max_nfev:
                call_kwargs['max_nfev'] = max_evaluations
            elif accepts_maxiter or accepts_var_kwargs:
                call_kwargs['maxiter'] = max_evaluations
            elif accepts_maxfev:
                call_kwargs['maxfev'] = max_evaluations
            elif accepts_maxfun:
                call_kwargs['maxfun'] = max_evaluations
        
        # We always strip nproc as no single-spaxel fit needs it, 
        # but we also strip inplace if not explicitly accepted to avoid errors.
        call_kwargs.pop('nproc', None)
        if not accepts_inplace:
            call_kwargs.pop('inplace', None)

        # Reject parameters with equal bounds (lower == upper).
        # scipy.optimize.least_squares requires strict inequality.
        # Use param.fixed = True instead of setting equal bounds.
        for pname in eval_model.param_names:
            p = getattr(eval_model, pname)
            lo, hi = p.bounds
            if lo is not None and hi is not None and lo == hi:
                raise ValueError(
                    f"Parameter '{pname}' has equal lower and upper bounds "
                    f"({lo}). Use `param.fixed = True` instead."
                )

        if z is not None:
            result = original_call(self, eval_model, x, y, z=z, **call_kwargs)
        else:
            result = original_call(self, eval_model, x, y, **call_kwargs)

        if inplace and result is not model:
            model.parameters = result.parameters
            result = model

        # Enrich fit_info with standard diagnostic fields
        if hasattr(self, 'fit_info'):
            fi = self.fit_info
            from astropy.modeling.fitting import model_to_fit_params
            _, fit_indices_info, _ = model_to_fit_params(result)
            n_free = len(fit_indices_info)
            n_data = int(np.prod(np.shape(y))) if z is None else int(np.prod(np.shape(z)))

            if isinstance(fi, dict):
                fi.setdefault('nfree', n_free)
                fi.setdefault('ndata', n_data)
                fi.setdefault('dof', n_data - n_free)
                fi.setdefault('statmethod', statistic if (weights is not None or yerr is not None) else 'leastsq')
                # stat = full chi2 or sum(residuals^2). scipy cost = 0.5 * sum(r^2).
                if 'stat' not in fi:
                    scipy_cost = fi.get('cost', np.nan)
                    if np.isfinite(scipy_cost) and 'native_result' not in fi:
                        # Astropy OptimizeResult: cost = 0.5 * sum(r^2)
                        fi['stat'] = 2.0 * scipy_cost
                    elif np.isfinite(scipy_cost):
                        fi['stat'] = fi.get('stat', scipy_cost)
            else:
                if not hasattr(fi, 'nfree'):
                    fi['nfree'] = n_free
                if not hasattr(fi, 'ndata'):
                    fi['ndata'] = n_data
                if not hasattr(fi, 'dof'):
                    fi['dof'] = n_data - n_free
                if not hasattr(fi, 'statmethod'):
                    fi['statmethod'] = statistic if (weights is not None or yerr is not None) else 'leastsq'
                if not hasattr(fi, 'stat'):
                    cost_val = getattr(fi, 'cost', np.nan)
                    fi['stat'] = 2.0 * cost_val if np.isfinite(cost_val) else np.nan

        if verbose:
            msg = getattr(self, 'fit_info', {}).get('message', '')
            print(f"{self.__class__.__name__} fit completed: {msg}")

        if hasattr(self, 'fit_info') and 'param_cov' in self.fit_info and self.fit_info['param_cov'] is not None:
            cov = self.fit_info['param_cov']
            std = np.sqrt(np.maximum(np.diagonal(np.asarray(cov), axis1=-2, axis2=-1), 0))
            
            # Find free parameters and attach
            from astropy.modeling.fitting import model_to_fit_params
            init_values, fit_indices, _ = model_to_fit_params(result)
            
            idx_map = {}
            cumulative_idx = 0
            for param_name in result.param_names:
                param = getattr(result, param_name)
                for offset in range(param.size):
                    idx_map[cumulative_idx + offset] = (param, offset)
                cumulative_idx += param.size

            for free_index, sigma in zip(fit_indices, std):
                if free_index not in idx_map:
                    continue
                param, offset = idx_map[free_index]
                if param.size == 1:
                    param.std = sigma
                else:
                    if not hasattr(param, 'std') or param.std is None or np.isscalar(param.std):
                        param.std = np.full(param.size, np.nan)
                    param.std[offset] = sigma

            # Attach covariance matrix to the model for downstream propagation
            result._param_cov = np.asarray(cov)

        # 5. Ensure direct fitted_model.save()/to_fits() methods are installed
        from ..io import install_model_io_methods
        install_model_io_methods()

        return result

    return wrapped_call


def _fitter_stdevs(self):
    """Dynamic property returning the standard deviations array if available."""
    cov = getattr(self, 'covariance', None)
    if cov is not None:
        var = np.diagonal(np.asarray(cov), axis1=-2, axis2=-1).copy()
        var[var < 0] = np.nan
        return np.sqrt(var)
    return None


def _fitter_covariance(self):
    """Dynamic property returning the covariance matrix if available."""
    if hasattr(self, 'fit_info'):
        return self.fit_info.get('param_cov', None)
    return None


def _patch_astropy_tied_jacobian_wrapper():
    """Patch Astropy's nonlinear fitter Jacobian wrapper once."""
    fitter_cls = getattr(ast_fit, '_NonLinearLSQFitter', None)
    if fitter_cls is None or getattr(fitter_cls, '_prism_tied_jacobian_patch', False):
        return

    original = fitter_cls.__dict__['_wrap_deriv']
    original_func = original.__func__ if isinstance(original, staticmethod) else original

    def _patched_wrap_deriv(params, model, weights, x, y, z=None, fit_param_indices=None):
        if not model.has_tied:
            return original_func(params, model, weights, x, y, z=z, fit_param_indices=fit_param_indices)

        if weights is None:
            weights = 1.0

        if not hasattr(model, '_prism_warn_tied_analytic_jacobian'):
            _prime_tied_analytic_jacobian_warning(model)
        ast_fit.fitter_to_model_params(model, params)
        if z is None:
            full = model.fit_deriv(x, *model.parameters)
        else:
            full = [np.ravel(item) for item in model.fit_deriv(x, y, *model.parameters)]

        corrected = _reduce_tied_analytic_jacobian(
            model,
            full,
            warn=True,
            stacklevel=3,
        )
        weighted = np.ravel(weights) * corrected
        fit_indices = fit_param_indices
        if fit_indices is None:
            fit_indices = _free_parameter_indices(model)
        fit_indices = np.asarray(fit_indices, dtype=int)

        if not model.col_fit_deriv:
            residues = weighted[fit_indices].T
        else:
            residues = weighted[fit_indices]

        return [np.ravel(item) for item in residues]

    fitter_cls._wrap_deriv = staticmethod(_patched_wrap_deriv)
    fitter_cls._prism_tied_jacobian_patch = True


def enable_astropy_fitting_compatibility_patch():
    """Enable the narrow Astropy fitting compatibility patch Prism still needs."""
    _patch_astropy_tied_jacobian_wrapper()


def patch_astropy_fitters():
    """Deprecated compatibility shim for the old broad fitter patch API."""
    warnings.warn(
        "patch_astropy_fitters() no longer patches native Astropy fitter classes. "
        "Use prism.modeling.fitting.TRFLSQFitter, DogBoxLSQFitter, "
        "LevMarLSQFitter, and related Prism wrappers instead. "
        "Only the narrow tied-analytic-Jacobian compatibility patch is retained.",
        DeprecationWarning,
        stacklevel=2,
    )
    enable_astropy_fitting_compatibility_patch()
