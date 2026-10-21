"""Shared call, diagnostic and covariance helpers for Prism fitters."""

import inspect
import warnings
import numpy as np
from .data import _prepare_fit_data, _prepare_spatial_data

from .utils import (
    _free_parameter_indices,

)


_DEFAULT_MAX_EVALUATIONS = None


def _validate_spectral_inputs(model, x, y, z=None, yerr=None, weights=None, *, statistic='chi2'):
    """Keep the numerical spectral fitting boundary explicit."""
    if z is not None:
        raise NotImplementedError('Numerical spectral fitters are one-dimensional; use an Astropy spatial adapter for z.')
    if statistic not in ('chi2', 'leastsq'):
        raise NotImplementedError("Numerical fitters minimize weighted least squares; statistic must be 'chi2' or 'leastsq'.")
    _validate_native_inputs(model, x, y, yerr, weights)


def _validate_native_inputs(model, *values):
    """Require explicit numeric units at the native fitting boundary."""
    from ..composition import Model
    if isinstance(model, Model) and any(hasattr(value, 'unit') for value in values if value is not None):
        raise TypeError('Native fitting expects numeric data in consistent native units; convert Quantities explicitly before fitting.')


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
        _validate_native_inputs(model, x, y, z, yerr, weights)
        if z is None:
            x, y, yerr, weights, valid = _prepare_fit_data(
                x, y, yerr, weights, kwargs.pop('mask', None),
                filter_non_finite=filter_non_finite)
            if not np.all(valid):
                x, y = x[valid], y[valid]
                yerr = None if yerr is None else yerr[valid]
                weights = None if weights is None else weights[valid]
        else:
            x, y, z, yerr, weights = _prepare_spatial_data(
                x, y, z, yerr, weights, kwargs.pop('mask', None),
                filter_non_finite=filter_non_finite)

        # 2. Compute weights
        if weights is None and yerr is not None:
            weights = compute_weights(y, yerr, statistic)

        if not inplace:
            eval_model = model.copy()
        else:
            eval_model = model

        prepare_links = getattr(self, '_prepare_links', None)
        if prepare_links is not None:
            prepare_links(eval_model, x, y, z)

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

        restore_units = getattr(self, '_restore_link_units', None)
        if restore_units is not None:
            restore_units(result)

        if inplace and result is not model:
            for name in model.param_names:
                fitted = getattr(result, name)
                original = getattr(model, name)
                if fitted.unit is not None and original.unit is not None:
                    original.value = fitted.quantity.to_value(original.unit)
                elif fitted.unit is not None:
                    setattr(model, name, fitted.quantity)
                else:
                    original.value = fitted.value
            result = model

        # Enrich fit_info with standard diagnostic fields
        if hasattr(self, 'fit_info'):
            fi = self.fit_info
            fit_indices_info = _free_parameter_indices(result)
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

        _attach_covariance(result, getattr(self, 'fit_info', {}).get('param_cov'))

        return result

    return wrapped_call


def _clear_fit_uncertainties(model):
    """Remove uncertainty estimates belonging to a previous fit."""
    for name in model.param_names:
        parameter = getattr(model, name)
        for attribute in ('std', 'median', 'lolim', 'uplim', 'percentiles', 'samples'):
            if hasattr(parameter, attribute):
                try:
                    delattr(parameter, attribute)
                except (AttributeError, TypeError):
                    setattr(parameter, attribute, None)
    for attribute in ('_param_cov', '_fit_indices'):
        if hasattr(model, attribute):
            delattr(model, attribute)


def _attach_covariance(model, covariance):
    """Replace uncertainty estimates in the shared free-parameter order."""
    if covariance is None:
        _clear_fit_uncertainties(model)
        return model
    covariance = np.asarray(covariance)
    free = _free_parameter_indices(model)
    if covariance.shape != (len(free), len(free)):
        raise ValueError('Covariance shape does not match the free parameter vector.')
    _clear_fit_uncertainties(model)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    free_position = 0
    for name in model.param_names:
        param = getattr(model, name)
        if not param.fixed and not param.tied:
            values = std[free_position:free_position + param.size]
            param.std = values[0] if param.size == 1 else values.reshape(param.shape)
            free_position += param.size
    model._param_cov = covariance
    model._fit_indices = free
    return model


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
