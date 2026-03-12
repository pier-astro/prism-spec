"""
Base fitter class for prism.modeling.fitting

This module provides the FitterBase class that consolidates all common
fitting logic and provides a clean interface for specific fitter implementations.
"""

import abc
import time
import warnings
import numpy as np
from multiprocess import Pool
from tqdm.auto import tqdm

from astropy.modeling.fitting import Fitter, model_to_fit_params
from astropy.modeling import CompoundModel, Model

_MULTIFIT_STAT_KEYS = (
    'cost', 'optimality', 'ndata', 'nparam', 'nfree', 'dof', 'status'
)

__all__ = [
    'FitterBase', 'JointFitResult', 'MultiFitParameter', 'SpectrumFitResult', 'MultiFitResult',
    'tie', 'apply_tied', '_get_tied_info', '_apply_tied_fast'
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


# =============================================================================
# BASE CLASS
# =============================================================================

class JointFitResult:
    """Placeholder for future joint optimization across multiple datasets."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Joint fitting is not implemented yet. Use multifit() for independent fits over multiple spectra."
        )


class MultiFitParameter:
    """Array-like parameter view exposing value and std arrays for a MultiFitResult."""

    def __init__(self, parent, name):
        self._parent = parent
        self.name = name

    @property
    def value(self):
        return self._parent._param_array(self.name)

    @property
    def std(self):
        return self._parent._std_array(self.name)

    @property
    def bounds(self):
        return self._parent.get_bounds(self.name)

    @property
    def min(self):
        return self._parent.get_min(self.name)

    @property
    def max(self):
        return self._parent.get_max(self.name)

    @property
    def fixed(self):
        return self._parent.get_fixed(self.name)

    @property
    def tied(self):
        return self._parent.get_tied(self.name)

    @property
    def unit(self):
        return self._parent.get_unit(self.name)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, index):
        return self.value[index]

    def __iter__(self):
        return iter(self.value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"MultiFitParameter(name={self.name!r}, shape={self.shape}, dtype={self.value.dtype})"

class MultiFitResult:
    """
    Stores the result of a 2D/3D fitting over multiple spectra.
    Provides parameter maps, per-spectrum diagnostics, and model reconstruction.
    """
    def __init__(self, shape, param_names, template_model=None, x=None,
                 has_cov=False, spectral_axis=-1, fit_indices=None,
                 parameter_metadata=None):
        self.shape = tuple(shape)
        self.param_names = tuple(param_names)
        self._template_model = template_model.copy() if template_model is not None else None
        self._x = None if x is None else np.asarray(x)
        self.spectral_axis = spectral_axis
        self._fit_indices = None if fit_indices is None else np.asarray(fit_indices)
        self._parameter_metadata = parameter_metadata or {}
        self.fit_info = {}
        
        # We store flattened arrays initially and reshape upon access
        self.n_spaxels = np.prod(shape)
        self._params = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._errs = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._stats = {name: np.full(self.n_spaxels, np.nan) for name in _MULTIFIT_STAT_KEYS}
        self._success = np.zeros(self.n_spaxels, dtype=bool)
        self._nfev = np.zeros(self.n_spaxels, dtype=float)
        self._messages = np.full(self.n_spaxels, '', dtype=object)
        
        self.has_cov = has_cov
        if has_cov:
            self._cov = np.full((self.n_spaxels, len(param_names), len(param_names)), np.nan)

    def _normalize_spaxel_index(self, index):
        """Normalize flat or spatial indices to a flat index."""
        if isinstance(index, tuple):
            if len(index) != len(self.shape):
                raise IndexError(
                    f"Expected {len(self.shape)} spatial indices, got {len(index)}."
                )
            return np.ravel_multi_index(index, self.shape)

        flat_index = int(index)
        if flat_index < 0:
            flat_index += self.n_spaxels
        if flat_index < 0 or flat_index >= self.n_spaxels:
            raise IndexError(f"Spaxel index {index} is out of bounds for {self.shape}.")
        return flat_index

    def _parameter_vector(self, index):
        flat_index = self._normalize_spaxel_index(index)
        return np.array([self._params[name][flat_index] for name in self.param_names], dtype=float)

    def _error_vector(self, index):
        flat_index = self._normalize_spaxel_index(index)
        return np.array([self._errs[name][flat_index] for name in self.param_names], dtype=float)

    def _param_array(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._params[name].reshape(self.shape)

    def _std_array(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._errs[name].reshape(self.shape)

    def _parameter_meta(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._parameter_metadata.get(name, {})

    def _template_parameter(self, name):
        if self._template_model is None:
            return None
        return getattr(self._template_model, name)

    def _metadata_value(self, name, key, default=None):
        meta = self._parameter_meta(name)
        if key in meta:
            return meta[key]
        param = self._template_parameter(name)
        if param is None:
            return default
        return getattr(param, key, default)

    def _full_bounds_array(self, bounds):
        expanded = np.empty(self.shape, dtype=object)
        bound_pair = tuple(bounds)
        flat = expanded.reshape(-1)
        for idx in range(self.n_spaxels):
            flat[idx] = bound_pair
        return expanded

    def _bounds_component_array(self, name, component_index):
        bounds = self.get_bounds(name)
        values = [pair[component_index] for pair in bounds.flat]
        if any(value is None for value in values):
            component = np.empty(self.shape, dtype=object)
            component.flat[:] = values
            return component
        return np.asarray(values, dtype=float).reshape(self.shape)

    def _metadata_scalar(self, value, flat_index):
        if isinstance(value, np.ndarray):
            scalar = value.reshape(-1)[flat_index]
            if np.issubdtype(value.dtype, np.floating) and np.isnan(scalar):
                return None
            return scalar.item() if hasattr(scalar, 'item') else scalar
        return value

    def get_param(self, name):
        """Return the parameter view for one fitted parameter across all spectra."""
        return MultiFitParameter(self, name)

    def get_std(self, name):
        """Return the standard deviation array for one fitted parameter."""
        return self._std_array(name)

    def get_bounds(self, name):
        """Return a per-spectrum array of (lower, upper) tuples for one parameter."""
        bounds = self._metadata_value(name, 'bounds', default=None)
        if isinstance(bounds, np.ndarray):
            return bounds.reshape(self.shape)
        if bounds is not None:
            return self._full_bounds_array(bounds)
        param = self._template_parameter(name)
        return self._full_bounds_array((None, None) if param is None else param.bounds)

    def get_min(self, name):
        """Return the per-spectrum lower-bound array for one parameter."""
        return self._bounds_component_array(name, 0)

    def get_max(self, name):
        """Return the per-spectrum upper-bound array for one parameter."""
        return self._bounds_component_array(name, 1)

    def get_fixed(self, name):
        """Return the fixed state for one parameter."""
        return bool(self._metadata_value(name, 'fixed', default=False))

    def get_tied(self, name):
        """Return the tied callable or False for one parameter."""
        return self._metadata_value(name, 'tied', default=False)

    def get_unit(self, name):
        """Return the parameter unit, if present."""
        return self._metadata_value(name, 'unit', default=None)

    def get_stat(self, name):
        """Return a reshaped statistic array by statistic name."""
        if name not in _MULTIFIT_STAT_KEYS:
            raise KeyError(f"Unknown statistic '{name}'. Available: {list(_MULTIFIT_STAT_KEYS)}")
        return self._stats[name].reshape(self.shape)

    def get_parameter(self, name):
        return self.get_param(name)

    def get_model(self, index, copy_model=True):
        """Return the fitted model for a specific spaxel."""
        if self._template_model is None:
            raise ValueError("Model reconstruction is unavailable without a template model.")

        flat_index = self._normalize_spaxel_index(index)
        model = self._template_model.copy() if copy_model else self._template_model
        self._apply_parameter_metadata(model, flat_index)
        model.parameters = self._parameter_vector(flat_index)
        self._attach_model_stds(model, flat_index, self._error_vector(flat_index))
        return model

    def _apply_parameter_metadata(self, model, flat_index):
        for param_name in self.param_names:
            param = getattr(model, param_name)
            bound_pair = self.get_bounds(param_name).reshape(-1)[flat_index]
            param.bounds = tuple(bound_pair)
            param.fixed = self.get_fixed(param_name)
            param.tied = self.get_tied(param_name)
        return model

    def _attach_model_stds(self, model, flat_index, stds):
        if stds is None:
            return model
        for param_name, std in zip(self.param_names, stds):
            param = getattr(model, param_name)
            param.std = std
        if self.has_cov:
            model._param_cov = self._cov[flat_index]
        if self._fit_indices is not None:
            model._fit_indices = self._fit_indices
        return model

    def get_fit_info(self, index):
        """Return normalized fit information for a single spaxel."""
        flat_index = self._normalize_spaxel_index(index)
        spatial_index = np.unravel_index(flat_index, self.shape)
        info = {
            'index': spatial_index,
            'success': bool(self._success[flat_index]),
            'nfev': self._nfev[flat_index],
            'message': self._messages[flat_index],
        }
        for name in _MULTIFIT_STAT_KEYS:
            info[name] = self._stats[name][flat_index]
        cov = self.covariance
        info['param_cov'] = None if cov is None else cov[spatial_index]
        return info

    def evaluate(self, x=None, spectral_axis=None):
        """Evaluate the full set of fitted models on the supplied grid."""
        if self._template_model is None:
            raise ValueError("Model evaluation is unavailable without a template model.")

        eval_x = self._x if x is None else np.asarray(x)
        if eval_x is None:
            raise ValueError("x must be provided when the fit result was created without one.")

        evaluated = np.empty(self.shape + (len(eval_x),), dtype=float)
        for flat_index in range(self.n_spaxels):
            spatial_index = np.unravel_index(flat_index, self.shape)
            evaluated[spatial_index] = self.get_model(flat_index)(eval_x)

        target_axis = self.spectral_axis if spectral_axis is None else spectral_axis
        if target_axis < 0:
            target_axis += evaluated.ndim
        if target_axis != evaluated.ndim - 1:
            evaluated = np.moveaxis(evaluated, -1, target_axis)
        return evaluated

    def __call__(self, x=None, spectral_axis=None):
        """Alias for evaluate()."""
        return self.evaluate(x=x, spectral_axis=spectral_axis)

    def update(self, idx, model_params, stdevs, fit_info, cov=None):
        """Update a single spaxel result at flattened index idx."""
        for i, name in enumerate(self.param_names):
            self._params[name][idx] = model_params[i]

        if stdevs is not None:
            if self._fit_indices is None:
                for i, name in enumerate(self.param_names[:len(stdevs)]):
                    self._errs[name][idx] = stdevs[i]
            else:
                for free_index, sigma in zip(self._fit_indices, stdevs):
                    self._errs[self.param_names[int(free_index)]][idx] = sigma

        self._success[idx] = bool(fit_info.get('success', False))
        nfev = fit_info.get('nfev', np.nan)
        self._nfev[idx] = np.nan if nfev is None else nfev
        self._messages[idx] = fit_info.get('message', '')

        for name in _MULTIFIT_STAT_KEYS:
            value = fit_info.get(name, np.nan)
            self._stats[name][idx] = np.nan if value is None else value
        
        if self.has_cov and cov is not None:
            if self._fit_indices is None or cov.shape == self._cov[idx].shape:
                self._cov[idx] = cov
            else:
                expanded_cov = np.full_like(self._cov[idx], np.nan)
                fit_indices = self._fit_indices.astype(int)
                expanded_cov[np.ix_(fit_indices, fit_indices)] = cov
                self._cov[idx] = expanded_cov

    def __getitem__(self, index):
        return SpectrumFitResult(self, index)

    def __getattr__(self, name):
        """Allow direct access to parameter maps e.g. result.amplitude_0"""
        if name in self.param_names:
            return self.get_param(name)
        if name in _MULTIFIT_STAT_KEYS:
            return self.get_stat(name)
        if name in {'message', 'messages'}:
            return self.messages
        raise AttributeError(f"'MultiFitResult' object has no attribute '{name}'")

    @property
    def parameters(self):
        return np.stack([self._param_array(name) for name in self.param_names], axis=-1)

    @property
    def stds(self):
        return np.stack([self._std_array(name) for name in self.param_names], axis=-1)

    @property
    def errors(self):
        return self.stds

    @property
    def messages(self):
        return self._messages.reshape(self.shape)

    @property
    def statistics(self):
        return {name: self.get_stat(name) for name in _MULTIFIT_STAT_KEYS}
        
    @property
    def success(self):
        return self._success.reshape(self.shape)

    @property
    def nfev(self):
        return self._nfev.reshape(self.shape)

    @property
    def covariance(self):
        if not self.has_cov:
            return None
        return self._cov.reshape(self.shape + (len(self.param_names), len(self.param_names)))

    @property
    def stdevs(self):
        return self.stds


class SpectrumFitResult:
    """View of a single fitted spectrum extracted from a MultiFitResult."""

    def __init__(self, parent, index):
        self._parent = parent
        self._flat_index = parent._normalize_spaxel_index(index)
        self.index = np.unravel_index(self._flat_index, parent.shape)
        self._model = None

    def _scalar(self, values):
        return values[self.index] if hasattr(values, 'shape') else values

    @property
    def model(self):
        if self._model is None:
            self._model = self._parent.get_model(self._flat_index)
        return self._model

    @property
    def parameters(self):
        return self._parent._parameter_vector(self._flat_index)

    @property
    def errors(self):
        return self._parent._error_vector(self._flat_index)

    @property
    def stds(self):
        return self.errors

    @property
    def covariance(self):
        covariance = self._parent.covariance
        return None if covariance is None else covariance[self.index]

    @property
    def stdevs(self):
        return self.stds

    @property
    def fit_info(self):
        return self._parent.get_fit_info(self._flat_index)

    @property
    def success(self):
        return bool(self._parent.success[self.index])

    @property
    def nfev(self):
        return self._parent.nfev[self.index]

    @property
    def message(self):
        return self._parent.messages[self.index]

    def __getattr__(self, name):
        if name in self._parent.param_names:
            return getattr(self.model, name)
        if name in _MULTIFIT_STAT_KEYS:
            return self._parent.get_stat(name)[self.index]
        raise AttributeError(f"'SpectrumFitResult' object has no attribute '{name}'")


class FitterBase(Fitter):
    """
    Base class for all prism fitters.
    
    Provides:
    - Consistent interface across all fitter types
    - Standardized fit_info dictionary with scipy.optimize.OptimizeResult-like structure
    - Common parameter handling (bounds, ties, weights)
    - Covariance matrix computation and storage
    - Clean separation between generic logic and specific algorithms
    
    Subclasses only need to implement _fit_impl() method that receives
    prepared model, data, and parameters and returns the fitted result.
    
    Parameters
    ----------
    calc_uncertainties : bool
        Whether to compute covariance matrix.
    force_numerical_covariance : bool
        If True, compute expensive numerical Hessian when native fails.
    verbose : bool
        Print diagnostic information during fitting.
    """
    
    def __init__(self, calc_uncertainties=False, force_numerical_covariance=False,
                 verbose=False, filter_non_finite=False):
        self.calc_uncertainties = calc_uncertainties
        self.force_numerical_covariance = force_numerical_covariance
        self.verbose = verbose
        self.filter_non_finite = filter_non_finite
        self.fit_info = {}
        
    @property
    def covariance(self):
        """Covariance matrix of fitted parameters."""
        return self.fit_info.get('param_cov', None)
        
    @property
    def stdevs(self):
        """Standard deviations of fitted parameters."""
        cov = self.covariance
        if cov is not None:
            var = np.diagonal(np.asarray(cov), axis1=-2, axis2=-1).copy()
            var[var < 0] = np.nan
            return np.sqrt(var)
        return None

    def _attach_param_stds(self, model, covariance, fit_indices):
        """Attach .std values directly to fitted model parameters when covariance is available."""
        if covariance is None:
            return model

        std = np.diagonal(np.asarray(covariance), axis1=-2, axis2=-1).copy()
        std[std < 0] = np.nan
        std = np.sqrt(std)

        if len(std) != len(fit_indices):
            return model

        model._param_cov = covariance
        model._fit_indices = fit_indices

        idx_map = {}
        cumulative_idx = 0
        for param_name in model.param_names:
            param = getattr(model, param_name)
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
        return model

    def _resolve_spectral_axis(self, y, wave_len, spectral_axis=None):
        """Resolve the spectral axis, rejecting ambiguous layouts."""
        if spectral_axis is not None:
            axis = np.lib.array_utils.normalize_axis_index(spectral_axis, y.ndim)
            if y.shape[axis] != wave_len:
                raise ValueError(
                    f"spectral_axis={spectral_axis} has length {y.shape[axis]}, expected {wave_len}."
                )
            return axis

        matches = [axis for axis, size in enumerate(y.shape) if size == wave_len]
        if not matches:
            raise ValueError(
                f"Could not match wave axis length ({wave_len}) to any dimension in y shape {y.shape}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous spectral axis for y shape {y.shape}: axes {matches} all match len(x)={wave_len}. "
                "Pass spectral_axis explicitly."
            )
        return matches[0]

    def _set_batch_fit_info(self, result, total_time):
        """Store aggregate fit information for the last multi-fit call."""
        n_converged = int(result._success.sum())
        n_failed = result.n_spaxels - n_converged
        summary = {
            'time': total_time,
            'success': n_failed == 0,
            'nfev': result.nfev,
            'message': f"{n_converged}/{result.n_spaxels} spectra converged",
            'messages': result.messages,
            'result': result,
            'param_cov': result.covariance,
            'n_success': n_converged,
            'n_failed': n_failed,
            'success_fraction': n_converged / result.n_spaxels if result.n_spaxels else np.nan,
            'statistic': 'batch',
        }
        for name in _MULTIFIT_STAT_KEYS:
            summary[name] = result.get_stat(name)
        result.fit_info = summary
        self.fit_info = summary

    def _result_value(self, native_result, *keys):
        """Extract a field from dict-like or attribute-style optimizer results."""
        if native_result is None:
            return None

        for key in keys:
            value = None
            if isinstance(native_result, dict):
                value = native_result.get(key)
            else:
                value = getattr(native_result, key, None)
                if value is None and hasattr(native_result, 'get'):
                    value = native_result.get(key)
            if value is not None:
                return value
        return None

    def _compute_fit_diagnostics(self, model, prep_data, result, statistic):
        """Collect cheap fit metadata and native optimizer diagnostics."""
        native_result = result.get('native_result')
        native_fun = self._result_value(native_result, 'fun')
        native_cost = self._result_value(native_result, 'cost')
        if native_cost is None and np.isscalar(native_fun):
            native_cost = float(native_fun)

        ndata = float(np.asarray(prep_data['y']).size)
        nparam = float(len(model.parameters))
        nfree = float(len(prep_data['fit_indices']))
        dof = max(ndata - nfree, 0.0)

        diagnostics = {
            'statistic': statistic,
            'cost': np.nan if native_cost is None else native_cost,
            'optimality': self._result_value(native_result, 'optimality'),
            'ndata': ndata,
            'nparam': nparam,
            'nfree': nfree,
            'dof': dof,
            'status': self._result_value(native_result, 'status', 'ier', 'info'),
        }

        if diagnostics['optimality'] is None:
            diagnostics['optimality'] = np.nan
        if diagnostics['status'] is None:
            diagnostics['status'] = np.nan
        return diagnostics

    def _multifit_entry_from_fit_info(self):
        """Extract the per-spectrum subset needed by MultiFitResult."""
        entry = {
            'success': self.fit_info.get('success', False),
            'nfev': self.fit_info.get('nfev', np.nan),
            'message': self.fit_info.get('message', ''),
        }
        for name in _MULTIFIT_STAT_KEYS:
            entry[name] = self.fit_info.get(name, np.nan)
        return entry

    def _compute_weights(self, y, yerr=None, statistic='chi2'):
        """
        Compute weights from data uncertainties and statistic type.
        
        Parameters
        ----------
        y : array-like
            Data values.
        yerr : array-like, optional
            Data uncertainties (sigma). If None, uses uniform weights.
        statistic : str
            'chi2' (default) or 'poisson'.
            
        Returns
        -------
        weights : array-like or None
            Computed weights, or None if uniform weighting.
        """
        if yerr is None:
            return None
            
        yerr = np.asarray(yerr)
        
        if statistic.lower() == 'chi2':
            weights = 1.0 / yerr
        elif statistic.lower() == 'poisson':
            y_arr = np.asarray(y)
            weights = 1.0 / np.sqrt(np.maximum(y_arr, 1e-10))
        else:
            if self.verbose:
                print(f"Warning: Unknown statistic '{statistic}'. Using uniform weights.")
            return None
            
        return weights

    def _validate_multifit_param_name(self, model, name, control_name):
        if name not in model.param_names:
            raise KeyError(
                f"Unknown parameter '{name}' in {control_name}. Available: {list(model.param_names)}"
            )
        param = getattr(model, name)
        if param.size != 1:
            raise ValueError(
                f"{control_name} currently supports only scalar parameters. "
                f"Parameter '{name}' has size {param.size}."
            )
        return param

    def _normalize_multifit_value(self, value, spatial_shape, control_name, param_name):
        if np.isscalar(value) or value is None:
            return value, None

        array = np.asarray(value)
        if array.shape != tuple(spatial_shape):
            raise ValueError(
                f"{control_name} for parameter '{param_name}' must be a scalar or have shape "
                f"{tuple(spatial_shape)}, got {array.shape}."
            )
        return None, array

    def _normalize_multifit_bounds_pair(self, bounds, spatial_shape, param_name):
        if not isinstance(bounds, (tuple, list, np.ndarray)) or len(bounds) != 2:
            raise ValueError(
                f"bounds for parameter '{param_name}' must be a 2-tuple of (lower, upper)."
            )
        lower_shared, lower_map = self._normalize_multifit_value(
            bounds[0], spatial_shape, 'bounds', param_name
        )
        upper_shared, upper_map = self._normalize_multifit_value(
            bounds[1], spatial_shape, 'bounds', param_name
        )
        return (lower_shared, lower_map), (upper_shared, upper_map)

    def _build_multifit_bounds_array(self, spatial_shape, lower_shared, lower_map,
                                     upper_shared, upper_map):
        bounds = np.empty(tuple(spatial_shape), dtype=object)
        n_spaxels = int(np.prod(spatial_shape))
        flat_bounds = bounds.reshape(-1)
        lower_flat = None if lower_map is None else np.asarray(lower_map, dtype=object).reshape(-1)
        upper_flat = None if upper_map is None else np.asarray(upper_map, dtype=object).reshape(-1)

        for idx in range(n_spaxels):
            lower = lower_shared if lower_flat is None else lower_flat[idx]
            upper = upper_shared if upper_flat is None else upper_flat[idx]
            flat_bounds[idx] = (lower, upper)
        return bounds

    def _apply_scalar_multifit_overrides(self, model, initpars=None, bounds=None):
        if initpars:
            for name, value in initpars.items():
                getattr(model, name).value = value
        if bounds:
            for name, bound_pair in bounds.items():
                getattr(model, name).bounds = bound_pair
        return model

    def _prepare_multifit_configuration(self, model, spatial_shape,
                                        initpars=None, bounds=None,
                                        fixed=None, tied=None):
        template_model = model.copy()
        init_shared = {}
        init_maps = {}
        bounds_shared = {}
        bounds_maps = {}
        parameter_metadata = {}

        if fixed is not None:
            if not isinstance(fixed, dict):
                raise TypeError("fixed must be a dict mapping parameter names to booleans.")
            for name, value in fixed.items():
                self._validate_multifit_param_name(template_model, name, 'fixed')
                if isinstance(value, np.ndarray) or (not np.isscalar(value) and value is not None):
                    raise ValueError(
                        f"fixed for parameter '{name}' must be a shared boolean, not a per-spectrum array."
                    )
                bool_value = bool(value)
                getattr(template_model, name).fixed = bool_value
                parameter_metadata.setdefault(name, {})['fixed'] = bool_value

        if tied is not None:
            if not isinstance(tied, dict):
                raise TypeError("tied must be a dict mapping parameter names to callables or False.")
            for name, value in tied.items():
                self._validate_multifit_param_name(template_model, name, 'tied')
                if isinstance(value, np.ndarray):
                    raise ValueError(
                        f"tied for parameter '{name}' must be a shared callable or False, not a per-spectrum array."
                    )
                if value not in (False, None) and not callable(value):
                    raise TypeError(
                        f"tied for parameter '{name}' must be a callable, False, or None."
                    )
                tied_value = False if value in (False, None) else value
                getattr(template_model, name).tied = tied_value
                parameter_metadata.setdefault(name, {})['tied'] = tied_value

        if initpars is not None:
            if not isinstance(initpars, dict):
                raise TypeError("initpars must be a dict mapping parameter names to values.")
            for name, value in initpars.items():
                self._validate_multifit_param_name(template_model, name, 'initpars')
                shared_value, mapped_value = self._normalize_multifit_value(
                    value, spatial_shape, 'initpars', name
                )
                if mapped_value is None:
                    init_shared[name] = shared_value
                else:
                    init_maps[name] = np.asarray(mapped_value, dtype=float).reshape(-1)

        if bounds is not None:
            if not isinstance(bounds, dict):
                raise TypeError("bounds must be a dict mapping parameter names to (lower, upper) tuples.")
            for name, value in bounds.items():
                self._validate_multifit_param_name(template_model, name, 'bounds')
                (lower_shared, lower_map), (upper_shared, upper_map) = self._normalize_multifit_bounds_pair(
                    value, spatial_shape, name
                )

                if lower_map is None and upper_map is None:
                    shared_pair = (lower_shared, upper_shared)
                    getattr(template_model, name).bounds = shared_pair
                    bounds_shared[name] = shared_pair
                    parameter_metadata.setdefault(name, {})['bounds'] = self._build_multifit_bounds_array(
                        spatial_shape, lower_shared, None, upper_shared, None
                    )
                else:
                    lower_flat = None if lower_map is None else np.asarray(lower_map, dtype=object).reshape(-1)
                    upper_flat = None if upper_map is None else np.asarray(upper_map, dtype=object).reshape(-1)
                    bounds_maps[name] = (lower_shared, lower_flat, upper_shared, upper_flat)
                    parameter_metadata.setdefault(name, {})['bounds'] = self._build_multifit_bounds_array(
                        spatial_shape, lower_shared, lower_map, upper_shared, upper_map
                    )

        for name in template_model.param_names:
            param = getattr(template_model, name)
            meta = parameter_metadata.setdefault(name, {})
            meta.setdefault('fixed', param.fixed)
            meta.setdefault('tied', param.tied)
            meta.setdefault('bounds', param.bounds)
            meta.setdefault('unit', getattr(param, 'unit', None))

        return {
            'template_model': template_model,
            'init_shared': init_shared,
            'init_maps': init_maps,
            'bounds_shared': bounds_shared,
            'bounds_maps': bounds_maps,
            'parameter_metadata': parameter_metadata,
        }

    def _multifit_task_overrides(self, idx, config):
        initpars = {}
        bounds = {}

        for name, value in config['init_shared'].items():
            initpars[name] = value
        for name, values in config['init_maps'].items():
            initpars[name] = values[idx]

        for name, bound_pair in config['bounds_shared'].items():
            bounds[name] = bound_pair
        for name, bound_info in config['bounds_maps'].items():
            lower_shared, lower_flat, upper_shared, upper_flat = bound_info
            lower = lower_shared if lower_flat is None else lower_flat[idx]
            upper = upper_shared if upper_flat is None else upper_flat[idx]
            lower = None if lower is None or (np.isscalar(lower) and np.isnan(lower)) else lower
            upper = None if upper is None or (np.isscalar(upper) and np.isnan(upper)) else upper
            bounds[name] = (lower, upper)

        return initpars, bounds

    def _prepare_fitting(self, model, x, y, yerr=None, statistic='chi2', weights=None, inplace=False):
        """
        Prepare model and data for fitting.
        
        Returns
        -------
        prep_data : dict
            Dictionary with prepared fitting data:
            - model: Model to fit (copied if not inplace)
            - x, y: Data arrays
            - weights: Computed weights
            - init_values: Initial parameter values for free parameters
            - fit_indices: Indices of free parameters
            - param_bounds: Parameter bounds for free parameters
            - tied_info: Cached tied parameter information
            - params_cache: Parameter array cache for fast updates
        """
        if not inplace:
            model = model.copy()

        # Handle non-finite values
        if self.filter_non_finite:
            finite = np.isfinite(y)
            if weights is not None:
                finite &= np.isfinite(np.asarray(weights))
            elif yerr is not None:
                finite &= np.isfinite(np.asarray(yerr))
            x = np.asarray(x)[finite]
            y = y[finite]
            if weights is not None:
                weights = np.asarray(weights)[finite]
            if yerr is not None:
                yerr = np.asarray(yerr)[finite]
        else:
            if not np.all(np.isfinite(y)):
                raise ValueError(
                    "Non-finite values detected in the data. "
                    "Clean the input or set filter_non_finite=True."
                )
            if weights is not None and not np.all(np.isfinite(np.asarray(weights))):
                raise ValueError(
                    "Non-finite values detected in the weights. "
                    "Clean the input or set filter_non_finite=True."
                )
            if yerr is not None and not np.all(np.isfinite(np.asarray(yerr))):
                raise ValueError(
                    "Non-finite values detected in the uncertainties. "
                    "Clean the input or set filter_non_finite=True."
                )

        # Compute weights if not explicitly provided
        if weights is None and yerr is not None:
            weights = self._compute_weights(y, yerr, statistic=statistic)
        
        # Get free parameters and bounds
        init_values, fit_indices, _ = model_to_fit_params(model)
        
        # Extract bounds for free parameters only
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                b[1] if b[1] is not None else np.inf) 
                               for b in bounds_list])
        param_bounds = all_bounds[fit_indices]
        
        # Cache tied parameter information
        tied_info = _get_tied_info(model)
        params_cache = model.parameters.copy()
        
        return {
            'model': model,
            'x': x,
            'y': y,
            'weights': weights,
            'init_values': init_values,
            'fit_indices': fit_indices,
            'param_bounds': param_bounds,
            'tied_info': tied_info,
            'params_cache': params_cache,
        }

    def estimate_covariance(self, model, x, y, weights=None, epsilon=1e-5):
        """
        Estimate covariance matrix using numerical differentiation (Hessian).
        """
        init_values, fit_indices, _ = model_to_fit_params(model)
        n_params = len(init_values)
        
        def func(params):
            full_params = model.parameters.copy()
            full_params[fit_indices] = params
            model.parameters = full_params
            
            if model.has_tied:
                apply_tied(model)
                 
            diff = model(x) - y
            if weights is not None:
                resid = diff * weights
            else:
                resid = diff
            return np.sum(resid**2)
            
        # Calculate Hessian
        hessian = np.zeros((n_params, n_params))
        steps = np.maximum(np.abs(init_values * epsilon), epsilon)
        
        for i in range(n_params):
            for j in range(i, n_params):
                if i == j:
                    p_plus = init_values.copy(); p_plus[i] += steps[i]
                    p_minus = init_values.copy(); p_minus[i] -= steps[i]
                    hessian[i, i] = (func(p_plus) - 2*func(init_values) + func(p_minus)) / (steps[i]**2)
                else:
                    p_pp = init_values.copy(); p_pp[i] += steps[i]; p_pp[j] += steps[j]
                    p_pm = init_values.copy(); p_pm[i] += steps[i]; p_pm[j] -= steps[j]
                    p_mp = init_values.copy(); p_mp[i] -= steps[i]; p_mp[j] += steps[j]
                    p_mm = init_values.copy(); p_mm[i] -= steps[i]; p_mm[j] -= steps[j]
                    hessian[i, j] = (func(p_pp) - func(p_pm) - func(p_mp) + func(p_mm)) / (4 * steps[i] * steps[j])
                    hessian[j, i] = hessian[i, j]
                    
        # Restore parameters
        model.parameters[fit_indices] = init_values
        if model.has_tied:
            apply_tied(model)
        
        try:
            cov = 2.0 * np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("Warning: Hessian is singular.")
            cov = np.full((n_params, n_params), np.nan)
            
        return cov

    @abc.abstractmethod
    def _fit_impl(self, prep_data, **kwargs):
        """
        Implement the specific fitting algorithm.
        
        Parameters  
        ----------
        prep_data : dict
            Prepared fitting data from _prepare_fitting()
        **kwargs : dict
            Additional algorithm-specific parameters
            
        Returns
        -------
        result : dict
            Dictionary with fitting results:
            - fitted_params: Array of fitted parameter values
            - success: Boolean convergence status
            - nfev: Number of function evaluations
            - message: Termination message
            - native_result: Original backend result object (optional)
            - native_cov: Native covariance matrix (optional)
        """
        pass

    def _fit_single_target(self, args):
        """Helper for multiprocess mapping. Unpacks and runs a single 1D fit."""
        idx, model, x, y_1d, yerr_1d, statistic, weights_1d, initpars_1d, bounds_1d, kwargs = args
        try:
            local_model = model.copy()
            self._apply_scalar_multifit_overrides(local_model, initpars=initpars_1d, bounds=bounds_1d)
            fitted = self(model=local_model, x=x, y=y_1d, yerr=yerr_1d,
                          statistic=statistic, weights=weights_1d, inplace=True, nproc=1, **kwargs)
            return (idx, fitted.parameters, self.stdevs, self._multifit_entry_from_fit_info(), self.covariance)
        except Exception as e:
            if self.verbose:
                print(f"Worker {idx} failed: {e}")
            return (
                idx,
                np.full(len(model.parameters), np.nan),
                None,
                {
                    'success': False,
                    'nfev': 0,
                    'message': str(e),
                    **{name: np.nan for name in _MULTIFIT_STAT_KEYS},
                },
                None,
            )

    def _fit_multi(self, model, x, y, yerr=None, statistic='chi2', weights=None,
                   nproc=1, spectral_axis=None, progress=None,
                   initpars=None, bounds=None, fixed=None, tied=None,
                   **kwargs):
        """
        Handle multi-spectra (2D/3D) fitting by iterating over spaxels.
        Expects y shape: (N_axes..., N_wave) or (N_wave, N_axes...).
        We assume the dimension matching len(x) is the wave axis unless spectral_axis is given.
        """
        y = np.asarray(y)
        if y.ndim < 2:
            raise ValueError("Multi-spectrum fitting requires y.ndim >= 2.")
        
        # 1. Identify and move spectral axis to the end
        wave_len = len(x)
        resolved_axis = self._resolve_spectral_axis(y, wave_len, spectral_axis=spectral_axis)
        y = np.moveaxis(y, resolved_axis, -1)
        mapped_yerr = np.moveaxis(np.asarray(yerr), resolved_axis, -1) if yerr is not None else None
        mapped_weights = np.moveaxis(np.asarray(weights), resolved_axis, -1) if weights is not None else None
        spatial_shape = y.shape[:-1]

        multifit_config = self._prepare_multifit_configuration(
            model,
            spatial_shape,
            initpars=initpars,
            bounds=bounds,
            fixed=fixed,
            tied=tied,
        )
        template_model = multifit_config['template_model']

        n_spaxels = np.prod(spatial_shape)
        y_flat = y.reshape((n_spaxels, wave_len))
        yerr_flat = mapped_yerr.reshape((n_spaxels, wave_len)) if mapped_yerr is not None else [None]*n_spaxels
        weights_flat = mapped_weights.reshape((n_spaxels, wave_len)) if mapped_weights is not None else [None]*n_spaxels
        
        # 2. Setup MultiFitResult
        res = MultiFitResult(
            spatial_shape,
            template_model.param_names,
            template_model=template_model,
            x=x,
            has_cov=self.calc_uncertainties,
            spectral_axis=resolved_axis,
            fit_indices=model_to_fit_params(template_model)[1],
            parameter_metadata=multifit_config['parameter_metadata'],
        )
        
        # 3. Create task generator
        def task_generator():
            for idx in range(n_spaxels):
                initpars_1d, bounds_1d = self._multifit_task_overrides(idx, multifit_config)
                yield (
                    idx,
                    template_model,
                    x,
                    y_flat[idx],
                    yerr_flat[idx],
                    statistic,
                    weights_flat[idx],
                    initpars_1d,
                    bounds_1d,
                    kwargs,
                )

        show_progress = self.verbose if progress is None else progress

        if self.verbose:
            print(f"Fitting {n_spaxels} spectra across {nproc} cores...")

        t0 = time.perf_counter()

        # 4. Execute standard or parallel mapping
        if nproc <= 1:
            iterator = task_generator()
            if show_progress:
                iterator = tqdm(iterator, total=n_spaxels, desc="Fitting spectra")
            for task in iterator:
                result = self._fit_single_target(task)
                res.update(*result)
        else:
            with Pool(nproc) as pool:
                iterator = pool.imap_unordered(self._fit_single_target, task_generator(), chunksize=max(1, n_spaxels//(nproc*4)))
                if show_progress:
                    iterator = tqdm(iterator, total=n_spaxels, desc="Fitting spectra")
                for result in iterator:
                    res.update(*result)

        total_time = time.perf_counter() - t0
        self._set_batch_fit_info(res, total_time)

        if self.verbose or show_progress:
            n_converged = self.fit_info['n_success']
            n_failed = self.fit_info['n_failed']
            print(f"Done in {total_time:.1f}s - {n_converged}/{n_spaxels} converged"
                  + (f", {n_failed} failed" if n_failed else "") + ".")

        return res

    def multifit(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None,
                 inplace=False, nproc=1, spectral_axis=None, progress=None,
                 initpars=None, bounds=None, fixed=None, tied=None,
                 **kwargs) -> MultiFitResult:
        """Fit the same model independently across multiple spectra sharing the same x grid."""
        if z is not None:
            raise NotImplementedError("2D fitting not yet implemented")
        return self._fit_multi(model, x, y, yerr=yerr, statistic=statistic,
                               weights=weights, nproc=nproc,
                               spectral_axis=spectral_axis, progress=progress,
                               initpars=initpars, bounds=bounds,
                               fixed=fixed, tied=tied,
                               **kwargs)

    def jointfit(self, *args, **kwargs):
        """Placeholder for future joint optimization on multiple datasets with shared parameters."""
        raise NotImplementedError(
            "Joint fitting is not implemented yet. Use multifit() for independent fits over multiple spectra."
        )

    def __call__(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None,
                 inplace=False, nproc=1, spectral_axis=None, progress=None,
                 initpars=None, bounds=None, fixed=None, tied=None,
                 **kwargs) -> Model:
        """
        Fit model to data.
        
        Parameters
        ----------
        model : astropy.modeling.Model
            Model to fit to the data.
        x : array_like
            Independent variable.
        y : array_like
            Dependent variable.
        z : array_like, optional
            Second independent variable (for 2D models).
        yerr : array_like, optional
            Uncertainties on y values.
        statistic : {'chi2', 'poisson'}
            Statistic for computing weights.
        weights : array_like, optional
            Explicit weights (overrides yerr).
        inplace : bool
            Modify model in-place.
        nproc : int
            Number of worker processes for multi-spectrum fitting.
        spectral_axis : int, optional
            Axis of y matching x for multi-spectrum fitting. Auto-detected when omitted.
        progress : bool, optional
            Show a progress bar during multi-spectrum fitting. Defaults to verbose.
        **kwargs : dict
            Additional arguments passed to the specific fitter implementation.
            
        Returns
        -------
        fitted_model : astropy.modeling.Model
            Fitted model for 1D data. For multiple spectra prefer multifit().
        """
        if z is not None:
            raise NotImplementedError("2D fitting not yet implemented")

        y = np.asarray(y)
        if y.ndim > 1:
            return self.multifit(model, x, y, yerr=yerr, statistic=statistic,
                                 weights=weights, nproc=nproc,
                                 spectral_axis=spectral_axis, progress=progress,
                                 initpars=initpars, bounds=bounds,
                                 fixed=fixed, tied=tied,
                                 **kwargs)

        for name, value in {
            'initpars': initpars,
            'bounds': bounds,
            'fixed': fixed,
            'tied': tied,
        }.items():
            if value is not None:
                raise ValueError(f"{name} is only supported for multifit() and multi-spectrum inputs.")

        # Prepare fitting data using common logic
        t0 = time.perf_counter()
        prep_data = self._prepare_fitting(model, x, y, yerr=yerr, statistic=statistic, 
                                        weights=weights, inplace=inplace)

        # Call specific fitting implementation
        try:
            result = self._fit_impl(prep_data, **kwargs)
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} failed: {str(e)}") from e
            
        fit_time = time.perf_counter() - t0

        # Update model with fitted parameters
        model = prep_data['model']  # Use the prepared model
        fit_indices = prep_data['fit_indices']
        tied_info = prep_data['tied_info']
        params_cache = prep_data['params_cache']
        
        params_cache[fit_indices] = result['fitted_params']
        model.parameters = params_cache
        if tied_info:
            _apply_tied_fast(model, tied_info, params_cache)

        # Store standardized fit_info
        self.fit_info = {
            'time': fit_time,
            'success': result.get('success', False),
            'nfev': result.get('nfev', None),
            'message': result.get('message', ''),
            'result': result.get('native_result'),
            'param_cov': None,
        }

        self.fit_info.update(self._compute_fit_diagnostics(model, prep_data, result, statistic))

        # Preserve native covariance whenever the backend already returned it.
        cov = result.get('native_cov')
        if cov is not None:
            self.fit_info['param_cov'] = cov
        elif self.calc_uncertainties and self.force_numerical_covariance:
            warnings.warn(
                "Falling back to numerical covariance estimation. This is expensive and should only be used explicitly.",
                UserWarning,
                stacklevel=2,
            )
            if self.verbose:
                print("Computing expensive numerical covariance...")
            cov = self.estimate_covariance(model, x, y, weights=prep_data['weights'])
            self.fit_info['param_cov'] = cov
        elif self.calc_uncertainties:
            warnings.warn(
                "Native covariance is unavailable for this fit. No covariance was attached. "
                "Set force_numerical_covariance=True to opt into the expensive numerical fallback.",
                UserWarning,
                stacklevel=2,
            )

        if self.fit_info['param_cov'] is not None:
            self._attach_param_stds(model, self.fit_info['param_cov'], fit_indices)

        # Verbose output
        if self.verbose:
            status = "✓ converged" if self.fit_info['success'] else "✗ FAILED"
            nfev = self.fit_info['nfev'] or '?'
            print(f"{self.__class__.__name__}: {fit_time*1000:.1f}ms, nfev={nfev}, {status}")

        return model