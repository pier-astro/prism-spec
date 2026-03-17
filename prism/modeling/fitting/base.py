"""
Base fitter class for prism.modeling.fitting

This module provides the FitterBase class that consolidates all common
fitting logic and provides a clean interface for specific fitter implementations.
MultiFit functionality (MultiFitMixin, MultiFitResult, SpectrumFitResult,
MultiFitParameter) lives in the companion `multifit` module.
"""

import abc
import time
import warnings
import numpy as np

from astropy.modeling.fitting import Fitter, model_to_fit_params
from astropy.modeling import CompoundModel, Model

from .multifit import (MultiFitMixin, MultiFitResult, SpectrumFitResult,
                       MultiFitParameter, _MULTIFIT_STAT_KEYS)

__all__ = [
    'FitterBase',
    'tie', 'apply_tied', '_get_tied_info', '_apply_tied_fast',
    'MultiFitMixin', 'MultiFitResult', 'SpectrumFitResult', 'MultiFitParameter',
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

class FitterBase(Fitter, MultiFitMixin):
    """
    Base class for all prism fitters.

    Provides:
    - Consistent interface across all fitter types
    - Standardized fit_info dictionary with scipy.optimize.OptimizeResult-like structure
    - Common parameter handling (bounds, ties, weights)
    - Covariance matrix computation and storage
    - Clean separation between generic logic and specific algorithms
    - Batched multi-spectrum fitting via MultiFitMixin (multifit() / fitter(model, x, cube))

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
    filter_non_finite : bool
        Silently discard non-finite data points before fitting.
    """

    def __init__(self, calc_uncertainties=False, force_numerical_covariance=False,
                 verbose=False, filter_non_finite=False):
        self.calc_uncertainties = calc_uncertainties
        self.force_numerical_covariance = force_numerical_covariance
        self.verbose = verbose
        self.filter_non_finite = filter_non_finite
        self.fit_info = {}

    def get_unit(self, param_name):
        """Get the unit associated with a parameter, falling back to None."""
        # This method assumes self._template_model is set by the fitter's __call__ method
        # or similar, representing the model being fitted.
        if not hasattr(self, '_template_model') or self._template_model is None:
            warnings.warn("'_template_model' not set on fitter. Cannot resolve parameter unit.",
                          RuntimeWarning)
            return None

        model = self._template_model
        param = getattr(model, param_name)

        # First, try the direct param.unit attribute
        if hasattr(param, 'unit') and param.unit is not None:
            return param.unit

        # Duck-typing lookup to bypass astropy strict initialization or for custom models
        sub_model = model
        # Check if param_name belongs to a sub-model in a compound model
        # This logic assumes param._model_name exists for submodel parameters
        if hasattr(param, '_model_name') and hasattr(model, param._model_name):
            sub_model = getattr(model, param._model_name)

        # It might be an array parameter (e.g., in a LineGroup)
        # Extract the base name (e.g., 'amplitude' from 'amplitude_0' or 'amplitude[0]')
        base_name = param_name.split('_')[0] if '_' in param_name else param_name
        base_name = base_name.split('[')[0]

        # Check _parameter_units dictionary (common in custom models)
        if hasattr(sub_model, '_parameter_units'):
            if base_name in sub_model._parameter_units:
                return sub_model._parameter_units[base_name]

        # Check _shared_units dictionary (e.g., for LineGroup components)
        if hasattr(sub_model, '_shared_units'):
            if base_name in sub_model._shared_units:
                return sub_model._shared_units[base_name]

        return None

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

    def __call__(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None,
                 inplace=False, nproc=1, spectral_axis=None, progress=None,
                 initpars=None, bounds=None, fixed=None, tied=None,
                 **kwargs) -> Model:
        """
        Fit model to data.

        For 1-D ``y``, performs a single-spectrum fit and returns the fitted model.
        For multi-dimensional ``y``, automatically dispatches to ``multifit()`` and
        returns a ``MultiFitResult``.

        Parameters
        ----------
        model : astropy.modeling.Model
            Model to fit to the data.
        x : array_like
            Independent variable.
        y : array_like
            Dependent variable. If ndim > 1, triggers batch fitting.
        yerr : array_like, optional
            Uncertainties on y values.
        statistic : {'chi2', 'poisson'}
            Statistic for computing weights from yerr.
        weights : array_like, optional
            Explicit weights (overrides yerr).
        inplace : bool
            Modify model in-place (single-spectrum only). Ignored in
            multi-spectrum mode.
        nproc : int
            Number of worker processes for multi-spectrum fitting.
        spectral_axis : int, optional
            Axis of y matching x. Auto-detected when omitted.
        progress : bool, optional
            Show a progress bar during batch fitting. Defaults to verbose.
        initpars : dict, optional
            Per-parameter initial value overrides (batch fitting only).
        bounds : dict, optional
            Per-parameter bound overrides (batch fitting only).
        fixed : dict, optional
            Per-parameter fixed-state overrides (batch fitting only).
        tied : dict, optional
            Per-parameter tie-function overrides (batch fitting only).
        **kwargs
            Additional arguments passed to the specific fitter implementation.

        Returns
        -------
        fitted_model : astropy.modeling.Model or MultiFitResult
        """
        if z is not None:
            raise NotImplementedError("2D fitting not yet implemented")

        y = np.asarray(y)
        if y.ndim > 1:
            if inplace:
                warnings.warn(
                    "'inplace=True' has no effect in multifit mode: each spectrum "
                    "is fit on an independent model copy.",
                    UserWarning,
                    stacklevel=2,
                )
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
        model = prep_data['model']
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

        # Ensure direct fitted_model.save()/to_fits() methods are installed
        # at class level (pickle-safe for multiprocessing workflows).
        from ..io import install_model_io_methods
        install_model_io_methods()

        return model