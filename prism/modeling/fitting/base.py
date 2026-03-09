"""
Base fitter class for prism.modeling.fitting

This module provides the FitterBase class that consolidates all common
fitting logic and provides a clean interface for specific fitter implementations.
"""

import abc
import time
import copy
import numpy as np
from multiprocess import Pool
from tqdm import tqdm

from astropy.modeling.fitting import Fitter, model_to_fit_params
from astropy.modeling import CompoundModel

__all__ = ['FitterBase', 'tie', 'apply_tied', '_get_tied_info', '_apply_tied_fast']

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

class MultiFitResult:
    """
    Stores the result of a 2D/3D fitting over multiple spectra.
    Provides easy access to parameter maps and statistics.
    """
    def __init__(self, shape, param_names, has_cov=False):
        self.shape = shape
        self.param_names = param_names
        
        # We store flattened arrays initially and reshape upon access
        self.n_spaxels = np.prod(shape)
        self._params = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._errs = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._success = np.zeros(self.n_spaxels, dtype=bool)
        self._nfev = np.zeros(self.n_spaxels, dtype=float)
        
        self.has_cov = has_cov
        if has_cov:
            self._cov = np.full((self.n_spaxels, len(param_names), len(param_names)), np.nan)

    def update(self, idx, model_params, stdevs, success, nfev, cov=None):
        """Update a single spaxel result at flattened index idx."""
        for i, name in enumerate(self.param_names):
            self._params[name][idx] = model_params[i]
            if stdevs is not None:
                self._errs[name][idx] = stdevs[i]
                
        self._success[idx] = success
        self._nfev[idx] = nfev
        
        if self.has_cov and cov is not None:
            self._cov[idx] = cov

    def __getattr__(self, name):
        """Allow direct access to parameter maps e.g. result.amplitude_0"""
        if name in self.param_names:
            return self._params[name].reshape(self.shape)
        if name.endswith("_err") and name[:-4] in self.param_names:
            return self._errs[name[:-4]].reshape(self.shape)
        raise AttributeError(f"'MultiFitResult' object has no attribute '{name}'")
        
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
    
    def __init__(self, calc_uncertainties=False, force_numerical_covariance=False, verbose=False):
        self.calc_uncertainties = calc_uncertainties
        self.force_numerical_covariance = force_numerical_covariance
        self.verbose = verbose
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
            var = np.diag(cov).copy()
            var[var < 0] = np.nan
            return np.sqrt(var)
        return None

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
        idx, model, x, y_1d, yerr_1d, statistic, weights_1d, kwargs = args
        try:
            # We enforce inplace=False to avoid corrupting shared state (though dill makes a copy anyway)
            fitted = self(model=model.copy(), x=x, y=y_1d, yerr=yerr_1d, 
                          statistic=statistic, weights=weights_1d, inplace=False, nproc=1, **kwargs)
            return (idx, fitted.parameters, self.stdevs, self.fit_info['success'], self.fit_info['nfev'], self.covariance)
        except Exception as e:
            if self.verbose:
                print(f"Worker {idx} failed: {e}")
            return (idx, np.full(len(model.parameters), np.nan), None, False, 0, None)

    def _fit_multi(self, model, x, y, yerr=None, statistic='chi2', weights=None, nproc=1, **kwargs):
        """
        Handle multi-spectra (2D/3D) fitting by iterating over spaxels.
        Expects y shape: (N_axes..., N_wave) or (N_wave, N_axes...).
        We assume the dimension matching len(x) is the wave axis.
        """
        y = np.asarray(y)
        
        # 1. Identify spectral axis
        wave_len = len(x)
        if y.shape[0] == wave_len and y.shape[-1] != wave_len:
            # (N_wave, spatial) -> Transpose to (spatial..., N_wave)
            y = np.moveaxis(y, 0, -1)
            mapped_yerr = np.moveaxis(np.asarray(yerr), 0, -1) if yerr is not None else None
            mapped_weights = np.moveaxis(np.asarray(weights), 0, -1) if weights is not None else None
            spatial_shape = y.shape[:-1]
        elif y.shape[-1] == wave_len:
            # (spatial..., N_wave)
            mapped_yerr = np.asarray(yerr) if yerr is not None else None
            mapped_weights = np.asarray(weights) if weights is not None else None
            spatial_shape = y.shape[:-1]
        else:
            raise ValueError(f"Could not match wave axis length ({wave_len}) to any dimension in y shape {y.shape}")

        n_spaxels = np.prod(spatial_shape)
        y_flat = y.reshape((n_spaxels, wave_len))
        yerr_flat = mapped_yerr.reshape((n_spaxels, wave_len)) if mapped_yerr is not None else [None]*n_spaxels
        weights_flat = mapped_weights.reshape((n_spaxels, wave_len)) if mapped_weights is not None else [None]*n_spaxels
        
        # 2. Setup MultiFitResult
        res = MultiFitResult(spatial_shape, model.param_names, has_cov=self.calc_uncertainties)
        
        # 3. Create task generator
        def task_generator():
            for idx in range(n_spaxels):
                # Skip all-NaN or all-zero spectra if you want, but astropy should fail fast
                yield (idx, model, x, y_flat[idx], yerr_flat[idx], statistic, weights_flat[idx], kwargs)

        if self.verbose:
            print(f"Fitting {n_spaxels} spectra across {nproc} cores...")

        t0 = time.perf_counter()
        
        # 4. Execute standard or parallel mapping
        if nproc <= 1:
            # Single process, nice progress bar
            for task in tqdm(task_generator(), total=n_spaxels, disable=not self.verbose, desc="Fitting Spectra"):
                result = self._fit_single_target(task)
                res.update(*result)
        else:
            # Multi process
            with Pool(nproc) as pool:
                iterator = pool.imap_unordered(self._fit_single_target, task_generator(), chunksize=max(1, n_spaxels//(nproc*4)))
                for result in tqdm(iterator, total=n_spaxels, disable=not self.verbose, desc="Fitting Spectra Par"):
                    res.update(*result)
                    
        total_time = time.perf_counter() - t0
        if self.verbose:
            print(f"Finished multi-fit in {total_time:.2f}s ({total_time/n_spaxels*1000:.1f}ms/spaxel)")

        return res

    def __call__(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None, 
                 inplace=False, nproc=1, **kwargs):
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
        **kwargs : dict
            Additional arguments passed to the specific fitter implementation.
            
        Returns
        -------
        fitted_model : astropy.modeling.Model
            Model with fitted parameters.
        """
        if z is not None:
            raise NotImplementedError("2D fitting not yet implemented")

        y = np.asarray(y)
        if y.ndim > 1:
            return self._fit_multi(model, x, y, yerr=yerr, statistic=statistic, 
                                   weights=weights, nproc=nproc, **kwargs)

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
            'param_cov': None
        }

        # Handle covariance matrix
        if self.calc_uncertainties:
            cov = result.get('native_cov')
            if cov is not None:
                self.fit_info['param_cov'] = cov
            elif self.force_numerical_covariance:
                if self.verbose:
                    print("Computing expensive numerical covariance...")
                cov = self.estimate_covariance(model, x, y, weights=prep_data['weights'])
                self.fit_info['param_cov'] = cov
            else:
                if self.verbose:
                    print("Native covariance unavailable. Set force_numerical_covariance=True for fallback.")

        # Verbose output
        if self.verbose:
            status = "✓ converged" if self.fit_info['success'] else "✗ FAILED"
            nfev = self.fit_info['nfev'] or '?'
            print(f"{self.__class__.__name__}: {fit_time*1000:.1f}ms, nfev={nfev}, {status}")

        return model