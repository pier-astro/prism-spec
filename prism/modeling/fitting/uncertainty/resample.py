"""
Resampling-based uncertainty estimation for FantasyLab.

Provides parametric bootstrap for parameter uncertainty estimation by
refitting synthetic datasets generated from the best-fit model.

Usage
-----
    from prism.modeling.fitting import AstropyTRF
    from fantasylab.uncertainty.resample import bootstrap, attach
    
    # Fit model
    fitter = AstropyTRF()
    fitted = fitter(model, x, y, yerr=yerr)
    
    # Bootstrap uncertainty estimation
    samples = bootstrap(fitted, fitter, x, y, yerr=yerr, n_samples=1000)
    
    # Attach to parameters
    attach(fitted, samples, confidence=68)
    
    # Access uncertainties
    print(f"Amplitude: {fitted.amplitude_0.value} ± {fitted.amplitude_0.std}")
    print(f"68% CI: [{fitted.amplitude_0.lolim}, {fitted.amplitude_0.uplim}]")
    print(f"Median: {fitted.amplitude_0.median}")
"""

import numpy as np
import warnings
import joblib
from tqdm.auto import tqdm
from copy import deepcopy

__all__ = [
    'bootstrap',
    'attach',
    'detach',
    'ResampleError'
]


class ResampleError(Exception):
    """Raised when resampling fails or inputs are invalid."""
    pass


def _bootstrap_worker(model, fitter, x, y_model, yerr, weights, statistic, fitter_kwargs, seed):
    """Worker function for parallel bootstrap."""
    np.random.seed(seed)
    if statistic.lower() == 'gauss':
        noise = np.random.normal(0, yerr)
        y_synth = y_model + noise
    else:
        y_synth = np.random.poisson(np.maximum(y_model, 0))
        
    try:
        model_copy = model.copy()
        fitted_synth = fitter(model_copy, x, y_synth, weights=weights, **fitter_kwargs)
        from astropy.modeling.fitting import model_to_fit_params
        fitted_params, _, _ = model_to_fit_params(fitted_synth)
        return fitted_params
    except Exception:
        return None



def _extract_limits(param, param_samples, lower_percentile, upper_percentile,
                    boundary_epsilon=1e-10, boundary_tolerance=0.5):
    """
    Extract confidence limits with two-stage boundary detection.
    
    Stage 1 (Fast): Check if ANY samples touch boundaries
    Stage 2 (Smart): If triggered, test if distribution is truncated using IQR-relative logic
    
    This avoids false positives where parameter has wide bounds but tight, well-constrained
    distribution (e.g., bounds=(0, 1000) but samples ~0.56±0.005).
    
    Parameters
    ----------
    param : Parameter
        Model parameter with bounds
    param_samples : array
        Bootstrap samples for this parameter
    lower_percentile : float
        Lower percentile for confidence interval
    upper_percentile : float
        Upper percentile for confidence interval
    boundary_epsilon : float
        Stage 1: "at boundary" threshold (default: 1e-10, ~machine precision)
    boundary_tolerance : float
        Stage 2: flag if percentile within (tolerance × IQR) of boundary (default: 0.5)
        
    Returns
    -------
    lolim, uplim : float
        Confidence limits, or np.nan if distribution is truncated at boundary
    """
    # Remove NaN samples
    valid_samples = param_samples[~np.isnan(param_samples)]
    
    if len(valid_samples) == 0:
        return np.nan, np.nan
    
    # Compute percentiles
    lolim = np.percentile(valid_samples, lower_percentile)
    uplim = np.percentile(valid_samples, upper_percentile)
    
    # Check if parameter has bounds
    has_lower_bound = hasattr(param, 'min') and param.min is not None
    has_upper_bound = hasattr(param, 'max') and param.max is not None
    
    # STAGE 1: Quick screening - are ANY samples at boundaries?
    at_lower = 0
    at_upper = 0
    
    if has_lower_bound:
        at_lower = np.sum(valid_samples <= param.min + boundary_epsilon)
    
    if has_upper_bound:
        at_upper = np.sum(valid_samples >= param.max - boundary_epsilon)
    
    # If no samples at boundaries, distribution is clearly interior → done
    if at_lower == 0 and at_upper == 0:
        return lolim, uplim
    
    # STAGE 2: Samples touched boundary → check if it's meaningful truncation
    # Use IQR (interquartile range) to represent distribution spread
    iqr = np.percentile(valid_samples, 75) - np.percentile(valid_samples, 25)
    tolerance = max(boundary_tolerance * iqr, 1e-10)
    
    # Check if confidence limit is within tolerance of boundary
    if has_lower_bound and at_lower > 0:
        distance_to_bound = lolim - param.min
        if distance_to_bound < tolerance:
            lolim = np.nan  # Unconstrained from below
    
    if has_upper_bound and at_upper > 0:
        distance_to_bound = param.max - uplim
        if distance_to_bound < tolerance:
            uplim = np.nan  # Unconstrained from above
    
    return lolim, uplim


def _prepare_noise_and_weights(y, yerr, weights, statistic):
    """
    Normalize yerr/weights inputs and return both arrays.
    - If only weights are provided, derive yerr for noise generation.
    - If only yerr is provided, compute chi-squared style weights.
    - If neither is provided and statistic is Poisson, synthesize yerr/weights.
    """
    stat = statistic.lower()

    if yerr is None:
        if weights is not None:
            weights = np.asarray(weights)
            if np.any(~np.isfinite(weights)):
                raise ResampleError("weights contains non-finite values.")
            safe_weights = np.maximum(weights, 1e-20)
            yerr = np.sqrt(1.0 / safe_weights)
        elif stat == 'poisson':
            base = np.maximum(np.abs(y), 1)
            yerr = np.sqrt(base)
        else:
            raise ResampleError(
                "yerr or weights required for Gaussian bootstrap. "
                "Provide data uncertainties or weights for noise generation."
            )

    yerr = np.asarray(yerr)
    yerr = np.maximum(yerr, 1e-10)
    if np.any(~np.isfinite(yerr)):
        raise ResampleError("yerr contains non-finite values.")

    # Always pass weights to fitter (native Astropy + FitterBases).
    if weights is None:
        if stat == 'poisson':
            weights = 1.0 / np.maximum(np.abs(y), 1e-10)
        else:
            weights = 1.0 / (yerr ** 2)
    else:
        weights = np.asarray(weights)
        if np.any(~np.isfinite(weights)):
            raise ResampleError("weights contains non-finite values.")
        weights = np.maximum(weights, 1e-20)

    return yerr, weights


def bootstrap(model, fitter, x, y, yerr=None, weights=None, n_samples=1000, statistic='gauss',
             fitter_kwargs=None, seed=None, verbose=True, njobs=1):
    """
    Parametric bootstrap uncertainty estimation.
    
    Generates synthetic datasets from best-fit model + noise, refits each,
    and collects parameter distributions.
    
    Algorithm:
    1. For each bootstrap iteration:
       - Generate synthetic data: y_synth = model(x) + noise
       - Noise from Gaussian or Poisson depending on statistic
       - Refit model to synthetic data
       - Store fitted parameters
    2. Return parameter samples
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Best-fit model (starting point for bootstrap)
    fitter : Fitter instance
        Fitter to use for each iteration (e.g., AstropyTRF, ScipyTRF).
        Should be same type used for original fit.
    x : array
        Independent variable
    y : array
        Observed data (not used, just for reference)
    yerr : array, optional
        Data uncertainties for noise generation. Used only to draw noise;
        weights for fitting are derived internally from these values.
    weights : array, optional
        Explicit chi-squared weights to send to the fitter. If provided,
        they override yerr for weighting; yerr is derived from weights for
        noise generation.
    n_samples : int
        Number of bootstrap iterations (default: 1000)
    statistic : str
        Noise model: 'gauss' (Gaussian) or 'poisson' (Poisson).
        Default: 'gauss'
    fitter_kwargs : dict, optional
        Additional kwargs passed to fitter (e.g., {'maxiter': 5000})
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Show progress bar (default: True)
    njobs : int, optional
        Number of parallel jobs to run. Default: 1 (serial).
        If > 1, uses joblib for parallel execution.

    Notes
    -----
    Weights are always passed to the fitter (derived from yerr when needed)
    so both native Astropy fitters and FitterBases receive a consistent
    interface.
        
    Returns
    -------
    samples : dict
        Dictionary mapping parameter names to arrays of shape (n_samples,)
        containing bootstrap samples for each free parameter.
        
    Raises
    ------
    ResampleError
        If yerr is None or inputs are invalid.
        
    Notes
    -----
    Parametric bootstrap assumes:
    - Model is correct
    - Errors follow specified distribution (Gaussian or Poisson)
    - Independent data points
    
    For complex models, consider reducing fitter_kwargs['maxiter'] to speed up
    bootstrap (looser convergence is often acceptable).
    
    Examples
    --------
    >>> from prism.modeling.fitting import AstropyTRF
    >>> from fantasylab.uncertainty.resample import bootstrap, attach
    >>> 
    >>> fitter = AstropyTRF()
    >>> fitted = fitter(model, x, y, yerr=yerr, maxiter=10000)
    >>> 
    >>> # Bootstrap with Gaussian noise
    >>> samples = bootstrap(fitted, fitter, x, y, yerr, n_samples=1000, njobs=4)
    >>> attach(fitted, samples)
    >>> 
    >>> print(f"Amplitude: {fitted.amplitude_0.value:.3f} ± {fitted.amplitude_0.std:.3f}")
    """
    from astropy.modeling.fitting import model_to_fit_params
    
    if statistic.lower() not in ['gauss', 'poisson']:
        raise ResampleError(
            f"Unknown statistic '{statistic}'. Use 'gauss' or 'poisson'."
        )

    # Prepare noise scale (yerr) and fitter weights once up-front
    yerr, weights = _prepare_noise_and_weights(y, yerr, weights, statistic)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Prepare fitter kwargs
    if fitter_kwargs is None:
        fitter_kwargs = {}
    
    # Get model evaluation at best-fit
    y_model = model(x)
    
    # Get parameter names for free parameters only
    _, fit_indices, _ = model_to_fit_params(model)
    
    # Map parameter indices to names
    param_names = []
    cumulative_idx = 0
    for pname in model.param_names:
        param = getattr(model, pname)
        for offset in range(param.size):
            if cumulative_idx in fit_indices:
                if param.size == 1:
                    param_names.append(pname)
                else:
                    param_names.append(f"{pname}[{offset}]")
            cumulative_idx += 1
    
    n_params = len(param_names)
    
    # Initialize storage
    param_samples = {name: np.zeros(n_samples) for name in param_names}
    n_success = 0
    n_failed = 0
    
    # Parallel execution
    if njobs > 1:
        # Generate seeds for reproducibility in parallel
        seeds = np.random.randint(0, 2**32 - 1, size=n_samples)
        
        # Use joblib for parallel execution
        # return_as='generator' allows tqdm to update as tasks complete
        with joblib.Parallel(n_jobs=njobs, return_as='generator') as parallel:
            results_gen = parallel(
                joblib.delayed(_bootstrap_worker)(
                    model, fitter, x, y_model, yerr, weights, statistic, fitter_kwargs, s
                ) for s in seeds
            )
            
            # Collect results with progress bar
            for i, fitted_params in enumerate(tqdm(results_gen, total=n_samples, desc="Bootstrap (Parallel)", disable=not verbose)):
                if fitted_params is not None:
                    for j, (param_name, param_value) in enumerate(zip(param_names, fitted_params)):
                        param_samples[param_name][i] = param_value
                    n_success += 1
                else:
                    # Fit failed - use NaN
                    for param_name in param_names:
                        param_samples[param_name][i] = np.nan
                    n_failed += 1
    else:
        # Serial execution
        pbar = tqdm(range(n_samples), desc="Bootstrap", disable=not verbose)
        
        for i in pbar:
            # Generate synthetic data
            if statistic.lower() == 'gauss':
                # Gaussian noise: y_synth = model(x) + N(0, yerr)
                noise = np.random.normal(0, yerr)
                y_synth = y_model + noise
            else:  # poisson
                # Poisson: y_synth ~ Poisson(model(x))
                # Scale by yerr to match data variance structure
                y_synth = np.random.poisson(np.maximum(y_model, 0))
            
            try:
                # Refit model to synthetic data using explicit weights
                model_copy = model.copy()
                fitted_synth = fitter(model_copy, x, y_synth, weights=weights, **fitter_kwargs)
                
                # Extract fitted parameters
                fitted_params, _, _ = model_to_fit_params(fitted_synth)
                
                # Store parameters
                for j, (param_name, param_value) in enumerate(zip(param_names, fitted_params)):
                    param_samples[param_name][i] = param_value
                
                n_success += 1
                
            except Exception as e:
                # Fit failed - use NaN
                for param_name in param_names:
                    param_samples[param_name][i] = np.nan
                n_failed += 1
                
                if verbose and n_failed <= 5:
                    warnings.warn(
                        f"Bootstrap iteration {i} failed: {e}",
                        RuntimeWarning
                    )
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'success': n_success,
                    'failed': n_failed
                })
        
        pbar.close()
    
    # Check success rate
    success_rate = n_success / n_samples
    if success_rate < 0.5:
        warnings.warn(
            f"Only {success_rate*100:.1f}% of bootstrap iterations succeeded. "
            f"Results may be unreliable. Consider:\n"
            f"  - Loosening fitter convergence (fitter_kwargs={{'maxiter': 1000}})\n"
            f"  - Checking model/data quality\n"
            f"  - Using simpler model",
            RuntimeWarning
        )
    
    if verbose:
        print(f"✓ Bootstrap complete: {n_success}/{n_samples} successful")
    
    return param_samples


def attach(model, samples, confidence=68, percentiles=None, verbose=False, set_values=True):
    """
    Attach bootstrap uncertainties to model parameters.
    
    Attaches the following attributes to each parameter:
    - param.std: Standard deviation from bootstrap samples
    - param.median: 50th percentile (median) from bootstrap samples
    - param.lolim: Lower confidence limit
    - param.uplim: Upper confidence limit  
    - param.percentiles: Dictionary with all percentiles
    - param.samples: Bootstrap samples (if store_samples=True)
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Fitted model whose parameters receive uncertainty info
    samples : dict
        Output from bootstrap(): parameter names -> sample arrays
    confidence : float
        Confidence level for limits (default: 68 for 1-sigma)
        E.g., 68 -> [16th, 84th] percentiles
              95 -> [2.5th, 97.5th] percentiles
    percentiles : list, optional
        Additional percentiles to compute. Default: [16, 50, 84]
    verbose : bool
        Print parameter uncertainties
    set_values : bool, optional
        If True (default), update each parameter value to its bootstrap median.
        
    Returns
    -------
    None
        Modifies model in-place.
        
    Examples
    --------
    >>> samples = bootstrap(fitted, fitter, x, y, yerr, n_samples=1000)
    >>> attach(fitted, samples, confidence=95)
    >>> print(f"{fitted.amplitude_0.value:.3f} [{fitted.amplitude_0.lolim:.3f}, {fitted.amplitude_0.uplim:.3f}]")
    >>> print(f"Median: {fitted.amplitude_0.median:.3f}")
    """
    if percentiles is None:
        percentiles = [16, 50, 84]
    
    # Ensure 50th percentile is always computed for median
    if 50 not in percentiles:
        percentiles = list(percentiles) + [50]
    
    # Compute confidence limits
    alpha = (100 - confidence) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    for param_name, param_samples in samples.items():
        # Handle array-indexed parameters (e.g., "param[0]")
        if '[' in param_name:
            base_name = param_name.split('[')[0]
            idx = int(param_name.split('[')[1].rstrip(']'))
            param = getattr(model, base_name)
            
            # Initialize arrays if needed
            if not hasattr(param, 'std') or param.std is None or np.isscalar(param.std):
                param.std = np.full(param.size, np.nan)
                param.median = np.full(param.size, np.nan)
                param.lolim = np.full(param.size, np.nan)
                param.uplim = np.full(param.size, np.nan)
                param.percentiles = {}
            
            # Remove NaN values
            valid_samples = param_samples[~np.isnan(param_samples)]
            
            if len(valid_samples) > 0:
                param.std[idx] = np.std(valid_samples)
                param.median[idx] = np.percentile(valid_samples, 50)
                
                # Extract limits with boundary detection
                lolim, uplim = _extract_limits(param, param_samples, 
                                               lower_percentile, upper_percentile)
                param.lolim[idx] = lolim
                param.uplim[idx] = uplim
                
                # Store percentiles
                for p in percentiles:
                    if p not in param.percentiles:
                        param.percentiles[p] = np.full(param.size, np.nan)
                    param.percentiles[p][idx] = np.percentile(valid_samples, p)

                if set_values and np.isfinite(param.median[idx]):
                    updated = np.array(param.value, copy=True)
                    updated.flat[idx] = param.median[idx]
                    param.value = updated
        else:
            # Scalar parameter
            param = getattr(model, param_name)
            
            # Remove NaN values
            valid_samples = param_samples[~np.isnan(param_samples)]
            
            if len(valid_samples) > 0:
                param.std = np.std(valid_samples)
                param.median = np.percentile(valid_samples, 50)
                
                # Extract limits with boundary detection
                lolim, uplim = _extract_limits(param, param_samples,
                                               lower_percentile, upper_percentile)
                param.lolim = lolim
                param.uplim = uplim
                
                # Store percentiles
                param.percentiles = {
                    p: np.percentile(valid_samples, p) for p in percentiles
                }

                if set_values and np.isfinite(param.median):
                    param.value = param.median
                
                if verbose:
                    print(f"  {param_name}: {param.value:.6g} ± {param.std:.6g} "
                          f"(median: {param.median:.6g}) "
                          f"[{param.lolim:.6g}, {param.uplim:.6g}]")
    
    if verbose:
        print(f"✓ Attached uncertainties with {confidence}% confidence intervals")


def detach(model):
    """
    Remove bootstrap uncertainty attributes from model parameters.
    
    Removes .std, .median, .lolim, .uplim, .percentiles, .samples attributes.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to clean up
        
    Returns
    -------
    None
        Modifies model in-place.
    """
    for param_name in model.param_names:
        param = getattr(model, param_name)
        
        for attr in ['std', 'median', 'lolim', 'uplim', 'percentiles', 'samples']:
            if hasattr(param, attr):
                try:
                    delattr(param, attr)
                except (AttributeError, TypeError):
                    # If it's a property or can't be deleted, set to None
                    setattr(param, attr, None)
