"""
Covariance-based uncertainty estimation for FantasyLab fitters.

This module provides utilities to extract parameter uncertainties from covariance
matrices computed during fitting. It works with:
- All FantasyLab fitters (AstropyTRF, ScipyTRF, SherpaLevMar, etc.)
- Native Astropy fitters (TRFLSQFitter, LevMarLSQFitter, etc.)

Uses Astropy's standard naming: covariance stored in fit_info['param_cov'].

Usage
-----
With FantasyLab fitters:

    from fantasylab.fitting import AstropyTRF
    from fantasylab.uncertainty.covariance import attach
    
    fitter = AstropyTRF(calc_uncertainties=True)
    fitted_model = fitter(model, x, y)
    attach(fitted_model, fitter)
    
    print(fitted_model.amplitude_0.std)

With native Astropy fitters:

    from astropy.modeling import fitting
    from fantasylab.uncertainty.covariance import attach
    
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    fitted_model = fitter(model, x, y, weights=weights)
    attach(fitted_model, fitter)
    
    print(fitted_model.amplitude_0.std)

Or use lower-level functions:

    from fantasylab.uncertainty import covariance
    
    cov_matrix = covariance.covar(fitter)
    std = covariance.stdevs(cov_matrix)
    corr = covariance.correlation(cov_matrix)
"""

import numpy as np
import warnings

__all__ = [
    'attach',
    'detach',
    'covar',
    'stdevs',
    'correlation',
    'CovarianceError'
]


class CovarianceError(Exception):
    """Raised when covariance matrix is not available in fit_info."""
    pass


def covar(fitter):
    """
    Extract covariance matrix from fitter's fit_info.
    
    Parameters
    ----------
    fitter : FantasyFitter or astropy fitter
        Fitter instance with fit_info dictionary populated after fitting.
        
    Returns
    -------
    covariance : ndarray
        Covariance matrix of free parameters (n_free_params, n_free_params).
        
    Raises
    ------
    CovarianceError
        If covariance matrix is not present in fit_info.
        
    Notes
    -----
    Extracts covariance from fit_info['param_cov'], following Astropy's standard.
    Compatible with both FantasyLab fitters and native Astropy fitters.
    """
    if not hasattr(fitter, 'fit_info'):
        raise CovarianceError(
            "Fitter does not have fit_info. Did you run the fit?"
        )
    
    cov = fitter.fit_info.get('param_cov', None)
    
    if cov is None:
        raise CovarianceError(
            "Covariance matrix not found in fit_info. "
            "Please fit with calc_uncertainties=True:\n"
            "    fitter = YourFitter(calc_uncertainties=True)\n"
            "    fitted_model = fitter(model, x, y)"
        )
    
    return cov


def stdevs(covariance_matrix):
    """
    Compute standard deviations from covariance matrix.
    
    Parameters
    ----------
    covariance_matrix : ndarray
        Covariance matrix (n_params, n_params).
        
    Returns
    -------
    stdevs : ndarray
        Standard deviations (sqrt of diagonal elements).
        NaN for negative variances (shouldn't happen but handled gracefully).
        
    Notes
    -----
    This computes sqrt(diag(covariance_matrix)). Negative diagonal elements
    (which indicate numerical issues) are set to NaN with a warning.
    """
    if covariance_matrix is None:
        return None
    
    variances = np.diag(covariance_matrix).copy()
    
    # Check for negative variances (numerical issues)
    if np.any(variances < 0):
        n_negative = np.sum(variances < 0)
        warnings.warn(
            f"Covariance matrix has {n_negative} negative variance(s). "
            f"This indicates numerical issues. Setting these to NaN.",
            RuntimeWarning
        )
        variances[variances < 0] = np.nan
    
    return np.sqrt(variances)


def correlation(covariance_matrix):
    """
    Compute correlation matrix from covariance matrix.
    
    Parameters
    ----------
    covariance_matrix : ndarray
        Covariance matrix (n_params, n_params).
        
    Returns
    -------
    correlation : ndarray
        Correlation matrix (n_params, n_params).
        Elements are in range [-1, 1].
        
    Notes
    -----
    Correlation[i, j] = Covariance[i, j] / (std[i] * std[j])
    """
    if covariance_matrix is None:
        return None
    
    std = stdevs(covariance_matrix)
    
    # Avoid division by zero
    std_safe = np.where(std > 0, std, np.nan)
    
    # Correlation = cov / (std_i * std_j)
    corr = covariance_matrix / np.outer(std_safe, std_safe)
    
    return corr


def attach(model, fitter, verbose=False):
    """
    Attach parameter uncertainties from fitter's covariance to model parameters.
    
    This function extracts the covariance matrix from the fitter's fit_info,
    computes standard deviations, and attaches them as .std attributes to
    the model's parameters.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Fitted model whose parameters will receive .std attributes.
    fitter : FantasyFitter
        Fitter instance with fit_info containing covariance matrix.
    verbose : bool, optional
        Print diagnostic information. Default is False.
        
    Raises
    ------
    CovarianceError
        If covariance matrix is not available in fitter.fit_info.
        
    Returns
    -------
    None
        Modifies model in-place by attaching .std to parameters.
        
    Notes
    -----
    After calling this function, parameters will have a .std attribute:
    
        attach(fitted_model, fitter)
        print(fitted_model.amplitude_0.std)  # Standard deviation
        
    The function handles both scalar and array-valued parameters correctly.
    Only free (fitted) parameters receive uncertainties; fixed parameters
    will not have .std attributes or will have existing ones unchanged.
    
    Examples
    --------
    >>> fitter = AstropyTRF(calc_uncertainties=True)
    >>> fitted = fitter(model, x, y)
    >>> attach(fitted, fitter, verbose=True)
    >>> print(f"Amplitude: {fitted.amplitude_0.value:.3f} ± {fitted.amplitude_0.std:.3f}")
    """
    from astropy.modeling.fitting import model_to_fit_params
    
    # Extract covariance matrix
    try:
        cov = covar(fitter)
    except CovarianceError as e:
        raise e
    
    # Compute standard deviations
    std = stdevs(cov)
    
    if std is None:
        raise CovarianceError("Could not compute standard deviations from covariance.")
    
    # Get indices of free parameters
    _, fit_indices, _ = model_to_fit_params(model)
    
    if len(std) != len(fit_indices):
        warnings.warn(
            f"Mismatch: covariance has {len(std)} parameters, "
            f"but model has {len(fit_indices)} free parameters. "
            f"Uncertainties may be incorrect.",
            RuntimeWarning
        )
        return
    
    # Store covariance info in model (following fit_info naming convention)
    model._param_cov = cov
    model._fit_indices = fit_indices
    
    # Create a map: global_param_index -> (param_name, param_object, offset_within_param)
    idx_map = {}
    cumulative_idx = 0
    for param_name in model.param_names:
        param = getattr(model, param_name)
        param_size = param.size
        for offset in range(param_size):
            idx_map[cumulative_idx + offset] = (param_name, param, offset)
        cumulative_idx += param_size
    
    # Attach uncertainties to parameters
    attached_count = 0
    for i, s in zip(fit_indices, std):
        if i not in idx_map:
            if verbose:
                warnings.warn(f"Parameter index {i} not found in model. Skipping.", RuntimeWarning)
            continue
        
        param_name, param, offset = idx_map[i]
        
        # For scalar parameters
        if param.size == 1:
            param.std = s
            attached_count += 1
            if verbose:
                print(f"  {param_name}: {param.value:.6g} ± {s:.6g}")
        
        # For array-valued parameters
        else:
            # Initialize .std array if needed
            if not hasattr(param, 'std') or param.std is None or np.isscalar(param.std):
                param.std = np.full(param.size, np.nan)
            
            param.std[offset] = s
            attached_count += 1
            if verbose:
                print(f"  {param_name}[{offset}]: {param.value[offset]:.6g} ± {s:.6g}")
    
    if verbose:
        print(f"✓ Attached uncertainties to {attached_count} parameter(s)")


def detach(model):
    """
    Remove .std attributes and covariance info from model parameters.
    
    This is a utility function to clean up uncertainty information,
    useful when refitting or comparing different fits.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model whose parameter uncertainties will be removed.
        
    Returns
    -------
    None
        Modifies model in-place.
        
    Notes
    -----
    Removes:
    - .std attributes from parameters
    - ._param_cov (covariance matrix)
    - ._fit_indices (free parameter indices)
    
    Sets .std to None rather than deleting it, since astropy Parameter
    objects may have .std as a property that cannot be deleted.
    """
    for param_name in model.param_names:
        param = getattr(model, param_name)
        if hasattr(param, 'std'):
            try:
                delattr(param, 'std')
            except (AttributeError, TypeError):
                # If std is a property or can't be deleted, set to None
                param.std = None
    
    # Remove stored covariance info
    if hasattr(model, '_param_cov'):
        delattr(model, '_param_cov')
    if hasattr(model, '_fit_indices'):
        delattr(model, '_fit_indices')
