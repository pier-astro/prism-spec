
import abc
import numpy as np
from astropy.modeling.fitting import Fitter
from astropy.modeling import CompoundModel

__all__ = ['FantasyFitter', 'tie', 'apply_tied', '_get_tied_info', '_apply_tied_fast']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def tie(submodel_name, func):
    """
    Create a tie function for parameter constraints in compound models.
    
    This is the recommended way to tie parameters in FantasyLab models.
    
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
                # Fallback: try to find submodel by traversing
                # This handles cases where the model structure might be nested differently
                # or if we are operating on the submodel directly (though less likely for this function)
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
    """
    Apply all tied parameter constraints to a model.
    
    Updates parameter values based on their tie functions. Useful when you
    want model(x) to reflect tied constraints without fitting.
    
    Parameters
    ----------
    model : Model
        The model to update.
        
    Returns
    -------
    model : Model
        Same model with updated values.
    """
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
    """
    Extract and cache tied parameter information for fast evaluation.
    Returns a list of (tied_func, slice) tuples for tied parameters only.
    """
    tied_info = []
    if not model.has_tied:
        return tied_info
        
    param_metrics = model._param_metrics
    tied_dict = model.tied  # Access once
    
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

class FantasyFitter(Fitter):
    """
    Base class for FantasyLab fitters.
    
    Provides:
    - Consistent interface across all fitter types
    - fit_info dictionary with scipy.optimize.OptimizeResult-like structure
    - Covariance matrix computation and storage in fit_info['param_cov']
    - Native covariance extraction with optional numerical fallback
    
    Note: Parameter uncertainty attachment (.std attributes) is handled by
    the fantasylab.uncertainty.covariance module for clean separation.
    
    Parameters
    ----------
    calc_uncertainties : bool
        Whether to compute covariance matrix. Results stored in fit_info['param_cov'].
    force_numerical_covariance : bool
        If True, compute expensive numerical Hessian when native fails.
        WARNING: O(n²) function evaluations. Default is False.
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
        """
        Standard deviations of fitted parameters.
        
        Note: For parameter attachment, use fantasylab.uncertainty.covariance module:
            from fantasylab.uncertainty.covariance import attach
            attach(model, fitter)
        """
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
            # Chi-squared: weights = 1 / sigma^2
            weights = 1.0 / (yerr ** 2)
        elif statistic.lower() == 'poisson':
            # Poisson: weights = 1 / y (variance = mean for Poisson)
            # Use yerr as Poisson lambda parameter if it represents counts
            y_arr = np.asarray(y)
            # For Poisson, variance = mean, so sigma = sqrt(mean)
            # weights = 1/sigma^2 = 1/mean
            weights = 1.0 / np.maximum(y_arr, 1e-10)
        else:
            if self.verbose:
                print(f"Warning: Unknown statistic '{statistic}'. Using uniform weights.")
            return None
            
        return weights

    def estimate_covariance(self, model, x, y, weights=None, epsilon=1e-5):
        """
        Estimate covariance matrix using numerical differentiation (Hessian).
        Returns the covariance matrix of the fitted parameters.
        """
        from astropy.modeling.fitting import model_to_fit_params
        
        # Get current parameters
        _, fit_param_indices, _ = model_to_fit_params(model)
        p0 = model.parameters[fit_param_indices]
        n_params = len(p0)
        
        # Define function to minimize (Chi2)
        def func(params):
            # Correctly update model parameters
            full_params = model.parameters.copy()
            full_params[fit_param_indices] = params
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
        steps = np.maximum(np.abs(p0 * epsilon), epsilon)
        
        for i in range(n_params):
            for j in range(i, n_params):
                if i == j:
                    p_plus = p0.copy(); p_plus[i] += steps[i]
                    p_minus = p0.copy(); p_minus[i] -= steps[i]
                    hessian[i, i] = (func(p_plus) - 2*func(p0) + func(p_minus)) / (steps[i]**2)
                else:
                    p_pp = p0.copy(); p_pp[i] += steps[i]; p_pp[j] += steps[j]
                    p_pm = p0.copy(); p_pm[i] += steps[i]; p_pm[j] -= steps[j]
                    p_mp = p0.copy(); p_mp[i] -= steps[i]; p_mp[j] += steps[j]
                    p_mm = p0.copy(); p_mm[i] -= steps[i]; p_mm[j] -= steps[j]
                    hessian[i, j] = (func(p_pp) - func(p_pm) - func(p_mp) + func(p_mm)) / (4 * steps[i] * steps[j])
                    hessian[j, i] = hessian[i, j]
                    
        # Restore parameters
        full_params = model.parameters.copy()
        full_params[fit_param_indices] = p0
        model.parameters = full_params
        
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
    def __call__(self, model, x, y, z=None, weights=None, **kwargs):
        """Fit model to data."""
        pass
