"""
Astropy-based fitters for prism.modeling.fitting

This module provides wrappers around native Astropy fitters with
standardized interfaces and improved error handling.
"""

import warnings
from astropy.modeling import fitting as astropy_fitting
from .base import FitterBase

__all__ = ['AstroFitter', 'AstroTRF', 'AstroDogBox', 'AstroLM', 'AstroSimplex']


class AstroFitter(FitterBase):
    """
    Base wrapper around native Astropy fitters.
    
    Provides standardized fit_info structure and covariance storage using
    Astropy's native optimization algorithms with clean error handling.
    
    **Supported methods:**
    - 'TRF': Trust Region Reflective (supports bounds & ties)
    - 'DogBox': Dogleg with rectangular trust regions (supports bounds & ties)
    - 'LevMar': Levenberg-Marquardt (**NO BOUNDS OR TIES SUPPORT**)
    - 'Simplex': Nelder-Mead simplex (**NO BOUNDS OR TIES SUPPORT**)
    
    **Covariance Extraction:**
    Uses native covariance from Astropy's `param_cov` (computed analytically 
    during optimization). More reliable than numerical estimation.
    
    Parameters
    ----------
    method : str
        'TRF' (default), 'DogBox', 'LevMar', or 'Simplex'.
    calc_uncertainties : bool
        Calculate covariance matrix using native Astropy method.
    verbose : bool
        Print diagnostic information.
    """
    
    _methods = {
        'trf': astropy_fitting.TRFLSQFitter,
        'dogbox': astropy_fitting.DogBoxLSQFitter,
        'levmar': astropy_fitting.LevMarLSQFitter,
        'simplex': astropy_fitting.SimplexLSQFitter,
    }
    
    _no_bounds_methods = {'levmar', 'simplex'}
    _no_uncertainties_methods = {'simplex'}
    
    def __init__(self, method='TRF', calc_uncertainties=False, force_numerical_covariance=False,
                 verbose=False, filter_non_finite=False):
        super().__init__(calc_uncertainties=calc_uncertainties,
                        force_numerical_covariance=force_numerical_covariance,
                        verbose=verbose,
                        filter_non_finite=filter_non_finite)
        
        key = method.lower()
        if key not in self._methods:
            raise ValueError(f"Unknown method: {method}. Options: {list(self._methods.keys())}")
        
        # Warn if calc_uncertainties requested for methods that don't support it
        if calc_uncertainties and key in self._no_uncertainties_methods:
            warnings.warn(
                f"{method.upper()} does not support native uncertainty calculation. "
                f"Set calc_uncertainties=False or use a different fitter.",
                UserWarning
            )
            
        self.method = method
        self._fitter_class = self._methods[key]

    def _check_bounds_and_ties(self, model):
        """Check for bounds/ties compatibility with the selected method."""
        method_key = self.method.lower()
        if method_key not in self._no_bounds_methods:
            return  # Method supports bounds and ties
            
        # Check if model has bounds
        has_bounds = any(
            getattr(model, pname).bounds != (None, None)
            for pname in model.param_names
        )
        
        # Check if model has tied parameters
        has_tied = any(
            getattr(model, pname).tied is not False
            for pname in model.param_names
        )
        
        if has_bounds:
            warnings.warn(
                f"\\n{'='*70}\\n"
                f"⚠️  WARNING: {self.method.upper()} DOES NOT SUPPORT PARAMETER BOUNDS!\\n"
                f"    Your model has bounded parameters, but {self.method} will IGNORE them.\\n"
                f"    Use AstroTRF or AstroDogBox for bounds support.\\n"
                f"{'='*70}",
                UserWarning, stacklevel=3
            )
        
        if has_tied:
            warnings.warn(
                f"\\n{'='*70}\\n"
                f"⚠️  WARNING: {self.method.upper()} DOES NOT SUPPORT TIED PARAMETERS!\\n"
                f"    Your model has tied parameters, but {self.method} will IGNORE them.\\n"
                f"    Use AstroTRF or AstroDogBox for tied parameter support.\\n"
                f"{'='*70}",
                UserWarning, stacklevel=3
            )

    def _fit_impl(self, prep_data, estimate_jacobian=None, **kwargs):
        """
        Implement Astropy fitting using the prepared data.
        
        Parameters
        ----------
        prep_data : dict
            Prepared fitting data from _prepare_fitting()
        estimate_jacobian : bool, optional  
            Use numerical Jacobian. If None, auto-detect.
        **kwargs : dict
            Additional arguments to pass to Astropy fitter
            
        Returns
        -------
        result : dict
            Fitting results with standardized keys
        """
        model = prep_data['model']
        x = prep_data['x']
        y = prep_data['y']
        weights = prep_data['weights']
        
        # Check bounds/ties compatibility
        self._check_bounds_and_ties(model)
        
        # Create Astropy fitter
        method_key = self.method.lower()
        if method_key in self._no_uncertainties_methods:
            fitter = self._fitter_class()
        else:
            fitter = self._fitter_class(calc_uncertainties=self.calc_uncertainties)
        
        # Auto-detect jacobian mode if not specified
        use_numeric = estimate_jacobian if estimate_jacobian is not None else False
        
        # Handle method-specific parameters
        fit_kwargs = kwargs.copy()
        
        # Increase default iterations/evaluations to 1000 for spectral fitting
        # (Astropy default is 100, which is often too low)
        if 'maxiter' not in fit_kwargs:
            fit_kwargs['maxiter'] = 1000
            
        if method_key == 'simplex':
            # Simplex doesn't support estimate_jacobian parameter
            fit_kwargs.pop('estimate_jacobian', None)
        else:
            fit_kwargs['estimate_jacobian'] = use_numeric
        
        try:
            fitted = fitter(model, x, y, weights=weights, **fit_kwargs)
        except ValueError as e:
            # Handle fit_deriv shape mismatch for compound models
            if "broadcast" in str(e) and not use_numeric:
                if self.verbose:
                    print(f"Note: fit_deriv shape mismatch, using numeric derivatives")
                # Retry with numeric jacobian
                fit_kwargs['estimate_jacobian'] = True
                fitted = fitter(model, x, y, weights=weights, **fit_kwargs)
            else:
                raise
        
        # Extract results
        success = fitter.fit_info.get('success', True)
        nfev = fitter.fit_info.get('nfev', None)
        message = fitter.fit_info.get('message', '')
        native_cov = fitter.fit_info.get('param_cov')
        
        return {
            'fitted_params': fitted.parameters[prep_data['fit_indices']],
            'success': success,
            'nfev': nfev,
            'message': message,
            'native_result': fitter.fit_info,
            'native_cov': native_cov
        }


class AstroTRF(AstroFitter):
    """
    Trust Region Reflective via Astropy (supports bounds & ties).
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 1000)
    - acc : float - Accuracy for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    Note: Default maxiter has been increased from Astropy's 100 to 1000.
    Increase further for complex multi-component fits: maxiter=10000
    
    Example:
        fitter = AstroTRF(calc_uncertainties=True, verbose=True)
        result = fitter(model, x, y, maxiter=10000, acc=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='TRF', **kwargs)


class AstroDogBox(AstroFitter):
    """
    Dogleg via Astropy (supports bounds & ties).
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 1000)
    - acc : float - Accuracy for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    Note: Default maxiter has been increased from Astropy's 100 to 1000.
    For complex models, consider increasing further: maxiter=5000
    
    Example:
        fitter = AstroDogBox(calc_uncertainties=True)
        result = fitter(model, x, y, maxiter=5000)
    """
    def __init__(self, **kwargs):
        super().__init__(method='DogBox', **kwargs)


class AstroLM(AstroFitter):
    """
    Levenberg-Marquardt via Astropy.
    
    ⚠️  **WARNING: DOES NOT SUPPORT PARAMETER BOUNDS OR TIED PARAMETERS!**
    
    This is the classic LM algorithm without bounds support. If your model has
    bounded or tied parameters, they will be IGNORED. Use AstroTRF or 
    AstroDogBox for bounds/ties support.
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 1000)
    - acc : float - Relative error for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    Note: Default maxiter has been increased from Astropy's 100 to 1000.
    
    Example:
        fitter = AstroLM(calc_uncertainties=True)
        result = fitter(model, x, y, maxiter=5000)
    """
    def __init__(self, **kwargs):
        super().__init__(method='LevMar', **kwargs)


class AstroSimplex(AstroFitter):
    """
    Nelder-Mead Simplex via Astropy (derivative-free).
    
    ⚠️  **WARNING: DOES NOT SUPPORT PARAMETER BOUNDS OR TIED PARAMETERS!**
    
    This is a derivative-free simplex algorithm. If your model has bounded or 
    tied parameters, they will be IGNORED. Use AstroTRF or AstroDogBox for 
    bounds/ties support.
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 1000)
    - acc : float - Convergence tolerance (default: 1e-7)
    - Note: Does NOT support estimate_jacobian (derivative-free method)
    
    Note: Default maxiter has been increased from Astropy's 100 to 1000.
    
    Example:
        fitter = AstroSimplex(verbose=True)
        result = fitter(model, x, y, maxiter=5000, acc=1e-6)
    """
    def __init__(self, **kwargs):
        super().__init__(method='Simplex', **kwargs)