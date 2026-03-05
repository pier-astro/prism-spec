
import time
import warnings
from astropy.modeling import fitting as astropy_fitting
from .fitting import FantasyFitter

__all__ = ['AstropyFitter', 'AstropyTRF', 'AstropyDogBox', 'AstropyLevMar', 'AstropySimplex']

class AstropyFitter(FantasyFitter):
    """
    Wrapper around native Astropy fitters.
    
    This wrapper provides standardized fit_info structure and covariance storage:
    - fitter.fit_info: Complete fit information (Astropy's structure)
    - fitter.covariance: Covariance matrix (from Astropy's param_cov)
    - fitter.stdevs: Standard deviations (convenience property)
    
    **Parameter Uncertainties:**
    To attach uncertainties to model parameters, use the uncertainty module:
        from fantasylab.uncertainty.covariance import attach
        attach(model, fitter)
    
    **Native Astropy Structure:**
    All fit information is stored in `fitter.fit_info`, matching Astropy's structure:
    - x: Fitted parameters
    - success: Convergence status  
    - nfev: Function evaluations
    - message: Termination message
    - param_cov: Covariance matrix (if calc_uncertainties=True)
    - (method-specific keys: cost, jac, grad, etc.)
    
    **Covariance Extraction:**
    Uses native covariance from Astropy's `param_cov` (computed analytically during
    optimization, not numerically estimated). If unavailable, raises warning.
    
    **Supported methods:**
    - 'TRF': Trust Region Reflective (supports bounds & ties)
    - 'DogBox': Dogleg with rectangular trust regions (supports bounds & ties)
    - 'LevMar': Levenberg-Marquardt (**NO BOUNDS OR TIES SUPPORT**)
    - 'Simplex': Nelder-Mead simplex (**NO BOUNDS OR TIES SUPPORT**)
    
    Parameters
    ----------
    method : str
        'TRF' (default), 'DogBox', 'LevMar', or 'Simplex'.
    calc_uncertainties : bool
        Calculate covariance matrix using native Astropy method.
    maxiter : int, optional
        Maximum iterations (pass in __call__). Astropy default: 100 (too low!).
        Increase for complex models: fitter(model, x, y, maxiter=10000)
    """
    
    _methods = {
        'trf': astropy_fitting.TRFLSQFitter,
        'dogbox': astropy_fitting.DogBoxLSQFitter,
        'levmar': astropy_fitting.LevMarLSQFitter,
        'simplex': astropy_fitting.SimplexLSQFitter,
    }
    
    _no_bounds_methods = {'levmar', 'simplex'}  # Methods that don't support bounds/ties
    _no_uncertainties_methods = {'simplex'}  # Methods that don't support calc_uncertainties
    
    def __init__(self, method='TRF', calc_uncertainties=False, force_numerical_covariance=False, verbose=False):
        super().__init__(calc_uncertainties=calc_uncertainties,
                        force_numerical_covariance=force_numerical_covariance,
                        verbose=verbose)
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
        
    def __call__(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None, inplace=False, 
                 estimate_jacobian=None, **kwargs):
        """
        Fit model to data.
        
        Parameters
        ----------
        yerr : array-like, optional
            Uncertainties on y data.
        statistic : str
            Fit statistic: 'chi2' (default) or 'poisson'.
        weights : array-like, optional
            Explicit weights. Overrides yerr-based calculation.
        estimate_jacobian : bool, optional
            If True, use numerical Jacobian. If None, auto-detect.
        """
        # Check for bounds/ties with incompatible methods
        method_key = self.method.lower()
        if method_key in self._no_bounds_methods:
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
                    f"    Use AstropyTRF or AstropyDogBox for bounds support.\\n"
                    f"{'='*70}",
                    UserWarning, stacklevel=2
                )
            
            if has_tied:
                warnings.warn(
                    f"\\n{'='*70}\\n"
                    f"⚠️  WARNING: {self.method.upper()} DOES NOT SUPPORT TIED PARAMETERS!\\n"
                    f"    Your model has tied parameters, but {self.method} will IGNORE them.\\n"
                    f"    Use AstropyTRF or AstropyDogBox for tied parameter support.\\n"
                    f"{'='*70}",
                    UserWarning, stacklevel=2
                )
        
        if not inplace:
            model = model.copy()
            
        # Compute weights if not explicitly provided
        if weights is None and yerr is not None:
            weights = self._compute_weights(y, yerr, statistic=statistic)
            
        # Create Astropy fitter (minimal wrapper)
        # Some fitters don't support calc_uncertainties parameter
        method_key = self.method.lower()
        if method_key in self._no_uncertainties_methods:
            fitter = self._fitter_class()
        else:
            fitter = self._fitter_class(calc_uncertainties=self.calc_uncertainties)
        
        # Auto-detect jacobian mode if not specified
        use_numeric = estimate_jacobian if estimate_jacobian is not None else False
        
        t0 = time.perf_counter()
        try:
            # Simplex doesn't support estimate_jacobian parameter
            if method_key in {'simplex'}:
                if z is None:
                    fitted = fitter(model, x, y, weights=weights, **kwargs)
                else:
                    fitted = fitter(model, x, y, z, weights=weights, **kwargs)
            else:
                if z is None:
                    fitted = fitter(model, x, y, weights=weights, 
                                   estimate_jacobian=use_numeric, **kwargs)
                else:
                    fitted = fitter(model, x, y, z, weights=weights, 
                                   estimate_jacobian=use_numeric, **kwargs)
        except ValueError as e:
            # Handle fit_deriv shape mismatch for compound models with custom components
            if "broadcast" in str(e) and "fit_deriv" not in str(kwargs.get('estimate_jacobian', '')):
                if self.verbose:
                    print(f"Note: fit_deriv shape mismatch, using numeric derivatives")
                # Retry with numeric jacobian
                if z is None:
                    fitted = fitter(model, x, y, weights=weights, 
                                   estimate_jacobian=True, **kwargs)
                else:
                    fitted = fitter(model, x, y, z, weights=weights, 
                                   estimate_jacobian=True, **kwargs)
            else:
                raise
        
        elapsed = time.perf_counter() - t0
        
        # Store native Astropy fit_info directly (minimal wrapper)
        self.fit_info = fitter.fit_info.copy()
        self.fit_info['time'] = elapsed
        self.fit_info['result'] = fitter.fit_info  # Reference for compatibility
        
        # Extract and store covariance matrix
        if self.calc_uncertainties:
            cov = fitter.fit_info.get('param_cov')
            if cov is not None:
                # Store as param_cov (Astropy standard)
                self.fit_info['param_cov'] = cov
            else:
                if self.verbose:
                    print("⚠ Astropy fitter did not provide native covariance matrix.")
                self.fit_info['param_cov'] = None
        
        if self.verbose:
            nfev = self.fit_info.get('nfev', '?')
            success = self.fit_info.get('success', True)
            status = "✓ converged" if success else "✗ FAILED"
            print(f"AstropyFitter({self.method}): {elapsed*1000:.1f}ms, nfev={nfev}, {status}")
        
        return fitted


class AstropyTRF(AstropyFitter):
    """
    Trust Region Reflective via Astropy (supports bounds & ties).
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 100, often too low!)
    - acc : float - Accuracy for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    ⚠️  Default maxiter=100 might be low for complex models!
    Always increase for multi-component fits: maxiter=10000
    
    Example:
        fitter = AstropyTRF(calc_uncertainties=True, verbose=True)
        result = fitter(model, x, y, maxiter=10000, acc=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='TRF', **kwargs)

class AstropyDogBox(AstropyFitter):
    """
    Dogleg via Astropy (supports bounds & ties).
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 100, often TOO LOW!)
    - acc : float - Accuracy for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    ⚠️  **IMPORTANT:** Default maxiter=100 is too low for complex models!
    Always increase for multi-component fits: maxiter=10000
    
    Example:
        fitter = AstropyDogBox(calc_uncertainties=True)
        result = fitter(model, x, y, maxiter=5000)
    """
    def __init__(self, **kwargs):
        super().__init__(method='DogBox', **kwargs)

class AstropyLevMar(AstropyFitter):
    """
    Levenberg-Marquardt via Astropy.
    
    ⚠️  **WARNING: DOES NOT SUPPORT PARAMETER BOUNDS OR TIED PARAMETERS!**
    
    This is the classic LM algorithm without bounds support. If your model has
    bounded or tied parameters, they will be IGNORED. Use AstropyTRF or 
    AstropyDogBox for bounds/ties support.
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 100)
    - acc : float - Relative error for termination (default: 1e-7)
    - epsilon : float - Step size for numerical derivatives (default: 1.49e-8)
    - estimate_jacobian : bool - Use numerical Jacobian (default: False)
    
    Example:
        fitter = AstropyLevMar(calc_uncertainties=True)
        result = fitter(model, x, y, maxiter=1000, acc=1e-9)
    """
    def __init__(self, **kwargs):
        super().__init__(method='LevMar', **kwargs)

class AstropySimplex(AstropyFitter):
    """
    Nelder-Mead Simplex via Astropy (derivative-free).
    
    ⚠️  **WARNING: DOES NOT SUPPORT PARAMETER BOUNDS OR TIED PARAMETERS!**
    
    This is a derivative-free simplex algorithm. If your model has bounded or 
    tied parameters, they will be IGNORED. Use AstropyTRF or AstropyDogBox for 
    bounds/ties support.
    
    **Key Parameters (pass in __call__):**
    - maxiter : int - Maximum iterations (default: 100)
    - acc : float - Convergence tolerance (default: 1e-7)
    - Note: Does NOT support estimate_jacobian (derivative-free method)
    
    Example:
        fitter = AstropySimplex(verbose=True)
        result = fitter(model, x, y, maxiter=5000, acc=1e-6)
    """
    def __init__(self, **kwargs):
        super().__init__(method='Simplex', **kwargs)

