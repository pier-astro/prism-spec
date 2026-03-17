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
    Wrapper around native Astropy fitters.

    Supported methods
    -----------------
    - ``'TRF'``: trust-region reflective (supports bounds and ties)
    - ``'DogBox'``: dogleg with rectangular trust regions (supports bounds and ties)
    - ``'LevMar'``: Levenberg-Marquardt (ignores bounds/ties)
    - ``'Simplex'``: Nelder-Mead simplex (ignores bounds/ties)

    Covariance uses Astropy's native ``fit_info['param_cov']`` when available.
    
    Parameters
    ----------
    method : str
        'TRF' (default), 'DogBox', 'LevMar', or 'Simplex'.
    calc_uncertainties : bool
        Calculate covariance matrix using native Astropy method.
    verbose : bool
        Print diagnostic information.

    Common call-time parameters
    ---------------------------
    yerr, statistic, weights
        Standard weighting controls from ``FitterBase.__call__``.
    maxiter : int, optional
        Maximum optimizer iterations (default 1000 in this wrapper).
    estimate_jacobian : bool, optional
        Use numerical Jacobian (where supported).

    Examples
    --------
    >>> fitter = AstroFitter(method='TRF', calc_uncertainties=True)
    >>> fitted = fitter(model, x, y, yerr=yerr, maxiter=5000)

    >>> fitter = AstroFitter(method='DogBox')
    >>> result = fitter(model, x, cube, yerr=cube_err, nproc=4, spectral_axis=0)
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
    Astropy TRF fitter (bounds/ties supported).

    Call-time parameters
    --------------------
    ``maxiter``, ``acc``, ``epsilon``, ``estimate_jacobian``.

    Example
    -------
    >>> fitter = AstroTRF(calc_uncertainties=True, verbose=True)
    >>> fitted = fitter(model, x, y, yerr=yerr, maxiter=10000, acc=1e-10)
    """
    def __init__(self, **kwargs):
        super().__init__(method='TRF', **kwargs)


class AstroDogBox(AstroFitter):
    """
    Astropy DogBox fitter (bounds/ties supported).

    Call-time parameters
    --------------------
    ``maxiter``, ``acc``, ``epsilon``, ``estimate_jacobian``.

    Example
    -------
    >>> fitter = AstroDogBox(calc_uncertainties=True)
    >>> fitted = fitter(model, x, y, yerr=yerr, maxiter=5000)
    """
    def __init__(self, **kwargs):
        super().__init__(method='DogBox', **kwargs)


class AstroLM(AstroFitter):
    """
    Astropy Levenberg-Marquardt fitter.

    Notes
    -----
    Does not enforce parameter bounds/ties; prefer ``AstroTRF`` or
    ``AstroDogBox`` for constrained fits.

    Call-time parameters
    --------------------
    ``maxiter``, ``acc``, ``epsilon``, ``estimate_jacobian``.

    Example
    -------
    >>> fitter = AstroLM(calc_uncertainties=True)
    >>> fitted = fitter(model, x, y, yerr=yerr, maxiter=5000)
    """
    def __init__(self, **kwargs):
        super().__init__(method='LevMar', **kwargs)


class AstroSimplex(AstroFitter):
    """
    Astropy Nelder-Mead simplex fitter (derivative-free).

    Notes
    -----
    Does not enforce parameter bounds/ties and does not use
    ``estimate_jacobian``.

    Call-time parameters
    --------------------
    ``maxiter``, ``acc``.

    Example
    -------
    >>> fitter = AstroSimplex(verbose=True)
    >>> fitted = fitter(model, x, y, maxiter=5000, acc=1e-6)
    """
    def __init__(self, **kwargs):
        super().__init__(method='Simplex', **kwargs)