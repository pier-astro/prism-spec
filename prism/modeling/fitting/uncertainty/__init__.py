"""
Parameter uncertainty estimation for FantasyLab.

This module provides tools for extracting and working with parameter uncertainties
from fitted models.

Available Methods
-----------------
- **Covariance**: Fast, from Jacobian at best-fit (covariance submodule)
  Assumes Gaussian errors and linear model behavior near minimum.
  
- **Bootstrap**: Robust, via parametric resampling (resample submodule)
  Generates synthetic datasets and refits to estimate parameter distributions.

Usage
-----
**Covariance-based (fast):**

    from prism.modeling.fitting import AstropyTRF
    from fantasylab.uncertainty import covariance
    
    fitter = AstropyTRF(calc_uncertainties=True)
    fitted = fitter(model, x, y, yerr=yerr)
    
    covariance.attach(fitted, fitter)
    print(f"{fitted.amplitude_0.value} ± {fitted.amplitude_0.std}")

**Bootstrap (robust):**

    from fantasylab.uncertainty import resample
    
    fitter = AstropyTRF()
    fitted = fitter(model, x, y, yerr=yerr)
    
    samples = resample.bootstrap(fitted, fitter, x, y, yerr, n_samples=1000)
    resample.attach(fitted, samples)
    print(f"{fitted.amplitude_0.value} [{fitted.amplitude_0.lolim}, {fitted.amplitude_0.uplim}]")
"""

from .covariance import (
    attach as covar_attach,
    detach as covar_detach,
    attach_uncertainties as covar_attach_uncertainties,
    detach_uncertainties as covar_detach_uncertainties,
    covar,
    stdevs,
    correlation,
    CovarianceError
)

from .resample import (
    Bootstrap,
    bootstrap,
    attach as resample_attach,
    detach as resample_detach,
    ResampleError
)

__all__ = [
    # Covariance
    'covar_attach',
    'covar_detach', 
    'covar_attach_uncertainties',
    'covar_detach_uncertainties',
    'covar',
    'stdevs',
    'correlation',
    'CovarianceError',
    # Resample
    'Bootstrap',
    'bootstrap',
    'resample_attach',
    'resample_detach',
    'ResampleError',
]

__version__ = '0.2.0'
