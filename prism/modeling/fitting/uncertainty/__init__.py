"""
prism.modeling.fitting.uncertainty — parameter uncertainty utilities.

Two complementary approaches are provided:

- covariance: fast, local approximation from the Jacobian/Hessian at best fit.
- bootstrap: robust resampling-based intervals via repeated synthetic refits.

Typical usage
-------------
Covariance-based uncertainties:

        from prism.modeling.fitting import AstroTRF
        from prism.modeling.fitting.uncertainty import covar_attach

        fitter = AstroTRF(calc_uncertainties=True)
        fitted = fitter(model, x, y, yerr=yerr)
        covar_attach(fitted, fitter)

Bootstrap intervals:

        from prism.modeling.fitting import AstroTRF
        from prism.modeling.fitting.uncertainty import bootstrap, resample_attach

        fitter = AstroTRF()
        fitted = fitter(model, x, y, yerr=yerr)
        samples = bootstrap(fitted, fitter, x, y, yerr, n_samples=500)
        resample_attach(fitted, samples)
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
