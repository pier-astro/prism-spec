"""
prism.modeling.fitting.uncertainty — parameter uncertainty utilities.

Two complementary approaches are provided:

- covariance: fast, local approximation from the Jacobian/Hessian at best fit.
- bootstrap: robust resampling-based intervals via repeated synthetic refits.

Typical usage
-------------
Covariance-based uncertainties:

        from prism.modeling.fitting import TRF
        from prism.modeling.fitting.uncertainty import covar_attach

        fitter = TRF(covariance=True)
        result = fitter(model, x, y, yerr=yerr)
        result.params
        result.covariance

        # Optional attachment for a model-only export.
        fitted = result.model
        covar_attach(fitted, result)

Bootstrap intervals:

        from prism.modeling.fitting import TRF
        from prism.modeling.fitting.uncertainty import bootstrap

        fitter = TRF()
        result = fitter(model, x, y, yerr=yerr)
        fitted = result.model
        samples = bootstrap(fitted, fitter, x, y, yerr, n_samples=500)
        samples.params
        samples.show()
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
    BootstrapResult,
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
    'BootstrapResult',
    'ResampleError',
]

__version__ = '0.2.0'
