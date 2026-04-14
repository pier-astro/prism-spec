"""
prism.modeling — spectral models, operators, fitters, and I/O.

Submodules
----------
models
    Astropy-compliant spectral models: emission/absorption lines
    (Gaussian, Lorentzian, Voigt), power-law and Balmer continua,
    empirical templates (B-spline, fixed interpolation table), and
    pre-built AGN component factories.
operators
    Linear operators applied to source models: ``LinearOperatorCompoundModel``
    for matrix transforms and ``InstrumentResponse`` / ``SpectralResponse``
    for LSF-based instrumental broadening.
fitting
    Fitter hierarchy built on Astropy's ``Fitter`` base.  Single-spectrum
    and batched multi-spectrum fitting, outlier removal, covariance and
    bootstrap uncertainty estimation.
io
    FITS I/O for fitted models and ``MultiFitResult`` output.
"""
from .operators.convolved import LinearOperator, MatrixLinearOperator, LinearOperatorCompoundModel
from .operators.instrument import InstrumentResponse, SpectralResponse, ResponseOperator
