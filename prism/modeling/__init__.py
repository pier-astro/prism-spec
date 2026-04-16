"""
prism.modeling — spectral models, model-aware analysis, fitters, and I/O.

Submodules
----------
models
    Astropy-compliant spectral models and component factories.
line_analysis
    Post-fit line selection, measurement, and uncertainty sampling.
operators
    Linear operators applied to source models.
fitting
    Single-spectrum and multi-spectrum fitting tools.
io
    FITS I/O for fitted models and multi-fit results.
"""
from . import line_analysis
from .line_analysis import *
from .operators.convolved import LinearOperator, MatrixLinearOperator, LinearOperatorCompoundModel
from .operators.instrument import InstrumentResponse, SpectralResponse, ResponseOperator
from .models.display import (
    enable_model_display,
    disable_model_display,
    get_model_expression,
    format_model_text,
    format_model_html,
    format_fit_text,
    format_fit_html,
    show_model,
)
