"""
prism.modeling — general components, fitting, domain models and I/O.

Submodules
----------
models
    Astropy-compliant spectral models and component factories.
lines
    Post-fit line selection, measurement, and uncertainty sampling.
operators
    Linear operators applied to source models.
fitting
    Single-spectrum and multi-spectrum fitting tools.
io
    FITS I/O for fitted models and multi-fit results.
"""
from .composition import Model
from .component import Component
from .lines import LineSet
from .lines.measurements import (
    LineResult, Metric, MultiLineMeasurements, SelectedLineProfile,
    SelectedLineCollection, measure_line, profile_metrics,
    sample_line_measurements, select_line,
)
from .operators.matop import LinearOperatorModel, is_linear_operator_pipe
from .operators.instrument import InstrumentResponse, SpectralResponse
from .display import (
    get_model_expression,
    format_model_text,
    format_model_html,
    format_fit_text,
    format_fit_html,
    show_model,
)
