"""
prism.modeling.models.lines — emission-line spectral models.
"""
from .base import (
    Metric, LineModelBase,
    set_wavelength_range, setup_local_lines,
    c_kms, sigma2fwhm,
)
from .single import GaussianLine, LorentzianLine, VoigtLine
from .groups import (
    LineGroupBase,
    GaussianLines, LorentzianLines, VoigtLines,
)

# Allow attribute access to mutable module-level state in base
from . import base as _base

def __getattr__(name):
    if name in ('csv_lines_path', '_wmin', '_wmax', 'resource_path'):
        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
