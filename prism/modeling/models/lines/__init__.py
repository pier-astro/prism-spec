"""
prism.modeling.models.lines — emission-line spectral models.
"""
from ...metrics import Metric
from .base import LineModelBase
from .tables import (
    set_wavelength_range,
    convert_linetable_domain, convert_linetable_medium,
    convert_domain_csv, convert_medium_csv,
    setup_local_lines, trim_line_lists,
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
    if name in ('linetable_path', '_wmin', '_wmax', 'resource_path'):
        return getattr(_base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Activate YAML hooks even when users import line models directly.
from .. import serialization as _serialization  # noqa: E402,F401
