"""prism.modeling.operators — linear operators applied to spectral models."""

from .matop import LinearOperatorModel, is_linear_operator_pipe
from .instrument import InstrumentResponse, SpectralResponse

__all__ = [
    'LinearOperatorModel',
    'is_linear_operator_pipe',
    'InstrumentResponse',
    'SpectralResponse',
]
