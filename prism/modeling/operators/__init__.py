"""prism.modeling.operators — linear operators applied to spectral models."""

from .matop import LinearOperatorModel, install_pipe_patch, is_linear_operator_pipe
from .instrument import InstrumentResponse, SpectralResponse

__all__ = [
    'LinearOperatorModel',
    'install_pipe_patch',
    'is_linear_operator_pipe',
    'InstrumentResponse',
    'SpectralResponse',
]
