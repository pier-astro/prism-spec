"""
prism.modeling.operators — linear operators applied to spectral models.

Exports
-------
ConvolvedModel
    Generic wrapper for applying a fixed linear operator matrix to a
    source model while preserving analytic derivatives via chain rule.
InstrumentResponse
    Container / I/O helper for instrumental response matrices.
SpectralResponse
    Callable operator that wraps a source model into a ``ResponseModel``.
ResponseModel
    Instrument-specific ``ConvolvedModel`` subclass.
ResponseOperator
    Low-level helper used by ``SpectralResponse``.
"""

from .convolved import ConvolvedModel
from .instrument import InstrumentResponse, SpectralResponse, ResponseOperator, ResponseModel

__all__ = [
    'ConvolvedModel',
    'InstrumentResponse',
    'SpectralResponse',
    'ResponseOperator',
    'ResponseModel',
]
