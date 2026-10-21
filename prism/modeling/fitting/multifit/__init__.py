"""Independent multi-spectrum fitting and its structured results.

Implementation: configuration prepares parameter controls, execution runs and
collects independent fits, results owns arrays and component views.
"""
from .results import MultiFitParameter, MultiFitResult, SpectrumFitResult, _MULTIFIT_STAT_KEYS
from .execution import MultiFitMixin

__all__ = ['MultiFitParameter', 'MultiFitResult', 'SpectrumFitResult', 'MultiFitMixin']
