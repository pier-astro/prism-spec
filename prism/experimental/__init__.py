"""Experimental solvers: bounded LM and variable projection need further validation.

These implementations are retained for development and explicit experiments;
SciPy and Sherpa remain the supported numerical fitting paths.
"""
from .lmfit import LMFitter
from .separable import SeparableTRF, inspect_separable

__all__ = ['LMFitter', 'SeparableTRF', 'inspect_separable']
