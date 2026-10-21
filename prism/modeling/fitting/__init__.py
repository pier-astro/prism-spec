"""Numerical single/joint fits, rejection, uncertainties and independent batches.

SciPy and Sherpa use Prism's common numerical problem. Astropy fitter adapters
are available in modeling.interop.astropy; experimental solvers in prism.experimental.
"""
from . import utils, scipyfit, sherpafit, outremoval, multifit, uncertainty
from .utils import apply_tied
from .data import Dataset
from .results import FitResult
from .scipyfit import *
from .sherpafit import *
from .multifit import *
from .outremoval import *

__all__ = ['Dataset', 'FitResult', 'apply_tied', 'MultiFitResult', 'SpectrumFitResult', 'MultiFitParameter']
__all__ += scipyfit.__all__ + sherpafit.__all__ + outremoval.__all__

from .extension import get_max_evaluations, set_max_evaluations
__all__ += ['get_max_evaluations', 'set_max_evaluations']
