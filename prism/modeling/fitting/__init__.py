"""
prism.modeling.fitting — fitter backends, Astropy-style wrappers, and batch fitting utilities.

This package exposes Prism-native fitters together with Prism-owned wrappers
around selected Astropy fitters. Native ``astropy.modeling.fitting`` classes
are no longer patched broadly.

Main classes
------------
ScipyFitter / ScipyTRF / ScipyDogBox
	Wrappers around ``scipy.optimize.least_squares``.
LMFitter
	Classic Levenberg–Marquardt with internal parameter transforms for bounds.
SherpaFitter / SherpaLM / SherpaSimplex / SherpaMonCar
	Sherpa backend fitters (when Sherpa is installed).
FittingWithOutlierRemoval
	Iterative sigma-clipping wrapper around any fitter.

Batch objects
-------------
MultiFitResult, SpectrumFitResult, MultiFitParameter
	Structured outputs for batched fits across cubes/maps.
"""

from . import astropyfitters, utils, lmfit, scipyfit, sherpafit, outremoval, multifit, uncertainty
from .astropyfitters import *
from .utils import *
from .scipyfit import *
from .sherpafit import *
from .lmfit import *
from .multifit import *
from .outremoval import *

from .extension import enable_astropy_fitting_compatibility_patch
enable_astropy_fitting_compatibility_patch()

from ..io import install_model_io_methods
install_model_io_methods()

__all__ = [
    'tie', 'apply_tied', 'MultiFitResult', 'SpectrumFitResult', 'MultiFitParameter'
]
__all__.extend(astropyfitters.__all__)
__all__.extend(scipyfit.__all__)
__all__.extend(sherpafit.__all__)
__all__.extend(lmfit.__all__)
__all__.extend(outremoval.__all__)
