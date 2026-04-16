"""
prism.modeling.fitting — fitter backends and batch fitting utilities.

This package dynamically extends Astropy fitters to seamlessly support prism's
features like weights from `yerr` and `.multifit()`.

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

from . import utils, lmfit, scipyfit, sherpafit, outremoval, multifit, uncertainty
from .utils import *
from .scipyfit import *
from .sherpafit import *
from .lmfit import *
from .multifit import *
from .outremoval import *

from .extension import patch_astropy_fitters
patch_astropy_fitters()

from ..io import install_model_io_methods
install_model_io_methods()

__all__ = [
    'tie', 'apply_tied', 'MultiFitResult', 'SpectrumFitResult', 'MultiFitParameter'
]
__all__.extend(scipyfit.__all__)
__all__.extend(sherpafit.__all__)
__all__.extend(lmfit.__all__)
__all__.extend(outremoval.__all__)
