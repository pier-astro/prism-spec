"""
prism.modeling.fitting — fitter backends and batch fitting utilities.

This package provides Astropy-compatible fitters for single spectra and
batched multi-spectrum fitting with a shared API.

Main classes
------------
FitterBase
	Common fitting pipeline (weights, ties, bounds, diagnostics, covariance).
AstroFitter / AstroTRF / AstroDogBox / AstroLM / AstroSimplex
	Wrappers around native Astropy fitters.
ScipyFitter / ScipyTRF / ScipyDogBox
	Wrappers around ``scipy.optimize.least_squares``.
LMFitter
	Classic Levenberg–Marquardt with internal parameter transforms for bounds.
SherpaFitter / SherpaLM / SherpaSimplex / SherpaMonCar
	Sherpa backend fitters (when Sherpa is installed).
FittingWithOutlierRemoval
	Iterative sigma-clipping wrapper around any prism fitter.

Batch objects
-------------
MultiFitResult, SpectrumFitResult, MultiFitParameter
	Structured outputs for batched fits across cubes/maps.
"""

from . import astrofit, base, lmfit, scipyfit, sherpafit
from .base import *
from .astrofit import *
from .scipyfit import *
from .sherpafit import *
from .lmfit import *

__all__ = []
__all__.extend(base.__all__)
__all__.extend(astrofit.__all__)
__all__.extend(scipyfit.__all__)
__all__.extend(sherpafit.__all__)
__all__.extend(lmfit.__all__)
from . import outremoval
from .outremoval import *
__all__.extend(outremoval.__all__)

from . import uncertainty
