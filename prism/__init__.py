"""
prism-spec: spectral fitting framework for optical/NIR spectroscopy.

Provides tools for spectral data handling, model construction, and
fitting from single spectra to full IFU data cubes.

Subpackages
-----------
data
    Spectral data containers: ``Spectrum``, ``Cube``, ``Image``.
modeling
    Spectral models, line-analysis helpers, operators, fitters, and model I/O.
utils
    Low-level wavelength tools, resampling helpers, and plotting utilities.
"""
from ._version import __version__

# Import main subpackages
from . import config
from . import data
from . import modeling
from . import utils