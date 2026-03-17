"""
prism.data — spectral data containers.

Classes
-------
Spectrum
    1-D spectrum with wavelength, flux, and optional error arrays.
    Supports cropping, rebinning, redshift correction, dereddening, and
    vacuum/air wavelength conversion.
Cube
    3-D IFU data cube (wavelength × spatial × spatial) with Astropy WCS
    support and FITS I/O.  Designed for spaxel-by-spaxel fitting.
Image
    2-D map container used to store and display spatially resolved
    fitting results, white-light collapse images or general 2-D data.
"""
from .spectrum import Spectrum
from .cube import Cube
from .image import Image
