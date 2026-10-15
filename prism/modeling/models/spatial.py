"""Spatial Astropy models with directly fitted astronomical PSF widths.

Widths and positions use a common spatial coordinate unit; the Prism PSF workflow
uses local pixels and angles in radians. Amplitude is peak surface brightness,
not integrated flux. Ellipse widths refer to its rotated principal axes.
"""

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter

__all__ = ['MoffatPSF', 'EllipticalMoffatPSF']


class MoffatPSF(Fittable2DModel):
    """Circular Moffat profile parameterized by FWHM and wing exponent beta.

    Parameters
    ----------
    amplitude : float, optional
        Peak brightness. Default 1.
    x_0, y_0 : float, optional
        Centroid coordinates. Default zero.
    fwhm : float, optional
        Full width at half maximum, in the same units as the coordinates.
        Default 1. Must be positive.
    beta : float, optional
        Wing exponent. Default 2.5; beta > 1 gives a finite total flux.
    **kwargs
        Standard Astropy model options, including fixed and bounds mappings.

    Notes
    -----
    The radial scale is fwhm / (2*sqrt(2**(1/beta)-1)). The infinite-plane
    integral is pi*amplitude*scale**2/(beta-1). The profile is not normalized
    automatically; ChromaticPSF handles normalization separately.

    Examples
    --------
    >>> model = MoffatPSF(amplitude=100, x_0=12, y_0=13, fwhm=3, beta=2.5)
    >>> model.fwhm.fixed = True
    """

    amplitude = Parameter(default=1, bounds=(0, None))
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    fwhm = Parameter(default=1, bounds=(np.finfo(float).tiny, None))
    beta = Parameter(default=2.5, bounds=(np.nextafter(1., np.inf), None))

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, fwhm, beta):
        scale = fwhm / (2 * np.sqrt(np.expm1(np.log(2) / beta)))
        return amplitude * (1 + ((x - x_0)**2 + (y - y_0)**2) / scale**2)**(-beta)


class EllipticalMoffatPSF(Fittable2DModel):
    """Rotated elliptical Moffat with independently fitted principal-axis FWHM.

    Parameters
    ----------
    amplitude : float, optional
        Peak brightness. Default 1.
    x_0, y_0 : float, optional
        Centroid coordinates. Default zero.
    fwhm_x, fwhm_y : float, optional
        Positive FWHM along the two principal axes. Default 1. The names
        identify the unrotated axes, not necessarily major/minor order.
    beta : float, optional
        Shared wing exponent. Default 2.5; must exceed 1 for finite total flux.
    theta : float, optional
        Counterclockwise rotation of the principal x axis from image +x toward
        +y, in radians. Default zero. Has period pi. It is not a sky position angle.
    **kwargs
        Standard Astropy model options, including fixed and bounds mappings.

    Notes
    -----
    Uses the circular Moffat radial law in rotated, independently scaled
    coordinates. Its integral is pi*amplitude*scale_x*scale_y/(beta-1).
    The model remains centrosymmetric. Swapping the widths and adding pi/2
    to theta gives an equivalent profile; near circularity theta is poorly
    constrained. Use consistent initial axes or fix theta when appropriate.

    Examples
    --------
    >>> model = EllipticalMoffatPSF(fwhm_x=3, fwhm_y=2.7, beta=2.5, theta=0.2)
    >>> model.theta.bounds = (-0.5, 0.5)
    """

    amplitude = Parameter(default=1, bounds=(0, None))
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    fwhm_x = Parameter(default=1, bounds=(np.finfo(float).tiny, None))
    fwhm_y = Parameter(default=1, bounds=(np.finfo(float).tiny, None))
    beta = Parameter(default=2.5, bounds=(np.nextafter(1., np.inf), None))
    theta = Parameter(default=0)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, fwhm_x, fwhm_y, beta, theta):
        scale_x = fwhm_x / (2 * np.sqrt(np.expm1(np.log(2) / beta)))
        scale_y = fwhm_y / (2 * np.sqrt(np.expm1(np.log(2) / beta)))
        dx, dy = x - x_0, y - y_0
        xp = np.cos(theta) * dx + np.sin(theta) * dy
        yp = -np.sin(theta) * dx + np.cos(theta) * dy
        return amplitude * (1 + (xp / scale_x)**2 + (yp / scale_y)**2)**(-beta)
