"""
prism.modeling.models.profiles — analytic line profile functions.

Provides vectorised 1-D profile functions and their analytic derivatives
used by the line model classes.  All functions work with scalar or
broadcast-compatible array inputs.

Functions
---------
gaussian / gaussian_deriv
    Gaussian profile and its derivatives w.r.t. amplitude, center, sigma.
lorentzian / lorentzian_deriv
    Lorentzian (Cauchy) profile and derivatives.
voigt / voigt_deriv
    Voigt profile (Gaussian ⊗ Lorentzian) via ``scipy.special.wofz``
    and its complex-plane derivatives.
gaussian_flux / lorentzian_flux / voigt_flux
    Integrated flux from profile parameters.
"""
import numpy as np
from scipy.special import wofz

# Constants
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SQRT_2PI = np.sqrt(2 * np.pi)
SQRT_LN2 = np.sqrt(np.log(2))

def gaussian(x, amplitude, center, sigma):
    """
    1-D Gaussian profile.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    amplitude : float
        Peak value.
    center : float
        Center position.
    sigma : float
        Standard deviation (width parameter).

    Returns
    -------
    ndarray
        Profile values at ``x``.
    """
    if sigma == 0:
        return np.where(x == center, amplitude, 0.0)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def gaussian_deriv(x, amplitude, center, sigma):
    """
    Gaussian profile and its analytic derivatives.

    Returns
    -------
    tuple
        ``(val, d_amp, d_center, d_sigma)`` — profile value and partial
        derivatives w.r.t. amplitude, center, and sigma.
    """
    if sigma == 0:
        val = np.where(x == center, amplitude, 0.0)
        return val, val / amplitude, np.zeros_like(x), np.zeros_like(x)

    z = (x - center) / sigma
    val = amplitude * np.exp(-0.5 * z ** 2)
    
    d_amp = val / amplitude
    d_center = val * z / sigma
    d_sigma = val * (z ** 2) / sigma
    
    return val, d_amp, d_center, d_sigma

def lorentzian(x, amplitude, center, gamma):
    """
    1-D Lorentzian (Cauchy) profile.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    amplitude : float
        Peak value.
    center : float
        Line center.
    gamma : float
        Half-width at half-maximum (HWHM).

    Returns
    -------
    ndarray
    """
    if gamma == 0:
        return np.where(x == center, amplitude, 0.0)
    return amplitude * (gamma**2) / ((x - center)**2 + gamma**2)

def lorentzian_deriv(x, amplitude, center, gamma):
    """
    Lorentzian profile and its analytic derivatives.

    Returns
    -------
    tuple
        ``(val, d_amp, d_center, d_gamma)``.
    """
    if gamma == 0:
        val = np.where(x == center, amplitude, 0.0)
        return val, val / amplitude, np.zeros_like(x), np.zeros_like(x)

    denom = (x - center)**2 + gamma**2
    val = amplitude * (gamma**2) / denom
    
    d_amp = val / amplitude
    d_center = 2 * val * (x - center) / denom
    d_gamma = 2 * amplitude * gamma / denom - 2 * val * gamma / denom 
    # Simplified d_gamma:
    # val = A * g^2 * D^-1
    # d_val/dg = A * (2g * D^-1 + g^2 * (-1) * D^-2 * 2g)
    #          = A * 2g / D - A * 2g^3 / D^2
    #          = (2/g) * val - (2/g) * val * (g^2/D)
    #          = (2/g) * val * (1 - val/A)
    
    return val, d_amp, d_center, d_gamma

def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile evaluated via ``scipy.special.wofz``.

    The profile is peak-normalised so that ``voigt(center, ...) = amplitude``.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    amplitude : float
        Peak value.
    center : float
        Line center.
    sigma : float
        Gaussian standard deviation.
    gamma : float
        Lorentzian HWHM.

    Returns
    -------
    ndarray
    """
    if sigma == 0:
        return lorentzian(x, amplitude, center, gamma)
    if gamma == 0:
        return gaussian(x, amplitude, center, sigma)
        
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    w = wofz(z)
    # The real part of w(z) / (sigma * sqrt(2pi)) is the normalized Voigt profile.
    # We need to scale it to match the amplitude definition.
    # Usually amplitude is the peak. But Voigt peak is complex.
    # The user's previous implementation scaled by amplitude.
    # Let's assume amplitude is a scaling factor for the normalized profile?
    # No, the user's code: return w.real * sqrt_ln2pi / fwhm_G * fwhm_L * amplitude
    # That looks like a specific normalization.
    # Let's stick to the standard definition where amplitude is the peak or integral?
    # The user's previous code:
    # w = hum2zpf16c(z)
    # return w.real * sqrt_ln2pi / fwhm_G * fwhm_L * amplitude
    # This seems to try to match the peak to amplitude roughly?
    # Let's use a standard definition: V(x) = amplitude * V_norm(x) / V_norm(0) ?
    # Or just V(x) = amplitude * V_norm(x).
    
    # Let's look at the user's code again.
    # sqrt_ln2pi = np.sqrt(np.log(2) * np.pi)
    # z = ... * sqrt_ln2 / fwhm_G  <-- this is 1/(sigma*sqrt(2))
    # return w.real * sqrt_ln2pi / fwhm_G * fwhm_L * amplitude
    # This is: w.real * (sigma * sqrt(2pi))^-1 * (something?)
    
    # Let's implement the standard Voigt and maybe a scaling factor.
    # If the user wants 'amplitude' to be the peak value, we need to normalize.
    # But calculating the peak of a Voigt is expensive (requires solving derivative).
    # Standard practice: amplitude is the area, or amplitude is the value at center (approx).
    # The user's code seems to treat 'amplitude' as a scaling factor for the Lorentzian part?
    
    # Let's stick to the user's normalization if possible, or switch to a standard one.
    # User's: amplitude * (hwhm^2) / ((x-mean)^2 + hwhm^2) for Lorentzian.
    # User's: amplitude * exp(...) for Gaussian.
    # These are peak-normalized.
    # For Voigt, we want V(center) = amplitude.
    # V_norm(0) = Re[wofz(i gamma / (sigma sqrt(2)))] / (sigma sqrt(2pi))
    #           = wofz(i rho).real / (sigma sqrt(2pi))
    # So we can normalize by V_norm(0).
    
    val = w.real / (sigma * SQRT_2PI)
    
    # Normalization factor to make peak = amplitude
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    norm = wofz(z0).real / (sigma * SQRT_2PI)
    
    return amplitude * val / norm

def voigt_deriv(x, amplitude, center, sigma, gamma):
    """
    Voigt profile and its analytic derivatives.

    Uses the identity ``dw/dz = -2z w + 2i/√π`` for the Faddeeva function.

    Returns
    -------
    tuple
        ``(val, d_amp, d_center, d_sigma, d_gamma)``.
    """
    if sigma == 0:
        return lorentzian_deriv(x, amplitude, center, gamma)
        
    sqrt2 = np.sqrt(2)
    z = (x - center + 1j * gamma) / (sigma * sqrt2)
    w = wofz(z)
    V = w.real / (sigma * SQRT_2PI)
    
    # Derivatives of V w.r.t parameters (unnormalized)
    # dV/dx:
    # dw/dx = dw/dz * dz/dx = (-2z w + 2i/sqrt(pi)) * (1/(sigma*sqrt2))
    dw_dz = -2 * z * w + 2j / np.sqrt(np.pi)
    dz_dx = 1 / (sigma * sqrt2)
    dV_dx = (dw_dz * dz_dx).real / (sigma * SQRT_2PI)
    
    # dV/d_center = -dV/dx
    dV_dcenter = -dV_dx
    
    # dV/d_sigma:
    # dz/d_sigma = -z / sigma
    # d(1/sigma)/d_sigma = -1/sigma^2
    # dV/d_sigma = (dw_dz * dz_dsigma).real / (sigma*sqrt2pi) + w.real * (-1/(sigma^2*sqrt2pi))
    #            = (dw_dz * (-z/sigma)).real / (sigma*sqrt2pi) - V/sigma
    dV_dsigma = (dw_dz * (-z / sigma)).real / (sigma * SQRT_2PI) - V / sigma
    
    # dV/d_gamma:
    # dz/d_gamma = i / (sigma*sqrt2)
    # dV/d_gamma = (dw_dz * dz_dgamma).real / (sigma*sqrt2pi)
    dV_dgamma = (dw_dz * (1j / (sigma * sqrt2))).real / (sigma * SQRT_2PI)
    
    # Now handle the normalization: F = amplitude * V / V0
    # V0 = V(center) (depends on sigma, gamma)
    z0 = (1j * gamma) / (sigma * sqrt2)
    w0 = wofz(z0)
    V0 = w0.real / (sigma * SQRT_2PI)
    
    # Derivatives of V0
    # dV0/d_sigma: same formula as dV/d_sigma evaluated at x=center (z=z0)
    dw0_dz0 = -2 * z0 * w0 + 2j / np.sqrt(np.pi)
    dV0_dsigma = (dw0_dz0 * (-z0 / sigma)).real / (sigma * SQRT_2PI) - V0 / sigma
    
    # dV0/d_gamma:
    dV0_dgamma = (dw0_dz0 * (1j / (sigma * sqrt2))).real / (sigma * SQRT_2PI)
    
    # Final derivatives
    f = amplitude * V / V0
    
    d_amp = V / V0
    d_center = (amplitude / V0) * dV_dcenter
    d_sigma = (amplitude / V0) * dV_dsigma - (amplitude * V / V0**2) * dV0_dsigma
    d_gamma = (amplitude / V0) * dV_dgamma - (amplitude * V / V0**2) * dV0_dgamma
    
    return f, d_amp, d_center, d_sigma, d_gamma

def gaussian_flux(amplitude, sigma):
    """
    Calculate the integrated flux of a Gaussian profile.
    """
    return amplitude * sigma * SQRT_2PI

def lorentzian_flux(amplitude, gamma):
    """
    Calculate the integrated flux of a Lorentzian profile.
    """
    return amplitude * gamma * np.pi

def voigt_flux(amplitude, sigma, gamma):
    """
    Calculate the integrated flux of a Voigt profile.
    """
    if sigma == 0:
        return lorentzian_flux(amplitude, gamma)
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    norm = wofz(z0).real / (sigma * SQRT_2PI)
    return amplitude / norm
