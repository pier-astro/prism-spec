"""
prism.modeling.models.lines.profiles — analytic line profile functions.

Provides vectorised 1-D profile functions, their analytic derivatives,
integrated-flux formulas, and velocity-parameterised wrappers used by the
line model classes.
"""
import numpy as np
import astropy.units as u
from scipy.special import wofz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SQRT_2PI = np.sqrt(2 * np.pi)
SQRT_LN2 = np.sqrt(np.log(2))
C_KMS = 299792.458  # speed of light in km/s (IAU)
C_AA_S = 2.99792458e18  # speed of light in Angstrom / s
HC_AA_EV = 12398.419843320027  # h * c in Angstrom * eV
DOMAIN_FAMILIES = ('wavelength', 'linear')

# ---------------------------------------------------------------------------
# Pure profile functions
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, center, sigma):
    """1-D Gaussian profile (peak-normalised)."""
    if sigma == 0:
        return np.where(x == center, amplitude, 0.0)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def gaussian_deriv(x, amplitude, center, sigma):
    """Gaussian profile and analytic derivatives.

    Returns ``(val, d_amp, d_center, d_sigma)``.
    """
    if sigma == 0:
        val = np.where(x == center, amplitude, 0.0)
        d_amp = np.where(x == center, 1.0, 0.0)
        return val, d_amp, np.zeros_like(x), np.zeros_like(x)
    z = (x - center) / sigma
    shape = np.exp(-0.5 * z ** 2)
    val = amplitude * shape
    d_amp = shape
    d_center = val * z / sigma
    d_sigma = val * (z ** 2) / sigma
    return val, d_amp, d_center, d_sigma


def lorentzian(x, amplitude, center, gamma):
    """1-D Lorentzian (Cauchy) profile (peak-normalised)."""
    if gamma == 0:
        return np.where(x == center, amplitude, 0.0)
    return amplitude * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)


def lorentzian_deriv(x, amplitude, center, gamma):
    """Lorentzian profile and analytic derivatives.

    Returns ``(val, d_amp, d_center, d_gamma)``.
    """
    if gamma == 0:
        val = np.where(x == center, amplitude, 0.0)
        d_amp = np.where(x == center, 1.0, 0.0)
        return val, d_amp, np.zeros_like(x), np.zeros_like(x)
    denom = (x - center) ** 2 + gamma ** 2
    shape = (gamma ** 2) / denom
    val = amplitude * shape
    d_amp = shape
    d_center = 2 * val * (x - center) / denom
    d_gamma = 2 * amplitude * gamma / denom - 2 * val * gamma / denom
    return val, d_amp, d_center, d_gamma


def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile evaluated via ``wofz``; peak-normalised so that
    ``voigt(center, ...) ≈ amplitude``."""
    if sigma == 0:
        return lorentzian(x, amplitude, center, gamma)
    if gamma == 0:
        return gaussian(x, amplitude, center, sigma)
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    w = wofz(z)
    val = w.real / (sigma * SQRT_2PI)
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    norm = wofz(z0).real / (sigma * SQRT_2PI)
    return amplitude * val / norm


def voigt_deriv(x, amplitude, center, sigma, gamma):
    """Voigt profile and analytic derivatives.

    Always returns a 5-tuple ``(val, d_amp, d_center, d_sigma, d_gamma)``.
    """
    if sigma == 0:
        val, d_amp, d_center, d_gamma = lorentzian_deriv(x, amplitude, center, gamma)
        return val, d_amp, d_center, np.zeros_like(val), d_gamma
    if gamma == 0:
        val, d_amp, d_center, d_sigma = gaussian_deriv(x, amplitude, center, sigma)
        return val, d_amp, d_center, d_sigma, np.zeros_like(val)

    sqrt2 = np.sqrt(2)
    z = (x - center + 1j * gamma) / (sigma * sqrt2)
    w = wofz(z)
    V = w.real / (sigma * SQRT_2PI)

    dw_dz = -2 * z * w + 2j / np.sqrt(np.pi)
    dz_dx = 1 / (sigma * sqrt2)
    dV_dx = (dw_dz * dz_dx).real / (sigma * SQRT_2PI)
    dV_dcenter = -dV_dx
    dV_dsigma = (dw_dz * (-z / sigma)).real / (sigma * SQRT_2PI) - V / sigma
    dV_dgamma = (dw_dz * (1j / (sigma * sqrt2))).real / (sigma * SQRT_2PI)

    z0 = (1j * gamma) / (sigma * sqrt2)
    w0 = wofz(z0)
    V0 = w0.real / (sigma * SQRT_2PI)
    dw0_dz0 = -2 * z0 * w0 + 2j / np.sqrt(np.pi)
    dV0_dsigma = (dw0_dz0 * (-z0 / sigma)).real / (sigma * SQRT_2PI) - V0 / sigma
    dV0_dgamma = (dw0_dz0 * (1j / (sigma * sqrt2))).real / (sigma * SQRT_2PI)

    f = amplitude * V / V0
    d_amp = V / V0
    d_center = (amplitude / V0) * dV_dcenter
    d_sigma = (amplitude / V0) * dV_dsigma - (amplitude * V / V0 ** 2) * dV0_dsigma
    d_gamma = (amplitude / V0) * dV_dgamma - (amplitude * V / V0 ** 2) * dV0_dgamma
    return f, d_amp, d_center, d_sigma, d_gamma

# ---------------------------------------------------------------------------
# Integrated flux
# ---------------------------------------------------------------------------

def gaussian_flux(amplitude, sigma):
    return amplitude * sigma * SQRT_2PI


def lorentzian_flux(amplitude, gamma):
    return amplitude * gamma * np.pi


def voigt_flux(amplitude, sigma, gamma):
    if sigma == 0:
        return lorentzian_flux(amplitude, gamma)
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    norm = wofz(z0).real / (sigma * SQRT_2PI)
    return amplitude / norm

# ---------------------------------------------------------------------------
# Velocity-parameterised helpers
# ---------------------------------------------------------------------------

def observed_center(pos, offset, redshift, domain_family='wavelength'):
    """Observed center wavelength/energy from rest position, velocity offset, and redshift."""
    if domain_family == 'wavelength':
        return pos * (1.0 + redshift) * np.exp(offset / C_KMS)
    return pos / ((1.0 + redshift) * np.exp(offset / C_KMS))


def gaussian_profile_params(center, amplitude, weight, fwhm, redshift, instfwhm_val):
    """Gaussian profile parameters from velocity parameterisation.

    Returns ``(amplitude_eff, sigma_eff)``.
    """
    fwhm_A = fwhm / C_KMS * center
    instfwhm_A = instfwhm_val / C_KMS * center
    sigma_int = fwhm_A / SIGMA2FWHM
    sigma_inst = instfwhm_A / SIGMA2FWHM
    sigma_eff = np.sqrt(sigma_int ** 2 + sigma_inst ** 2)
    amplitude_eff = (amplitude * weight) / (1.0 + redshift)
    return amplitude_eff, sigma_eff


def lorentzian_profile_params(center, amplitude, weight, fwhm, redshift, instfwhm_val):
    """Lorentzian profile parameters from velocity parameterisation.

    Returns ``(amplitude_eff, sigma_inst, gamma)``.
    """
    fwhm_A = fwhm / C_KMS * center
    gamma = fwhm_A / 2.0
    sigma_inst = (instfwhm_val / C_KMS * center) / SIGMA2FWHM
    amplitude_eff = (amplitude * weight) / (1.0 + redshift)
    return amplitude_eff, sigma_inst, gamma


def voigt_profile_params(center, amplitude, weight, fwhm_G, fwhm_L, redshift, instfwhm_val):
    """Voigt profile parameters from velocity parameterisation.

    Returns ``(amplitude_eff, sigma_eff, gamma)``.
    """
    fwhm_G_A = fwhm_G / C_KMS * center
    fwhm_L_A = fwhm_L / C_KMS * center
    instfwhm_A = instfwhm_val / C_KMS * center
    sigma_int = fwhm_G_A / SIGMA2FWHM
    sigma_inst = instfwhm_A / SIGMA2FWHM
    sigma_eff = np.sqrt(sigma_int ** 2 + sigma_inst ** 2)
    gamma = fwhm_L_A / 2.0
    amplitude_eff = (amplitude * weight) / (1.0 + redshift)
    return amplitude_eff, sigma_eff, gamma

# ---------------------------------------------------------------------------
# Chain-rule derivatives in velocity parameterisation
# ---------------------------------------------------------------------------

def gaussian_velocity_deriv(x, pos, amplitude, weight, offset, fwhm, redshift,
                            instfwhm_val, instfwhm_deriv_val, domain_family='wavelength'):
    """Full chain-rule derivatives for a Gaussian line in velocity parameterisation.

    Parameters
    ----------
    instfwhm_val : float or array
        Instrumental FWHM (km/s) evaluated at observed center.
    instfwhm_deriv_val : float or array
        d(instfwhm_kms)/d(center_x) evaluated at observed center.

    Returns
    -------
    val, d_amp, d_pos, d_off, d_fwhm, d_z : arrays
    """
    center = observed_center(pos, offset, redshift, domain_family=domain_family)
    amp_eff, sigma_eff = gaussian_profile_params(
        center, amplitude, weight, fwhm, redshift, instfwhm_val)

    val, d_amp_eff, d_center, d_sigma_eff = gaussian_deriv(x, amp_eff, center, sigma_eff)

    d_amp = d_amp_eff * weight / (1.0 + redshift)

    # Velocity-to-native Jacobian
    sign = 1.0 if domain_family == 'wavelength' else -1.0
    d_center_d_pos = (1.0 + redshift) * np.exp(offset / C_KMS) if domain_family == 'wavelength' else 1.0 / ((1.0 + redshift) * np.exp(offset / C_KMS))
    d_center_d_off = sign * center / C_KMS

    # Width Jacobian
    fwhm_x = fwhm / C_KMS * center
    instfwhm_x = instfwhm_val / C_KMS * center
    sigma_int = fwhm_x / SIGMA2FWHM
    sigma_inst = instfwhm_x / SIGMA2FWHM

    safe_se = np.where(sigma_eff != 0, sigma_eff, 1.0)
    dse_dsi = np.where(sigma_eff != 0, sigma_int / safe_se, 0.0)
    dse_dsn = np.where(sigma_eff != 0, sigma_inst / safe_se, 0.0)
    dsn_dc = (instfwhm_val + instfwhm_deriv_val * center) / (C_KMS * SIGMA2FWHM)

    dsi_dp = (fwhm / C_KMS * d_center_d_pos) / SIGMA2FWHM
    dsi_do = (fwhm / C_KMS * d_center_d_off) / SIGMA2FWHM
    dsi_df = (center / C_KMS) / SIGMA2FWHM

    dse_dp = dse_dsi * dsi_dp + dse_dsn * dsn_dc * d_center_d_pos
    dse_do = dse_dsi * dsi_do + dse_dsn * dsn_dc * d_center_d_off
    dse_df = dse_dsi * dsi_df

    d_pos = d_center * d_center_d_pos + d_sigma_eff * dse_dp
    d_off = d_center * d_center_d_off + d_sigma_eff * dse_do
    d_fwhm = d_sigma_eff * dse_df

    # Redshift derivatives
    d_center_d_z = sign * center / (1.0 + redshift)
    dsi_dz = sign * sigma_int / (1.0 + redshift)
    dsn_dz = dsn_dc * d_center_d_z
    dse_dz = dse_dsi * dsi_dz + dse_dsn * dsn_dz

    d_ampeff_dz = -amp_eff / (1.0 + redshift)
    d_z = d_amp_eff * d_ampeff_dz + d_center * d_center_d_z + d_sigma_eff * dse_dz

    return val, d_amp, d_pos, d_off, d_fwhm, d_z


def voigt_velocity_deriv(x, pos, amplitude, weight, offset, fwhm_G, fwhm_L,
                         redshift, instfwhm_val, instfwhm_deriv_val, domain_family='wavelength'):
    """Full chain-rule derivatives for a Voigt line in velocity parameterisation.

    Returns ``(val, d_amp, d_pos, d_off, d_fwhm_G, d_fwhm_L, d_z)``.
    """
    center = observed_center(pos, offset, redshift, domain_family=domain_family)
    amp_eff, sigma_eff, gamma = voigt_profile_params(
        center, amplitude, weight, fwhm_G, fwhm_L, redshift, instfwhm_val)

    val, d_amp_eff, d_center, d_sigma_eff, d_gamma = voigt_deriv(
        x, amp_eff, center, sigma_eff, gamma)

    d_amp = d_amp_eff * weight / (1.0 + redshift)

    sign = 1.0 if domain_family == 'wavelength' else -1.0
    d_center_d_pos = (1.0 + redshift) * np.exp(offset / C_KMS) if domain_family == 'wavelength' else 1.0 / ((1.0 + redshift) * np.exp(offset / C_KMS))
    d_center_d_off = sign * center / C_KMS

    # Sigma effective Jacobian
    fwhm_G_x = fwhm_G / C_KMS * center
    instfwhm_x = instfwhm_val / C_KMS * center
    sigma_int = fwhm_G_x / SIGMA2FWHM
    sigma_inst = instfwhm_x / SIGMA2FWHM

    safe_se = np.where(sigma_eff != 0, sigma_eff, 1.0)
    dse_dsi = np.where(sigma_eff != 0, sigma_int / safe_se, 0.0)
    dse_dsn = np.where(sigma_eff != 0, sigma_inst / safe_se, 0.0)
    dsn_dc = (instfwhm_val + instfwhm_deriv_val * center) / (C_KMS * SIGMA2FWHM)

    dsi_dp = (fwhm_G / C_KMS * d_center_d_pos) / SIGMA2FWHM
    dsi_do = (fwhm_G / C_KMS * d_center_d_off) / SIGMA2FWHM
    dsi_dfG = (center / C_KMS) / SIGMA2FWHM

    dse_dp = dse_dsi * dsi_dp + dse_dsn * dsn_dc * d_center_d_pos
    dse_do = dse_dsi * dsi_do + dse_dsn * dsn_dc * d_center_d_off
    dse_dfG = dse_dsi * dsi_dfG

    # Gamma Jacobian
    dg_dp = (fwhm_L / C_KMS * d_center_d_pos) / 2.0
    dg_do = (fwhm_L / C_KMS * d_center_d_off) / 2.0
    dg_dfL = (center / C_KMS) / 2.0

    d_pos = d_center * d_center_d_pos + d_sigma_eff * dse_dp + d_gamma * dg_dp
    d_off = d_center * d_center_d_off + d_sigma_eff * dse_do + d_gamma * dg_do
    d_fwhm_G = d_sigma_eff * dse_dfG
    d_fwhm_L = d_gamma * dg_dfL

    # Redshift derivatives
    d_center_d_z = sign * center / (1.0 + redshift)
    dsi_dz = sign * sigma_int / (1.0 + redshift)
    dsn_dz = dsn_dc * d_center_d_z
    dse_dz = dse_dsi * dsi_dz + dse_dsn * dsn_dz
    dg_dz = sign * gamma / (1.0 + redshift)

    d_ampeff_dz = -amp_eff / (1.0 + redshift)
    d_z = (d_amp_eff * d_ampeff_dz + d_center * d_center_d_z
           + d_sigma_eff * dse_dz + d_gamma * dg_dz)

    return val, d_amp, d_pos, d_off, d_fwhm_G, d_fwhm_L, d_z
