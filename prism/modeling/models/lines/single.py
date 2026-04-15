"""
prism.modeling.models.lines.single — single-line spectral models.

``GaussianLine``, ``LorentzianLine``, ``VoigtLine``.
"""
import numpy as np
import astropy.units as u
from astropy.modeling import Parameter

from . import profiles
from .base import LineModelBase


class GaussianLine(LineModelBase):
    """Gaussian line model parameterised by rest position, velocity offset, and FWHM."""
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm = Parameter(default=1000.0)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm')
    _parameter_units = {'position': u.AA, 'offset': u.km / u.s, 'fwhm': u.km / u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        center = profiles.observed_center(position, offset, redshift)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amplitude, 1.0, fwhm, redshift, self.instfwhm_val(center))
        return profiles.gaussian(x, amp_eff, center, sigma_eff)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        center = profiles.observed_center(position, offset, redshift)
        _, d_amp, d_pos, d_off, d_fwhm, d_z = profiles.gaussian_velocity_deriv(
            x, position, amplitude, 1.0, offset, fwhm, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return [d_amp, d_pos, d_off, d_fwhm, d_z]

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, self.amplitude.value, 1.0, self.fwhm.value,
            self.redshift.value, self.instfwhm_val(center))
        return profiles.gaussian_flux(amp_eff, sigma_eff)


class LorentzianLine(LineModelBase):
    """Lorentzian line model (Voigt profile when instrument broadening is present)."""
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm = Parameter(default=1000.0)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm')
    _parameter_units = {'position': u.AA, 'offset': u.km / u.s, 'fwhm': u.km / u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        center = profiles.observed_center(position, offset, redshift)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amplitude, 1.0, fwhm, redshift, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            return profiles.lorentzian(x, amp_eff, center, gamma)
        return profiles.voigt(x, amp_eff, center, sigma_inst, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        params = [amplitude, position, offset, fwhm, redshift]
        base = self.evaluate(x, *params)
        derivs = []
        eps = 1e-6
        for i, pval in enumerate(params):
            delta = abs(pval) * eps if pval != 0 else eps
            shifted = list(params)
            shifted[i] = pval + delta
            derivs.append((self.evaluate(x, *shifted) - base) / delta)
        return derivs

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, self.amplitude.value, 1.0, self.fwhm.value,
            self.redshift.value, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amp_eff, gamma)
        return profiles.voigt_flux(amp_eff, sigma_inst, gamma)


class VoigtLine(LineModelBase):
    """Voigt line model with independent Gaussian and Lorentzian widths."""
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm_G = Parameter(default=1000.0)
    fwhm_L = Parameter(default=1000.0)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm_G', 'fwhm_L')
    _parameter_units = {'position': u.AA, 'offset': u.km / u.s,
                        'fwhm_G': u.km / u.s, 'fwhm_L': u.km / u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(position, offset, redshift)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amplitude, 1.0, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        return profiles.voigt(x, amp_eff, center, sigma_eff, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(position, offset, redshift)
        _, d_amp, d_pos, d_off, d_fG, d_fL, d_z = profiles.voigt_velocity_deriv(
            x, position, amplitude, 1.0, offset, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return [d_amp, d_pos, d_off, d_fG, d_fL, d_z]

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, self.amplitude.value, 1.0, self.fwhm_G.value,
            self.fwhm_L.value, self.redshift.value, self.instfwhm_val(center))
        return profiles.voigt_flux(amp_eff, sigma_eff, gamma)
