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
    position = Parameter(default=5000.0, fixed=True, unit=u.AA)
    offset = Parameter(default=0.0, unit=u.km / u.s)
    fwhm = Parameter(default=1000.0, unit=u.km / u.s)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm')

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        x_native, jacobian, is_scalar = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm = self._as_value(fwhm, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, amplitude_unit = self._native_amplitude_value(amplitude, center)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amplitude, 1.0, fwhm, redshift, self.instfwhm_val(center))
        values = profiles.gaussian(x_native, amp_eff, center, sigma_eff)
        return self._finalize_output(values, jacobian, is_scalar, amplitude_unit)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        x_native, jacobian, _ = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm = self._as_value(fwhm, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, _ = self._native_amplitude_value(amplitude, center, for_deriv=True)
        _, d_amp, d_pos, d_off, d_fwhm, d_z = profiles.gaussian_velocity_deriv(
            x_native, position, amplitude, 1.0, offset, fwhm, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return [self._apply_domain_jacobian(d, jacobian)
                for d in [d_amp, d_pos, d_off, d_fwhm, d_z]]

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amplitude, _ = self._native_amplitude_value(self._parameter_payload(self.amplitude), center)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amplitude, 1.0, self.fwhm.value,
            self.redshift.value, self.instfwhm_val(center))
        return profiles.gaussian_flux(amp_eff, sigma_eff)


class LorentzianLine(LineModelBase):
    """Lorentzian line model (Voigt profile when instrument broadening is present)."""
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True, unit=u.AA)
    offset = Parameter(default=0.0, unit=u.km / u.s)
    fwhm = Parameter(default=1000.0, unit=u.km / u.s)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm')

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        x_native, jacobian, is_scalar = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm = self._as_value(fwhm, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, amplitude_unit = self._native_amplitude_value(amplitude, center)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amplitude, 1.0, fwhm, redshift, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            values = profiles.lorentzian(x_native, amp_eff, center, gamma)
        else:
            values = profiles.voigt(x_native, amp_eff, center, sigma_inst, gamma)
        return self._finalize_output(values, jacobian, is_scalar, amplitude_unit)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        x_native, jacobian, _ = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm = self._as_value(fwhm, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, _ = self._native_amplitude_value(amplitude, center, for_deriv=True)
        params = [amplitude, position, offset, fwhm, redshift]
        base = np.asarray(self.evaluate(x_native, *params), dtype=float)
        derivs = []
        eps = 1e-6
        for i, pval in enumerate(params):
            delta = abs(pval) * eps if pval != 0 else eps
            shifted = list(params)
            shifted[i] = pval + delta
            derivs.append((np.asarray(self.evaluate(x_native, *shifted), dtype=float) - base) / delta)
        return [self._apply_domain_jacobian(d, jacobian) for d in derivs]

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amplitude, _ = self._native_amplitude_value(self._parameter_payload(self.amplitude), center)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amplitude, 1.0, self.fwhm.value,
            self.redshift.value, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amp_eff, gamma)
        return profiles.voigt_flux(amp_eff, sigma_inst, gamma)


class VoigtLine(LineModelBase):
    """Voigt line model with independent Gaussian and Lorentzian widths."""
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True, unit=u.AA)
    offset = Parameter(default=0.0, unit=u.km / u.s)
    fwhm_G = Parameter(default=1000.0, unit=u.km / u.s)
    fwhm_L = Parameter(default=1000.0, unit=u.km / u.s)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm_G', 'fwhm_L')

    def evaluate(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        x_native, jacobian, is_scalar = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm_G = self._as_value(fwhm_G, u.km / u.s)
        fwhm_L = self._as_value(fwhm_L, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, amplitude_unit = self._native_amplitude_value(amplitude, center)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amplitude, 1.0, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        values = profiles.voigt(x_native, amp_eff, center, sigma_eff, gamma)
        return self._finalize_output(values, jacobian, is_scalar, amplitude_unit)

    def fit_deriv(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        x_native, jacobian, _ = self._prepare_input_grid(x)
        position = self._as_value(position, u.AA)
        offset = self._as_value(offset, u.km / u.s)
        fwhm_G = self._as_value(fwhm_G, u.km / u.s)
        fwhm_L = self._as_value(fwhm_L, u.km / u.s)
        redshift = self._as_value(redshift)
        center = profiles.observed_center(position, offset, redshift)
        amplitude, _ = self._native_amplitude_value(amplitude, center, for_deriv=True)
        _, d_amp, d_pos, d_off, d_fG, d_fL, d_z = profiles.voigt_velocity_deriv(
            x_native, position, amplitude, 1.0, offset, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return [self._apply_domain_jacobian(d, jacobian)
                for d in [d_amp, d_pos, d_off, d_fG, d_fL, d_z]]

    def _calc_flux(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        amplitude, _ = self._native_amplitude_value(self._parameter_payload(self.amplitude), center)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amplitude, 1.0, self.fwhm_G.value,
            self.fwhm_L.value, self.redshift.value, self.instfwhm_val(center))
        return profiles.voigt_flux(amp_eff, sigma_eff, gamma)
