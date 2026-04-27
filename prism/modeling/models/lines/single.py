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
    """Single Gaussian emission or absorption line.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest wavelength of the transition. Default is ``5000 AA`` and is fixed by
        default.
    offset : float or astropy.units.Quantity, optional
        Velocity shift relative to the rest wavelength. Default is ``0 km / s``.
    fwhm : float or astropy.units.Quantity, optional
        Intrinsic Gaussian full width at half maximum in velocity units. Default
        is ``1000 km / s``.
    redshift : float, optional
        Source redshift applied on top of ``position``. Default is ``0`` and is
        fixed by default.

    Notes
    -----
    Prism parameterises line centroids and widths in velocity space while keeping
    the physical transition anchored at a rest wavelength. Instrumental Gaussian
    broadening is folded into the intrinsic profile before evaluation, so the
    analytic flux and Jacobian remain available for fitting and uncertainty
    propagation.

    Examples
    --------
    >>> line = GaussianLine(amplitude=5.0, position=5007.0, fwhm=400.0)
    >>> y = line(x)
    >>> line.flux.value
    """
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
    """Single Lorentzian line with optional instrumental Gaussian broadening.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest wavelength of the transition. Default is ``5000 AA``.
    offset : float or astropy.units.Quantity, optional
        Velocity shift relative to the rest wavelength. Default is ``0 km / s``.
    fwhm : float or astropy.units.Quantity, optional
        Intrinsic Lorentzian full width at half maximum in velocity units.
        Default is ``1000 km / s``.
    redshift : float, optional
        Source redshift. Default is ``0``.

    Notes
    -----
    With zero instrumental broadening this class evaluates a pure Lorentzian.
    When ``instfwhm`` is non-zero the observed profile becomes Voigt-like because
    the intrinsic Lorentzian is convolved with a Gaussian line-spread function.
    """
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
    """Single Voigt line with independent Gaussian and Lorentzian widths.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest wavelength of the transition. Default is ``5000 AA``.
    offset : float or astropy.units.Quantity, optional
        Velocity shift relative to the rest wavelength. Default is ``0 km / s``.
    fwhm_G : float or astropy.units.Quantity, optional
        Gaussian contribution to the intrinsic width in velocity units. Default is
        ``1000 km / s``.
    fwhm_L : float or astropy.units.Quantity, optional
        Lorentzian contribution to the intrinsic width in velocity units. Default
        is ``1000 km / s``.
    redshift : float, optional
        Source redshift. Default is ``0``.

    Notes
    -----
    ``VoigtLine`` is the most flexible analytic single-line profile in Prism. It
    is useful when the astrophysical broadening cannot be approximated by a purely
    Gaussian or purely Lorentzian kernel and when the separate core and wing widths
    need to remain explicit fit parameters.
    """
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
