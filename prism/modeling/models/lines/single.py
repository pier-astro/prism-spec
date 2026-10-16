"""
prism.modeling.models.lines.single — single-line spectral models.

``GaussianLine``, ``LorentzianLine``, ``VoigtLine``.
"""
import numpy as np
import astropy.units as u
from astropy.modeling import Parameter

from . import profiles
from .base import LineModelBase
from ...metrics import Metric
from . import _measurements


class _SingleLineBase(LineModelBase):
    """Astropy adapter for one free-position line and its analytic measures."""

    def evaluate(self, x, amplitude, position, *shared):
        """Evaluate one profile after normalizing coordinates and kinematics."""
        x_native = np.atleast_1d(self._coerce_domain_axis(x))
        args = self._single_profile_args(
            self._as_value(position), self._as_value(amplitude), 1.0,
            *self._shared_values(shared))
        values = self._profile_func(x_native, *args)
        return values[0] if np.ndim(x) == 0 else values

    def fit_deriv(self, x, amplitude, position, *shared):
        """Return the parameter-major Jacobian in the model parameter units."""
        x_native = np.atleast_1d(self._coerce_domain_axis(x))
        result = self._profile_derivatives(
            x_native, self._as_value(position), self._as_value(amplitude), 1.0,
            *self._shared_values(shared))
        derivatives = list(result[1:])
        if self._to_kms != 1.0:
            for i, name in enumerate(self.param_names):
                if name in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    derivatives[i] = derivatives[i] * self._to_kms
        return derivatives

    def _calc_flux(self):
        return self._profile_integral(
            self.position.value, self._parameter_payload(self.amplitude), 1.0,
            *self._shared_values())

    @property
    def flux(self) -> Metric:
        """Analytic integrated flux with local covariance or independent errors."""
        return _measurements._flux(
            self, self._calc_flux, self.param_names, self._flux_governing_params)

    def eqw(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096):
        """Equivalent width using a fixed continuum or finite-grid integration.

        An omitted continuum uses the model value at its shifted center.
        Continuum uncertainty is not propagated. Integration has no error
        estimate and uses the explicit grid, or a window of six effective FWHM.
        """
        if method not in ('constant-continuum', 'integration'):
            raise ValueError("method must be 'constant-continuum' or 'integration'.")
        if continuum is None:
            center, _ = self._infer_center_fwhm_x()
            continuum = self(center)
        if method == 'constant-continuum':
            result = _measurements._constant_eqw(self.flux, continuum, self._position_unit)
        else:
            center, width = self._infer_center_fwhm_x()
            grid = self._default_domain_grid(center, width, window, num) if x is None else self._coerce_domain_axis(x)
            line = self(grid)
            line = line.value if isinstance(line, u.Quantity) else line
            result = _measurements._integrated_eqw(
                grid, line, continuum, self._position_unit, self.amplitude.unit)
        self.ew = result
        return result

    def _infer_center_fwhm_x(self):
        args = self._single_profile_args(self.position.value, 1.0, 1.0, *self._shared_values())
        return args[1], float(self._profile_width(*args[2:]))


class GaussianLine(_SingleLineBase):
    """Single Gaussian emission or absorption line.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest spectral position of the transition. Default is ``5000 AA`` and is fixed by
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
    the physical transition anchored at a rest spectral position. Instrumental Gaussian
    broadening is folded into the intrinsic profile before evaluation, so the
    analytic flux and Jacobian remain available for fitting and uncertainty
    propagation.
    The ``amplitude`` parameter is also marked as a separable linear parameter,
    so ``prism.modeling.fitting.SeparableTRF`` can solve amplitudes exactly
    while optimizing only the non-linear shape parameters outside.

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
    _separable_linear_params = ('amplitude',)
    _profile_func = staticmethod(profiles.gaussian)
    _profile_params = staticmethod(profiles.gaussian_profile_params)
    _profile_deriv = staticmethod(profiles.gaussian_velocity_deriv)
    _profile_flux = staticmethod(profiles.gaussian_flux)
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}


class LorentzianLine(_SingleLineBase):
    """Single Lorentzian line with optional instrumental Gaussian broadening.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest spectral position of the transition. Default is ``5000 AA``.
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
    The ``amplitude`` parameter is treated as a separable linear parameter by
    ``prism.modeling.fitting.SeparableTRF``.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True, unit=u.AA)
    offset = Parameter(default=0.0, unit=u.km / u.s)
    fwhm = Parameter(default=1000.0, unit=u.km / u.s)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm')
    _separable_linear_params = ('amplitude',)
    _profile_func = staticmethod(profiles.voigt)
    _profile_params = staticmethod(profiles.lorentzian_profile_params)
    _profile_deriv = staticmethod(profiles._lorentzian_velocity_deriv)
    _profile_flux = staticmethod(profiles.voigt_flux)
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}


class VoigtLine(_SingleLineBase):
    """Single Voigt line with independent Gaussian and Lorentzian widths.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Peak amplitude in the chosen output domain. Default is ``1.0``.
    position : float or astropy.units.Quantity, optional
        Rest spectral position of the transition. Default is ``5000 AA``.
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
    Its ``amplitude`` remains separable and is therefore solved in the inner
    linear step by ``prism.modeling.fitting.SeparableTRF``.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True, unit=u.AA)
    offset = Parameter(default=0.0, unit=u.km / u.s)
    fwhm_G = Parameter(default=1000.0, unit=u.km / u.s)
    fwhm_L = Parameter(default=1000.0, unit=u.km / u.s)
    redshift = Parameter(default=0.0, fixed=True)
    _flux_governing_params = ('amplitude', 'fwhm_G', 'fwhm_L')
    _separable_linear_params = ('amplitude',)
    _profile_func = staticmethod(profiles.voigt)
    _profile_params = staticmethod(profiles.voigt_profile_params)
    _profile_deriv = staticmethod(profiles.voigt_velocity_deriv)
    _profile_flux = staticmethod(profiles.voigt_flux)
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0, 'fwhm_L': 1000.0, 'redshift': 0.0}
