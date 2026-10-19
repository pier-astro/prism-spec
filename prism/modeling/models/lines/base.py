"""Shared base class for analytic line models."""
import numpy as np
from scipy.interpolate import CubicSpline

import astropy.units as u
from astropy.modeling import Fittable1DModel
from ...parameters import _Parameter as Parameter

from . import profiles


class LineModelBase(Fittable1DModel):
    """Common Astropy adapter for spectral units, instrumental widths and profiles.

    ``velocity_unit`` defaults to km/s and sets the numeric unit of ``offset``,
    ``fwhm``, ``fwhm_G``, and ``fwhm_L``, including their bounds and uncertainties.
    Quantity inputs are converted to that unit at construction. Omitted velocity
    parameters retain their physical class defaults. Construct a new model to
    change ``velocity_unit``; it is not a unit-conversion setter.

    Numeric ``instfwhm`` values are always in km/s, independently of
    ``velocity_unit``. A scalar velocity Quantity is also accepted. In a
    two-column instrumental table, column one uses the native spectral unit
    and column two uses km/s.
    """

    input_units_allow_dimensionless = {'x': True}
    input_units_equivalencies = {'x': u.spectral()}
    _profile_width = staticmethod(profiles._profile_fwhm)

    @staticmethod
    def _normalize_instfwhm(instfwhm):
        if isinstance(instfwhm, u.Quantity):
            if not instfwhm.isscalar:
                raise ValueError(
                    "instfwhm Quantity must be a scalar velocity; use a numeric "
                    "two-column table for native spectral positions and km/s widths.")
            instfwhm = float(instfwhm.to_value(u.km/u.s))
        if np.isscalar(instfwhm):
            value = float(instfwhm)
            if not np.isfinite(value) or value < 0:
                raise ValueError('Instrumental FWHM must be finite and non-negative.')
            return value, 'constant', value, None

        arr = np.asarray(instfwhm, dtype=float)
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(
                    "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array.")
            arr = arr.reshape(1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array.")

        if len(arr) == 0 or not np.isfinite(arr).all() or np.any(arr[:, 1] < 0):
            raise ValueError('Instrumental table must contain finite positions and non-negative FWHM values.')

        if arr.shape[0] == 1:
            return arr.copy(), 'constant', float(arr[0, 1]), None

        order = np.argsort(arr[:, 0])
        arr = arr[order]
        wl, fwhm = arr[:, 0], arr[:, 1]
        if np.any(np.diff(wl) <= 0):
            raise ValueError("instfwhm wavelengths must be strictly increasing.")

        spline = CubicSpline(wl, fwhm, extrapolate=True)
        return arr.copy(), 'spline', spline, spline.derivative()

    def __init__(self, *args, instfwhm=0.0, velocity_unit=u.km/u.s, **kwargs):
        raw, mode, value_or_spline, deriv = self._normalize_instfwhm(instfwhm)

        vel_unit = u.Unit(velocity_unit)
        if not vel_unit.is_equivalent(u.km/u.s):
            raise ValueError("velocity_unit must be a unit of velocity (e.g. km/s, m/s).")
        self.velocity_unit = vel_unit
        self._to_kms = vel_unit.to(u.km/u.s)

        self._instfwhm_raw = raw
        self._instfwhm_mode = mode
        if mode == 'constant':
            self._instfwhm_const = float(value_or_spline)
            self._instfwhm_spline = None
            self._instfwhm_spline_deriv = None
        else:
            self._instfwhm_const = None
            self._instfwhm_spline = value_or_spline
            self._instfwhm_spline_deriv = deriv
        self.instfwhm = raw

        # Keep physical defaults when only the selected velocity unit changes.
        param_names = type(self).param_names
        for pname in param_names[len(args):]:
            param = getattr(type(self), pname)
            if param.unit is not None and param.unit.is_equivalent(u.km/u.s):
                kwargs.setdefault(pname, param.default * param.unit)
        if 'bounds' in kwargs:
            kwargs['bounds'] = dict(kwargs['bounds'])
            for pname, limits in kwargs['bounds'].items():
                if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    kwargs['bounds'][pname] = tuple(
                        limit.to_value(vel_unit) if isinstance(limit, u.Quantity) else limit
                        for limit in limits)

        # Normalize velocity quantities once, before repeated numerical calls.
        coerced_args = [
            self._coerce_init_parameter(pname, val) for pname, val in zip(param_names, args)
        ]
        coerced_kwargs = {
            key: self._coerce_init_parameter(key, val) if key in param_names else val
            for key, val in kwargs.items()
        }
        super().__init__(*coerced_args, **coerced_kwargs)
        # Cache the position unit now, while it is still intact.
        # During fitting Astropy replaces parameter values with raw floats,
        # stripping instance-level units; reading from the cache keeps
        # domain_family and input_units stable throughout the fit loop.
        # Group models set _position_unit before calling this __init__, so
        # we only write it here for single-line models (no pre-existing value).
        if not hasattr(self, '_position_unit') or self._position_unit is None:
            pos_param = getattr(self, 'position', None)
            if pos_param is not None:
                raw_unit = getattr(pos_param, 'unit', None)
                self._position_unit = raw_unit if (raw_unit and raw_unit != u.dimensionless_unscaled) else None

    @property
    def domain_family(self):
        """Kinematic family inferred from the physical type of the position unit.

        The reference unit is recorded from the ``position`` parameter for a
        single line, or from the position column when a group is built. The
        inferred family is reused until that reference unit changes.

        Returns
        -------
        str
            ``'wavelength'`` if the position unit measures length, or
            ``'linear'`` if it measures frequency, energy, or temperature.

        Raises
        ------
        ValueError
            If no unit can be determined.
        """
        # Both single-line and group models cache the resolved spectral unit as
        # _position_unit — set in __init__ (single) or from_templates (group).
        pos_unit = self._position_unit
        cached = self.__dict__.get('_domain_cache')
        if cached is not None and cached[0] is pos_unit:
            return cached[1]

        if pos_unit is None or pos_unit == u.dimensionless_unscaled:
            raise ValueError(
                "Cannot infer domain_family: no spectral unit found on the position "
                "parameter or line table. Provide positions with an explicit unit "
                "(e.g. position=5000*u.AA)."
            )
        ptype = pos_unit.physical_type
        if ptype == 'length':
            family = 'wavelength'
        elif ptype in ('frequency', 'energy', 'temperature'):
            family = 'linear'
        else:
            raise ValueError(
                f"Cannot infer domain_family from position unit '{pos_unit}' "
                f"(physical_type='{ptype}'). Use a spectral unit such as u.AA, u.nm, "
                "u.Hz, u.eV, or u.K."
            )
        # Astropy units are immutable; a different reference invalidates the cache.
        self._domain_cache = (pos_unit, family)
        return family

    @property
    def input_units(self):
        """Map the first model input to the position unit for Astropy unit handling.

        When this property is defined, Astropy automatically intercepts any
        `~astropy.units.Quantity` passed to ``model(x)`` and converts it to
        ``self.position.unit`` before forwarding the bare numeric array to
        :meth:`evaluate`. This removes the need for manual unit stripping inside
        ``evaluate`` and enables transparent cross-unit evaluation, e.g. passing
        ``x`` in nm when the model is defined in Å.

        Returns
        -------
        dict or None
            ``{input_name: position_unit}`` if the position parameter carries
            a unit; ``None`` otherwise (so unitless models still accept plain
            floats without Astropy raising an error).
        """
        pos_unit = self._position_unit
        if pos_unit is None or pos_unit == u.dimensionless_unscaled:
            return None
        return {self.inputs[0]: pos_unit}

    @staticmethod
    def _as_value(value, unit=None):
        if isinstance(value, u.Quantity):
            raw = value.to_value(unit) if unit is not None else value.value
        else:
            raw = value
        arr = np.asarray(raw)
        if arr.size != 1:
            raise ValueError("Line-model parameters must be scalar values.")
        return float(arr.reshape(-1)[0])
        
    def _as_kms(self, velocity_value):
        """Converts the flexible velocity input into km/s for internal math."""
        if (isinstance(velocity_value, u.Quantity)
                and velocity_value.unit is not self.velocity_unit):
            return self._as_value(velocity_value, u.km/u.s)
        return self._as_value(velocity_value) * self._to_kms

    def _coerce_init_parameter(self, name, value):
        """Normalize velocities; wrap other numbers in their declared unit."""
        param = getattr(type(self), name, None)
        unit = getattr(param, 'unit', None)
        if unit is None:
            return value
        if unit.is_equivalent(u.km/u.s):
            unit = self.velocity_unit
            if isinstance(value, Parameter):
                value = self._parameter_payload(value)
            if isinstance(value, u.Quantity):
                return value.to(unit)
        if isinstance(value, (u.Quantity, Parameter)):
            return value
        return value * unit

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        mapping = {}
        output_unit = outputs_unit[self.outputs[0]]
        for pname in self.param_names:
            param = getattr(self, pname)
            if pname == 'amplitude' or pname.startswith('amp_'):
                mapping[pname] = output_unit
                continue
            unit = getattr(param, 'unit', None)
            if unit is not None:
                mapping[pname] = unit
        return mapping

    def _model_output_unit(self):
        """Retrieve the output unit implied by the amplitude parameter(s)."""
        for pname in self.param_names:
            if pname == 'amplitude' or pname.startswith('amp_'):
                return getattr(getattr(self, pname), 'unit', None)
        return None

    @property
    def return_units(self):
        output_unit = self._model_output_unit()
        if output_unit is not None:
            return {self.outputs[0]: output_unit}
        return None

    @staticmethod
    def _parameter_payload(param):
        return param.quantity if getattr(param, 'unit', None) is not None else param.value

    def _coerce_domain_axis(self, x):
        if isinstance(x, u.Quantity):
            x = x.to_value(self._position_unit, equivalencies=u.spectral())
        return np.asarray(x, dtype=float)

    def _default_domain_grid(self, center_x, fwhm_x, window=None, num=4096):
        if window is None:
            if not np.isfinite(fwhm_x) or fwhm_x <= 0:
                raise ValueError('A zero-width profile requires an explicit integration grid or window.')
            hw = 6.0 * fwhm_x
            low, high = center_x - hw, center_x + hw
        else:
            low, high = window
            if isinstance(low, u.Quantity):
                low = low.to_value(self._position_unit, equivalencies=u.spectral())
            if isinstance(high, u.Quantity):
                high = high.to_value(self._position_unit, equivalencies=u.spectral())
            low, high = float(low), float(high)
        return np.linspace(low, high, int(num), dtype=float)

    def instfwhm_val(self, center):
        if self._instfwhm_mode == 'constant':
            return np.asarray(center, dtype=float) * 0.0 + self._instfwhm_const
        return self._instfwhm_spline(center)

    def instfwhm_deriv(self, center):
        if self._instfwhm_mode == 'constant':
            return np.asarray(center, dtype=float) * 0.0
        return self._instfwhm_spline_deriv(center)

    def _shared_values(self, values=None):
        """Return scalar shared parameters, converting only velocities to km/s.

        ``values`` follows ``_shared_params`` order. When omitted, read the
        current model parameters; this path is also used by post-fit routines.
        """
        if values is None:
            values = (getattr(self, name).value for name in self._shared_params)
        return [self._as_kms(value) if name in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L')
                else self._as_value(value)
                for name, value in zip(self._shared_params, values)]

    def _single_profile_args(self, pos, amp, weight, *shared):
        """Map native position and km/s kinematics to peak, center and widths."""
        center = profiles.observed_center(
            pos, shared[0], shared[-1], domain_family=self.domain_family)
        values = self._profile_params(
            center, amp, weight, *shared[1:], self.instfwhm_val(center))
        return (values[0], center, *values[1:])

    def _profile_derivatives(self, x, pos, amp, weight, *shared):
        """Return value and derivatives in amplitude, position, shared order."""
        center = profiles.observed_center(
            pos, shared[0], shared[-1], domain_family=self.domain_family)
        return self._profile_deriv(
            x, pos, amp, weight, *shared,
            self.instfwhm_val(center), self.instfwhm_deriv(center),
            domain_family=self.domain_family)

    def _single_profile_deriv(self, x, pos, amp, weight, *shared):
        result = self._profile_derivatives(x, pos, amp, weight, *shared)
        return result[0], result[1], *result[3:]  # rest positions are tabulated

    def _profile_integral(self, pos, amp, weight, *shared):
        """Full-profile integral in amplitude times native spectral units."""
        args = self._single_profile_args(pos, amp, weight, *shared)
        value = self._profile_flux(args[0], *args[2:])
        return value*self._position_unit if isinstance(value, u.Quantity) else value
