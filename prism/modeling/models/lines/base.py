"""Shared base class for analytic line models."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

from ...metrics import Metric
from . import profiles

from .tables import (
    c_kms,
    _get_param_limits,
    _has_param_std,
)

class LineModelBase(Fittable1DModel):
    """Base class for line models providing instfwhm handling and flux/EW/propagation."""

    input_units_allow_dimensionless = {'x': True}
    input_units_equivalencies = {'x': u.spectral()}

    @staticmethod
    def _normalize_instfwhm(instfwhm):
        if np.isscalar(instfwhm):
            value = float(instfwhm)
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

        if arr.shape[0] == 1:
            return arr.copy(), 'constant', float(arr[0, 1]), None

        order = np.argsort(arr[:, 0])
        arr = arr[order]
        wl, fwhm = arr[:, 0], arr[:, 1]
        if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(fwhm)):
            raise ValueError("instfwhm wavelength/FWHM values must be finite.")
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

        # Coerce plain scalar inputs to Quantities for parameters that define a default unit
        param_names = type(self).param_names
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

        For single-line models the unit comes from the ``position`` parameter.
        For group models (which have no ``position`` parameter) the unit is
        read from the ``_linetable`` position column, which always carries the
        unit the table was built with.

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

        if pos_unit is None or pos_unit == u.dimensionless_unscaled:
            raise ValueError(
                "Cannot infer domain_family: no spectral unit found on the position "
                "parameter or line table. Provide positions with an explicit unit "
                "(e.g. position=5000*u.AA)."
            )
        ptype = pos_unit.physical_type
        if ptype == 'length':
            return 'wavelength'
        if ptype in ('frequency', 'energy', 'temperature'):
            return 'linear'
        raise ValueError(
            f"Cannot infer domain_family from position unit '{pos_unit}' "
            f"(physical_type='{ptype}'). Use a spectral unit such as u.AA, u.nm, "
            "u.Hz, u.eV, or u.K."
        )

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
        if isinstance(velocity_value, u.Quantity):
            return velocity_value.to_value(u.km/u.s)
        return velocity_value * self._to_kms

    @classmethod
    def _coerce_init_parameter(cls, name, value):
        """Wrap plain numeric inputs in the parameter's default unit (if defined)."""
        param = getattr(cls, name, None)
        unit = getattr(param, 'unit', None)
        if unit is None or isinstance(value, (u.Quantity, Parameter)):
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
    def _output_unit_from_amplitude(amplitude):
        return amplitude.unit if isinstance(amplitude, u.Quantity) else None

    def _common_output_unit(self, amplitudes):
        output_unit = None
        for amplitude in amplitudes:
            unit = self._output_unit_from_amplitude(amplitude)
            if unit is None:
                continue
            if output_unit is None:
                output_unit = unit
                continue
            try:
                (1.0 * unit).to(output_unit)
            except Exception as exc:
                raise ValueError(
                    "All quantity amplitudes in a line group must share compatible output units.") from exc
        return output_unit

    @staticmethod
    def _parameter_payload(param):
        return param.quantity if getattr(param, 'unit', None) is not None else param.value

    def _coerce_domain_axis(self, x):
        return np.asarray(x, dtype=float)

    def _default_domain_grid(self, center_x, fwhm_x, window=None, num=4096):
        if window is None:
            hw = 6.0 * fwhm_x
            low, high = center_x - hw, center_x + hw
        else:
            low, high = window
            low = float(low.value) if isinstance(low, u.Quantity) else float(low)
            high = float(high.value) if isinstance(high, u.Quantity) else float(high)
        return np.linspace(low, high, int(num), dtype=float)

    @staticmethod
    def _metric_frame(rows, index):
        frame = pd.DataFrame(rows, index=list(index), columns=['value', 'std', 'lolim', 'uplim', 'unit'])
        frame['unit'] = [None if unit in (None, '') else u.Unit(unit) for unit in frame['unit']]
        return frame

    def instfwhm_val(self, center):
        if self._instfwhm_mode == 'constant':
            return np.asarray(center, dtype=float) * 0.0 + self._instfwhm_const
        return self._instfwhm_spline(center)

    def instfwhm_deriv(self, center):
        if self._instfwhm_mode == 'constant':
            return np.asarray(center, dtype=float) * 0.0
        return self._instfwhm_spline_deriv(center)

    _flux_governing_params = ('amplitude',)

    @property
    def flux(self) -> Metric:
        """Return a Metric with flux value, uncertainty, and confidence limits."""
        flux_val = self._calc_flux()
        flux_std = self._propagate_flux_std(flux_val)
        flux_lo, flux_hi = self._propagate_flux_limits()
        return Metric(
            value=float(flux_val),
            std=float(flux_std) if flux_std is not None else float('nan'),
            lolim=float(flux_lo) if flux_lo is not None else float('nan'),
            uplim=float(flux_hi) if flux_hi is not None else float('nan'),
        )

    def eqw(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096) -> Metric:
        """Compute the equivalent width of the line.

        Parameters
        ----------
        continuum : float, callable, or None
            Local continuum level.
        method : {'constant-continuum', 'integration'}
        x : array-like, optional
        window : (float, float), optional
        num : int

        Returns
        -------
        Metric
        """
        if continuum is None:
            center, _ = self._infer_center_fwhm_x()
            continuum = self(center)

        if method == 'constant-continuum':
            continuum_val = float(continuum.value) if isinstance(continuum, u.Quantity) else float(continuum)
            if not np.isfinite(continuum_val) or continuum_val == 0.0:
                result = Metric(value=float('nan'))
                self.ew = result
                return result
            f = self.flux
            ew_val = f.value / continuum_val
            ew_std = None if f.std is None else (
                float(f.std / abs(continuum_val)) if np.isfinite(f.std) else float('nan')
            )
            ew_lo = None if f.lolim is None else (
                float(f.lolim / continuum_val) if np.isfinite(f.lolim) else float('nan')
            )
            ew_hi = None if f.uplim is None else (
                float(f.uplim / continuum_val) if np.isfinite(f.uplim) else float('nan')
            )
            result = Metric(
                value=ew_val,
                std=ew_std,
                lolim=ew_lo,
                uplim=ew_hi,
            )

        elif method == 'integration':
            if x is None:
                center, fwhm_x = self._infer_center_fwhm_x()
                x_arr = self._default_domain_grid(center, fwhm_x, window=window, num=num)
            else:
                x_arr = self._coerce_domain_axis(x)
            y_line = np.asarray(self(x_arr), dtype=float)
            if callable(continuum):
                y_cont = np.asarray(continuum(x_arr), dtype=float)
            else:
                continuum_val = float(continuum.value) if isinstance(continuum, u.Quantity) else float(continuum)
                y_cont = np.full_like(x_arr, continuum_val, dtype=float)
            safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
            try:
                ew_val = float(np.trapezoid(y_line / safe, x_arr))
            except AttributeError:
                ew_val = float(np.trapz(y_line / safe, x_arr))
            result = Metric(value=ew_val)

        else:
            raise ValueError("method must be 'constant-continuum' or 'integration'.")

        self.ew = result
        return result

    def _infer_center_fwhm_x(self):
        center = profiles.observed_center(
            self.position.value, self._as_kms(self.offset.value), self.redshift.value, domain_family=self.domain_family)
        if hasattr(self, 'fwhm'):
            fwhm_x = self._as_kms(self.fwhm.value) / c_kms * center
        elif hasattr(self, 'fwhm_G'):
            fwhm_x = self._as_kms(self.fwhm_G.value) / c_kms * center
        else:
            fwhm_x = 10.0
        return center, max(fwhm_x, 0.1)

    def _flux_jacobian(self, flux_val):
        from astropy.modeling.fitting import model_to_fit_params
        _, fit_indices, _ = model_to_fit_params(self)
        eps = 1e-6
        jac = np.zeros(len(fit_indices))
        all_params = self.parameters
        for j, idx in enumerate(fit_indices):
            orig = float(all_params[idx])
            delta = abs(orig) * eps if orig != 0 else eps
            all_params[idx] = orig + delta
            self.parameters = all_params
            jac[j] = (self._calc_flux() - flux_val) / delta
            all_params[idx] = orig
        self.parameters = all_params
        return jac, fit_indices

    def _propagate_flux_std(self, flux_val) -> float | None:
        cov = getattr(self, '_param_cov', None)
        if cov is not None:
            jac, _ = self._flux_jacobian(flux_val)
            var = float(jac @ cov @ jac)
            return np.sqrt(var) if var > 0 else None

        eps = 1e-6
        var_flux = 0.0
        has_any = False
        for pname in self.param_names:
            param = getattr(self, pname)
            if not _has_param_std(param):
                continue
            has_any = True
            orig = param.value
            delta = abs(orig) * eps if orig != 0 else eps
            param.value = orig + delta
            df = (self._calc_flux() - flux_val) / delta
            param.value = orig
            var_flux += (df * param.std) ** 2
        return np.sqrt(var_flux) if has_any and var_flux > 0 else None

    def _propagate_flux_limits(self):
        governing = list(self._flux_governing_params)
        amp_pname = governing[0]
        amp_param = getattr(self, amp_pname)
        _, _, amp_has_lo, amp_has_hi = _get_param_limits(amp_param)
        if not amp_has_lo and not amp_has_hi:
            return None, None

        def _eval_at(overrides):
            orig = {}
            for p, v in overrides.items():
                orig[p] = getattr(self, p).value
                getattr(self, p).value = v
            f = self._calc_flux()
            for p, v in orig.items():
                getattr(self, p).value = v
            return f

        def _compute(use_lo):
            has_amp = amp_has_lo if use_lo else amp_has_hi
            if not has_amp:
                return None
            overrides = {amp_pname: amp_param.lolim if use_lo else amp_param.uplim}
            for p in governing[1:]:
                _, _, has_lo, has_hi = _get_param_limits(getattr(self, p))
                if use_lo and has_lo:
                    overrides[p] = getattr(self, p).lolim
                elif not use_lo and has_hi:
                    overrides[p] = getattr(self, p).uplim
            return _eval_at(overrides)

        return _compute(True), _compute(False)
