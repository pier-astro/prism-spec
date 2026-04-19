"""
prism.modeling.models.lines.groups — multi-line (template-group) models.

``GaussianLines``, ``LorentzianLines``, ``VoigtLines``
built dynamically from CSV tables via ``LineGroupBase``.
"""
import os
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.modeling import Parameter

from . import profiles
from . import base as _base
from .base import (
    LineModelBase, Metric,
    _clean_name, _make_unique,
    _has_param_std, _get_param_limits,
    c_kms,
)


class LineGroupBase(LineModelBase):
    """Base class for models defined from CSVs with columns: name, pos, weight."""

    @classmethod
    def from_csv(cls, csv_files, name=None, dirpath=None, bounds=None,
                 amplitude=None, instfwhm=0.0, domain='wavelength',
                 medium=None, position_unit=None, **init_kwargs):
        if not isinstance(csv_files, (list, tuple)):
            csv_files = [csv_files]
        if dirpath is None:
            dirpath = _base.csv_lines_path

        tables = []
        for f in csv_files:
            path = f if os.path.isabs(f) else os.path.join(dirpath, f)
            tables.append(_base.read_linetable(path, medium=medium, position_unit=position_unit))

        linetable = _base.stack_linetables(tables, medium=medium, position_unit=position_unit)
        return cls.from_templates(
            linetable, name=name, bounds=bounds, amplitude=amplitude,
            instfwhm=instfwhm, domain=domain, medium=medium,
            position_unit=position_unit, **init_kwargs)

    @classmethod
    def from_arrays(cls, names, pos, weights=None, name=None, bounds=None,
                    amplitude=None, instfwhm=0.0, domain='wavelength',
                    medium=None, position_unit=None, **init_kwargs):
        names = np.atleast_1d(names)
        pos = np.atleast_1d(pos)
        if weights is None:
            weights = np.ones_like(pos)
            names = _make_unique(names)
        else:
            weights = np.atleast_1d(weights)
        linetable = _base.normalize_linetable(
            {'name': names, 'position': pos, 'weight': weights},
            medium=medium,
            position_unit=position_unit,
        )
        return cls.from_templates(
            linetable, name=name, bounds=bounds, amplitude=amplitude,
            instfwhm=instfwhm, domain=domain, medium=medium,
            position_unit=position_unit, **init_kwargs)

    @classmethod
    def from_templates(cls, linetable, name=None, bounds=None, amplitude=None,
                       instfwhm=0.0, domain='wavelength',
                       medium=None, position_unit=None, **init_kwargs):
        linetable = _base.normalize_linetable(
            linetable,
            medium=medium,
            position_unit=position_unit,
        )
        position = linetable['position']
        linetable = linetable[
            (position >= _base._wmin * u.AA) & (position <= _base._wmax * u.AA)]
        if len(linetable) == 0:
            raise ValueError(f"No lines found in the range [{_base._wmin}, {_base._wmax}]")

        templates = np.asarray(list(dict.fromkeys(np.asarray(linetable['name'], dtype=str))))
        _raw_param_names = [f"amp_{_clean_name(tmpl)}" for tmpl in templates]
        param_names = _make_unique(_raw_param_names)
        n_templates = len(templates)

        _tmpl_positions = []
        _tmpl_weights = []
        for tmpl in templates:
            mask = np.asarray(linetable['name'], dtype=str) == tmpl
            table_tmpl = linetable[mask]
            _tmpl_positions.append(table_tmpl['position'].to_value(u.AA))
            _tmpl_weights.append(np.asarray(table_tmpl['weight'], dtype=float))

        for pname in param_names:
            if amplitude is not None and pname not in init_kwargs:
                init_kwargs[pname] = amplitude

        bounds = dict(bounds) if bounds else {}
        amp_bounds = bounds.pop('amplitude', None)
        param_bounds = {}
        for pname in param_names:
            if amp_bounds is not None:
                param_bounds[pname] = amp_bounds
            if pname in bounds:
                param_bounds[pname] = bounds[pname]
        for k, v in bounds.items():
            if k not in param_bounds:
                param_bounds[k] = v

        params = {pname: Parameter(default=1.0) for pname in param_names}
        for pname, default in cls._shared_params.items():
            is_fixed = True if pname == 'redshift' else False
            params[pname] = Parameter(
                default=default, fixed=is_fixed, unit=cls._shared_units.get(pname))

        def evaluate(self, x, *args):
            x_native, jacobian, is_scalar = self._prepare_input_grid(x)
            amplitudes = args[:n_templates]
            amplitude_unit = self._common_output_unit(amplitudes)
            shared = args[n_templates:]
            shared_values = []
            for idx, value in enumerate(shared):
                pname = list(self._shared_params.keys())[idx]
                punit = cls._shared_units.get(pname)
                shared_values.append(self._as_value(value, punit))

            total = np.zeros_like(x_native, dtype=float)
            for i in range(n_templates):
                for j in range(len(_tmpl_positions[i])):
                    profile_args = self._single_profile_args(
                        _tmpl_positions[i][j], amplitudes[i],
                        _tmpl_weights[i][j], *shared_values, output_unit=amplitude_unit)
                    total += cls._profile_func(x_native, *profile_args)
            return self._finalize_output(total, jacobian, is_scalar, amplitude_unit)

        def fit_deriv(self, x, *args):
            x_native, jacobian, _ = self._prepare_input_grid(x)
            amplitudes = args[:n_templates]
            amplitude_unit = self._common_output_unit(amplitudes)
            shared = args[n_templates:]
            if amplitude_unit is not None and self.domain != 'wavelength':
                raise NotImplementedError(
                    "fit_deriv with quantity amplitudes is not supported for frequency or energy domains.")
            shared_values = []
            for idx, value in enumerate(shared):
                pname = list(self._shared_params.keys())[idx]
                punit = cls._shared_units.get(pname)
                shared_values.append(self._as_value(value, punit))
            n_params = len(args)
            n_x = len(x_native)
            grad = np.zeros((n_params, n_x))
            d_shared = [np.zeros(n_x) for _ in shared_values]
            for i in range(n_templates):
                for j in range(len(_tmpl_positions[i])):
                    derivs = self._single_profile_deriv(
                        x_native, _tmpl_positions[i][j], amplitudes[i],
                        _tmpl_weights[i][j], *shared_values, output_unit=amplitude_unit)
                    grad[i] += derivs[1]
                    for k, d in enumerate(derivs[2:]):
                        d_shared[k] += d
            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d
            return [self._apply_domain_jacobian(d, jacobian) for d in grad]

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            'n_inputs': 1,
            'n_outputs': 1,
            '_linetable': linetable,
            '_templates': templates,
            '_n_templates': n_templates,
            '_param_names_list': param_names,
            '_tmpl_positions': _tmpl_positions,
            '_tmpl_weights': _tmpl_weights,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(
            bounds=param_bounds,
            instfwhm=instfwhm,
            domain=domain,
            medium=linetable.meta.get('medium', medium),
            **init_kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lines(self):
        """QTable containing the physical lines in this model group."""
        return self._linetable

    @property
    def flux(self) -> pd.DataFrame:
        """Theoretical fluxes for all templates with uncertainties."""
        shared_values = [getattr(self, pname).value
                         for pname in self._shared_params.keys()]
        rows = []
        for tmpl, pname in zip(self._templates, self._param_names_list):
            amp = getattr(self, pname)
            idx = self._param_names_list.index(pname)
            positions = self._tmpl_positions[idx]
            weights = self._tmpl_weights[idx]
            amp_payload = self._parameter_payload(amp)
            flux_unit = self._flux_unit(getattr(amp, 'unit', None))

            total_flux = sum(
                self._calc_flux(pos, amp_payload * wt, *shared_values)
                for pos, wt in zip(positions, weights))

            flux_std = self._propagate_template_flux_std(
                pname, total_flux, positions, weights)
            flux_lo, flux_hi = self._propagate_template_flux_limits(
                pname, positions, weights)

            rows.append({
                'value': total_flux,
                'std': float(flux_std) if flux_std is not None else float('nan'),
                'lolim': float(flux_lo) if flux_lo is not None else float('nan'),
                'uplim': float(flux_hi) if flux_hi is not None else float('nan'),
                'unit': flux_unit,
            })
        return self._metric_frame(rows, self._templates)

    # ------------------------------------------------------------------
    # Equivalent width
    # ------------------------------------------------------------------

    def eqw(self, continuum=None, method='constant-continuum', x=None,
            window=None, num=4096) -> 'pd.DataFrame':
        """Compute and store the equivalent width for all templates."""
        try:
            _trap = np.trapezoid
        except AttributeError:
            _trap = np.trapz

        if continuum is None:
            shared_values = [getattr(self, pname).value
                             for pname in self._shared_params.keys()]
            redshift = shared_values[-1]
            flux_df = self.flux
            rows = []
            for idx, (tmpl, pname) in enumerate(
                    zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude_param = getattr(self, pname)
                amplitude = self._parameter_payload(amplitude_param)
                output_unit = getattr(amplitude_param, 'unit', None)
                center_aa = float(np.mean(positions)) * (1.0 + redshift)
                peak = 0.0
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(
                        pos, amplitude, weight, *shared_values, output_unit=output_unit)
                    peak += float(self._profile_func(np.array([center_aa]), *args).ravel()[0])
                peak *= self._center_jacobian(center_aa)
                if np.isfinite(peak) and peak > 0.0:
                    f_row = flux_df.loc[tmpl]
                    rows.append({
                        'value': float(f_row['value']) / peak,
                        'std': float(f_row['std']) / peak
                              if np.isfinite(f_row['std']) else float('nan'),
                        'lolim': float(f_row['lolim']) / peak
                                if np.isfinite(f_row['lolim']) else float('nan'),
                        'uplim': float(f_row['uplim']) / peak
                                if np.isfinite(f_row['uplim']) else float('nan'),
                        'unit': self._eqw_unit(output_unit),
                    })
                else:
                    rows.append({'value': float('nan'), 'std': float('nan'),
                                 'lolim': float('nan'), 'uplim': float('nan'),
                                 'unit': self._eqw_unit(output_unit)})
            result = self._metric_frame(rows, self._templates)
            self.ew = result
            return result

        if method == 'constant-continuum':
            flux_df = self.flux
            rows = []
            for tmpl, pname in zip(self._templates, self._param_names_list):
                amp = getattr(self, pname)
                output_unit = getattr(amp, 'unit', None)
                continuum_val = self._coerce_output_scalar(continuum, output_unit, name='continuum')
                if not np.isfinite(continuum_val) or continuum_val == 0.0:
                    rows.append({'value': float('nan'), 'std': float('nan'),
                                 'lolim': float('nan'), 'uplim': float('nan'),
                                 'unit': self._eqw_unit(output_unit)})
                    continue
                f_row = flux_df.loc[tmpl]
                rows.append({
                    'value': float(f_row['value']) / continuum_val,
                    'std': float(f_row['std']) / abs(continuum_val)
                           if np.isfinite(f_row['std']) else float('nan'),
                    'lolim': float(f_row['lolim']) / continuum_val
                             if np.isfinite(f_row['lolim']) else float('nan'),
                    'uplim': float(f_row['uplim']) / continuum_val
                             if np.isfinite(f_row['uplim']) else float('nan'),
                    'unit': self._eqw_unit(output_unit),
                })
            result = self._metric_frame(rows, self._templates)
            self.ew = result
            return result

        elif method == 'integration':
            shared_values = [getattr(self, pname).value
                             for pname in self._shared_params.keys()]
            rows = []
            for idx, (tmpl, pname) in enumerate(
                    zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude_param = getattr(self, pname)
                amplitude = self._parameter_payload(amplitude_param)
                output_unit = getattr(amplitude_param, 'unit', None)

                if x is None:
                    centers = [pos * (1.0 + shared_values[-1])
                               for pos in positions]
                    mean_c = float(np.mean(centers))
                    if window is None:
                        sp = self._shared_params
                        fwhm_key = 'fwhm' if 'fwhm' in sp else list(sp.keys())[0]
                        shared_fwhm = shared_values[
                            list(sp.keys()).index(fwhm_key)] if sp else 10.0
                        fwhm_aa = (shared_fwhm / c_kms * mean_c) if sp else 10.0
                        hw = 6.0 * max(fwhm_aa, 0.1)
                        _window = (mean_c - hw, mean_c + hw)
                    else:
                        _window = window
                    x_arr = self._default_domain_grid(mean_c, max(fwhm_aa, 0.1), window=_window, num=num)
                else:
                    x_arr = self._coerce_domain_axis(x)

                x_native = profiles.to_wavelength_values(x_arr, self.domain)
                jacobian = profiles.domain_jacobian(x_native, self.domain)
                y_line = np.zeros_like(x_arr, dtype=float)
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(
                        pos, amplitude, weight, *shared_values, output_unit=output_unit)
                    y_line += self._profile_func(x_native, *args) * jacobian

                if callable(continuum):
                    y_cont = self._coerce_output_array(continuum(x_arr), output_unit, name='continuum')
                else:
                    continuum_val = self._coerce_output_scalar(continuum, output_unit, name='continuum')
                    y_cont = np.full_like(x_arr, continuum_val, dtype=float)
                safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
                ew_val = float(_trap(y_line / safe, x_arr))
                rows.append({'value': ew_val, 'std': float('nan'),
                             'lolim': float('nan'), 'uplim': float('nan'),
                             'unit': self._eqw_unit(output_unit)})

            result = self._metric_frame(rows, self._templates)
            self.ew = result
            return result

        else:
            raise ValueError(
                "method must be 'constant-continuum' or 'integration'.")

    # ------------------------------------------------------------------
    # Flux propagation helpers
    # ------------------------------------------------------------------

    def _template_flux(self, amp_pname, positions, weights):
        shared = [getattr(self, sp).value for sp in self._shared_params.keys()]
        param = getattr(self, amp_pname)
        amp = self._parameter_payload(param)
        return sum(
            self._calc_flux(pos, amp * wt, *shared)
            for pos, wt in zip(positions, weights))

    def _propagate_template_flux_std(self, amp_pname, flux_val,
                                     positions, weights):
        cov = getattr(self, '_param_cov', None)
        if cov is not None:
            return self._propagate_template_flux_std_cov(
                amp_pname, flux_val, positions, weights, cov)

        eps = 1e-6
        var_flux = 0.0
        has_any = False
        relevant = [amp_pname] + list(self._shared_params.keys())
        for p_name in relevant:
            param = getattr(self, p_name)
            if not _has_param_std(param):
                continue
            has_any = True
            orig = param.value
            delta = abs(orig) * eps if orig != 0 else eps
            param.value = orig + delta
            df = (self._template_flux(amp_pname, positions, weights)
                  - flux_val) / delta
            param.value = orig
            var_flux += (df * param.std) ** 2
        return np.sqrt(var_flux) if has_any and var_flux > 0 else None

    def _propagate_template_flux_std_cov(self, amp_pname, flux_val,
                                         positions, weights, cov):
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
            jac[j] = (self._template_flux(
                amp_pname, positions, weights) - flux_val) / delta
            all_params[idx] = orig
        self.parameters = all_params
        var = float(jac @ cov @ jac)
        return np.sqrt(var) if var > 0 else None

    def _propagate_template_flux_limits(self, amp_pname, positions, weights):
        width_params = [sp for sp in self._shared_params
                        if sp not in ('offset', 'redshift')]
        amp_param = getattr(self, amp_pname)
        _, _, amp_has_lo, amp_has_hi = _get_param_limits(amp_param)
        if not amp_has_lo and not amp_has_hi:
            return None, None

        def _eval_at(overrides):
            orig = {}
            for p, v in overrides.items():
                orig[p] = getattr(self, p).value
                getattr(self, p).value = v
            f = self._template_flux(amp_pname, positions, weights)
            for p, v in orig.items():
                getattr(self, p).value = v
            return f

        def _compute(use_lo):
            has_amp = amp_has_lo if use_lo else amp_has_hi
            if not has_amp:
                return None
            overrides = {
                amp_pname: amp_param.lolim if use_lo else amp_param.uplim}
            for p in width_params:
                _, _, has_lo, has_hi = _get_param_limits(getattr(self, p))
                if use_lo and has_lo:
                    overrides[p] = getattr(self, p).lolim
                elif not use_lo and has_hi:
                    overrides[p] = getattr(self, p).uplim
            return _eval_at(overrides)

        return _compute(True), _compute(False)


# ======================================================================
# Concrete line-group classes
# ======================================================================

class GaussianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.gaussian)

    def _single_profile_args(self, pos, amp, weight, offset, fwhm, redshift, output_unit=None):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amp, weight, fwhm, redshift, self.instfwhm_val(center))
        return (amp_eff, center, sigma_eff)

    def _single_profile_deriv(self, x, pos, amp, weight, offset, fwhm, redshift, output_unit=None):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit, for_deriv=True)
        val, d_amp, _, d_off, d_fwhm, d_z = profiles.gaussian_velocity_deriv(
            x, pos, amp, weight, offset, fwhm, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return val, d_amp, d_off, d_fwhm, d_z

    def _calc_flux(self, pos, amp, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amp, 1.0, fwhm, redshift, self.instfwhm_val(center))
        return profiles.gaussian_flux(amp_eff, sigma_eff)


class LorentzianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)

    def _single_profile_args(self, pos, amp, weight, offset, fwhm, redshift, output_unit=None):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amp, weight, fwhm, redshift, self.instfwhm_val(center))
        return (amp_eff, center, sigma_inst, gamma)

    def _single_profile_deriv(self, x, pos, amp, weight, offset, fwhm, redshift, output_unit=None):
        eps = 1e-6
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit, for_deriv=True)

        def _eval(a, o, f, z):
            args = self._single_profile_args(pos, a, weight, o, f, z, output_unit=output_unit)
            return self._profile_func(x, *args)

        val = _eval(amp, offset, fwhm, redshift)
        d_amp_step = abs(amp) * eps if amp != 0 else eps
        d_off_step = abs(offset) * eps if offset != 0 else eps
        d_fwhm_step = abs(fwhm) * eps if fwhm != 0 else eps
        d_z_step = abs(redshift) * eps if redshift != 0 else eps

        d_amp_true = (_eval(amp + d_amp_step, offset, fwhm, redshift)
                      - val) / d_amp_step
        d_offset = (_eval(amp, offset + d_off_step, fwhm, redshift)
                    - val) / d_off_step
        d_fwhm = (_eval(amp, offset, fwhm + d_fwhm_step, redshift)
                  - val) / d_fwhm_step
        d_redshift = (_eval(amp, offset, fwhm, redshift + d_z_step)
                      - val) / d_z_step

        d_amp = d_amp_true / weight if weight != 0 else np.zeros_like(val)
        return val, d_amp, d_offset, d_fwhm, d_redshift

    def _calc_flux(self, pos, amp, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amp, 1.0, fwhm, redshift, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amp_eff, gamma)
        return profiles.voigt_flux(amp_eff, sigma_inst, gamma)


class VoigtLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0,
                      'fwhm_L': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm_G': u.km / u.s,
                     'fwhm_L': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)

    def _single_profile_args(self, pos, amp, weight, offset,
                             fwhm_G, fwhm_L, redshift, output_unit=None):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amp, weight, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        return (amp_eff, center, sigma_eff, gamma)

    def _single_profile_deriv(self, x, pos, amp, weight, offset,
                              fwhm_G, fwhm_L, redshift, output_unit=None):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center, output_unit=output_unit, for_deriv=True)
        val, d_amp, _, d_off, d_fG, d_fL, d_z = profiles.voigt_velocity_deriv(
            x, pos, amp, weight, offset, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center))
        return val, d_amp, d_off, d_fG, d_fL, d_z

    def _calc_flux(self, pos, amp, offset, fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(pos, offset, redshift)
        amp, _ = self._native_amplitude_value(amp, center)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amp, 1.0, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        return profiles.voigt_flux(amp_eff, sigma_eff, gamma)
