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

from ...metrics import Metric
from . import profiles
from . import tables as _tables
from .base import LineModelBase
from .tables import (
    _clean_name, _make_unique,
    _has_param_std, _get_param_limits,
    c_kms,
)


class LineGroupBase(LineModelBase):
    """Base class for template-driven multi-line models.

    Notes
    -----
    Concrete subclasses are built from line tables whose rows define the physical
    transitions belonging to each template. Lines sharing the same template name
    are tied through one amplitude parameter, while kinematic parameters such as
    ``offset`` and width are shared across the whole group. This is the natural
    parameterisation for narrow-line regions, doublets, and multiplets where
    relative positions are fixed by atomic physics and relative weights are either
    tabulated or user supplied.

    Main constructors
    -----------------
    ``from_csv``
        Read one or more line tables from disk.
    ``from_arrays``
        Build a group directly from arrays of names, positions, and weights.
    ``from_templates``
        Build from an already normalized Astropy table.
    """

    @classmethod
    def from_csv(cls, csv_files, name=None, dirpath=None, bounds=None,
                 amplitude=None, instfwhm=0.0,
                 position_unit=None, **init_kwargs):
        """Build a line-group model from one or more ECSV/CSV line-list files.

        Parameters
        ----------
        csv_files : str or list of str
            File name(s) relative to *dirpath* (or absolute paths).  ECSV
            files carry their own unit metadata; plain CSV files require
            ``position_unit`` to be supplied.
        name : str, optional
            Label attached to the returned model instance.
        dirpath : str, optional
            Directory to search when *csv_files* are relative names.
            Defaults to the built-in packaged line-list directory.
        bounds : dict, optional
            Parameter bounds forwarded to the model constructor.
            Use ``{'amplitude': (0, None)}`` to constrain all template
            amplitudes simultaneously.
        amplitude : float, optional
            Initial value for every template amplitude parameter.
        instfwhm : float or array-like, optional
            Instrumental FWHM in the same velocity unit as the model
            (default 0 — no instrumental broadening).
        position_unit : `~astropy.units.Unit`, optional
            Spectral unit to assign to positions read from unitless CSV files.
            Not needed for ECSV files that already carry unit metadata.
        **init_kwargs
            Additional keyword arguments forwarded verbatim to the model
            ``__init__`` (e.g. ``fwhm``, ``redshift``, ``velocity_unit``).

        Returns
        -------
        model : instance of *cls*
            A fully initialised line-group model whose
            :attr:`domain_family` is inferred automatically from the
            spectral unit of the loaded positions.

        Raises
        ------
        ValueError
            If a plain CSV file is passed without ``position_unit``.
        ValueError
            If no lines survive the global wavelength-range filter after
            loading.  Call :func:`~prism.modeling.models.lines.base.set_wavelength_range`
            to expand the filter window.

        Examples
        --------
        Load an ECSV file that already stores positions in Angstrom:

        >>> model = GaussianLines.from_csv(
        ...     'optical.ecsv',
        ...     fwhm=300.0,          # km/s (default velocity_unit)
        ...     redshift=0.5,
        ... )

        Load a unit-less CSV file by supplying the unit explicitly:

        >>> model = GaussianLines.from_csv(
        ...     'optical.csv',
        ...     position_unit=u.AA,
        ...     fwhm=300.0,
        ... )

        Load X-ray lines stored in eV:

        >>> model = GaussianLines.from_csv(
        ...     'xray.ecsv',         # positions tagged as eV in the file
        ...     fwhm=500.0,
        ... )
        >>> model.domain_family      # inferred from eV unit
        'linear'
        """
        if not isinstance(csv_files, (list, tuple)):
            csv_files = [csv_files]
        if dirpath is None:
            dirpath = _tables.linetable_path

        tables = []
        for f in csv_files:
            path = f if os.path.isabs(f) else os.path.join(dirpath, f)
            tables.append(_tables.read_linetable(path, position_unit=position_unit))

        linetable = _tables.stack_linetables(tables, position_unit=position_unit)
        return cls.from_templates(
            linetable, name=name, bounds=bounds, amplitude=amplitude,
            instfwhm=instfwhm,
            position_unit=None, **init_kwargs)

    @classmethod
    def from_arrays(cls, names, pos, weights=None, name=None, bounds=None,
                    amplitude=None, instfwhm=0.0,
                    position_unit=None, **init_kwargs):
        """Build a line-group model directly from Python arrays.

        Parameters
        ----------
        names : array-like of str
            Line (or template) names.  Duplicate names create a single
            template whose constituent lines are summed with their
            respective weights.
        pos : array-like or `~astropy.units.Quantity`
            Rest-frame line positions.  Pass a bare numeric array together
            with *position_unit*, **or** pass an Astropy Quantity which
            already carries its spectral unit — but not both.
        weights : array-like of float, optional
            Relative weights for each line within a template.  Defaults
            to 1.0 for every line.
        name : str, optional
            Label for the returned model instance.
        bounds : dict, optional
            Parameter bounds forwarded to the model constructor.
        amplitude : float, optional
            Shared initial value for all template amplitude parameters.
        instfwhm : float or array-like, optional
            Instrumental FWHM in velocity units (default 0).
        position_unit : `~astropy.units.Unit`, optional
            Spectral unit to assign to *pos* when it is a plain numeric
            array.  **Required** when *pos* carries no unit; omit when
            *pos* is already an Astropy Quantity.
        **init_kwargs
            Forwarded to the model ``__init__`` (e.g. ``fwhm``,
            ``redshift``, ``velocity_unit``).

        Returns
        -------
        model : instance of *cls*
            A fully initialised line-group model.  :attr:`domain_family`
            is inferred automatically from the spectral unit.

        Raises
        ------
        ValueError
            If *pos* is a plain array and *position_unit* is not supplied.
        ValueError
            If no lines survive the global wavelength-range filter.

        Examples
        --------
        Optical lines in Angstrom — pass ``position_unit`` explicitly:

        >>> model = GaussianLines.from_arrays(
        ...     names=['Hb', 'OIII4959', 'OIII5007'],
        ...     pos=[4861.33, 4958.92, 5006.84],
        ...     position_unit=u.AA,
        ...     fwhm=200.0,       # km/s
        ...     redshift=0.1,
        ... )
        >>> model.domain_family
        'wavelength'

        UV lines as an Astropy Quantity (no ``position_unit`` needed):

        >>> pos_uv = np.array([1548.2, 1550.8]) * u.AA
        >>> model = GaussianLines.from_arrays(
        ...     names=['CIV1548', 'CIV1551'],
        ...     pos=pos_uv,
        ...     fwhm=300.0,
        ... )

        X-ray lines in eV — domain inferred automatically:

        >>> pos_ev = np.array([6.40, 6.97]) * u.keV
        >>> model = GaussianLines.from_arrays(
        ...     names=['Fe_Ka', 'Fe_Kb'],
        ...     pos=pos_ev,
        ...     fwhm=500.0,
        ...     velocity_unit=u.km / u.s,
        ... )
        >>> model.domain_family
        'linear'
        """
        names = np.atleast_1d(names)
        pos = np.atleast_1d(pos)
        if weights is None:
            weights = np.ones_like(pos)
            names = _make_unique(names)
        else:
            weights = np.atleast_1d(weights)

        # Require an explicit unit when pos carries none — no silent AA default.
        if not isinstance(pos, u.Quantity) and position_unit is None:
            raise ValueError(
                "from_arrays requires an explicit position_unit when 'pos' is a "
                "plain array (e.g. position_unit=u.AA). Alternatively, pass 'pos' "
                "as an astropy Quantity (e.g. pos * u.AA)."
            )

        # If pos is a Quantity honour its own unit; position_unit is then ignored.
        _norm_unit = None if isinstance(pos, u.Quantity) else position_unit
        linetable = _tables.normalize_linetable(
            {'name': names, 'position': pos, 'weight': weights},
            position_unit=_norm_unit,
        )
        # Pass position_unit=None so from_templates trusts the already-normalised
        # linetable unit rather than re-coercing it.
        return cls.from_templates(
            linetable, name=name, bounds=bounds, amplitude=amplitude,
            instfwhm=instfwhm,
            position_unit=None, **init_kwargs)

    @classmethod
    def from_templates(cls, linetable, name=None, bounds=None, amplitude=None,
                       instfwhm=0.0,
                       position_unit=None, **init_kwargs):
        """Build a line-group model from an Astropy `~astropy.table.QTable`.

        This is the lowest-level constructor; :meth:`from_csv` and
        :meth:`from_arrays` both delegate to it.  Call it directly when
        you already have a ``QTable`` with ``name``, ``position``, and
        ``weight`` columns.

        Parameters
        ----------
        linetable : `~astropy.table.QTable` or dict
            Table with at minimum ``'name'``, ``'position'``, and
            ``'weight'`` columns.  The ``'position'`` column must be an
            Astropy `~astropy.units.Quantity` (with a spectral unit) or
            the table must be accompanied by *position_unit*.
            Rows sharing the same ``'name'`` are grouped into a single
            template and evaluated as a weighted sum.
        name : str, optional
            Label for the returned model instance.
        bounds : dict, optional
            Parameter bounds.  The special key ``'amplitude'`` applies
            to every template amplitude::

                bounds={'amplitude': (0, None), 'fwhm': (10, 2000)}

        amplitude : float, optional
            Shared initial value for all template amplitude parameters.
        instfwhm : float or array-like, optional
            Instrumental FWHM in velocity units (default 0).
        position_unit : `~astropy.units.Unit`, optional
            Override or assign the spectral unit of the position column.
            Leave ``None`` (the default) to trust the unit already
            present in *linetable* — this is the normal path when called
            from :meth:`from_arrays` or :meth:`from_csv`.
        **init_kwargs
            Forwarded to the model ``__init__`` (e.g. ``fwhm``,
            ``redshift``, ``velocity_unit``).

        Returns
        -------
        model : instance of *cls*
            A dynamically-created subclass of *cls* with one amplitude
            parameter per template plus the shared kinematic parameters
            (``offset``, ``fwhm``, ``redshift``, etc.).
            :attr:`domain_family` is inferred automatically from the
            spectral unit of the position column.

        Raises
        ------
        ValueError
            If no lines fall inside the global wavelength-range filter.
            Call :func:`~prism.modeling.models.lines.base.set_wavelength_range`
            to expand the window.

        Notes
        -----
        The ``'position'`` column is stored internally in Angstrom for
        kinematic evaluation, but the original spectral unit is preserved
        as ``_position_unit`` and used by :attr:`domain_family` and
        :attr:`input_units`.

        Examples
        --------
        Build from a QTable with Angstrom positions:

        >>> from astropy.table import QTable
        >>> import astropy.units as u
        >>> lines = QTable()
        >>> lines['name']     = ['Ha', 'NII6548', 'NII6583']
        >>> lines['position'] = [6562.8, 6548.0, 6583.0] * u.AA
        >>> lines['weight']   = [1.0, 0.34, 1.0]
        >>> model = GaussianLines.from_templates(
        ...     lines,
        ...     fwhm=200.0,
        ...     redshift=0.3,
        ... )
        >>> model.domain_family
        'wavelength'

        Build from a QTable with eV positions (e.g. X-ray):

        >>> lines['position'] = [6.40, 6.70, 6.97] * u.keV
        >>> model = GaussianLines.from_templates(lines, fwhm=500.0)
        >>> model.domain_family
        'linear'
        """
        linetable = _tables.normalize_linetable(
            linetable,
            position_unit=position_unit,
        )
        # Capture the spectral unit before any conversion.
        _resolved_position_unit = linetable['position'].unit
        position = linetable['position']
        # Express the global wavelength bounds in the position's native unit so
        # the comparison stays in that unit without an intermediate AA conversion.
        bound1 = (_tables._wmin * u.AA).to(_resolved_position_unit, equivalencies=u.spectral())
        bound2 = (_tables._wmax * u.AA).to(_resolved_position_unit, equivalencies=u.spectral())
        lo, hi = (min(bound1, bound2), max(bound1, bound2))
        linetable = linetable[(position >= lo) & (position <= hi)]
        if len(linetable) == 0:
            raise ValueError(
                f"No lines found in the range [{lo:.6g}, {hi:.6g}]. "
                f"Call set_wavelength_range() to expand the global filter."
            )

        templates = np.asarray(list(dict.fromkeys(np.asarray(linetable['name'], dtype=str))))
        _raw_param_names = [f"amp_{_clean_name(tmpl)}" for tmpl in templates]
        param_names = _make_unique(_raw_param_names)
        n_templates = len(templates)

        _tmpl_positions = []
        _tmpl_weights = []
        for tmpl in templates:
            mask = np.asarray(linetable['name'], dtype=str) == tmpl
            table_tmpl = linetable[mask]
            _tmpl_positions.append(
                table_tmpl['position'].to(_resolved_position_unit, equivalencies=u.spectral()).value)
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
            x_native = self._coerce_domain_axis(x)
            is_scalar = np.ndim(x) == 0
            x_native = np.atleast_1d(x_native)
            
            # Extract template amplitudes
            amplitudes = [self._as_value(a) for a in args[:n_templates]]
            
            # Extract shared kinematic/width parameters
            shared = args[n_templates:]
            shared_values = []
            for idx, value in enumerate(shared):
                pname = list(self._shared_params.keys())[idx]
                if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    shared_values.append(self._as_kms(self._as_value(value)))
                else:
                    shared_values.append(self._as_value(value))

            total = np.zeros_like(x_native, dtype=float)
            
            # Loop over each template and evaluate its constituent lines
            for i in range(n_templates):
                for j in range(len(_tmpl_positions[i])):
                    profile_args = self._single_profile_args(
                        _tmpl_positions[i][j], amplitudes[i],
                        _tmpl_weights[i][j], *shared_values)
                    total += cls._profile_func(x_native, *profile_args)
            
            return total[0] if is_scalar else total

        def fit_deriv(self, x, *args):
            x_native = self._coerce_domain_axis(x)
            x_native = np.atleast_1d(x_native)
            
            # Extract template amplitudes
            amplitudes = [self._as_value(a) for a in args[:n_templates]]
            
            # Extract shared kinematic/width parameters
            shared = args[n_templates:]
            
            shared_values = []
            for idx, value in enumerate(shared):
                pname = list(self._shared_params.keys())[idx]
                if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    shared_values.append(self._as_kms(self._as_value(value)))
                else:
                    shared_values.append(self._as_value(value))
            
            n_params = len(args)
            n_x = len(x_native)
            grad = np.zeros((n_params, n_x))
            d_shared = [np.zeros(n_x) for _ in shared_values]
            
            # Accumulate derivatives for each template and shared parameters
            for i in range(n_templates):
                for j in range(len(_tmpl_positions[i])):
                    derivs = self._single_profile_deriv(
                        x_native, _tmpl_positions[i][j], amplitudes[i],
                        _tmpl_weights[i][j], *shared_values)
                    grad[i] += derivs[1]
                    for k, d in enumerate(derivs[2:]):
                        d_shared[k] += d
                        
            # Map shared parameter derivatives back to the gradient array
            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d
                
            return grad

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            'n_inputs': 1,
            'n_outputs': 1,
            '_separable_linear_params': tuple(param_names),
            '_linetable': linetable,
            '_templates': templates,
            '_n_templates': n_templates,
            '_param_names_list': param_names,
            '_tmpl_positions': _tmpl_positions,
            '_tmpl_weights': _tmpl_weights,
            '_position_unit': _resolved_position_unit,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(
            bounds=param_bounds,
            instfwhm=instfwhm,
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
        shared_values = []
        for pname in self._shared_params.keys():
            if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                shared_values.append(self._as_kms(getattr(self, pname).value))
            else:
                shared_values.append(getattr(self, pname).value)
        rows = []
        for tmpl, pname in zip(self._templates, self._param_names_list):
            amp = getattr(self, pname)
            idx = self._param_names_list.index(pname)
            positions = self._tmpl_positions[idx]
            weights = self._tmpl_weights[idx]
            amp_payload = self._parameter_payload(amp)

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
                'unit': None,
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
            shared_values = []
            for pname in self._shared_params.keys():
                if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    shared_values.append(self._as_kms(getattr(self, pname).value))
                else:
                    shared_values.append(getattr(self, pname).value)
            redshift = shared_values[-1]
            flux_df = self.flux
            rows = []
            for idx, (tmpl, pname) in enumerate(
                    zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude_param = getattr(self, pname)
                amplitude = self._parameter_payload(amplitude_param)
                center_rest = float(np.mean(positions))
                center_obs = profiles.observed_center(center_rest, shared_values[0], redshift, domain_family=self.domain_family)
                peak = 0.0
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(
                        pos, amplitude, weight, *shared_values)
                    peak += float(self._profile_func(np.array([center_obs]), *args).ravel()[0])
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
                    })
                else:
                    rows.append({'value': float('nan'), 'std': float('nan'),
                                 'lolim': float('nan'), 'uplim': float('nan')})
            result = self._metric_frame(rows, self._templates)
            self.ew = result
            return result

        if method == 'constant-continuum':
            flux_df = self.flux
            rows = []
            for tmpl, pname in zip(self._templates, self._param_names_list):
                amp = getattr(self, pname)
                continuum_val = float(continuum.value) if isinstance(continuum, u.Quantity) else float(continuum)
                if not np.isfinite(continuum_val) or continuum_val == 0.0:
                    rows.append({'value': float('nan'), 'std': float('nan'),
                                 'lolim': float('nan'), 'uplim': float('nan')})
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
                })
            result = self._metric_frame(rows, self._templates)
            self.ew = result
            return result

        elif method == 'integration':
            shared_values = []
            for pname in self._shared_params.keys():
                if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
                    shared_values.append(self._as_kms(getattr(self, pname).value))
                else:
                    shared_values.append(getattr(self, pname).value)
            rows = []
            for idx, (tmpl, pname) in enumerate(
                    zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude_param = getattr(self, pname)
                amplitude = self._parameter_payload(amplitude_param)

                if x is None:
                    centers = [profiles.observed_center(pos, shared_values[0], shared_values[-1], domain_family=self.domain_family)
                               for pos in positions]
                    mean_c = float(np.mean(centers))
                    if window is None:
                        sp = self._shared_params
                        fwhm_key = 'fwhm' if 'fwhm' in sp else list(sp.keys())[0]
                        shared_fwhm = shared_values[
                            list(sp.keys()).index(fwhm_key)] if sp else 10.0
                        fwhm_x = (shared_fwhm / c_kms * mean_c) if sp else 10.0
                        hw = 6.0 * max(fwhm_x, 0.1)
                        _window = (mean_c - hw, mean_c + hw)
                    else:
                        _window = window
                    x_arr = self._default_domain_grid(mean_c, max(fwhm_x, 0.1), window=_window, num=num)
                else:
                    x_arr = self._coerce_domain_axis(x)

                y_line = np.zeros_like(x_arr, dtype=float)
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(
                        pos, amplitude, weight, *shared_values)
                    y_line += self._profile_func(x_arr, *args)

                if callable(continuum):
                    y_cont = np.asarray(continuum(x_arr), dtype=float)
                else:
                    continuum_val = float(continuum.value) if isinstance(continuum, u.Quantity) else float(continuum)
                    y_cont = np.full_like(x_arr, continuum_val, dtype=float)
                safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
                ew_val = float(_trap(y_line / safe, x_arr))
                rows.append({'value': ew_val, 'std': float('nan'),
                             'lolim': float('nan'), 'uplim': float('nan')})

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
    """Gaussian template group with shared velocity offset and FWHM.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity, optional
        Default amplitude assigned to every template when building the model.
    offset : float or astropy.units.Quantity, optional
        Shared velocity offset for all lines. Default is ``0 km / s``.
    fwhm : float or astropy.units.Quantity, optional
        Shared intrinsic Gaussian width in velocity units. Default is
        ``1000 km / s``.
    redshift : float, optional
        Shared redshift. Default is ``0``.

    Examples
    --------
    >>> group = GaussianLines.from_arrays(
    ...     names=['Hb4861', '[OIII]5007'],
    ...     pos=[4861.333, 5006.803],
    ...     amplitude=1.0,
    ...     fwhm=300.0,
    ... )
    """
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.gaussian)

    def _single_profile_args(self, pos, amp, weight, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amp, weight, fwhm, redshift, self.instfwhm_val(center))
        return (amp_eff, center, sigma_eff)

    def _single_profile_deriv(self, x, pos, amp, weight, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        val, d_amp, _, d_off, d_fwhm, d_z = profiles.gaussian_velocity_deriv(
            x, pos, amp, weight, offset, fwhm, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center), domain_family=self.domain_family)
        return val, d_amp, d_off, d_fwhm, d_z

    def _calc_flux(self, pos, amp, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_eff = profiles.gaussian_profile_params(
            center, amp, 1.0, fwhm, redshift, self.instfwhm_val(center))
        return profiles.gaussian_flux(amp_eff, sigma_eff)


class LorentzianLines(LineGroupBase):
    """Lorentzian template group with shared velocity offset and FWHM.

    Notes
    -----
    The intrinsic line core is Lorentzian for every member of the group. If an
    instrumental Gaussian broadening term is present, each member is evaluated as
    a Voigt profile after combining the intrinsic Lorentzian with the line-spread
    function.
    """
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)

    def _single_profile_args(self, pos, amp, weight, offset, fwhm, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amp, weight, fwhm, redshift, self.instfwhm_val(center))
        return (amp_eff, center, sigma_inst, gamma)

    def _single_profile_deriv(self, x, pos, amp, weight, offset, fwhm, redshift):
        eps = 1e-6
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)

        def _eval(a, o, f, z):
            args = self._single_profile_args(pos, a, weight, o, f, z)
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
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_inst, gamma = profiles.lorentzian_profile_params(
            center, amp, 1.0, fwhm, redshift, self.instfwhm_val(center))
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amp_eff, gamma)
        return profiles.voigt_flux(amp_eff, sigma_inst, gamma)


class VoigtLines(LineGroupBase):
    """Voigt template group with shared Gaussian and Lorentzian widths.

    Notes
    -----
    This class keeps separate Gaussian and Lorentzian width parameters while still
    tying amplitudes within each named template. It is useful when a line complex
    requires explicit wing control without giving up the compact multi-line Prism
    parameterisation.
    """
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0,
                      'fwhm_L': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm_G': u.km / u.s,
                     'fwhm_L': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)

    def _single_profile_args(self, pos, amp, weight, offset,
                             fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amp, weight, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        return (amp_eff, center, sigma_eff, gamma)

    def _single_profile_deriv(self, x, pos, amp, weight, offset,
                              fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        val, d_amp, _, d_off, d_fG, d_fL, d_z = profiles.voigt_velocity_deriv(
            x, pos, amp, weight, offset, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center), self.instfwhm_deriv(center), domain_family=self.domain_family)
        return val, d_amp, d_off, d_fG, d_fL, d_z

    def _calc_flux(self, pos, amp, offset, fwhm_G, fwhm_L, redshift):
        center = profiles.observed_center(pos, offset, redshift, domain_family=self.domain_family)
        amp_eff, sigma_eff, gamma = profiles.voigt_profile_params(
            center, amp, 1.0, fwhm_G, fwhm_L, redshift,
            self.instfwhm_val(center))
        return profiles.voigt_flux(amp_eff, sigma_eff, gamma)
