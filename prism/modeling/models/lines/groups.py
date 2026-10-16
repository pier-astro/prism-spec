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
from . import profiles, _measurements
from . import tables as _tables
from .base import LineModelBase
from .tables import (
    _clean_name, _make_unique,
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
        # Normalize once at the shared constructor boundary.
        return cls.from_templates(
            {'name': names, 'position': pos, 'weight': weights},
            name=name, bounds=bounds, amplitude=amplitude,
            instfwhm=instfwhm,
            position_unit=_norm_unit, **init_kwargs)

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
        linetable : Astropy table, pandas.DataFrame, or dict
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
        Positions and numerical evaluation retain the resolved native spectral
        unit. That unit is recorded as ``_position_unit`` and used by
        :attr:`domain_family` and :attr:`input_units`. The factory prepares the
        table metadata and Astropy parameter descriptors; evaluation and
        derivatives are ordinary methods shared by all groups.

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

        model_class = type(cls.__name__, (cls,), {
            **params,
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

    def evaluate(self, x, *args):
        """Sum the prepared template members on the native spectral grid."""
        x_native = self._coerce_domain_axis(x)
        is_scalar = np.ndim(x) == 0
        x_native = np.atleast_1d(x_native)

        # Extract template amplitudes
        amplitudes = [self._as_value(a) for a in args[:self._n_templates]]

        shared_values = self._shared_values(args[self._n_templates:])

        total = np.zeros_like(x_native, dtype=float)

        # Loop over each template and evaluate its constituent lines
        for amplitude, positions, weights in zip(
                amplitudes, self._tmpl_positions, self._tmpl_weights):
            for pos, weight in zip(positions, weights):
                profile_args = self._single_profile_args(
                    pos, amplitude, weight, *shared_values)
                total += self._profile_func(x_native, *profile_args)

        return total[0] if is_scalar else total

    def fit_deriv(self, x, *args):
        """Accumulate parameter-major derivatives in the model parameter units."""
        x_native = self._coerce_domain_axis(x)
        x_native = np.atleast_1d(x_native)

        # Extract template amplitudes
        amplitudes = [self._as_value(a) for a in args[:self._n_templates]]

        shared_values = self._shared_values(args[self._n_templates:])

        n_params = len(args)
        n_x = len(x_native)
        grad = np.zeros((n_params, n_x))
        d_shared = [np.zeros(n_x) for _ in shared_values]

        # Accumulate derivatives for each template and shared parameters
        for i, (amplitude, positions, weights) in enumerate(zip(
                amplitudes, self._tmpl_positions, self._tmpl_weights)):
            for pos, weight in zip(positions, weights):
                derivs = self._single_profile_deriv(
                    x_native, pos, amplitude, weight, *shared_values)
                grad[i] += derivs[1]
                for k, d in enumerate(derivs[2:]):
                    d_shared[k] += d

        # Map shared parameter derivatives back to the gradient array
        for j, (pname, d) in enumerate(zip(self._shared_params, d_shared)):
            if pname in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L') and self._to_kms != 1.0:
                d = d * self._to_kms
            grad[self._n_templates + j] = d

        return grad

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_frame(rows, index):
        frame = pd.DataFrame(rows, index=list(index), columns=['value', 'std', 'lolim', 'uplim', 'unit'])
        frame['unit'] = [None if unit in (None, '') else u.Unit(unit) for unit in frame['unit']]
        return frame

    def _calc_flux(self, pos, amp, *shared):
        return self._profile_integral(pos, amp, 1.0, *shared)

    @property
    def lines(self):
        """QTable containing the physical lines in this model group."""
        return self._linetable

    def _template_flux(self, amp_pname, positions, weights):
        amplitude = self._parameter_payload(getattr(self, amp_pname))
        shared = self._shared_values()
        return sum(self._calc_flux(pos, amplitude*weight, *shared)
                   for pos, weight in zip(positions, weights))

    @property
    def flux(self) -> pd.DataFrame:
        """Analytic template fluxes with local covariance or independent errors."""
        rows = []
        for name, positions, weights in zip(
                self._param_names_list, self._tmpl_positions, self._tmpl_weights):
            def evaluate():
                return self._template_flux(name, positions, weights)
            governing = [name]+[p for p in self._shared_params if p not in ('offset', 'redshift')]
            result = _measurements._flux(
                self, evaluate, [name]+list(self._shared_params), governing)
            rows.append(vars(result))
        return self._metric_frame(rows, self._templates)

    def eqw(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096):
        """Template EWs from a fixed continuum or finite-grid integration.

        An omitted continuum uses each template's own profile at the mean
        rest position shifted by the shared kinematics. A fixed continuum is
        treated as exact; integration does not estimate errors. Results are in
        the native spectral unit when amplitudes carry units.
        """
        if method not in ('constant-continuum', 'integration'):
            raise ValueError("method must be 'constant-continuum' or 'integration'.")
        shared = self._shared_values()
        flux = self.flux if continuum is None or method == 'constant-continuum' else None
        rows = []
        for i, name in enumerate(self._param_names_list):
            positions, weights = self._tmpl_positions[i], self._tmpl_weights[i]
            amplitude = getattr(self, name)
            centers = [profiles.observed_center(pos, shared[0], shared[-1],
                       domain_family=self.domain_family) for pos in positions]
            center = float(np.mean(centers))
            local_continuum = continuum
            if local_continuum is None:
                local_continuum = self._template_profile(name, positions, weights, np.array([center]))[0]
                # Preserve the established self-profile normalization for positive peaks.
                if not np.isfinite(local_continuum) or local_continuum <= 0:
                    local_continuum = np.nan
            if continuum is None or method == 'constant-continuum':
                result = _measurements._constant_eqw(
                    Metric(**flux.iloc[i].to_dict()), local_continuum, self._position_unit)
            else:
                if x is None:
                    widths = []
                    for pos in positions:
                        args = self._single_profile_args(pos, 1.0, 1.0, *shared)
                        widths.append(float(self._profile_width(*args[2:])))
                    bounds = window
                    if bounds is None:
                        if not any(width > 0 for width in widths):
                            raise ValueError('A zero-width template requires an explicit integration grid or window.')
                        bounds = (min(c-6*w for c, w in zip(centers, widths)),
                                  max(c+6*w for c, w in zip(centers, widths)))
                    grid = self._default_domain_grid(center, max(widths), bounds, num)
                else:
                    grid = self._coerce_domain_axis(x)
                line = self._template_profile(name, positions, weights, grid)
                result = _measurements._integrated_eqw(
                    grid, line, continuum, self._position_unit, amplitude.unit)
            rows.append(vars(result))
        result = self._metric_frame(rows, self._templates)
        self.ew = result
        return result

    def _template_profile(self, name, positions, weights, x):
        """Numeric template profile in the amplitude parameter's unit."""
        amplitude = getattr(self, name).value
        shared = self._shared_values()
        result = np.zeros_like(x, dtype=float)
        for pos, weight in zip(positions, weights):
            args = self._single_profile_args(pos, amplitude, weight, *shared)
            result += self._profile_func(x, *args)
        return result


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
    ...     pos=[4861.333, 5006.803] * u.AA,
    ...     amplitude=1.0,
    ...     fwhm=300.0,
    ... )
    """
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.gaussian)
    _profile_params = staticmethod(profiles.gaussian_profile_params)
    _profile_deriv = staticmethod(profiles.gaussian_velocity_deriv)
    _profile_flux = staticmethod(profiles.gaussian_flux)


class LorentzianLines(LineGroupBase):
    """Lorentzian template group with shared velocity offset and FWHM.

    Notes
    -----
    The intrinsic line core is Lorentzian for every member of the group. If an
    instrumental Gaussian broadening term is present, each member is evaluated as
    a Voigt profile after combining the intrinsic Lorentzian with the line-spread
    function. Profile derivatives use the analytic Voigt Jacobian with zero intrinsic
    Gaussian width; each member retains its relative template weight.
    """
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km / u.s, 'fwhm': u.km / u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)
    _profile_params = staticmethod(profiles.lorentzian_profile_params)
    _profile_deriv = staticmethod(profiles._lorentzian_velocity_deriv)
    _profile_flux = staticmethod(profiles.voigt_flux)


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
    _profile_params = staticmethod(profiles.voigt_profile_params)
    _profile_deriv = staticmethod(profiles.voigt_velocity_deriv)
    _profile_flux = staticmethod(profiles.voigt_flux)
