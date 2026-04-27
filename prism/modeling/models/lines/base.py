"""
prism.modeling.models.lines.base — base classes, Metric, and CSV utilities.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy.interpolate import CubicSpline

import astropy.units as u
import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter
from astropy.table import QTable, Table, vstack

from . import profiles
from ....utils.tools import air_to_vac, vac_to_air

c_kms = profiles.C_KMS
sigma2fwhm = profiles.SIGMA2FWHM
DEFAULT_LINETABLE_MEDIUM = 'air'
VALID_LINETABLE_MEDIA = {'air', 'vacuum'}
_default_medium = DEFAULT_LINETABLE_MEDIUM

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    """A single measured quantity with optional uncertainty bounds."""
    value: float
    std: float = float('nan')
    lolim: float = float('nan')
    uplim: float = float('nan')
    unit: object = None

    def __post_init__(self):
        if self.unit not in (None, ''):
            object.__setattr__(self, 'unit', u.Unit(self.unit))

    @property
    def quantity(self):
        return None if self.unit is None else self.value * self.unit

    @property
    def std_quantity(self):
        return None if self.unit is None or not np.isfinite(self.std) else self.std * self.unit

    @property
    def lolim_quantity(self):
        return None if self.unit is None or not np.isfinite(self.lolim) else self.lolim * self.unit

    @property
    def uplim_quantity(self):
        return None if self.unit is None or not np.isfinite(self.uplim) else self.uplim * self.unit

    def __repr__(self) -> str:
        unit_str = f" {self.unit}" if self.unit is not None else ""
        if not np.isfinite(self.std):
            return f"{self.value:.4g}{unit_str}"
        return f"{self.value:.4g} \u00b1 {self.std:.4g}{unit_str}  [{self.lolim:.4g}, {self.uplim:.4g}]"

# ---------------------------------------------------------------------------
# Line-table / wavelength-range globals
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(__file__)
resource_path = os.path.join(script_dir, "..", "..", "..", "..", "resources", "lines")
linetable_path = resource_path

_wmin = 0.0
_wmax = np.inf


def _scalar_float(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"{name} must be a scalar value.")
    scalar = float(arr.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _wavelength_limit_value(value, name):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        return _scalar_float(value.to_value(u.AA, equivalencies=u.spectral()), name)
    return _scalar_float(value, name)


def _resolve_filter_bounds(minimum, maximum, *, unit=None, target_unit=u.AA):
    lower_is_quantity = isinstance(minimum, u.Quantity)
    upper_is_quantity = isinstance(maximum, u.Quantity)
    if lower_is_quantity != upper_is_quantity:
        raise ValueError(
            "min and max must both be spectral Quantities or both be plain scalars.")

    target_unit = u.Unit(target_unit)
    if lower_is_quantity:
        lower = minimum.to(target_unit, equivalencies=u.spectral())
        upper = maximum.to(target_unit, equivalencies=u.spectral())
    else:
        resolved_unit = target_unit if unit is None else u.Unit(unit)
        lower = (_scalar_float(minimum, 'min') * resolved_unit).to(target_unit, equivalencies=u.spectral())
        upper = (_scalar_float(maximum, 'max') * resolved_unit).to(target_unit, equivalencies=u.spectral())

    lower_val = _scalar_float(lower.to_value(target_unit), 'min')
    upper_val = _scalar_float(upper.to_value(target_unit), 'max')
    if lower_val == upper_val:
        raise ValueError("min and max resolve to the same spectral position.")
    return (min(lower_val, upper_val) * target_unit,
            max(lower_val, upper_val) * target_unit)


def set_wavelength_range(wmin=None, wmax=None):
    """Set the global wavelength range for line filtering."""
    global _wmin, _wmax
    if wmin is not None:
        _wmin = _wavelength_limit_value(wmin, 'wmin')
    if wmax is not None:
        _wmax = _wavelength_limit_value(wmax, 'wmax')
    print(f"Wavelength range set to: [{_wmin}, {_wmax}]")


def set_medium(medium=DEFAULT_LINETABLE_MEDIUM):
    """Set the session default wavelength medium for line models and headerless line tables."""
    global _default_medium
    _default_medium = _validate_medium(medium)
    return _default_medium


def _validate_domain(domain):
    if domain not in profiles.DOMAIN_UNITS:
        raise ValueError("domain must be 'wavelength', 'frequency', or 'energy'.")
    return domain


def _validate_medium(medium):
    medium = str(medium).strip().lower()
    if medium not in VALID_LINETABLE_MEDIA:
        raise ValueError("medium must be 'air' or 'vacuum'.")
    return medium


def _resolve_default_medium(medium=None):
    return _default_medium if medium is None else _validate_medium(medium)


def _spectral_position_unit(values, position_unit=None):
    if position_unit is not None:
        return u.Unit(position_unit)
    if isinstance(values, u.Quantity):
        return u.Unit(values.unit)
    quantity = getattr(values, 'quantity', None)
    unit = getattr(quantity, 'unit', None)
    if unit not in (None, u.dimensionless_unscaled):
        return u.Unit(unit)
    unit = getattr(values, 'unit', None)
    if unit not in (None, u.dimensionless_unscaled):
        return u.Unit(unit)
    return None


def _resolve_source_linetable_medium(table_medium=None, position_unit=None, fallback_medium=None):
    unit = None if position_unit is None else u.Unit(position_unit)
    if unit is not None and not unit.is_equivalent(u.AA):
        if table_medium is not None and _validate_medium(table_medium) != 'vacuum':
            raise ValueError(
                "Line tables specified in frequency or energy units imply vacuum wavelengths; medium must be 'vacuum'.")
        return 'vacuum'
    if table_medium is not None:
        return _validate_medium(table_medium)
    return _resolve_default_medium(fallback_medium)


def _convert_line_positions(positions, source_medium, target_medium):
    source_medium = _validate_medium(source_medium)
    target_medium = _validate_medium(target_medium)
    positions = positions.to(u.AA)
    if source_medium == target_medium:
        return positions
    values = positions.to_value(u.AA)
    if source_medium == 'air' and target_medium == 'vacuum':
        converted = air_to_vac(values)
    elif source_medium == 'vacuum' and target_medium == 'air':
        converted = vac_to_air(values)
    else:
        raise ValueError("medium must be 'air' or 'vacuum'.")
    return np.asarray(converted, dtype=float) * u.AA


def _resolve_linetable_medium(medium=None, table_medium=None, position_unit=None):
    source_medium = _resolve_source_linetable_medium(
        table_medium=table_medium,
        position_unit=position_unit,
        fallback_medium=medium,
    )
    target_medium = source_medium if medium is None else _validate_medium(medium)
    if position_unit is not None and not u.Unit(position_unit).is_equivalent(u.AA) and target_medium != 'vacuum':
        raise ValueError(
            "Line tables specified in frequency or energy units imply vacuum wavelengths; medium must be 'vacuum'.")
    return source_medium, target_medium


def _position_quantity(values, position_unit=None):
    unit = _spectral_position_unit(values, position_unit=position_unit)
    if isinstance(values, u.Quantity):
        quantity = values if unit is None else values.to(unit, equivalencies=u.spectral())
    else:
        quantity = getattr(values, 'quantity', None)
        if getattr(quantity, 'unit', None) not in (None, u.dimensionless_unscaled):
            quantity = quantity if unit is None else quantity.to(unit, equivalencies=u.spectral())
        else:
            base_unit = u.AA if unit is None else unit
            quantity = np.asarray(values, dtype=float) * base_unit
    if quantity.unit.is_equivalent(u.AA):
        return quantity.to(u.AA)
    return quantity.to(u.AA, equivalencies=u.spectral())


def normalize_linetable(data, medium=None, position_unit=u.AA):
    """Return a QTable with columns name, position [AA], and weight."""
    if isinstance(data, QTable):
        table = data.copy(copy_data=True)
    elif isinstance(data, Table):
        table = QTable(data, copy=True)
    elif isinstance(data, pd.DataFrame):
        table = QTable({col: data[col].to_numpy() for col in data.columns})
    else:
        table = QTable(data)

    if 'position' not in table.colnames:
        raise ValueError("Line tables must define a 'position' column.")

    if 'name' not in table.colnames:
        raise ValueError("Line table format not recognized. Required column: name.")

    if 'weight' not in table.colnames:
        table['weight'] = np.ones(len(table), dtype=float)
        table['name'] = _make_unique(table['name'])

    table['name'] = np.asarray(table['name'], dtype=str)
    input_unit = _spectral_position_unit(table['position'], position_unit=position_unit)
    source_medium, target_medium = _resolve_linetable_medium(
        medium=medium,
        table_medium=table.meta.get('medium'),
        position_unit=input_unit,
    )
    table['position'] = _convert_line_positions(
        _position_quantity(table['position'], position_unit=position_unit),
        source_medium,
        target_medium,
    )
    table['weight'] = np.asarray(table['weight'], dtype=float)
    table.meta['medium'] = target_medium
    return table


def read_linetable(path, medium=None, position_unit=None):
    """Read an ECSV line table, or a plain CSV with an explicit position unit."""
    suffix = Path(path).suffix.lower()
    if suffix == '.ecsv':
        table = QTable.read(path, format='ascii.ecsv')
    elif suffix == '.csv':
        if position_unit is None:
            raise ValueError(
                "Plain CSV line tables require position_unit=... because CSV does not store units. Use ECSV for self-describing line tables.")
        table = QTable.read(path, format='ascii.csv')
    else:
        raise ValueError("Line table files must use either the .ecsv or .csv extension.")
    return normalize_linetable(
        table,
        medium=medium,
        position_unit=position_unit,
    )


def stack_linetables(tables, medium=None, position_unit=None):
    """Stack line tables while preserving a single explicit wavelength medium."""
    normed = [normalize_linetable(table, medium=medium, position_unit=position_unit) for table in tables]
    table = normed[0] if len(normed) == 1 else vstack(normed, metadata_conflicts='silent')
    source_mediums = {tbl.meta.get('medium', DEFAULT_LINETABLE_MEDIUM) for tbl in normed}
    resolved_medium = medium if medium is not None else source_mediums.pop()
    if medium is None and len(source_mediums) > 0:
        raise ValueError("All line tables must use the same wavelength medium.")
    table.meta['medium'] = _validate_medium(resolved_medium)
    return table


def _ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
        return True
    return False


def _linetable_file_paths(dirpath):
    paths = sorted(glob.glob(os.path.join(dirpath, '*.ecsv')))
    paths.extend(sorted(glob.glob(os.path.join(dirpath, '*.csv'))))
    stems = {}
    for path in paths:
        stem = Path(path).stem
        if stem in stems:
            raise ValueError(
                f"Ambiguous line-table inputs found for '{stem}'. Keep only one of .ecsv or .csv.")
        stems[stem] = path
    return paths


def _write_linetable(path, table, overwrite=False):
    table = normalize_linetable(table, medium=table.meta.get('medium'))
    out = QTable()
    out['name'] = np.asarray(table['name'], dtype=str)
    out['position'] = table['position'].to(u.AA)
    out['weight'] = np.asarray(table['weight'], dtype=float)
    out.meta['medium'] = table.meta.get('medium', DEFAULT_LINETABLE_MEDIUM)
    lines = [
        '# %ECSV 1.0',
        '# ---',
        '# datatype:',
        '# - {name: name, datatype: string}',
        '# - {name: position, unit: Angstrom, datatype: float64}',
        '# - {name: weight, datatype: float64}',
        '# meta:',
        f"#   medium: {out.meta['medium']}",
        'name position weight',
    ]
    for row in out:
        lines.append(
            f"{str(row['name'])} {float(row['position'].to_value(u.AA)):.15g} {float(row['weight']):.15g}")

    if os.path.exists(path) and not overwrite:
        raise OSError(f"File exists and overwrite=False: {path}")
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def _trim_linetable_file(filepath, output_dirpath, min, max, unit=None, position_unit=None,
                         overwrite=False):
    table = read_linetable(filepath, position_unit=position_unit)
    lower, upper = _resolve_filter_bounds(
        min,
        max,
        unit=unit,
        target_unit=table['position'].unit,
    )
    trimmed = table[(table['position'] >= lower) & (table['position'] <= upper)]
    output_path = os.path.join(output_dirpath, f"{Path(filepath).stem}.ecsv")
    _write_linetable(output_path, trimmed, overwrite=overwrite)


def setup_local_lines(min=4000, max=7000, unit=u.AA, dirpath='./lines', overwrite=False):
    """Copy packaged line lists locally after filtering by a spectral range.

    Parameters
    ----------
    min, max : float or `~astropy.units.Quantity`
        Spectral bounds used to trim the packaged Prism line lists. When plain
        scalars are provided, they are interpreted in ``unit``. When
        ``Quantity`` bounds are provided, ``unit`` is ignored and spectral
        equivalencies are used.
    unit : unit-like, optional
        Unit for scalar ``min``/``max`` inputs. Defaults to Angstrom.
    dirpath : str, optional
        Output directory for the trimmed CSV files.
    overwrite : bool, optional
        Rewrite files even if ``dirpath`` already exists.

    Notes
    -----
    Local copies are always written as ECSV with an explicit Angstrom unit and
    wavelength-medium metadata.
    """
    lower, upper = _resolve_filter_bounds(min, max, unit=unit, target_unit=u.AA)
    set_wavelength_range(wmin=lower, wmax=upper)

    is_created = _ensure_directory(dirpath)

    global linetable_path
    linetable_path = dirpath

    if overwrite or is_created:
        trim_line_lists(min, max, resource_path, dirpath, unit=unit, overwrite=True)


def trim_line_lists(min, max, input_dirpath, output_dirpath, unit=None, overwrite=False):
    """Trim user line-table files and rewrite them as self-describing ECSV.

    ECSV files carry their own units and are read directly. Plain CSV files are
    only accepted when ``unit`` is provided, because CSV does not store units.
    The trimmed outputs are always written as canonical ECSV files with the
    ``position`` column stored in Angstrom.
    """
    if not os.path.isdir(input_dirpath):
        raise ValueError(f"Input directory does not exist: {input_dirpath}")

    is_created = _ensure_directory(output_dirpath)
    if not overwrite and not is_created:
        return

    filepaths = _linetable_file_paths(input_dirpath)
    if not filepaths:
        raise ValueError(f"No line-table files were found in {input_dirpath}")

    for filepath in filepaths:
        suffix = Path(filepath).suffix.lower()
        csv_position_unit = unit if suffix == '.csv' else None
        _trim_linetable_file(
            filepath,
            output_dirpath,
            min=min,
            max=max,
            unit=unit,
            position_unit=csv_position_unit,
            overwrite=True,
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_name(name):
    """Convert scientific name to pythonic attribute name."""
    name = str(name).strip()
    name = re.sub(r'[\[\]<>]', '', name)
    name = re.sub(r'\s+', '', name)
    name = name.replace('-', '_')
    return name.lower()


def _make_unique(names):
    """Make a list of unique names by appending suffixes to duplicates."""
    import string
    counts = {}
    result = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            suffix = '_' + string.ascii_lowercase[counts[name]]
            result.append(name + suffix)
    return result


def _has_param_std(param) -> bool:
    return hasattr(param, 'std') and param.std is not None and np.isfinite(param.std)


def _get_param_limits(param):
    has_lo = hasattr(param, 'lolim') and param.lolim is not None and np.isfinite(param.lolim)
    has_hi = hasattr(param, 'uplim') and param.uplim is not None and np.isfinite(param.uplim)
    lo = param.lolim if has_lo else param.value
    hi = param.uplim if has_hi else param.value
    return lo, hi, has_lo, has_hi

# ---------------------------------------------------------------------------
# LineModelBase
# ---------------------------------------------------------------------------

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

    def __init__(self, *args, instfwhm=0.0, domain='wavelength', medium=None,
                 **kwargs):
        raw, mode, value_or_spline, deriv = self._normalize_instfwhm(instfwhm)
        self._domain = _validate_domain(domain)
        self._medium = _resolve_default_medium(medium)
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
        param_names = list(type(self).param_names)
        coerced_args = list(args)
        for index, value in enumerate(coerced_args[:len(param_names)]):
            coerced_args[index] = type(self)._coerce_init_parameter(param_names[index], value)
        coerced_kwargs = {
            key: type(self)._coerce_init_parameter(key, value) if key in param_names else value
            for key, value in kwargs.items()
        }
        super().__init__(*coerced_args, **coerced_kwargs)

    @property
    def input_units(self):
        return {self.inputs[0]: profiles.domain_unit(self.domain)}

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = _validate_domain(value)

    @property
    def medium(self):
        return self._medium

    @medium.setter
    def medium(self, value):
        self._medium = _resolve_default_medium(value)

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

    @classmethod
    def _coerce_init_parameter(cls, name, value):
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

    def _center_jacobian(self, center):
        jac = profiles.domain_jacobian(np.asarray([center], dtype=float), self.domain)
        return float(np.asarray(jac, dtype=float).reshape(-1)[0])

    def _native_amplitude_value(self, amplitude, center, output_unit=None, for_deriv=False):
        amp_unit = self._output_unit_from_amplitude(amplitude)
        if amp_unit is None:
            return self._as_value(amplitude), output_unit

        target_unit = amp_unit if output_unit is None else output_unit
        amp_value = self._as_value(amplitude, target_unit)
        if self.domain != 'wavelength':
            if for_deriv:
                raise NotImplementedError(
                    "fit_deriv with quantity amplitudes is not supported for frequency or energy domains.")
            amp_value /= self._center_jacobian(center)
        return amp_value, target_unit

    @staticmethod
    def _parameter_payload(param):
        return param.quantity if getattr(param, 'unit', None) is not None else param.value

    def _model_output_unit(self, amplitude_unit=None):
        if amplitude_unit is not None:
            return amplitude_unit
        if hasattr(self, 'amplitude'):
            return getattr(self.amplitude, 'unit', None)
        return None

    def _flux_unit(self, amplitude_unit=None):
        output_unit = self._model_output_unit(amplitude_unit)
        if output_unit is None:
            return None
        return output_unit * profiles.domain_unit(self.domain)

    def _eqw_unit(self, amplitude_unit=None):
        return None if self._model_output_unit(amplitude_unit) is None else profiles.domain_unit(self.domain)

    def _coerce_output_scalar(self, value, output_unit=None, name='value'):
        if isinstance(value, u.Quantity):
            if output_unit is None:
                raise ValueError(
                    f"{name} was provided as a Quantity, but this model has no output unit.")
            return self._as_value(value, output_unit)
        return self._as_value(value)

    def _coerce_output_array(self, values, output_unit=None, name='value'):
        if isinstance(values, u.Quantity):
            if output_unit is None:
                raise ValueError(
                    f"{name} was provided as a Quantity array, but this model has no output unit.")
            return np.asarray(values.to_value(output_unit), dtype=float)
        return np.asarray(values, dtype=float)

    def _coerce_domain_axis(self, x):
        if isinstance(x, u.Quantity):
            return np.asarray(
                x.to_value(profiles.domain_unit(self.domain), equivalencies=u.spectral()),
                dtype=float,
            )
        return np.asarray(x, dtype=float)

    def _default_domain_grid(self, center_aa, fwhm_aa, window=None, num=4096):
        if window is None:
            hw = 6.0 * fwhm_aa
            wave_window = np.array([center_aa - hw, center_aa + hw], dtype=float)
            edges = profiles.from_wavelength_values(wave_window, self.domain)
            low, high = float(np.min(edges)), float(np.max(edges))
        else:
            low, high = window
            if isinstance(low, u.Quantity):
                low = low.to_value(profiles.domain_unit(self.domain), equivalencies=u.spectral())
            if isinstance(high, u.Quantity):
                high = high.to_value(profiles.domain_unit(self.domain), equivalencies=u.spectral())
            low = float(low)
            high = float(high)
        return np.linspace(low, high, int(num), dtype=float)

    def _domain_center_value(self, center_aa):
        return float(profiles.from_wavelength_values(np.asarray([center_aa], dtype=float), self.domain)[0])

    @staticmethod
    def _metric_frame(rows, index):
        frame = pd.DataFrame(rows, index=list(index), columns=['value', 'std', 'lolim', 'uplim', 'unit'])
        frame['unit'] = [None if unit in (None, '') else u.Unit(unit) for unit in frame['unit']]
        return frame

    def _prepare_input_grid(self, x):
        is_scalar = np.ndim(x) == 0
        if isinstance(x, u.Quantity):
            x_values = x.to_value(profiles.domain_unit(self.domain), equivalencies=u.spectral())
        else:
            x_values = np.asarray(x, dtype=float)
        x_domain = np.atleast_1d(np.asarray(x_values, dtype=float))
        x_native = profiles.to_wavelength_values(x_domain, self.domain)
        jacobian = profiles.domain_jacobian(x_native, self.domain)
        return x_native, jacobian, is_scalar

    @staticmethod
    def _apply_domain_jacobian(values, jacobian):
        return np.asarray(values, dtype=float) * np.asarray(jacobian, dtype=float)

    def _finalize_output(self, values, jacobian, is_scalar, amplitude_unit=None):
        out = self._apply_domain_jacobian(values, jacobian)
        if amplitude_unit is not None:
            out = out * amplitude_unit
        return out[0] if is_scalar else out

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
            unit=self._flux_unit(),
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
        output_unit = self._model_output_unit()
        if continuum is None:
            center, _ = self._infer_center_fwhm_aa()
            continuum = self(self._domain_center_value(center))

        if method == 'constant-continuum':
            continuum_val = self._coerce_output_scalar(continuum, output_unit, name='continuum')
            if not np.isfinite(continuum_val) or continuum_val == 0.0:
                result = Metric(value=float('nan'), unit=self._eqw_unit(output_unit))
                self.ew = result
                return result
            f = self.flux
            ew_val = f.value / continuum_val
            ew_std = float(f.std / abs(continuum_val)) if np.isfinite(f.std) else float('nan')
            ew_lo = float(f.lolim / continuum_val) if np.isfinite(f.lolim) else float('nan')
            ew_hi = float(f.uplim / continuum_val) if np.isfinite(f.uplim) else float('nan')
            result = Metric(
                value=ew_val,
                std=ew_std,
                lolim=ew_lo,
                uplim=ew_hi,
                unit=self._eqw_unit(output_unit),
            )

        elif method == 'integration':
            if x is None:
                center, fwhm_aa = self._infer_center_fwhm_aa()
                x_arr = self._default_domain_grid(center, fwhm_aa, window=window, num=num)
            else:
                x_arr = self._coerce_domain_axis(x)
            y_line = self._coerce_output_array(self(x_arr), output_unit, name='line profile')
            if callable(continuum):
                y_cont = self._coerce_output_array(continuum(x_arr), output_unit, name='continuum')
            else:
                continuum_val = self._coerce_output_scalar(continuum, output_unit, name='continuum')
                y_cont = np.full_like(x_arr, continuum_val, dtype=float)
            safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
            try:
                ew_val = float(np.trapezoid(y_line / safe, x_arr))
            except AttributeError:
                ew_val = float(np.trapz(y_line / safe, x_arr))
            result = Metric(value=ew_val, unit=self._eqw_unit(output_unit))

        else:
            raise ValueError("method must be 'constant-continuum' or 'integration'.")

        self.ew = result
        return result

    def _infer_center_fwhm_aa(self):
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        if hasattr(self, 'fwhm'):
            fwhm_aa = self.fwhm.value / c_kms * center
        elif hasattr(self, 'fwhm_G'):
            fwhm_aa = self.fwhm_G.value / c_kms * center
        else:
            fwhm_aa = 10.0
        return center, max(fwhm_aa, 0.1)

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
