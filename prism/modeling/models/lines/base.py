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


def _validate_domain(domain):
    if domain not in profiles.DOMAIN_FAMILIES:
        raise ValueError("domain must be 'wavelength', 'frequency', or 'energy'.")
    return domain


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


def _position_quantity(values, position_unit=None):
    unit = _spectral_position_unit(values, position_unit=position_unit)
    if isinstance(values, u.Quantity):
        quantity = values if unit is None else values.to(unit, equivalencies=u.spectral())
    else:
        quantity = getattr(values, 'quantity', None)
        if getattr(quantity, 'unit', None) not in (None, u.dimensionless_unscaled):
            quantity = quantity if unit is None else quantity.to(unit, equivalencies=u.spectral())
        else:
            if unit is None:
                raise ValueError("Line table positions have no units and position_unit=None was passed. You must specify the position_unit.")
            quantity = np.asarray(values, dtype=float) * unit
    return quantity


def normalize_linetable(data, position_unit=None):
    """Return a QTable with columns name, position, and weight."""
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
    table['position'] = _position_quantity(table['position'], position_unit=position_unit)
    table['weight'] = np.asarray(table['weight'], dtype=float)
    return table


def _compute_spectral_jacobian(positions, target_unit):
    """Numerically evaluate the spectral density Jacobian |dx/dy| for weight scaling."""
    delta = 1e-5
    x1 = positions * (1.0 - delta)
    x2 = positions * (1.0 + delta)
    
    y1 = x1.to(target_unit, equivalencies=u.spectral())
    y2 = x2.to(target_unit, equivalencies=u.spectral())
    
    dx = np.abs((x2 - x1).value)
    dy = np.abs((y2 - y1).value)
    
    return dx / dy


def convert_linetable_domain(data, output_unit, input_unit=None):
    """Convert a line table to a new spectral domain, scaling weights accordingly to conserve flux.

    Parameters
    ----------
    data : QTable, Table, DataFrame, or dict
        Input line table.
    output_unit : unit-like
        Target spectral unit (e.g., u.eV, u.Hz, u.AA).
    input_unit : unit-like, optional
        Input unit for the 'position' column if not defined in the table.

    Returns
    -------
    QTable
        Converted line table with positions in the target domain and re-normalized weights.
    """
    table = normalize_linetable(data, position_unit=input_unit)
    target_unit = u.Unit(output_unit)
    
    positions = table['position']
    if positions.unit.is_equivalent(target_unit):
        table['position'] = positions.to(target_unit)
        return table
        
    converted_positions = positions.to(target_unit, equivalencies=u.spectral())
    
    # Scale weights by the Jacobian |dx_old / dx_new|
    jacobian = _compute_spectral_jacobian(positions, target_unit)
    table['weight'] = table['weight'] * jacobian
    
    # Normalize weights so max in each template group is 1.0
    for name in np.unique(table['name']):
        mask = table['name'] == name
        group_weights = table['weight'][mask]
        max_weight = np.max(group_weights)
        if max_weight > 0:
            table['weight'][mask] = group_weights / max_weight
            
    table['position'] = converted_positions
    return table


def convert_linetable_medium(data, target_medium, input_unit=None):
    """Convert line table wavelengths between air and vacuum.

    Parameters
    ----------
    data : QTable, Table, DataFrame, or dict
        Input line table.
    target_medium : {'air', 'vacuum'}
        Target wavelength medium for the line positions.
    input_unit : unit-like, optional
        Input unit for the 'position' column if not defined in the table.

    Returns
    -------
    QTable
        Converted line table with positions in the target wavelength medium.
    """
    target_medium = str(target_medium).strip().lower()
    if target_medium not in {'air', 'vacuum'}:
        raise ValueError("target_medium must be 'air' or 'vacuum'.")

    table = normalize_linetable(data, position_unit=input_unit)
    source_medium = table.meta.get('medium')
    
    # If already in target medium, return as-is
    if source_medium == target_medium:
        return table
    
    # Infer source medium if not explicitly set
    if source_medium is None:
        source_medium = 'vacuum' if target_medium == 'air' else 'air'
    
    positions = table['position'].to(u.AA)
    values = positions.to_value(u.AA)
    if source_medium == 'air' and target_medium == 'vacuum':
        converted = air_to_vac(values)
    else:
        converted = vac_to_air(values)
        
    table['position'] = np.asarray(converted, dtype=float) * u.AA
    table.meta['medium'] = target_medium
    
    return table


def read_linetable(path, position_unit=None):
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
        position_unit=position_unit,
    )


def stack_linetables(tables, position_unit=None):
    """Stack line tables."""
    normed = [normalize_linetable(table, position_unit=position_unit) for table in tables]
    return normed[0] if len(normed) == 1 else vstack(normed, metadata_conflicts='silent')


def convert_domain_csv(paths, output_unit, input_unit=None, output_dir=None, overwrite=False):
    """Convert line table files to a new spectral domain and rewrite as ECSV.

    Reads tabular line tables natively, inspecting columns for physical units. 
    Converts positions to the target spectral domain, re-normalizes weights 
    to conserve flux, and writes output as a self-describing ECSV.

    Parameters
    ----------
    paths : str or list of str
        File path(s) to CSV or ECSV line tables, or a directory path.
    output_unit : unit-like
        Target spectral unit (e.g., u.eV, u.Hz, u.AA).
    input_unit : unit-like, optional
        Input unit for the 'position' column. Required only if the input file 
        does not carry unit metadata natively.
    output_dir : str, optional
        Output directory for converted files. If None, files are written to the same
        location as input files.
    overwrite : bool, optional
        Whether to overwrite existing ECSV files. Default is False.
    """
    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = _linetable_file_paths(paths)
        else:
            paths = [paths]
    
    if output_dir is not None:
        _ensure_directory(output_dir)
    
    for path in paths:
        try:
            table = QTable.read(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        pos_col = table['position'] if 'position' in table.colnames else None
        if pos_col is None:
            print(f"Skipping {path}: no 'position' column found.")
            continue
            
        if getattr(pos_col, 'unit', None) is None:
            if input_unit is None:
                raise ValueError(
                    f"File '{path}' does not contain unit metadata for the 'position' column. "
                    "You must provide the `input_unit` parameter to define the starting domain."
                )
            table['position'] = np.asarray(pos_col, dtype=float) * u.Unit(input_unit)
            
        # Convert the domain
        converted = convert_linetable_domain(table, output_unit)
        
        # Determine output path
        if output_dir is not None:
            out_path = os.path.join(output_dir, Path(path).stem + '.ecsv')
        else:
            out_path = str(Path(path).with_suffix('.ecsv'))
        
        _write_linetable(out_path, converted, overwrite=overwrite)


def convert_medium_csv(paths, target_medium, input_unit=None, output_dir=None, overwrite=False):
    """Convert line table files between air and vacuum and rewrite as ECSV.

    Reads tabular line tables natively, inspecting columns for physical units. 
    Converts wavelengths between air and vacuum, and writes output as ECSV 
    with medium metadata.

    Parameters
    ----------
    paths : str or list of str
        File path(s) to CSV or ECSV line tables, or a directory path.
    target_medium : {'air', 'vacuum'}
        Target wavelength medium for the line positions.
    input_unit : unit-like, optional
        Input unit for the 'position' column. Required only if the input file 
        does not carry unit metadata natively.
    output_dir : str, optional
        Output directory for converted files. If None, files are written to the same
        location as input files.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is False.
    """
    target_medium = str(target_medium).strip().lower()
    if target_medium not in {'air', 'vacuum'}:
        raise ValueError("target_medium must be 'air' or 'vacuum'.")

    if isinstance(paths, str):
        if os.path.isdir(paths):
            paths = _linetable_file_paths(paths)
        else:
            paths = [paths]
    
    if output_dir is not None:
        _ensure_directory(output_dir)
    
    for path in paths:
        try:
            table = QTable.read(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        pos_col = table['position'] if 'position' in table.colnames else None
        if pos_col is None:
            print(f"Skipping {path}: no 'position' column found.")
            continue
            
        if getattr(pos_col, 'unit', None) is None:
            if input_unit is None:
                raise ValueError(
                    f"File '{path}' does not contain unit metadata for the 'position' column. "
                    "You must provide the `input_unit` parameter."
                )
            table['position'] = np.asarray(pos_col, dtype=float) * u.Unit(input_unit)
            
        # Convert the medium
        converted = convert_linetable_medium(table, target_medium)
        
        # Determine output path
        if output_dir is not None:
            out_path = os.path.join(output_dir, Path(path).stem + '.ecsv')
        else:
            out_path = str(Path(path).with_suffix('.ecsv'))
        
        _write_linetable(out_path, converted, overwrite=overwrite)


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
    table = normalize_linetable(table)
    out = QTable()
    out['name'] = np.asarray(table['name'], dtype=str)
    # Write natively to the unit the table was left in
    out['position'] = table['position']
    out['weight'] = np.asarray(table['weight'], dtype=float)
    if 'medium' in table.meta:
        out.meta['medium'] = table.meta['medium']
    
    pos_unit_str = out['position'].unit.to_string() if out['position'].unit else 'dimensionless'
    lines = [
        '# %ECSV 1.0',
        '# ---',
        '# datatype:',
        '# - {name: name, datatype: string}',
        f'# - {{name: position, unit: {pos_unit_str}, datatype: float64}}',
        '# - {name: weight, datatype: float64}',
    ]
    if 'medium' in out.meta:
        lines.extend([
            '# meta:',
            f"#   medium: {out.meta['medium']}",
        ])
    lines.append('name position weight')
    for row in out:
        lines.append(
            f"{str(row['name'])} {float(row['position'].value):.15g} {float(row['weight']):.15g}")

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
            ew_std = float(f.std / abs(continuum_val)) if np.isfinite(f.std) else float('nan')
            ew_lo = float(f.lolim / continuum_val) if np.isfinite(f.lolim) else float('nan')
            ew_hi = float(f.uplim / continuum_val) if np.isfinite(f.uplim) else float('nan')
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