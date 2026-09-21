"""Line-table and domain conversion utilities for analytic line models."""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable, Table, vstack

from ....utils.tools import air_to_vac, vac_to_air
from . import profiles

c_kms = profiles.C_KMS
sigma2fwhm = profiles.SIGMA2FWHM

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


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
