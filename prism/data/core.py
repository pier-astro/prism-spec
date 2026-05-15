"""Shared helpers and generic 1-D data containers for prism.data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import constants as const
from astropy.nddata import NDData, NDDataArray, StdDevUncertainty, VarianceUncertainty
from astropy import units as u
from astropy.cosmology import default_cosmology
from astropy.wcs.utils import proj_plane_pixel_scales


_C_ANG_PER_S = const.c.to_value(u.AA / u.s)
_H_EV_S = const.h.to_value(u.eV * u.s)
_HC_EV_ANG = (const.h * const.c).to_value(u.eV * u.AA)


def normalize_unit(unit):
    """Return an ``astropy`` unit or ``None``."""
    return u.Unit(unit) if unit is not None else None


def parse_celestial_coord(coord):
    """Parse RA or Dec input into degrees float."""
    if coord is None:
        return None
    if isinstance(coord, u.Quantity):
        return coord.to_value(u.deg)
    try:
        from astropy.coordinates import Angle
        return Angle(coord).deg
    except Exception:
        try:
            return float(coord)
        except (TypeError, ValueError):
            return coord


def infer_axis_type(axis_type, unit):
    if axis_type is not None:
        return axis_type
    normalized = normalize_unit(unit)
    if normalized is None:
        return None
    if normalized.is_equivalent(u.deg):
        return 'spatial'
    if normalized.is_equivalent(u.AA):
        return 'wavelength'
    if normalized.is_equivalent(u.Hz):
        return 'frequency'
    if normalized.is_equivalent(u.eV):
        return 'energy'
    return None


def infer_value_type(value_type, unit):
    if value_type is not None:
        return value_type
    normalized = normalize_unit(unit)
    if normalized is None:
        return None
    if normalized.is_equivalent(u.Jy) or normalized.is_equivalent(u.erg / (u.s * u.cm**2 * u.Hz)):
        return 'flux-density-nu'
    if normalized.is_equivalent(u.erg / (u.s * u.cm**2 * u.AA)):
        return 'flux-density-lambda'
    if normalized.is_equivalent(u.Jy):
        return 'flux-density'
    return None


def _as_1d_array(values, *, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    return array


def parse_err(err, *, shape, name='err', fill_value=1.0, dtype=float):
    """Return a canonical symmetric standard-deviation array."""
    if err is None:
        return np.full(shape, fill_value, dtype=dtype)

    array = np.asarray(err, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def parse_axis_err(err, *, size, name='xerr', dtype=float):
    """Validate optional symmetric or asymmetric axis/value uncertainties."""
    if err is None:
        return None

    array = np.asarray(err, dtype=dtype)
    if array.ndim == 1:
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},); got {array.shape}.")
    elif array.ndim == 2:
        if array.shape != (2, size):
            raise ValueError(f"{name} must have shape (2, {size}); got {array.shape}.")
    else:
        raise ValueError(f"{name} must be a 1-D or (2, N) array.")

    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def subset_axis_err(err, mask):
    if err is None:
        return None
    if err.ndim == 1:
        return err[mask]
    return err[:, mask]


def is_wavelength_like(axis_type, axis_unit):
    """Return ``True`` when the axis semantics support air/vacuum operations."""
    axis_type = (axis_type or '').lower()
    if axis_type in {'wavelength', 'lambda'}:
        return True
    unit = normalize_unit(axis_unit)
    if unit is None:
        return False
    try:
        return unit.is_equivalent(u.AA)
    except Exception:
        return False


def _require_unit(unit, *, axis_name):
    normalized = normalize_unit(unit)
    if normalized is None:
        raise ValueError(f"{axis_name} unit is required for this conversion.")
    return normalized


def parse_binmap(binmap, *, spatial_shape, name='binmap'):
    """Validate a VorBin-style integer bin map."""
    if binmap is None:
        return None

    array = np.asarray(binmap)
    expected_size = int(np.prod(spatial_shape))
    if array.ndim == 1 and array.size == expected_size:
        array = array.reshape(spatial_shape)
    if tuple(array.shape) != tuple(spatial_shape):
        raise ValueError(f"{name} must have shape {tuple(spatial_shape)}; got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise ValueError(f'{name} must contain integer labels.')
        array = rounded.astype(int)
    return array.astype(int, copy=False)


def nddata_uncertainty_from_err(err, *, is_var=False):
    if err is None:
        return None
    array = np.asarray(err, dtype=float)
    if is_var:
        return VarianceUncertainty(array ** 2 if array.ndim >= 1 else array)
    return StdDevUncertainty(array)


def err_from_nddata_uncertainty(uncertainty):
    if uncertainty is None:
        return None, False
    if isinstance(uncertainty, VarianceUncertainty):
        return np.sqrt(np.asarray(uncertainty.array, dtype=float)), True
    if isinstance(uncertainty, StdDevUncertainty):
        return np.asarray(uncertainty.array, dtype=float), False
    if hasattr(uncertainty, 'array'):
        return np.asarray(uncertainty.array, dtype=float), False
    raise TypeError(f'Unsupported NDData uncertainty type: {type(uncertainty).__name__}')


def valid_to_nddata_mask(mask):
    if mask is None:
        return None
    return ~np.asarray(mask, dtype=bool)


def nddata_mask_to_valid(mask, *, shape):
    if mask is None:
        return np.ones(shape, dtype=bool)
    return ~np.asarray(mask, dtype=bool)


def world_from_pixel_barycenter(wcs, x_center, y_center):
    if wcs is None:
        return None, None
    try:
        celestial = wcs.celestial if hasattr(wcs, 'celestial') else wcs
        world = celestial.all_pix2world([[x_center, y_center]], 0)[0]
        return float(world[0]), float(world[1])
    except Exception:
        return None, None


def pixel_scales_from_wcs_or_coords(*, wcs=None, x=None, y=None, xunit=None, yunit=None, unit='arcsec'):
    target_unit = normalize_unit(unit) or u.arcsec
    if wcs is not None:
        celestial = wcs.celestial if hasattr(wcs, 'celestial') else wcs
        scales = proj_plane_pixel_scales(celestial) * u.deg
        return scales.to(target_unit)

    x_unit = normalize_unit(xunit)
    y_unit = normalize_unit(yunit)
    if x is None or y is None or x_unit is None or y_unit is None:
        raise ValueError('Spatial scales require either a celestial WCS or angular x/y coordinates with units.')
    if not x_unit.is_equivalent(u.deg) or not y_unit.is_equivalent(u.deg):
        raise ValueError('Spatial coordinates must have angular units to infer pixel scales.')

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError('Need at least two x and y coordinates to infer pixel scales.')

    x_scale = np.mean(np.abs(np.diff(x))) * x_unit
    y_scale = np.mean(np.abs(np.diff(y))) * y_unit
    return u.Quantity([x_scale.to_value(target_unit), y_scale.to_value(target_unit)], target_unit)


def physical_scales_from_angular(scales, *, redshift=None, distance=None, unit='kpc', cosmology=None):
    target_unit = normalize_unit(unit) or u.kpc
    if distance is None:
        if redshift is None:
            raise ValueError('Provide either distance= or redshift= to compute physical scales.')
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        distance = cosmology.angular_diameter_distance(redshift)

    distance = u.Quantity(distance)
    scales = u.Quantity(scales)
    return (distance * scales.to(u.rad)).to(target_unit)


def _resolve_spectral_axis_type(axis_type, axis_unit):
    axis_type = (axis_type or '').lower()
    unit = _require_unit(axis_unit, axis_name='source')
    if axis_type != 'spectral':
        return axis_type, unit

    if unit.is_equivalent(u.AA):
        return 'wavelength', unit
    if unit.is_equivalent(u.Hz):
        return 'frequency', unit
    if unit.is_equivalent(u.eV):
        return 'energy', unit
    raise ValueError(
        "Generic spectral axes require wavelength-, frequency-, or energy-like units for conversion."
    )


def convert_spectral_axis(x, from_type, from_unit, to_type, to_unit=None, *, rest=None, convention='doppler'):
    """Convert a spectral axis between wavelength, frequency, energy, and velocity."""
    from_type, source_unit = _resolve_spectral_axis_type(from_type, from_unit)
    to_type = (to_type or '').lower()
    if from_type == to_type:
        target_unit = normalize_unit(to_unit) or source_unit
        return (np.asarray(x, dtype=float) * source_unit).to_value(target_unit), target_unit

    x_values = np.asarray(x, dtype=float)
    source_quantity = x_values * source_unit

    if from_type == 'wavelength':
        wavelength = source_quantity.to(u.AA)
    elif from_type == 'frequency':
        wavelength = (const.c / source_quantity).to(u.AA)
    elif from_type == 'energy':
        wavelength = ((const.h * const.c) / source_quantity).to(u.AA)
    elif from_type == 'velocity':
        if rest is None:
            raise ValueError("rest wavelength/frequency is required for velocity conversion.")
        rest_unit = normalize_unit(getattr(rest, 'unit', None))
        if rest_unit is None:
            raise ValueError("rest must be an astropy quantity for velocity conversion.")
        rest_quantity = rest.to(u.AA, equivalencies=u.spectral())
        equivalency = u.doppler_optical(rest_quantity)
        wavelength = source_quantity.to(u.AA, equivalencies=equivalency)
    else:
        raise ValueError(f"Unsupported spectral axis type '{from_type}'.")

    if to_type == 'wavelength':
        target_unit = normalize_unit(to_unit) or u.AA
        return wavelength.to_value(target_unit), target_unit
    if to_type == 'frequency':
        target_unit = normalize_unit(to_unit) or u.Hz
        frequency = (const.c / wavelength).to(target_unit)
        return frequency.value, target_unit
    if to_type == 'energy':
        target_unit = normalize_unit(to_unit) or u.eV
        energy = ((const.h * const.c) / wavelength).to(target_unit)
        return energy.value, target_unit
    if to_type == 'velocity':
        if rest is None:
            raise ValueError("rest wavelength/frequency is required for velocity conversion.")
        rest_quantity = rest.to(u.AA, equivalencies=u.spectral())
        target_unit = normalize_unit(to_unit) or (u.km / u.s)
        equivalency_name = (convention or 'doppler').lower()
        if equivalency_name in {'doppler', 'optical'}:
            equivalency = u.doppler_optical(rest_quantity)
        elif equivalency_name == 'radio':
            equivalency = u.doppler_radio(rest_quantity)
        elif equivalency_name == 'relativistic':
            equivalency = u.doppler_relativistic(rest_quantity)
        else:
            raise ValueError(f"Unsupported velocity convention '{convention}'.")
        velocity = wavelength.to(target_unit, equivalencies=equivalency)
        return velocity.value, target_unit

    raise ValueError(f"Unsupported target spectral axis type '{to_type}'.")


@dataclass
class AxisInfo:
    values: np.ndarray
    unit: u.UnitBase | None
    kind: str


class Data1D:
    """Generic 1-D container with coordinate/value semantics."""

    def __init__(
        self,
        x,
        y,
        *,
        xerr=None,
        yerr=None,
        xunit=None,
        yunit=None,
        xtype=None,
        ytype=None,
        name='data1d',
        meta=None,
        dtype=np.float32,
    ):
        self.name = name
        self.meta = {} if meta is None else dict(meta)
        self.dtype = np.dtype(dtype)
        self.xunit = normalize_unit(xunit)
        self.yunit = normalize_unit(yunit)
        self.xtype = infer_axis_type(xtype, self.xunit)
        self.ytype = infer_value_type(ytype, self.yunit)

        self.x = _as_1d_array(x, name='x')
        self.y = np.asarray(y, dtype=self.dtype)
        if self.y.ndim != 1:
            raise ValueError("y must be a 1-D array.")
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape.")

        self.xerr = parse_axis_err(xerr, size=self.x.size, name='xerr')
        self.yerr = parse_axis_err(yerr, size=self.y.size, name='yerr', dtype=self.dtype)

    @property
    def axis(self):
        """AxisInfo: A packaged representation of the primary spectral/spatial coordinate axis."""
        return AxisInfo(values=self.x, unit=self.xunit, kind=self.xtype)

    @property
    def values(self):
        """numpy.ndarray: The primary 1-D values (alias for `y`)."""
        return self.y

    @property
    def dx(self):
        """float or numpy.ndarray: The step size of the x-axis. 

        Returns a single float if the step is uniform, or an array of diffs if non-uniform.
        Returns an empty array if the axis has fewer than 2 elements.
        """
        if self.x.size < 2:
            return np.array([], dtype=float)
        delta = np.diff(self.x)
        if np.allclose(delta, delta[0]):
            return float(delta[0])
        return delta

    def copy(self):
        """Create a deep copy of the 1-D data container.

        Returns
        -------
        Data1D
            A new instance with independent arrays and metadata.
        """
        return self.__class__(
            x=self.x.copy(),
            y=self.y.copy(),
            xerr=None if self.xerr is None else self.xerr.copy(),
            yerr=None if self.yerr is None else self.yerr.copy(),
            xunit=self.xunit,
            yunit=self.yunit,
            xtype=self.xtype,
            ytype=self.ytype,
            name=self.name,
            meta=self.meta.copy(),
            dtype=self.dtype,
        )
