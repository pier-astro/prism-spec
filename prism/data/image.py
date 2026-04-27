"""Generic 2-D image container."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import NDData, NDDataArray
from astropy.wcs import WCS

from .. import config

from .core import (
    err_from_nddata_uncertainty,
    infer_axis_type,
    infer_value_type,
    nddata_mask_to_valid,
    nddata_uncertainty_from_err,
    normalize_unit,
    parse_binmap,
    parse_err,
    physical_scales_from_angular,
    pixel_scales_from_wcs_or_coords,
    valid_to_nddata_mask,
    world_from_pixel_barycenter,
)


def _axis_indices(indexer, size, *, axis_name):
    if indexer is None:
        return slice(None), True
    if isinstance(indexer, slice):
        return indexer, True

    array = np.asarray(indexer)
    if array.dtype == bool:
        if array.shape != (size,):
            raise ValueError(f'{axis_name} boolean mask must have shape ({size},); got {array.shape}.')
        return array, False
    if array.ndim != 1:
        raise ValueError(f'{axis_name} indexer must be a slice, 1-D integer array, or boolean mask.')
    return array.astype(int, copy=False), False


def _trim_axis_values(axis, new_size):
    if axis is None:
        return None
    if axis.shape[0] >= new_size:
        return np.asarray(axis[:new_size], dtype=float)
    return np.arange(new_size, dtype=float)


@dataclass
class RegionMeasurement:
    """Summary of a region extracted from an :class:`Image`.

    Attributes
    ----------
    value : float
        Aggregated value over the selected region.
    err : float
        Propagated uncertainty associated with ``value``.
    mask : object or None
        Boolean mask used for the extraction, if retained.
    ra, dec : float or None
        Sky coordinates of the region barycenter in degrees when available.
    area : float or None
        Region area in pixel units or a derived physical unit.
    n_pixels : int
        Number of contributing pixels.
    method : str
        Reduction method used to collapse the region.
    """
    value: float
    err: float
    mask: object | None
    ra: float | None
    dec: float | None
    area: float | None
    n_pixels: int
    method: str


class Image:
    """Two-dimensional image container with units, masks, and optional WCS.

    Parameters
    ----------
    values : array-like
        Image values with shape ``(y, x)`` or a 1-D map.
    err, var : array-like, optional
        Standard-deviation or variance arrays matching ``values``. Only one may
        be provided. Default is ``None``.
    x, y : array-like, optional
        Coordinate arrays. Defaults are pixel indices.
    mask : array-like of bool, optional
        Valid-data mask with the same shape as ``values``. Default is all ``True``.
    wcs : astropy.wcs.WCS, optional
        Celestial WCS attached to the image. Default is ``None``.
    header : mapping, optional
        FITS-style metadata. Default is an empty dict.
    unit, xunit, yunit : str or astropy.units.Unit, optional
        Units for the image values and coordinate axes. Defaults are ``None``.
    binmap : array-like, optional
        Integer spatial bin map for 2-D images. Default is ``None``.
    xtype, ytype, valuetype : str, optional
        Explicit semantic labels. Defaults are inferred from the units.
    is_var : bool, optional
        Whether the stored uncertainty should be interpreted as variance when
        round-tripping through ``NDData``. Default is inferred from ``var``.
    ra_ref, dec_ref : float, str, or astropy.units.Quantity, optional
        Reference sky position used for relative plotting offsets. Default is
        ``None``.

    Notes
    -----
    ``Image`` is the 2-D companion to :class:`prism.data.Cube`. It keeps the data
    model lightweight while supporting WCS-aware cropping, spatial-scale
    conversions, region extraction, and clean round-tripping through Astropy's
    ``NDData`` containers.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        values,
        *,
        err=None,
        var=None,
        x=None,
        y=None,
        mask=None,
        wcs=None,
        header=None,
        unit=None,
        xunit=None,
        yunit=None,
        binmap=None,
        xtype=None,
        ytype=None,
        valuetype=None,
        is_var=None,
        ra_ref=None,
        dec_ref=None,
    ):
        from .core import parse_celestial_coord
        values = np.asarray(values, dtype=float)
        if values.ndim not in {1, 2}:
            raise ValueError('values must be 1-D or 2-D.')

        self.shape = values.shape
        self._values = values

        if err is not None and var is not None:
            raise ValueError("Cannot provide both 'err' and 'var'.")
        if var is not None:
            var_array = np.asarray(var, dtype=float)
            if var_array.shape != self.shape:
                raise ValueError(f'var must have shape {self.shape}; got {var_array.shape}.')
            if np.any(var_array < 0):
                raise ValueError('var must be non-negative.')
            self._err = np.sqrt(var_array)
            self.is_var = True if is_var is None else is_var
        else:
            self._err = parse_err(err, shape=self.shape, name='err', fill_value=0.0)
            self.is_var = False if is_var is None else is_var

        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(self._values, dtype=bool)
        if self.mask.shape != self.shape:
            raise ValueError(f'mask must have shape {self.shape}; got {self.mask.shape}.')

        self.wcs = wcs
        self.header = header if header is not None else {}
        self.unit = normalize_unit(unit)
        self.xunit = normalize_unit(xunit)
        self.yunit = normalize_unit(yunit)
        self.xtype = infer_axis_type(xtype, self.xunit)
        self.ytype = infer_axis_type(ytype, self.yunit)
        self.valuetype = infer_value_type(valuetype, self.unit)

        if values.ndim == 2:
            self.y = np.asarray(y, dtype=float) if y is not None else np.arange(values.shape[0], dtype=float)
            self.x = np.asarray(x, dtype=float) if x is not None else np.arange(values.shape[1], dtype=float)
        else:
            self.y = None if y is None else np.asarray(y, dtype=float)
            self.x = np.asarray(x, dtype=float) if x is not None else np.arange(values.shape[0], dtype=float)

        self.binmap = parse_binmap(binmap, spatial_shape=self.shape, name='binmap') if values.ndim == 2 else None

        self.ra_ref = parse_celestial_coord(ra_ref)
        self.dec_ref = parse_celestial_coord(dec_ref)

    def set_reference_coordinates(self, ra, dec):
        """Set reference coordinates for relative offsets."""
        from .core import parse_celestial_coord
        self.ra_ref = parse_celestial_coord(ra)
        self.dec_ref = parse_celestial_coord(dec)

    def get_extent(self, unit='arcsec'):
        """
        Calculate the extent for imshow: [left, right, bottom, top] in relative offsets.
        Offsets are calculated relative to ra_ref and dec_ref.
        """
        if self.wcs is None:
            raise ValueError("WCS is required to calculate extent.")
        if self.ra_ref is None or self.dec_ref is None:
            raise ValueError("ra_ref and dec_ref must be set. Use set_reference_coordinates().")

        target_unit = normalize_unit(unit) or u.arcsec
        ny, nx = self.shape[:2]

        def get_offset(px, py):
            sky = self.wcs.pixel_to_world(px, py)
            dra = (sky.ra.deg - self.ra_ref) * np.cos(np.deg2rad(self.dec_ref))
            ddec = sky.dec.deg - self.dec_ref
            # East is positive offset, West is negative (RA increases East -> West)
            return (dra * u.deg).to_value(target_unit), (ddec * u.deg).to_value(target_unit)

        left, _ = get_offset(-0.5, (ny - 1) / 2)
        right, _ = get_offset(nx - 0.5, (ny - 1) / 2)
        _, bottom = get_offset((nx - 1) / 2, -0.5)
        _, top = get_offset((nx - 1) / 2, ny - 0.5)

        return [left, right, bottom, top]

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        array = np.asarray(value, dtype=float)
        if array.shape != self.shape:
            old_shape = self.shape
            warnings.warn(
                'Changing image shape via direct value assignment does not propagate WCS or bin maps. '
                'Prefer crop() for shape-changing operations.',
                UserWarning,
            )
            self.shape = array.shape
            if all(new <= old for new, old in zip(self.shape, old_shape)):
                slices = tuple(slice(0, size) for size in self.shape)
                self._err = self._err[slices]
                self.mask = self.mask[slices]
            else:
                self._err = np.full(self.shape, 0.0, dtype=float)
                self.mask = np.ones(self.shape, dtype=bool)

            if array.ndim == 2:
                self.y = _trim_axis_values(self.y, self.shape[0])
                self.x = _trim_axis_values(self.x, self.shape[1])
                if self.binmap is not None:
                    if all(new <= old for new, old in zip(self.shape, old_shape)):
                        self.binmap = self.binmap[:self.shape[0], :self.shape[1]]
                    else:
                        self.binmap = None
            else:
                self.y = None if self.y is None else _trim_axis_values(self.y, self.shape[0])
                self.x = _trim_axis_values(self.x, self.shape[0])
        self._values = array

    @property
    def data(self):
        return self._values

    @data.setter
    def data(self, value):
        self.values = value

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, value):
        self._err = parse_err(value, shape=self.shape, name='err', fill_value=0.0)
        self.is_var = False

    @property
    def var(self):
        return self._err ** 2

    @var.setter
    def var(self, value):
        array = np.asarray(value, dtype=float)
        if array.shape != self.shape:
            raise ValueError(f'var must have shape {self.shape}; got {array.shape}.')
        if np.any(array < 0):
            raise ValueError('var must be non-negative.')
        self._err = np.sqrt(array)
        self.is_var = True

    def copy(self):
        header = self.header.copy() if hasattr(self.header, 'copy') else dict(self.header)
        return Image(
            values=self.values.copy(),
            err=self.err.copy(),
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=self.mask.copy(),
            wcs=self.wcs.deepcopy() if hasattr(self.wcs, 'deepcopy') else self.wcs,
            header=header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            valuetype=self.valuetype,
            is_var=self.is_var,
        )

    def crop(self, yslice=None, xslice=None, inplace=False):
        target = self if inplace else self.copy()
        original_values = target.values
        original_err = target.err
        original_mask = target.mask

        if target.values.ndim == 2:
            yindex, y_is_slice = _axis_indices(yslice, target.shape[0], axis_name='yslice')
            xindex, x_is_slice = _axis_indices(xslice, target.shape[1], axis_name='xslice')
            indexers = (yindex, xindex)
            can_slice_wcs = y_is_slice and x_is_slice
            target._values = np.asarray(original_values[indexers], dtype=float)
            target._err = np.asarray(original_err[indexers], dtype=float)
            target.mask = np.asarray(original_mask[indexers], dtype=bool)
            target.shape = target._values.shape
            target.y = np.asarray(target.y[yindex], dtype=float)
            target.x = np.asarray(target.x[xindex], dtype=float)
            if target.binmap is not None:
                target.binmap = np.asarray(target.binmap[indexers], dtype=int)
        else:
            if yslice is not None:
                raise ValueError('yslice is not supported for 1-D images; use xslice only.')
            xindex, x_is_slice = _axis_indices(xslice, target.shape[0], axis_name='xslice')
            indexers = (xindex,)
            can_slice_wcs = x_is_slice
            target._values = np.asarray(original_values[indexers], dtype=float)
            target._err = np.asarray(original_err[indexers], dtype=float)
            target.mask = np.asarray(original_mask[indexers], dtype=bool)
            target.shape = target._values.shape
            target.x = np.asarray(target.x[xindex], dtype=float)
            if target.y is not None:
                target.y = np.asarray(target.y[xindex], dtype=float)

        if target.wcs is not None:
            if can_slice_wcs and target.values.ndim == 2:
                try:
                    target.wcs = target.wcs.slice(indexers)
                except Exception as exc:
                    warnings.warn(f'Failed to propagate WCS through crop: {exc}. Dropping WCS.', UserWarning)
                    target.wcs = None
            elif not can_slice_wcs:
                warnings.warn(
                    'Irregular image crops do not preserve WCS exactly. Dropping WCS on the cropped result.',
                    UserWarning,
                )
                target.wcs = None

        return target

    @property
    def unique_bins(self):
        if self.binmap is None:
            return np.arange(self.shape[0] * self.shape[1], dtype=int) if self.values.ndim == 2 else np.arange(self.shape[0], dtype=int)
        bins = np.unique(self.binmap)
        return bins[bins >= 0]

    def iter_bins(self):
        if self.values.ndim != 2:
            raise ValueError('Bin iteration is only supported for 2-D images.')
        if self.binmap is None:
            for flat_index in range(self.shape[0] * self.shape[1]):
                mask = np.zeros(self.shape, dtype=bool)
                mask.flat[flat_index] = True
                yield flat_index, mask
            return
        for bin_id in self.unique_bins:
            yield int(bin_id), self.binmap == bin_id

    def apply_binmap(self, binmap=None, method='mean', inplace=False):
        if self.values.ndim != 2:
            raise ValueError('Bin maps are only supported for 2-D images.')
        target = self if inplace else self.copy()
        target.binmap = parse_binmap(
            target.binmap if binmap is None else binmap,
            spatial_shape=target.shape,
            name='binmap',
        )

        if target.binmap is None:
            raise ValueError('No binmap was provided.')

        new_values = np.array(target.values, copy=True)
        new_err = np.array(target.err, copy=True)
        for _, mask in target.iter_bins():
            pixels = target.values[mask]
            pixel_err = target.err[mask]
            count = pixels.size
            if count == 0:
                continue
            if method == 'sum':
                value = np.sum(pixels)
                sigma = np.sqrt(np.sum(pixel_err ** 2))
            elif method == 'mean':
                value = np.mean(pixels)
                sigma = np.sqrt(np.sum(pixel_err ** 2)) / count
            elif method == 'median':
                value = np.median(pixels)
                sigma = np.sqrt(np.sum(pixel_err ** 2)) / count * 1.2533
            else:
                raise ValueError("method must be one of 'sum', 'mean', or 'median'.")
            new_values[mask] = value
            new_err[mask] = sigma

        target._values = new_values
        target._err = new_err
        return target

    def pixel_scales(self, unit='arcsec'):
        return pixel_scales_from_wcs_or_coords(
            wcs=self.wcs,
            x=self.x,
            y=self.y,
            xunit=self.xunit,
            yunit=self.yunit,
            unit=unit,
        )

    def spatial_scale(self, redshift=None, distance=None, unit='kpc', cosmology=None):
        return physical_scales_from_angular(
            self.pixel_scales(unit='arcsec'),
            redshift=redshift,
            distance=distance,
            unit=unit,
            cosmology=cosmology,
        )

    def pixel_area(self, unit='arcsec2', redshift=None, distance=None, cosmology=None):
        area_unit = normalize_unit(unit) or (u.arcsec ** 2)
        if area_unit.is_equivalent(u.arcsec ** 2):
            scales = self.pixel_scales(unit=u.arcsec)
            return (scales[0] * scales[1]).to(area_unit)
        scales = self.spatial_scale(redshift=redshift, distance=distance, unit=u.kpc, cosmology=cosmology)
        return (scales[0] * scales[1]).to(area_unit)

    def measure_region(self, mask, method='sum'):
        if self.values.ndim != 2:
            raise ValueError('Region measurements are only supported for 2-D images.')

        weights = np.asarray(mask, dtype=float)
        if weights.shape != self.shape:
            raise ValueError(f'mask must have shape {self.shape}; got {weights.shape}.')
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError('mask weights must lie between 0 and 1.')

        binary = weights > 0
        if not np.any(binary):
            raise ValueError('mask selects no pixels.')

        if method == 'sum':
            value = float(np.sum(self.values * weights))
            sigma = float(np.sqrt(np.sum((self.err * weights) ** 2)))
        elif method == 'mean':
            total_weight = float(np.sum(weights))
            value = float(np.sum(self.values * weights) / total_weight)
            sigma = float(np.sqrt(np.sum((self.err * weights) ** 2)) / total_weight)
        elif method == 'median':
            if not np.all((weights == 0) | (weights == 1)):
                raise ValueError('Fractional masks are not supported with method="median".')
            pixels = self.values[binary]
            pixel_err = self.err[binary]
            value = float(np.median(pixels))
            sigma = float(np.sqrt(np.sum(pixel_err ** 2)) / max(pixels.size, 1) * 1.2533)
        else:
            raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

        y_index, x_index = np.indices(self.shape)
        total_weight = float(np.sum(weights[binary]))
        x_center = float(np.sum(x_index[binary] * weights[binary]) / total_weight)
        y_center = float(np.sum(y_index[binary] * weights[binary]) / total_weight)
        ra, dec = world_from_pixel_barycenter(self.wcs, x_center, y_center)

        area = None
        try:
            area = self.pixel_area(unit=u.arcsec ** 2) * total_weight
        except Exception:
            area = None

        return RegionMeasurement(
            value=value,
            err=sigma,
            mask=mask,
            ra=ra,
            dec=dec,
            area=area,
            n_pixels=int(np.count_nonzero(binary)),
            method=method,
        )

    def _as_operand_array(self, other):
        if np.isscalar(other):
            return 'scalar', other
        if isinstance(other, Image):
            return 'image', other
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f'Shape mismatch: {self.shape} vs {other.shape}')
            return 'array', other
        raise TypeError(f'Operand must be an Image, scalar, or numpy.ndarray, got {type(other)}')

    def _check_compatibility(self, other):
        kind, parsed_other = self._as_operand_array(other)
        if kind in {'scalar', 'array'}:
            return True
        if self.shape != parsed_other.shape and not config.FORCE_OP_MISMATCH:
            raise ValueError(f'Shape mismatch: {self.shape} vs {parsed_other.shape}')
        if self.wcs is not None and parsed_other.wcs is not None:
            if self.wcs.to_header().tostring() != parsed_other.wcs.to_header().tostring():
                if not config.FORCE_OP_MISMATCH:
                    raise ValueError('WCS mismatch. Set prism.config.FORCE_OP_MISMATCH = True to ignore.')
        return True

    def _apply_op(self, other, op_data, op_err):
        self._check_compatibility(other)
        kind, parsed_other = self._as_operand_array(other)

        if kind == 'scalar':
            new_values = op_data(self.values, parsed_other)
            new_err = op_err(self.err, 0, self.values, parsed_other)
            new_mask = self.mask.copy()
        elif kind == 'array':
            new_values = op_data(self.values, parsed_other)
            new_err = op_err(self.err, 0, self.values, parsed_other)
            new_mask = self.mask.copy()
        else:
            new_values = op_data(self.values, parsed_other.values)
            new_err = op_err(self.err, parsed_other.err, self.values, parsed_other.values)
            new_mask = self.mask & parsed_other.mask

        return Image(
            values=new_values,
            err=new_err,
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=new_mask,
            wcs=self.wcs,
            header=self.header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            valuetype=self.valuetype,
            is_var=False,
        )

    def __add__(self, other):
        return self._apply_op(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __sub__(self, other):
        return self._apply_op(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __mul__(self, other):
        def err_mul(e1, e2, d1, d2):
            return np.sqrt((d2 * e1) ** 2 + (d1 * e2) ** 2)
        return self._apply_op(other, np.multiply, err_mul)

    def __truediv__(self, other):
        def err_div(e1, e2, d1, d2):
            return np.sqrt((e1 / d2) ** 2 + ((d1 * e2) / d2 ** 2) ** 2)
        return self._apply_op(other, np.divide, err_div)

    def _apply_rop(self, other, op_data, op_err):
        kind, parsed_other = self._as_operand_array(other)
        if kind == 'image':
            return NotImplemented
        if kind == 'scalar':
            left_values = parsed_other
            left_err = 0
            new_values = op_data(left_values, self.values)
            new_err = op_err(left_err, self.err, left_values, self.values)
        elif kind == 'array':
            left_values = parsed_other
            left_err = 0
            new_values = op_data(left_values, self.values)
            new_err = op_err(left_err, self.err, left_values, self.values)
        else:
            return NotImplemented
        return Image(
            values=new_values,
            err=new_err,
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=self.mask.copy(),
            wcs=self.wcs,
            header=self.header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            valuetype=self.valuetype,
            is_var=False,
        )

    def to_nddata(self):
        header = self.header.copy() if hasattr(self.header, 'copy') else dict(self.header)
        meta = {
            'prism_header': header,
            'prism_x': None if self.x is None else self.x.copy(),
            'prism_y': None if self.y is None else self.y.copy(),
            'prism_xunit': None if self.xunit is None else self.xunit.to_string(),
            'prism_yunit': None if self.yunit is None else self.yunit.to_string(),
            'prism_xtype': self.xtype,
            'prism_ytype': self.ytype,
            'prism_valuetype': self.valuetype,
            'prism_binmap': None if self.binmap is None else self.binmap.copy(),
        }
        return NDDataArray(
            data=self.values.copy(),
            uncertainty=nddata_uncertainty_from_err(self.err, is_var=self.is_var),
            mask=valid_to_nddata_mask(self.mask),
            wcs=self.wcs.deepcopy() if hasattr(self.wcs, 'deepcopy') else self.wcs,
            meta=meta,
            unit=self.unit,
        )

    @classmethod
    def from_nddata(
        cls,
        nddata,
        *,
        x=None,
        y=None,
        xunit=None,
        yunit=None,
        xtype=None,
        ytype=None,
        valuetype=None,
        binmap=None,
    ):
        if not isinstance(nddata, NDData):
            raise TypeError('nddata must implement the Astropy NDData interface.')

        values = np.asarray(nddata.data, dtype=float)
        err, is_var = err_from_nddata_uncertainty(getattr(nddata, 'uncertainty', None))
        meta = {} if getattr(nddata, 'meta', None) is None else dict(nddata.meta)

        return cls(
            values=values,
            err=err,
            x=meta.get('prism_x', x),
            y=meta.get('prism_y', y),
            mask=nddata_mask_to_valid(getattr(nddata, 'mask', None), shape=values.shape),
            wcs=getattr(nddata, 'wcs', None),
            header=meta.get('prism_header', meta),
            unit=getattr(nddata, 'unit', None),
            xunit=meta.get('prism_xunit', xunit),
            yunit=meta.get('prism_yunit', yunit),
            binmap=meta.get('prism_binmap', binmap),
            xtype=meta.get('prism_xtype', xtype),
            ytype=meta.get('prism_ytype', ytype),
            valuetype=meta.get('prism_valuetype', valuetype),
            is_var=is_var,
        )

    def __radd__(self, other):
        return self._apply_rop(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __rsub__(self, other):
        return self._apply_rop(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __rmul__(self, other):
        def err_mul(e1, e2, d1, d2):
            return np.sqrt((d2 * e1) ** 2 + (d1 * e2) ** 2)
        return self._apply_rop(other, np.multiply, err_mul)

    def __rtruediv__(self, other):
        def err_div(e1, e2, d1, d2):
            return np.sqrt((e1 / d2) ** 2 + ((d1 * e2) / d2 ** 2) ** 2)
        return self._apply_rop(other, np.divide, err_div)

    @classmethod
    def from_fits(cls, filename, ext_values=None, ext_err=None, ext_var=None, ext_mask=None, ext_wcs=None, ext_binmap=None):
        headers = {}
        with fits.open(filename) as hdul:
            ext_names = [hdu.name.upper() for hdu in hdul]
            if ext_values is None:
                ext_values = 'DATA' if 'DATA' in ext_names else 0
            if ext_err is None and ext_var is None:
                if 'STAT' in ext_names:
                    ext_var = 'STAT'
                elif 'VAR' in ext_names:
                    ext_var = 'VAR'
                elif 'ERR' in ext_names:
                    ext_err = 'ERR'
            if ext_mask is None:
                if 'DQ' in ext_names:
                    ext_mask = 'DQ'
                elif 'MASK' in ext_names:
                    ext_mask = 'MASK'
            if ext_binmap is None and 'BINMAP' in ext_names:
                ext_binmap = 'BINMAP'
            if ext_wcs is None:
                ext_wcs = ext_values

            try:
                values = np.squeeze(hdul[ext_values].data)
                headers['DATA'] = hdul[ext_values].header.copy()
            except KeyError as exc:
                raise ValueError(f'Could not find data extension {ext_values} in {filename}') from exc

            try:
                header_wcs = hdul[ext_wcs].header
                wcs = WCS(header_wcs)
                if wcs.naxis > 2:
                    wcs = wcs.celestial
            except KeyError as exc:
                raise ValueError(f'Could not find WCS extension {ext_wcs} in {filename}') from exc

            unit = None
            bunit_str = headers['DATA'].get('BUNIT')
            if bunit_str:
                try:
                    unit = u.Unit(bunit_str)
                except ValueError:
                    warnings.warn(f"Could not parse BUNIT '{bunit_str}' into astropy.units.")

            err = None
            var = None
            if ext_err is not None:
                try:
                    err = np.squeeze(hdul[ext_err].data)
                except Exception as exc:
                    warnings.warn(f'Failed to load error from extension {ext_err}: {exc}')
                    err = None
            elif ext_var is not None:
                try:
                    var = np.squeeze(hdul[ext_var].data)
                except Exception as exc:
                    warnings.warn(f'Failed to load variance from extension {ext_var}: {exc}')
                    var = None

            loaded_mask = None
            if ext_mask is not None:
                try:
                    loaded_mask = np.squeeze(hdul[ext_mask].data).astype(bool)
                except Exception as exc:
                    warnings.warn(f'Failed to load mask from extension {ext_mask}: {exc}')
                    loaded_mask = None

            loaded_binmap = None
            if ext_binmap is not None:
                try:
                    loaded_binmap = np.squeeze(hdul[ext_binmap].data).astype(int)
                except Exception as exc:
                    warnings.warn(f'Failed to load binmap from extension {ext_binmap}: {exc}')
                    loaded_binmap = None

        if loaded_mask is None:
            loaded_mask = np.isfinite(values)
            if err is not None:
                loaded_mask &= np.isfinite(err)
            elif var is not None:
                loaded_mask &= np.isfinite(var)

        if values.ndim == 2:
            y = np.arange(values.shape[0], dtype=float)
            x = np.arange(values.shape[1], dtype=float)
        else:
            y = None
            x = np.arange(values.shape[0], dtype=float)

        return cls(values=values, err=err, var=var, x=x, y=y, mask=loaded_mask, wcs=wcs, header=headers, unit=unit, binmap=loaded_binmap)

    def write(self, filename, overwrite=False, err=True, mask=False, cd_matrix=False, is_var=None, binmap=False):
        from prism.data.cube import wcs_to_cd_matrix

        primary_header = fits.Header()
        values_header = fits.Header()
        err_header = fits.Header()
        mask_header = fits.Header()

        if self.wcs is not None:
            wcs_header = self.wcs.to_header()
            if cd_matrix:
                wcs_to_cd_matrix(wcs_header)
            values_header.update(wcs_header)
            err_header.update(wcs_header)
            mask_header.update(wcs_header)

        if self.unit is not None:
            values_header['BUNIT'] = self.unit.to_string(format='fits')

        out_values = self.values.astype(np.float32).copy()
        out_values[~self.mask] = np.nan

        hdus = [fits.PrimaryHDU(header=primary_header), fits.ImageHDU(data=out_values, header=values_header, name='DATA')]

        if err:
            out_err = self.err.astype(np.float32).copy()
            out_err[~self.mask] = np.nan
            write_var = self.is_var if is_var is None else is_var
            if write_var:
                variance_header = err_header.copy()
                if self.unit is not None:
                    variance_header['BUNIT'] = (self.unit ** 2).to_string(format='fits')
                hdus.append(fits.ImageHDU(data=out_err ** 2, header=variance_header, name='STAT'))
            else:
                if self.unit is not None:
                    err_header['BUNIT'] = self.unit.to_string(format='fits')
                hdus.append(fits.ImageHDU(data=out_err, header=err_header, name='ERR'))

        if mask and self.mask is not None:
            hdus.append(fits.ImageHDU(data=self.mask.astype(np.uint8), header=mask_header, name='DQ'))

        if binmap and self.binmap is not None:
            hdus.append(fits.ImageHDU(data=self.binmap.astype(np.int32), header=mask_header, name='BINMAP'))

        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)