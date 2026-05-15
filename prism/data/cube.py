"""Generic 3-D IFU cube container."""

from __future__ import annotations

from datetime import datetime, timezone
import warnings

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import NDData, NDDataArray
from .. import config
from .core import (
    convert_spectral_axis,
    err_from_nddata_uncertainty,
    infer_axis_type,
    infer_value_type,
    is_wavelength_like,
    nddata_mask_to_valid,
    nddata_uncertainty_from_err,
    normalize_unit,
    parse_binmap,
    parse_celestial_coord,
    parse_err,
    physical_scales_from_angular,
    pixel_scales_from_wcs_or_coords,
    valid_to_nddata_mask,
    world_from_pixel_barycenter,
)

def wcs_to_cd_matrix(header):
    """Convert PC/CDELT WCS keywords into legacy CD matrix form."""
    naxis = header.get('WCSAXES', 2)
    has_pc = any(f'PC{i}_{j}' in header for i in range(1, naxis + 1) for j in range(1, naxis + 1))
    if has_pc:
        for i in range(1, naxis + 1):
            cdelt = header.get(f'CDELT{i}', 1.0)
            for j in range(1, naxis + 1):
                pc_key = f'PC{i}_{j}'
                if pc_key in header:
                    header[f'CD{i}_{j}'] = header[pc_key] * cdelt
                    del header[pc_key]
        for i in range(1, naxis + 1):
            header.pop(f'CDELT{i}', None)
    else:
        for i in range(1, naxis + 1):
            cdelt_key = f'CDELT{i}'
            if cdelt_key in header:
                header[f'CD{i}_{i}'] = header[cdelt_key]
                del header[cdelt_key]
    return header


def _trim_axis_values(axis, new_size):
    if axis is None:
        return None
    if axis.shape[0] >= new_size:
        return np.asarray(axis[:new_size], dtype=float)
    return np.arange(new_size, dtype=float)


def _require_slice(indexer, *, axis_name):
    if indexer is None:
        return slice(None)
    if not isinstance(indexer, slice):
        raise TypeError(f'{axis_name} must be a slice or None.')
    return indexer


def _mask_bounds(mask, shape, *, mask_name):
    array = np.asarray(mask)
    if array.dtype != bool:
        raise TypeError(f'{mask_name} must be a boolean mask.')
    if array.shape != tuple(shape):
        raise ValueError(f'{mask_name} must have shape {tuple(shape)}; got {array.shape}.')
    if not np.any(array):
        raise ValueError(f'{mask_name} must select at least one element.')

    axes = np.where(array)
    return tuple(slice(indices.min(), indices.max() + 1) for indices in axes), array


def _axis_bound_value(value, unit, *, name):
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        if unit is None:
            if value.unit is u.dimensionless_unscaled:
                return value.to_value(u.dimensionless_unscaled)
            raise ValueError(f'{name} has units but the spectral axis has no unit.')
        return value.to_value(unit)
    return value


def _preserve_wcs_axis_unit(header, axis, unit):
    if unit is None:
        return header

    cunit_key = f'CUNIT{axis}'
    old_unit = normalize_unit(header.get(cunit_key))
    if old_unit is None:
        header[cunit_key] = unit.to_string('fits')
        return header

    try:
        scale = (1.0 * old_unit).to_value(unit)
    except Exception:
        header[cunit_key] = unit.to_string('fits')
        return header

    if np.isclose(scale, 1.0):
        header[cunit_key] = unit.to_string('fits')
        return header

    if f'CRVAL{axis}' in header:
        header[f'CRVAL{axis}'] *= scale

    has_cd = any(f'CD{axis}_{j}' in header for j in range(1, header.get('WCSAXES', axis) + 1))
    if has_cd:
        for j in range(1, header.get('WCSAXES', axis) + 1):
            key = f'CD{axis}_{j}'
            if key in header:
                header[key] *= scale
    else:
        cdelt_key = f'CDELT{axis}'
        pc_axis_key = f'PC{axis}_{axis}'
        if pc_axis_key in header and np.isclose(header.get(cdelt_key, 1.0), 1.0) and not np.isclose(header[pc_axis_key], 1.0):
            for j in range(1, header.get('WCSAXES', axis) + 1):
                key = f'PC{axis}_{j}'
                if key in header:
                    header[key] *= scale
        else:
            header[cdelt_key] = header.get(cdelt_key, 1.0) * scale

    header[cunit_key] = unit.to_string('fits')
    return header


class Cube:
    """Spectral cube container for 3-D IFU data or flattened spectral stacks.

    Parameters
    ----------
    values : array-like
        Data array with shape ``(z, y, x)`` or ``(z, spaxel)``.
    z : array-like, optional
        Spectral coordinate grid. Default is ``arange(values.shape[0])``.
    err, var : array-like, optional
        Standard-deviation or variance arrays matching ``values``. Only one may
        be provided. Default is ``None``.
    x, y : array-like, optional
        Spatial coordinate grids. Defaults are pixel indices.
    mask : array-like of bool, optional
        Valid-data mask with the same shape as ``values`` where ``True`` marks
        usable samples and ``False`` marks masked samples. Default is all ``True``.
    wcs : astropy.wcs.WCS, optional
        Celestial or spectral WCS attached to the cube. Default is ``None``.
    header : mapping, optional
        FITS-style metadata. Default is an empty dict.
    unit, xunit, yunit, zunit : str or astropy.units.Unit, optional
        Units for the cube values and coordinate axes. Defaults are ``None``.
    binmap : array-like, optional
        Integer spatial bin map for 3-D cubes. Default is ``None``.
    xtype, ytype, ztype, valuetype : str, optional
        Explicit semantic labels. Defaults are inferred from the units.
    ra, dec : float, str, or astropy.units.Quantity, optional
        Sky coordinates associated with the cube. Default is ``None``.
    redshift : float, optional
        Source redshift stored with the cube. Default is ``None``.
    is_var : bool, optional
        Whether the stored uncertainty should be interpreted as variance when
        round-tripping through ``NDData``. Default is inferred from ``var``.
    dtype : numpy dtype, optional
        Floating-point dtype used for values and uncertainties. Default is
        ``numpy.float32``; pass ``numpy.float64`` for higher precision.

    Notes
    -----
    ``Cube`` is designed to stay close to Astropy containers while exposing the
    workflow conveniences needed for IFU fitting: spectral-axis conversion,
    WCS-aware cropping, FITS import/export, map extraction, and bin-map support.
    The first axis is always spectral, so methods such as ``multifit`` and line
    measurements can work on native cube outputs without extra reshaping.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        values,
        *,
        z=None,
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
        zunit=None,
        binmap=None,
        xtype=None,
        ytype=None,
        ztype=None,
        valuetype=None,
        ra=None,
        dec=None,
        redshift=None,
        is_var=None,
        dtype=np.float32,
    ):
        self.dtype = np.dtype(dtype)
        values = np.asarray(values, dtype=self.dtype)
        if values.ndim not in {2, 3}:
            raise ValueError('values must be either 3-D (z, y, x) or 2-D (z, spaxel).')

        self.shape = values.shape
        self._values = values

        if err is not None and var is not None:
            raise ValueError("Cannot provide both 'err' and 'var'.")
        if var is not None:
            var_array = np.asarray(var, dtype=self.dtype)
            if var_array.shape != self.shape:
                raise ValueError(f'var must have shape {self.shape}; got {var_array.shape}.')
            if np.any(var_array < 0):
                raise ValueError('var must be non-negative.')
            self._err = np.sqrt(var_array).astype(self.dtype, copy=False)
            self.is_var = True if is_var is None else is_var
        else:
            self._err = parse_err(err, shape=self.shape, name='err', fill_value=1.0, dtype=self.dtype)
            self.is_var = False if is_var is None else is_var

        self.mask = np.asarray(mask, dtype=bool) if mask is not None else (np.isfinite(self._values) & np.isfinite(self._err))
        if self.mask.shape != self.shape:
            raise ValueError(f'mask must have shape {self.shape}; got {self.mask.shape}.')

        self.wcs = wcs
        self.header = header if header is not None else {}
        self.unit = normalize_unit(unit)
        self.xunit = normalize_unit(xunit)
        self.yunit = normalize_unit(yunit)
        self.zunit = normalize_unit(zunit)
        self.xtype = infer_axis_type(xtype, self.xunit)
        self.ytype = infer_axis_type(ytype, self.yunit)
        self.ztype = infer_axis_type(ztype, self.zunit)
        self.valuetype = infer_value_type(valuetype, self.unit)

        self.z = np.asarray(z, dtype=float) if z is not None else np.arange(values.shape[0], dtype=float)
        if self.z.shape != (self.shape[0],):
            raise ValueError('z must have length matching the first dimension of values.')

        if values.ndim == 3:
            self.y = np.asarray(y, dtype=float) if y is not None else np.arange(values.shape[1], dtype=float)
            self.x = np.asarray(x, dtype=float) if x is not None else np.arange(values.shape[2], dtype=float)
        else:
            self.y = None if y is None else np.asarray(y, dtype=float)
            self.x = np.asarray(x, dtype=float) if x is not None else np.arange(values.shape[1], dtype=float)

        self.binmap = parse_binmap(binmap, spatial_shape=self.shape[1:], name='binmap') if values.ndim == 3 else None

        self.redshift = redshift
        self.ra = parse_celestial_coord(ra)
        self.dec = parse_celestial_coord(dec)

    @property
    def values(self):
        """numpy.ndarray: The cube data values."""
        return self._values

    @values.setter
    def values(self, value):
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            old_shape = self.shape
            warnings.warn(
                'Changing cube shape via direct value assignment does not propagate WCS or bin maps. '
                'Prefer cutout_slices(), cutout(), or cutout_spectral() for shape-changing operations.',
                UserWarning,
            )
            self.shape = array.shape
            if all(new <= old for new, old in zip(self.shape, old_shape)):
                slices = tuple(slice(0, size) for size in self.shape)
                self._err = self._err[slices]
                self.mask = self.mask[slices]
            else:
                self._err = np.full(self.shape, 1.0, dtype=self.dtype)
                self.mask = np.ones(self.shape, dtype=bool)

            self.z = _trim_axis_values(self.z, self.shape[0])
            if array.ndim == 3:
                self.y = _trim_axis_values(self.y, self.shape[1])
                self.x = _trim_axis_values(self.x, self.shape[2])
                if self.binmap is not None:
                    if all(new <= old for new, old in zip(self.shape[1:], old_shape[1:])):
                        self.binmap = self.binmap[:self.shape[1], :self.shape[2]]
                    else:
                        self.binmap = None
            else:
                self.y = None if self.y is None else _trim_axis_values(self.y, self.shape[1])
                self.x = _trim_axis_values(self.x, self.shape[1])
        self._values = array

    @property
    def data(self):
        """numpy.ndarray: Alias for the `values` property."""
        return self._values

    @data.setter
    def data(self, value):
        self.values = value

    @property
    def err(self):
        """numpy.ndarray: Standard deviation (1-sigma) uncertainties of the cube values."""
        return self._err

    @err.setter
    def err(self, value):
        self._err = parse_err(value, shape=self.shape, name='err', fill_value=1.0, dtype=self.dtype)
        self.is_var = False

    @property
    def var(self):
        """numpy.ndarray: Variance array of the cube values (square of `err`)."""
        return self._err ** 2

    @var.setter
    def var(self, value):
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            raise ValueError(f'var must have shape {self.shape}; got {array.shape}.')
        if np.any(array < 0):
            raise ValueError('var must be non-negative.')
        self._err = np.sqrt(array).astype(self.dtype, copy=False)
        self.is_var = True

    def _require_wavelength_axis(self, *, operation):
        if not is_wavelength_like(self.ztype, self.zunit):
            raise ValueError(f'{operation} is only defined for wavelength-like spectral axes.')

    def _copy_or_inplace(self, inplace):
        return self if inplace else self.copy()

    def copy(self):
        """Create a deep copy of the cube container.

        Returns
        -------
        Cube
            A new instance with independent arrays and metadata.
        """
        header = self.header.copy() if hasattr(self.header, 'copy') else dict(self.header)
        copied = Cube(
            values=self.values.copy(),
            z=self.z.copy(),
            err=self.err.copy(),
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=self.mask.copy(),
            wcs=self.wcs.deepcopy() if hasattr(self.wcs, 'deepcopy') else self.wcs,
            header=header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            zunit=self.zunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            ztype=self.ztype,
            valuetype=self.valuetype,
            ra=self.ra,
            dec=self.dec,
            redshift=self.redshift,
            is_var=self.is_var,
            dtype=self.dtype,
        )
        return copied

    def apply_mask(self, fill=np.nan, err_fill=None, inplace=False):
        """Materialize the validity mask into the stored data arrays.

        This is a convenience method for interoperability with workflows that
        expect masked samples to be represented directly in the numeric arrays.
        The cube mask itself is preserved; only the stored ``values`` and
        ``err`` arrays are modified at masked locations.

        Parameters
        ----------
        fill : float, optional
            Value written into ``values`` wherever ``mask`` is ``False``.
            Default is ``numpy.nan``.
        err_fill : float, optional
            Value written into ``err`` wherever ``mask`` is ``False``.
            Defaults to ``fill``.
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            The masked cube.
        """
        target = self if inplace else self.copy()
        invalid = ~target.mask
        target._values = np.array(target.values, copy=True)
        target._err = np.array(target.err, copy=True)
        target._values[invalid] = fill
        target._err[invalid] = fill if err_fill is None else err_fill
        return target

    def cutout_slices(self, z=None, y=None, x=None, inplace=False):
        """Extract a rectangular subcube using explicit axis slices.

        Parameters
        ----------
        z, y, x : slice, optional
            Slices for the spectral and spatial axes. Default is ``None`` for
            all axes, equivalent to keeping the full axis.
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            The cutout cube object. WCS is propagated through the slices.

        Notes
        -----
        Use :meth:`cutout` for spatial or spectral masks, and
        :meth:`cutout_spectral` for physical spectral bounds.
            
        Examples
        --------
        >>> subcube = cube.cutout_slices(y=slice(10, 50), x=slice(10, 50))
        >>> print(subcube.shape)
        """
        zindex = _require_slice(z, axis_name='z')
        if self.values.ndim == 3:
            yindex = _require_slice(y, axis_name='y')
            xindex = _require_slice(x, axis_name='x')
            indexers = (zindex, yindex, xindex)
        else:
            if y is not None:
                raise ValueError('y is not supported for 2-D cubes; use x for the spatial axis.')
            xindex = _require_slice(x, axis_name='x')
            indexers = (zindex, xindex)

        header = self.header.copy() if hasattr(self.header, 'copy') else dict(self.header)
        wcs = self.wcs.deepcopy() if hasattr(self.wcs, 'deepcopy') else self.wcs
        if wcs is not None:
            try:
                wcs = wcs.slice(indexers)
            except Exception as exc:
                warnings.warn(f'Failed to propagate WCS through cutout_slices: {exc}. Dropping WCS.', UserWarning)
                wcs = None

        if self.values.ndim == 3:
            binmap = None if self.binmap is None else np.array(self.binmap[yindex, xindex], dtype=int, copy=True)
            cutout = Cube(
                values=np.array(self.values[indexers], dtype=self.dtype, copy=True),
                z=np.array(self.z[zindex], dtype=float, copy=True),
                err=np.array(self.err[indexers], dtype=self.dtype, copy=True),
                x=np.array(self.x[xindex], dtype=float, copy=True),
                y=np.array(self.y[yindex], dtype=float, copy=True),
                mask=np.array(self.mask[indexers], dtype=bool, copy=True),
                wcs=wcs,
                header=header,
                unit=self.unit,
                xunit=self.xunit,
                yunit=self.yunit,
                zunit=self.zunit,
                binmap=binmap,
                xtype=self.xtype,
                ytype=self.ytype,
                ztype=self.ztype,
                valuetype=self.valuetype,
                ra=self.ra,
                dec=self.dec,
                redshift=self.redshift,
                is_var=self.is_var,
                dtype=self.dtype,
            )
        else:
            cutout = Cube(
                values=np.array(self.values[indexers], dtype=self.dtype, copy=True),
                z=np.array(self.z[zindex], dtype=float, copy=True),
                err=np.array(self.err[indexers], dtype=self.dtype, copy=True),
                x=np.array(self.x[xindex], dtype=float, copy=True),
                y=None if self.y is None else np.array(self.y[xindex], dtype=float, copy=True),
                mask=np.array(self.mask[indexers], dtype=bool, copy=True),
                wcs=wcs,
                header=header,
                unit=self.unit,
                xunit=self.xunit,
                yunit=self.yunit,
                zunit=self.zunit,
                xtype=self.xtype,
                ytype=self.ytype,
                ztype=self.ztype,
                valuetype=self.valuetype,
                ra=self.ra,
                dec=self.dec,
                redshift=self.redshift,
                is_var=self.is_var,
                dtype=self.dtype,
            )

        if inplace:
            self.__dict__.update(cutout.__dict__)
            return self
        return cutout

    def cutout(self, spatial=None, spectral=None, preserve_mask=True, inplace=False):
        """Extract a subcube from spatial and/or spectral boolean masks.

        Parameters
        ----------
        spatial : array-like of bool, optional
            Spatial mask with shape ``(y, x)`` for 3-D cubes.
        spectral : array-like of bool, optional
            Spectral mask with shape ``(z,)``.
        preserve_mask : bool, optional
            If ``True`` (default), keep the exact masked region by marking
            pixels outside the selected spatial/spectral mask as invalid in
            the returned cube. If ``False``, masks are used only to define the
            rectangular bounding box.
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            The cutout cube object.

        Notes
        -----
        The returned cube is always rectangular. Use :meth:`cutout_slices` for
        explicit rectangular slicing, and :meth:`cutout_spectral` for physical
        spectral bounds.
        """
        if spatial is None and spectral is None:
            return self if inplace else self.copy()

        zslice = slice(None)
        yslice = slice(None)
        xslice = slice(None)
        spectral_mask = None
        spatial_mask = None

        if spectral is not None:
            (zslice,), spectral_mask = _mask_bounds(spectral, (self.shape[0],), mask_name='spectral')

        if spatial is not None:
            if self.values.ndim != 3:
                raise ValueError('spatial masks are only supported for 3-D cubes.')
            (yslice, xslice), spatial_mask = _mask_bounds(spatial, self.shape[1:], mask_name='spatial')

        target = self.cutout_slices(z=zslice, y=yslice, x=xslice, inplace=inplace)
        if preserve_mask:
            if spectral_mask is not None:
                target.mask &= spectral_mask[zslice][(slice(None),) + (None,) * (target.mask.ndim - 1)]
            if spatial_mask is not None:
                target.mask &= spatial_mask[yslice, xslice][None, :, :]
        return target

    def cutout_spectral(self, z=None, min=None, max=None, inplace=False):
        """Extract a subcube along the spectral dimension.

        Parameters
        ----------
        z : slice or array-like bool, optional
            Spectral selector. If provided, ``min`` and ``max`` are ignored.
        min : float or astropy.units.Quantity, optional
            Minimum physical spectral value to keep.
        max : float or astropy.units.Quantity, optional
            Maximum physical spectral value to keep.
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            The cutout cube object.

        Notes
        -----
        This is a convenience wrapper around :meth:`cutout_slices` for
        contiguous ranges and :meth:`cutout` for spectral masks.
            
        Examples
        --------
        >>> # Cut out using a slice
        >>> cube_red = cube.cutout_spectral(z=slice(100, 200))
        >>> 
        >>> # Cut out using physical ranges
        >>> cube_range = cube.cutout_spectral(min=4000, max=5000)
        """
        if z is not None:
            if isinstance(z, slice):
                return self.cutout_slices(z=z, inplace=inplace)
            return self.cutout(spectral=z, inplace=inplace)

        if min is None and max is None:
            return self if inplace else self.copy()

        min_value = _axis_bound_value(min, self.zunit, name='min')
        max_value = _axis_bound_value(max, self.zunit, name='max')

        # Efficiently find indices handling both ascending and descending arrays
        ascending = self.z[-1] >= self.z[0] if len(self.z) > 1 else True
        
        if ascending:
            start = np.searchsorted(self.z, min_value) if min_value is not None else None
            end = np.searchsorted(self.z, max_value, side='right') if max_value is not None else None
        else:
            z_rev = self.z[::-1]
            start = len(self.z) - np.searchsorted(z_rev, max_value, side='right') if max_value is not None else None
            end = len(self.z) - np.searchsorted(z_rev, min_value) if min_value is not None else None

        zslice = np.s_[start:end]
        if len(range(*zslice.indices(self.shape[0]))) == 0:
            raise ValueError('The provided spectral bounds selected zero spectral channels.')
        return self.cutout_slices(z=zslice, inplace=inplace)

    @property
    def unique_bins(self):
        """numpy.ndarray: Unique valid spatial bin indices present in the ``binmap``.

        Excludes negative bin labels (commonly used to represent unbinned/ignored spaxels).
        """
        if self.values.ndim != 3:
            raise ValueError('Bin maps are only supported for 3-D cubes.')
        if self.binmap is None:
            return np.arange(self.shape[1] * self.shape[2], dtype=int)
        bins = np.unique(self.binmap)
        return bins[bins >= 0]

    def iter_bins(self):
        """Iterate over all spatial bins.

        Yields
        ------
        bin_id : int
            The integer label of the current bin.
        mask : numpy.ndarray of bool
            A 2D spatial boolean mask selecting spaxels belonging to the bin.
        """
        if self.values.ndim != 3:
            raise ValueError('Bin iteration is only supported for 3-D cubes.')
        if self.binmap is None:
            for flat_index in range(self.shape[1] * self.shape[2]):
                mask = np.zeros(self.shape[1:], dtype=bool)
                mask.flat[flat_index] = True
                yield flat_index, mask
            return
        for bin_id in self.unique_bins:
            yield int(bin_id), self.binmap == bin_id

    def apply_binmap(self, binmap=None, method='mean', inplace=False):
        """Rebin the spatial pixels based on the given bin map.

        Spaxels belonging to the same bin ID are aggregated into a single pseudo-spaxel
        using the specified method. This updates the cube flux and propagates the errors
        in quadrature.

        Parameters
        ----------
        binmap : array-like, optional
            A 2D integer array defining bin memberships. If ``None``, uses the cube's 
            existing ``binmap`` attribute.
        method : {'sum', 'mean', 'median'}, optional
            Aggregation method for the spaxels in a bin. Default is ``'mean'``.
            - 'sum': adds the flux values (errors add in quadrature).
            - 'mean': averages the flux values (errors add in quadrature and scale by 1/N).
            - 'median': calculates the median (errors approximate median standard error).
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            A new cube (or self if inplace) where all spaxels in a given bin are 
            replaced by the aggregated bin values.
            
        Examples
        --------
        >>> binned_cube = cube.apply_binmap(method='sum')
        """
        if self.values.ndim != 3:
            raise ValueError('Bin maps are only supported for 3-D cubes.')
        target = self if inplace else self.copy()
        target.binmap = parse_binmap(
            target.binmap if binmap is None else binmap,
            spatial_shape=target.shape[1:],
            name='binmap',
        )
        if target.binmap is None:
            raise ValueError('No binmap was provided.')

        new_values = np.array(target.values, copy=True)
        new_err = np.array(target.err, copy=True)
        new_mask = np.array(target.mask, copy=True)
        for _, mask in target.iter_bins():
            flat_mask = mask.reshape(-1)
            spectra = target.values.reshape(target.shape[0], -1)[:, flat_mask]
            spectra_err = target.err.reshape(target.shape[0], -1)[:, flat_mask]
            spectra_mask = target.mask.reshape(target.shape[0], -1)[:, flat_mask]
            count = spectra.shape[1]
            if count == 0:
                continue
            if method == 'sum':
                binned = np.sum(spectra, axis=1, dtype=target.dtype)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1, dtype=target.dtype))
            elif method == 'mean':
                binned = np.mean(spectra, axis=1, dtype=target.dtype)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1, dtype=target.dtype)) / count
            elif method == 'median':
                binned = np.median(spectra, axis=1)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1, dtype=target.dtype)) / count * 1.2533 # ONLY CORRECT FOR GAUSSIAN ERRORS ON THE MEDIAN, NOT OTHER DISTRIBUTIONS
            else:
                raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

            valid = np.all(spectra_mask, axis=1)
            binned = binned.astype(target.dtype, copy=False)
            sigma = sigma.astype(target.dtype, copy=False)
            binned[~valid] = np.nan
            sigma[~valid] = np.nan
            new_values[:, mask] = binned[:, None]
            new_err[:, mask] = sigma[:, None]
            new_mask[:, mask] = valid[:, None]

        target._values = new_values
        target._err = new_err
        target.mask = new_mask
        return target

    def pixel_scales(self, unit='arcsec'):
        """Calculate the spatial pixel scales based on WCS or coordinates.

        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            The angular unit for the output pixel scales. Default is ``'arcsec'``.

        Returns
        -------
        astropy.units.Quantity
            The ``(x_scale, y_scale)`` pixel size with physical units.
        """
        return pixel_scales_from_wcs_or_coords(
            wcs=self.wcs,
            x=self.x,
            y=self.y,
            xunit=self.xunit,
            yunit=self.yunit,
            unit=unit,
        )

    def spatial_scale(self, redshift=None, distance=None, unit='kpc', cosmology=None):
        """Calculate the physical spatial scales using the source redshift or distance.

        Parameters
        ----------
        redshift : float, optional
            The redshift of the object. Defaults to the cube's `redshift` attribute.
        distance : astropy.units.Quantity, optional
            The explicit angular diameter distance to the object.
        unit : str or astropy.units.Unit, optional
            The output physical unit. Default is ``'kpc'``.
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology model. Default is the current `astropy.cosmology.default_cosmology`.

        Returns
        -------
        astropy.units.Quantity
            The physical pixel dimensions as an array.
        """
        return physical_scales_from_angular(
            self.pixel_scales(unit='arcsec'),
            redshift=redshift or self.redshift,
            distance=distance,
            unit=unit,
            cosmology=cosmology,
        )

    def pixel_area(self, unit='arcsec2', redshift=None, distance=None, cosmology=None):
        """Compute the spatial area of a single pixel.

        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            The output area unit (e.g., 'arcsec2', 'kpc2'). Default is ``'arcsec2'``.
        redshift : float, optional
            Redshift for computing physical areas.
        distance : astropy.units.Quantity, optional
            Explicit distance for computing physical areas.
        cosmology : astropy.cosmology.Cosmology, optional
            Cosmology model.

        Returns
        -------
        astropy.units.Quantity
            The area per spaxel.
        """
        area_unit = normalize_unit(unit) or (u.arcsec ** 2)
        if area_unit.is_equivalent(u.arcsec ** 2):
            scales = self.pixel_scales(unit=u.arcsec)
            return (scales[0] * scales[1]).to(area_unit)
        scales = self.spatial_scale(redshift=redshift or self.redshift, distance=distance, unit=u.kpc, cosmology=cosmology)
        return (scales[0] * scales[1]).to(area_unit)

    def extract_spectrum(self, mask, method='sum'):
        """Extract a 1-D spectrum from the 3-D cube using a spatial mask.

        Collapses the spatial dimensions by applying weights. Propagates errors based
        on the chosen extraction method. The resulting spectrum is assigned a new spatial
        center calculated via the barycenter of the mask.

        Parameters
        ----------
        mask : array-like
            A 2D spatial array of weights (values between 0 and 1). Typically a boolean mask.
        method : {'sum', 'mean', 'median'}, optional
            Method used to extract the spectrum.
            - 'sum': simple unweighted/weighted sum of pixel values.
            - 'mean': weighted average of pixels.
            - 'median': median calculation (requires binary masks).

        Returns
        -------
        Spectrum
            A new `Spectrum` object containing the extracted flux and errors.
            
        Examples
        --------
        >>> import numpy as np
        >>> mask = np.zeros(cube.shape[1:], dtype=bool)
        >>> mask[10:20, 10:20] = True
        >>> spec = cube.extract_spectrum(mask, method='sum')
        """
        if self.values.ndim != 3:
            raise ValueError('Spectrum extraction is only supported for 3-D cubes.')

        weights = np.asarray(mask, dtype=float)
        if weights.shape != self.shape[1:]:
            raise ValueError(f'mask must have shape {self.shape[1:]}; got {weights.shape}.')
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError('mask weights must lie between 0 and 1.')

        binary = weights > 0
        if not np.any(binary):
            raise ValueError('mask selects no spaxels.')

        flat_weights = weights.reshape(-1)
        spectra = self.values.reshape(self.shape[0], -1)
        spectra_err = self.err.reshape(self.shape[0], -1)
        spectra_mask = self.mask.reshape(self.shape[0], -1)
        selected = flat_weights > 0

        if method == 'sum':
            flux = np.sum(spectra[:, selected] * flat_weights[selected], axis=1, dtype=self.dtype)
            sigma = np.sqrt(np.sum((spectra_err[:, selected] * flat_weights[selected]) ** 2, axis=1, dtype=self.dtype))
        elif method == 'mean':
            total_weight = np.sum(flat_weights[selected])
            flux = np.sum(spectra[:, selected] * flat_weights[selected], axis=1, dtype=self.dtype) / total_weight
            sigma = np.sqrt(np.sum((spectra_err[:, selected] * flat_weights[selected]) ** 2, axis=1, dtype=self.dtype)) / total_weight
        elif method == 'median':
            if not np.all((flat_weights == 0) | (flat_weights == 1)):
                raise ValueError('Fractional masks are not supported with method="median".')
            flux = np.median(spectra[:, selected], axis=1)
            sigma = np.sqrt(np.sum(spectra_err[:, selected] ** 2, axis=1, dtype=self.dtype)) / max(np.sum(selected), 1) * 1.2533
        else:
            raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

        valid = np.all(spectra_mask[:, selected], axis=1)
        flux = flux.astype(self.dtype, copy=False)
        sigma = sigma.astype(self.dtype, copy=False)
        flux[~valid] = np.nan
        sigma[~valid] = np.nan

        y_index, x_index = np.indices(self.shape[1:])
        total_weight = float(np.sum(weights[binary]))
        x_center = float(np.sum(x_index[binary] * weights[binary]) / total_weight)
        y_center = float(np.sum(y_index[binary] * weights[binary]) / total_weight)
        ra, dec = world_from_pixel_barycenter(self.wcs, x_center, y_center)

        from prism.data.spectrum import Spectrum

        return Spectrum(
            x=self.z.copy(),
            y=flux,
            yerr=sigma,
            ra=ra,
            dec=dec,
            xunit=self.zunit,
            yunit=self.unit,
            xtype=self.ztype,
            ytype=self.valuetype,
            meta={'parent': 'cube', 'method': method, 'redshift': self.redshift},
            dtype=self.dtype,
        )

    def to_nddata(self):
        """Convert the cube to an `astropy.nddata.NDDataArray` format.

        This packages the cube's values, uncertainties, mask, and metadata into a
        standardized Astropy data structure. Useful for exporting or passing into
        Astropy functions.

        Returns
        -------
        astropy.nddata.NDDataArray
            The equivalent Astropy data object.
        """
        header = self.header.copy() if hasattr(self.header, 'copy') else dict(self.header)
        meta = {
            'prism_header': header,
            'prism_x': None if self.x is None else self.x.copy(),
            'prism_y': None if self.y is None else self.y.copy(),
            'prism_z': self.z.copy(),
            'prism_xunit': None if self.xunit is None else self.xunit.to_string(),
            'prism_yunit': None if self.yunit is None else self.yunit.to_string(),
            'prism_zunit': None if self.zunit is None else self.zunit.to_string(),
            'prism_xtype': self.xtype,
            'prism_ytype': self.ytype,
            'prism_ztype': self.ztype,
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
        z=None,
        xunit=None,
        yunit=None,
        zunit=None,
        xtype=None,
        ytype=None,
        ztype=None,
        valuetype=None,
        binmap=None,
        redshift=None,
        dtype=np.float32,
    ):
        """Instantiate a Cube from an `astropy.nddata.NDData` object.

        Parameters
        ----------
        nddata : astropy.nddata.NDData
            The source NDData object containing arrays and uncertainties.
        x, y, z : array-like, optional
            Coordinate grids. Fallback values if not found in `nddata.meta`.
        xunit, yunit, zunit : str or astropy.units.Unit, optional
            Units for the coordinate grids.
        xtype, ytype, ztype, valuetype : str, optional
            Explicit semantic labels.
        binmap : array-like, optional
            Initial binmap structure.
        redshift : float, optional
            Source redshift.

        Returns
        -------
        Cube
            The newly created prism spectral cube.
        """
        if not isinstance(nddata, NDData):
            raise TypeError('nddata must implement the Astropy NDData interface.')

        values = np.asarray(nddata.data, dtype=dtype)
        err, is_var = err_from_nddata_uncertainty(getattr(nddata, 'uncertainty', None))
        meta = {} if getattr(nddata, 'meta', None) is None else dict(nddata.meta)

        return cls(
            values=values,
            err=err,
            x=meta.get('prism_x', x),
            y=meta.get('prism_y', y),
            z=meta.get('prism_z', z),
            mask=nddata_mask_to_valid(getattr(nddata, 'mask', None), shape=values.shape),
            wcs=getattr(nddata, 'wcs', None),
            header=meta.get('prism_header', meta),
            unit=getattr(nddata, 'unit', None),
            xunit=meta.get('prism_xunit', xunit),
            yunit=meta.get('prism_yunit', yunit),
            zunit=meta.get('prism_zunit', zunit),
            binmap=meta.get('prism_binmap', binmap),
            xtype=meta.get('prism_xtype', xtype),
            ytype=meta.get('prism_ytype', ytype),
            ztype=meta.get('prism_ztype', ztype),
            valuetype=meta.get('prism_valuetype', valuetype),
            is_var=is_var,
            redshift=meta.get('prism_redshift', redshift),
            dtype=dtype,
        )

    def _as_operand_array(self, other):
        if np.isscalar(other):
            return 'scalar', other
        if isinstance(other, Cube):
            return 'cube', other
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f'Shape mismatch: {self.shape} vs {other.shape}')
            return 'array', other
        raise TypeError(f'Operand must be a Cube, scalar, or numpy.ndarray, got {type(other)}')

    def _check_compatibility(self, other):
        kind, parsed_other = self._as_operand_array(other)
        if kind in {'scalar', 'array'}:
            return True
        if self.shape != parsed_other.shape and not config.FORCE_OP_MISMATCH:
            raise ValueError(f'Shape mismatch: {self.shape} vs {parsed_other.shape}')
        if not np.allclose(self.z, parsed_other.z, rtol=1e-5, atol=1e-5):
            if not config.FORCE_OP_MISMATCH:
                raise ValueError('Spectral-axis mismatch. Set prism.config.FORCE_OP_MISMATCH = True to ignore.')
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

        return Cube(
            values=new_values,
            z=self.z.copy(),
            err=new_err,
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=new_mask,
            wcs=self.wcs,
            header=self.header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            zunit=self.zunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            ztype=self.ztype,
            valuetype=self.valuetype,
            is_var=False,
            redshift=self.redshift,
            dtype=self.dtype,
        )

    def __add__(self, other):
        return self._apply_op(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __sub__(self, other):
        return self._apply_op(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __mul__(self, other):
        return self._apply_op(other, np.multiply, lambda e1, e2, d1, d2: np.sqrt((d2 * e1) ** 2 + (d1 * e2) ** 2))

    def __truediv__(self, other):
        return self._apply_op(other, np.divide, lambda e1, e2, d1, d2: np.sqrt((e1 / d2) ** 2 + ((d1 * e2) / d2 ** 2) ** 2))

    def _apply_rop(self, other, op_data, op_err):
        kind, parsed_other = self._as_operand_array(other)
        if kind == 'cube':
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
        return Cube(
            values=new_values,
            z=self.z.copy(),
            err=new_err,
            x=None if self.x is None else self.x.copy(),
            y=None if self.y is None else self.y.copy(),
            mask=self.mask.copy(),
            wcs=self.wcs,
            header=self.header,
            unit=self.unit,
            xunit=self.xunit,
            yunit=self.yunit,
            zunit=self.zunit,
            binmap=None if self.binmap is None else self.binmap.copy(),
            xtype=self.xtype,
            ytype=self.ytype,
            ztype=self.ztype,
            valuetype=self.valuetype,
            is_var=False,
            redshift=self.redshift,
            dtype=self.dtype,
        )

    def __radd__(self, other):
        return self._apply_rop(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __rsub__(self, other):
        return self._apply_rop(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1 ** 2 + e2 ** 2))

    def __rmul__(self, other):
        return self._apply_rop(other, np.multiply, lambda e1, e2, d1, d2: np.sqrt((d2 * e1) ** 2 + (d1 * e2) ** 2))

    def __rtruediv__(self, other):
        return self._apply_rop(other, np.divide, lambda e1, e2, d1, d2: np.sqrt((e1 / d2) ** 2 + ((d1 * e2) / d2 ** 2) ** 2))

    def get_masked(self):
        """Return flattened 1-D arrays of data selecting valid spectral channels.

        A channel is considered valid if all its spatial pixels are valid (i.e. not masked).

        Returns
        -------
        tuple of numpy.ndarray
            `(z_valid, values_valid, err_valid)` arrays selecting only non-masked spectral points.
        """
        spatial_axes = tuple(range(1, self.mask.ndim))
        channel_valid = self.mask.all(axis=spatial_axes)
        return self.z[channel_valid], self.values[channel_valid], self.err[channel_valid]

    @classmethod
    def from_fits(
        cls,
        filename,
        ext_values=None,
        ext_err=None,
        ext_var=None,
        ext_mask=None,
        ext_binmap=None,
        z=None,
        z_ext=None,
        ext_wcs=None,
        ztype=None,
        redshift=None,
        dtype=np.float32,
    ):
        """Construct a Cube instance by reading FITS files.

        Parameters
        ----------
        filename : str or path-like
            Path to the target FITS file.
        ext_values : str or int, optional
            Extension containing the data values (defaults to 'DATA', 'SCI', or 1).
        ext_err, ext_var : str or int, optional
            Extensions containing standard deviation or variance arrays.
        ext_mask : str or int, optional
            Extension containing the boolean or bit mask.
        ext_binmap : str or int, optional
            Extension containing the spatial bin labels.
        z : array-like, optional
            Explicit spectral coordinate array.
        z_ext : str or int, optional
            FITS extension or table from which to extract the spectral axis.
        ext_wcs : str or int, optional
            Extension from which to extract WCS information (defaults to `ext_values`).
        ztype : str, optional
            Type label for the spectral axis.
        redshift : float, optional
            Optional redshift metadata to store in the cube.
        dtype : numpy dtype, optional
            Floating-point dtype for values and uncertainties. Defaults to
            ``numpy.float32`` to match common IFU FITS products and keep large
            cubes compact. Use ``numpy.float64`` when higher precision is needed.

        Returns
        -------
        Cube
            The spectral cube loaded from disk.
        """
        with fits.open(filename) as hdul:
            headers = {'PRIMARY': hdul[0].header.copy()}
            ext_names = [hdu.name.upper() for hdu in hdul]

            if ext_values is None:
                if 'DATA' in ext_names:
                    ext_values = 'DATA'
                elif 'SCI' in ext_names:
                    ext_values = 'SCI'
                else:
                    ext_values = 1
            if ext_wcs is None:
                ext_wcs = ext_values
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

            try:
                values = np.asarray(hdul[ext_values].data, dtype=dtype)
                headers['DATA'] = hdul[ext_values].header.copy()
            except KeyError as exc:
                raise ValueError(f'Could not find data extension {ext_values} in {filename}') from exc

            try:
                header_wcs = hdul[ext_wcs].header.copy()
            except KeyError as exc:
                raise ValueError(f'Could not find WCS extension {ext_wcs} in {filename}') from exc

            header_zunit = normalize_unit(header_wcs.get('CUNIT3'))

            instrument = hdul[0].header.get('INSTRUME', '').strip().upper()
            if 'MUSE' in instrument:
                for key in list(header_wcs.keys()):
                    if key.endswith('A') or key.endswith('B'):
                        del header_wcs[key]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(header_wcs)

            unit = None
            bunit_str = headers['DATA'].get('BUNIT')
            if bunit_str:
                try:
                    unit = u.Unit(bunit_str)
                except ValueError:
                    warnings.warn(f"Could not parse BUNIT '{bunit_str}' into astropy.units.")

            zunit = None
            if z is None:
                if z_ext is not None:
                    z_data = hdul[z_ext].data
                    if isinstance(z_data, fits.FITS_rec):
                        columns = [name.upper() for name in z_data.columns.names]
                        matches = [name for name in columns if 'WAVE' in name or 'SPEC' in name]
                        z = z_data[matches[0]] if matches else z_data
                    else:
                        z = z_data
                elif wcs.naxis >= 3:
                    n_z = values.shape[0]
                    pix_coords = np.zeros((n_z, wcs.naxis))
                    pix_coords[:, wcs.naxis - 1] = np.arange(n_z)
                    world_coords = wcs.all_pix2world(pix_coords, 0)
                    z = world_coords[:, wcs.naxis - 1]
                    spectral_cunit = wcs.wcs.cunit[wcs.naxis - 1]
                    if spectral_cunit:
                        zunit = normalize_unit(spectral_cunit.to_string())
                    if header_zunit is not None and zunit is not None and header_zunit != zunit:
                        z = (z * zunit).to_value(header_zunit)
                        zunit = header_zunit

            if zunit is None:
                zunit = header_zunit

            if z is None:
                z = np.arange(values.shape[0], dtype=float)
            else:
                z = np.asarray(z, dtype=float)

            err = None
            var = None
            if ext_err is not None:
                try:
                    err = np.asarray(hdul[ext_err].data, dtype=dtype)
                except Exception as exc:
                    warnings.warn(f'Failed to load error from extension {ext_err}: {exc}')
                    err = None
            elif ext_var is not None:
                try:
                    var = np.asarray(hdul[ext_var].data, dtype=dtype)
                except Exception as exc:
                    warnings.warn(f'Failed to load variance from extension {ext_var}: {exc}')
                    var = None

            loaded_mask = None
            if ext_mask is not None:
                try:
                    loaded_mask = np.asarray(hdul[ext_mask].data, dtype=bool)
                except Exception as exc:
                    warnings.warn(f'Failed to load mask from extension {ext_mask}: {exc}')
                    loaded_mask = None

            loaded_binmap = None
            if ext_binmap is not None:
                try:
                    loaded_binmap = np.asarray(hdul[ext_binmap].data, dtype=int)
                except Exception as exc:
                    warnings.warn(f'Failed to load binmap from extension {ext_binmap}: {exc}')
                    loaded_binmap = None

        if loaded_mask is None:
            loaded_mask = np.isfinite(values)
            if err is not None:
                loaded_mask &= np.isfinite(err)
            elif var is not None:
                loaded_mask &= np.isfinite(var)

        if values.ndim == 3:
            y = np.arange(values.shape[1], dtype=float)
            x = np.arange(values.shape[2], dtype=float)
        else:
            y = None
            x = np.arange(values.shape[1], dtype=float)

        return cls(
            values=values,
            z=z,
            err=err,
            var=var,
            x=x,
            y=y,
            mask=loaded_mask,
            wcs=wcs,
            header=headers,
            unit=unit,
            zunit=zunit,
            binmap=loaded_binmap,
            ztype=ztype,
            redshift=redshift,
            dtype=dtype,
        )

    def wavelengths(self, unit=None):
        values, _ = convert_spectral_axis(self.z, self.ztype, self.zunit, 'wavelength', unit)
        return values

    def frequencies(self, unit=None):
        """Convert the spectral axis to frequency.
        
        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            The target frequency unit.
            
        Returns
        -------
        numpy.ndarray
            The spectral axis in frequency units.
            
        Examples
        --------
        >>> freqs = cube.frequencies(unit='GHz')
        """
        values, _ = convert_spectral_axis(self.z, self.ztype, self.zunit, 'frequency', unit)
        return values

    def energies(self, unit=None):
        values, _ = convert_spectral_axis(self.z, self.ztype, self.zunit, 'energy', unit)
        return values

    def velocities(self, rest=None, unit=None, convention='doppler'):
        values, _ = convert_spectral_axis(
            self.z,
            self.ztype,
            self.zunit,
            'velocity',
            unit,
            rest=rest,
            convention=convention,
        )
        return values
        
    def update_wcs_from_spectral_axis(self):
        """
        Update the WCS 3rd axis (spectral) to match the current `z` array.
        This assumes a linear dispersion. If `z` is non-linear, a linear
        approximation is constructed from the first two pixels.
        """
        if self.wcs is None:
            warnings.warn('No WCS attached to the cube. Cannot update WCS.', UserWarning)
            return

        if self.wcs.naxis < 3:
            warnings.warn('WCS has fewer than 3 axes. Cannot update spectral axis.', UserWarning)
            return

        if len(self.z) > 1:
            cdelt = self.z[1] - self.z[0]
            # Warn if the axis is significantly non-linear
            diffs = np.diff(self.z)
            if not np.allclose(diffs, cdelt, rtol=1e-3):
                warnings.warn('Spectral axis is non-linear. The updated WCS will only be a linear approximation.', UserWarning)
        else:
            cdelt = 1.0

        # Update the WCS object
        self.wcs.wcs.crval[2] = self.z[0]
        self.wcs.wcs.cdelt[2] = cdelt
        self.wcs.wcs.crpix[2] = 1.0  # FITS pixels are 1-indexed
        
        # Ensure we don't have skewed spectral coordinates if a PC matrix exists
        if self.wcs.wcs.has_pc():
            for i in range(2):
                self.wcs.wcs.pc[2, i] = 0.0
                self.wcs.wcs.pc[i, 2] = 0.0
            self.wcs.wcs.pc[2, 2] = 1.0

        if self.zunit is not None:
            try:
                self.wcs.wcs.cunit[2] = self.zunit.to_string('fits')
            except Exception as e:
                warnings.warn(f'Could not update WCS CUNIT3: {e}', UserWarning)

    def write(self, filename, overwrite=False, err=True, mask=False, binmap=False, cd_matrix=False, is_var=None, keep_keywords='default'):
        """Write the cube data to a FITS file.

        Parameters
        ----------
        filename : str or path-like
            Destination FITS file path.
        overwrite : bool, optional
            Overwrite the file if it exists. Default is ``False``.
        err : bool, optional
            Save the uncertainty array to the FITS file. Default is ``True``.
        mask : bool, optional
            Save the mask array. Default is ``False``.
        binmap : bool, optional
            Save the spatial binmap if present. Default is ``False``.
        cd_matrix : bool, optional
            Convert WCS keywords to the legacy CD matrix form. Default is ``False``.
        is_var : bool, optional
            If ``True``, save uncertainties as variance ('STAT'). Default infers from load.
        keep_keywords : str or list, optional
            FITS headers to keep. Default is ``'default'``. To keep all headers, use ``'all'``.

        Examples
        --------
        >>> cube.write('output.fits', overwrite=True, err=True, mask=True)
        >>> # Convert spectral axis to nm and save again
        >>> cube.z = (cube.z * cube.zunit).to_value('nm')
        >>> cube.zunit = u.nm
        >>> cube.write('output_nm.fits', overwrite=True)
        """
        wcs_header = self.wcs.to_header() if self.wcs else fits.Header()
        if self.wcs is not None and self.wcs.naxis >= 3:
            _preserve_wcs_axis_unit(wcs_header, self.wcs.naxis, self.zunit)

        if cd_matrix:
            wcs_to_cd_matrix(wcs_header)
            for i, j in [(1, 3), (2, 3), (3, 1), (3, 2)]:
                wcs_header.setdefault(f'CD{i}_{j}', 0.0)

        primary_header = fits.Header()
        values_header = wcs_header.copy()

        if keep_keywords:
            if keep_keywords == 'default' or keep_keywords is True:
                primary_keys = ['OBJECT', 'OBSERVER', 'PROG_ID', 'OBSTECH', 'TELESCOP', 'TELESCOPE', 'INSTRUME', 'INSTRUMENT', 'DATE-OBS', 'MJD-OBS', 'MJD-END', 'EXPTIME', 'TEXPTIME', 'DARKTIME', 'FLUXCAL', 'WAVELMIN', 'WAVELMAX']
                data_keys = ['BTYPE', 'SPEC_RES', 'SKY_RES', 'SKY_RERR', 'RA', 'DEC', 'EQUINOX', 'RADESYS', 'EPOCH', 'FILTER', 'GRISM', 'GRATING', 'SLIT']
            else:
                primary_keys = data_keys = keep_keywords

            if 'PRIMARY' in self.header:
                source_header = self.header['PRIMARY']
                if keep_keywords == 'all':
                    for key, value in source_header.items():
                        if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'COMMENT', 'HISTORY'] and not key.startswith('NAXIS'):
                            primary_header[key] = (value, source_header.comments[key])
                else:
                    for key in primary_keys:
                        if key in source_header:
                            primary_header[key] = (source_header[key], source_header.comments[key])

                if keep_keywords == 'all':
                    other_headers = [value for key, value in self.header.items() if key != 'PRIMARY']
                    for source_header in other_headers:
                        for key, value in source_header.items():
                            if key not in ['XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'] and not key.startswith('NAXIS') and key not in values_header:
                                values_header[key] = (value, source_header.comments[key])
                else:
                    for source_header in [value for key, value in self.header.items() if key != 'PRIMARY']:
                        for key in data_keys:
                            if key in source_header and key not in values_header:
                                values_header[key] = (source_header[key], source_header.comments[key])

        primary_header['AUTHOR'] = 'prism-spec'
        primary_header['DATE'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        primary_header['HISTORY'] = 'Processed and written by prism-spec'

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
                hdus.append(fits.ImageHDU(data=out_err ** 2, header=values_header.copy(), name='VAR'))
            else:
                hdus.append(fits.ImageHDU(data=out_err, header=values_header.copy(), name='ERR'))

        if mask and self.mask is not None:
            hdus.append(fits.ImageHDU(data=self.mask.astype(np.uint8), header=wcs_header, name='DQ'))

        if binmap and self.binmap is not None:
            hdus.append(fits.ImageHDU(data=self.binmap.astype(np.int32), header=wcs_header, name='BINMAP'))

        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)

    def get_image(self, min=None, max=None, weights=None, method='sum', propagate_err=True):
            """Collapse the cube along the spectral axis to create a 2-D Image.

            Parameters
            ----------
            min : float or astropy.units.Quantity, optional
                Minimum physical spectral value to include.
            max : float or astropy.units.Quantity, optional
                Maximum physical spectral value to include.
            weights : array-like, optional
                A 1-D array of weights (or a boolean mask) for the spectral channels. 
                Can be used to apply a filter transmission curve. Default is equal weights.
            method : {'sum', 'mean', 'median'}, optional
                Aggregation method for the spectral channels. Default is ``'sum'``.
                Note: 'median' is only supported if `weights` are purely boolean (0 or 1).
            propagate_err : bool, optional
                Whether to propagate cube uncertainties into the returned image.
                Default is ``True``. Set to ``False`` for faster quick-look images.

            Returns
            -------
            Image
                A new 2-D `Image` object.
                
            Examples
            --------
            >>> # Simple sum between 5000 and 5050
            >>> img = cube.get_image(min=5000, max=5050, method='sum')
            >>>
            >>> # Apply a synthetic filter transmission curve
            >>> transmission = my_filter_curve(cube.z)
            >>> img_band = cube.get_image(weights=transmission, method='sum')
            """
            from prism.data.image import Image

            if self.values.ndim != 3:
                raise ValueError('Image extraction is only supported for 3-D cubes.')

            try:
                image_wcs = self.wcs.celestial if self.wcs is not None else None
            except Exception:
                image_wcs = None

            # Handle weights and boolean masks natively
            if weights is not None:
                w = np.asarray(weights, dtype=self.dtype)
                if w.shape != (self.shape[0],):
                    raise ValueError(f"weights must be a 1-D array of length {self.shape[0]}")
            else:
                w = np.ones(self.shape[0], dtype=self.dtype)

            # Handle min/max boundaries
            axis_mask = w > 0  # Only process channels with non-zero weight to save memory
            min_value = _axis_bound_value(min, self.zunit, name='min')
            max_value = _axis_bound_value(max, self.zunit, name='max')
            if min_value is not None:
                axis_mask &= (self.z >= min_value)
            if max_value is not None:
                axis_mask &= (self.z <= max_value)

            if not np.any(axis_mask):
                raise ValueError("The provided min, max, or weights selected zero spectral channels.")

            selected = np.flatnonzero(axis_mask)
            if selected[-1] - selected[0] == selected.size - 1:
                spectral_index = slice(selected[0], selected[-1] + 1)
            else:
                spectral_index = selected

            valid_values = self.values[spectral_index]
            valid_w = w[spectral_index]
            pixel_mask = self.mask[spectral_index] if self.mask is not None else np.ones_like(valid_values, dtype=bool)
            weighted = not np.all(valid_w == 1)
            broadcast_w = valid_w[:, None, None]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                
                if method == 'sum':
                    value_terms = valid_values * broadcast_w if weighted else valid_values
                    value_terms = np.where(pixel_mask, value_terms, 0.0)
                    image_values = np.sum(value_terms, axis=0, dtype=self.dtype)
                    if propagate_err:
                        err_terms = self.err[spectral_index]
                        err_terms = (err_terms * broadcast_w) ** 2 if weighted else err_terms ** 2
                        err_terms = np.where(pixel_mask, err_terms, 0.0)
                        image_err = np.sqrt(np.sum(err_terms, axis=0, dtype=self.dtype))
                    else:
                        image_err = None
                    
                elif method == 'mean':
                    weight_sum = np.tensordot(valid_w, pixel_mask, axes=(0, 0)).astype(self.dtype, copy=False)
                    safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0) # Avoid division by zero
                    
                    value_terms = np.where(pixel_mask, valid_values * broadcast_w, 0.0)
                    image_values = np.sum(value_terms, axis=0, dtype=self.dtype) / safe_weight_sum
                    if propagate_err:
                        err_terms = (self.err[spectral_index] * broadcast_w) ** 2
                        err_terms = np.where(pixel_mask, err_terms, 0.0)
                        image_err = np.sqrt(np.sum(err_terms, axis=0, dtype=self.dtype)) / safe_weight_sum
                    else:
                        image_err = None
                    
                elif method == 'median':
                    if not np.all((valid_w == 1.0) | (valid_w == 0.0)):
                        raise ValueError("method='median' is only supported when weights are strictly boolean (0 or 1).")
                    
                    count = np.sum(pixel_mask, axis=0)
                    image_values = np.nanmedian(np.where(pixel_mask, valid_values, np.nan), axis=0).astype(self.dtype, copy=False)
                    if propagate_err:
                        err_terms = self.err[spectral_index] ** 2
                        image_err = np.sqrt(np.nansum(err_terms, axis=0, where=pixel_mask, dtype=self.dtype) / np.maximum(count, 1) ** 2) * 1.2533
                    else:
                        image_err = None
                    
                else:
                    raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

            # A collapsed pixel stays valid if at least one selected spectral
            # channel is valid. Masked spectral holes are excluded from the
            # collapse instead of invalidating the whole spatial pixel.
            image_mask = np.any(pixel_mask, axis=0)
            image_values[~image_mask] = np.nan
            if image_err is not None:
                image_err[~image_mask] = np.nan

            return Image(
                values=image_values,
                err=image_err,
                x=None if self.x is None else self.x.copy(),
                y=None if self.y is None else self.y.copy(),
                mask=image_mask,
                wcs=image_wcs,
                header=self.header.get('PRIMARY', self.header),
                unit=self.unit,
                xunit=self.xunit,
                yunit=self.yunit,
                binmap=None if self.binmap is None else self.binmap.copy(),
                xtype=self.xtype,
                ytype=self.ytype,
                valuetype=self.valuetype,
                is_var=False,
                dtype=self.dtype,
            )
