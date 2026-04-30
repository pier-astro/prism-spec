"""Generic 3-D IFU cube container."""

from __future__ import annotations

from datetime import datetime
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


def _axis_indices(indexer, size, *, axis_name):
    if indexer is None:
        return slice(None), True
    if isinstance(indexer, slice):
        return indexer, True

    array = np.asarray(indexer)
    if array.dtype == bool:
        if array.shape != (size,):
            raise ValueError(f'{axis_name} boolean mask must have shape ({size},); got {array.shape}.')
        
        # Check if the boolean mask represents a single contiguous chunk
        true_indices = np.where(array)[0]
        if len(true_indices) > 0:
            if true_indices[-1] - true_indices[0] == len(true_indices) - 1:
                return slice(true_indices[0], true_indices[-1] + 1), True
        elif len(true_indices) == 0:
            return slice(0, 0), True  # Empty mask
            
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
        Valid-data mask with the same shape as ``values``. Default is all ``True``.
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
    ):
        values = np.asarray(values, dtype=float)
        if values.ndim not in {2, 3}:
            raise ValueError('values must be either 3-D (z, y, x) or 2-D (z, spaxel).')

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
            self._err = parse_err(err, shape=self.shape, name='err', fill_value=1.0)
            self.is_var = False if is_var is None else is_var

        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(self._values, dtype=bool)
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
        array = np.asarray(value, dtype=float)
        if array.shape != self.shape:
            old_shape = self.shape
            warnings.warn(
                'Changing cube shape via direct value assignment does not propagate WCS or bin maps. '
                'Prefer crop()/crop_spectral() for shape-changing operations.',
                UserWarning,
            )
            self.shape = array.shape
            if all(new <= old for new, old in zip(self.shape, old_shape)):
                slices = tuple(slice(0, size) for size in self.shape)
                self._err = self._err[slices]
                self.mask = self.mask[slices]
            else:
                self._err = np.full(self.shape, 1.0, dtype=float)
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
        self._err = parse_err(value, shape=self.shape, name='err', fill_value=1.0)
        self.is_var = False

    @property
    def var(self):
        """numpy.ndarray: Variance array of the cube values (square of `err`)."""
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
        )
        return copied

    def crop(self, zslice=None, yslice=None, xslice=None, inplace=False):
        """Extract a subcube by applying index slices or boolean masks along the axes.

        Parameters
        ----------
        zslice, yslice, xslice : slice, array-like integer indices, or array-like bool, optional
            Indexing information for the spectral (z) and spatial (y, x) axes. 
            Default is ``None`` for all axes (i.e. no cropping).
        inplace : bool, optional
            Modify the cube in place. Default is ``False``.

        Returns
        -------
        Cube
            The cropped cube object. WCS is updated if slices are contiguous.
            
        Examples
        --------
        >>> cube_cropped = cube.crop(yslice=slice(10, 50), xslice=slice(10, 50))
        >>> print(cube_cropped.shape)
        """
        target = self if inplace else self.copy()
        original_values = target.values
        original_err = target.err
        original_mask = target.mask

        zindex, z_is_slice = _axis_indices(zslice, target.shape[0], axis_name='zslice')
        if target.values.ndim == 3:
            yindex, y_is_slice = _axis_indices(yslice, target.shape[1], axis_name='yslice')
            xindex, x_is_slice = _axis_indices(xslice, target.shape[2], axis_name='xslice')
            indexers = (zindex, yindex, xindex)
            can_slice_wcs = z_is_slice and y_is_slice and x_is_slice
        else:
            if yslice is not None:
                raise ValueError('yslice is not supported for 2-D cubes; use xslice for the spatial axis.')
            xindex, x_is_slice = _axis_indices(xslice, target.shape[1], axis_name='xslice')
            indexers = (zindex, xindex)
            can_slice_wcs = z_is_slice and x_is_slice

        target._values = np.asarray(original_values[indexers], dtype=float)
        target._err = np.asarray(original_err[indexers], dtype=float)
        target.mask = np.asarray(original_mask[indexers], dtype=bool)
        target.shape = target._values.shape
        target.z = np.asarray(target.z[zindex], dtype=float)
        if target.values.ndim == 3:
            target.y = np.asarray(target.y[yindex], dtype=float)
            target.x = np.asarray(target.x[xindex], dtype=float)
            if target.binmap is not None:
                target.binmap = np.asarray(target.binmap[yindex, xindex], dtype=int)
        else:
            target.x = np.asarray(target.x[xindex], dtype=float)
            if target.y is not None:
                target.y = np.asarray(target.y[xindex], dtype=float)

        if target.wcs is not None:
            if can_slice_wcs:
                try:
                    target.wcs = target.wcs.slice(indexers)
                except Exception as exc:
                    warnings.warn(f'Failed to propagate WCS through crop: {exc}. Dropping WCS.', UserWarning)
                    target.wcs = None
            else:
                warnings.warn(
                    'Irregular cube crops do not preserve WCS exactly. Dropping WCS on the cropped result.',
                    UserWarning,
                )
                target.wcs = None

        return target

    def crop_spectral(self, slice=None, min=None, max=None, inplace=False):
            """Crop the cube along its spectral dimension only.

            Parameters
            ----------
            slice : slice, array-like integer indices, or array-like bool, optional
                Indexing information for the spectral axis. If provided, `min` and `max` 
                are ignored.
            min : float, optional
                Minimum physical spectral value to keep.
            max : float, optional
                Maximum physical spectral value to keep.
            inplace : bool, optional
                Modify the cube in place. Default is ``False``.

            Returns
            -------
            Cube
                The cropped cube object.
                
            Examples
            --------
            >>> # Crop using a boolean mask or index slice
            >>> cube_red = cube.crop_spectral(slice(100, 200))
            >>> 
            >>> # Crop using physical ranges (preserves WCS efficiently)
            >>> cube_range = cube.crop_spectral(min=4000, max=5000)
            """
            if slice is None:
                if min is None and max is None:
                    return self if inplace else self.copy()
                
                # Efficiently find indices handling both ascending and descending arrays
                ascending = self.z[-1] >= self.z[0] if len(self.z) > 1 else True
                
                if ascending:
                    start = np.searchsorted(self.z, min) if min is not None else None
                    end = np.searchsorted(self.z, max, side='right') if max is not None else None
                else:
                    z_rev = self.z[::-1]
                    start = len(self.z) - np.searchsorted(z_rev, max, side='right') if max is not None else None
                    end = len(self.z) - np.searchsorted(z_rev, min) if min is not None else None
                    
                # Use np.s_ to safely generate a slice object even if start/end are None
                slice = np.s_[start:end]

            return self.crop(zslice=slice, inplace=inplace)

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
        for _, mask in target.iter_bins():
            flat_mask = mask.reshape(-1)
            spectra = target.values.reshape(target.shape[0], -1)[:, flat_mask]
            spectra_err = target.err.reshape(target.shape[0], -1)[:, flat_mask]
            count = spectra.shape[1]
            if count == 0:
                continue
            if method == 'sum':
                binned = np.sum(spectra, axis=1)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1))
            elif method == 'mean':
                binned = np.mean(spectra, axis=1)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1)) / count
            elif method == 'median':
                binned = np.median(spectra, axis=1)
                sigma = np.sqrt(np.sum(spectra_err ** 2, axis=1)) / count * 1.2533 # ONLY CORRECT FOR GAUSSIAN ERRORS ON THE MEDIAN, NOT OTHER DISTRIBUTIONS
            else:
                raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

            new_values[:, mask] = binned[:, None]
            new_err[:, mask] = sigma[:, None]

        target._values = new_values
        target._err = new_err
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
        selected = flat_weights > 0

        if method == 'sum':
            flux = np.sum(spectra[:, selected] * flat_weights[selected], axis=1)
            sigma = np.sqrt(np.sum((spectra_err[:, selected] * flat_weights[selected]) ** 2, axis=1))
        elif method == 'mean':
            total_weight = np.sum(flat_weights[selected])
            flux = np.sum(spectra[:, selected] * flat_weights[selected], axis=1) / total_weight
            sigma = np.sqrt(np.sum((spectra_err[:, selected] * flat_weights[selected]) ** 2, axis=1)) / total_weight
        elif method == 'median':
            if not np.all((flat_weights == 0) | (flat_weights == 1)):
                raise ValueError('Fractional masks are not supported with method="median".')
            flux = np.median(spectra[:, selected], axis=1)
            sigma = np.sqrt(np.sum(spectra_err[:, selected] ** 2, axis=1)) / max(np.sum(selected), 1) * 1.2533
        else:
            raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

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

        values = np.asarray(nddata.data, dtype=float)
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
                values = np.asarray(hdul[ext_values].data, dtype=float)
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
                    err = np.asarray(hdul[ext_err].data, dtype=float)
                except Exception as exc:
                    warnings.warn(f'Failed to load error from extension {ext_err}: {exc}')
                    err = None
            elif ext_var is not None:
                try:
                    var = np.asarray(hdul[ext_var].data, dtype=float)
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
            FITS headers to keep. Default is ``'default'``.

        Examples
        --------
        >>> cube.write('output.fits', overwrite=True, err=True, mask=True)
        """
        wcs_header = self.wcs.to_header() if self.wcs else fits.Header()

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
        primary_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
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

    def get_image(self, min=None, max=None, weights=None, method='sum'):
            """Collapse the cube along the spectral axis to create a 2-D Image.

            Parameters
            ----------
            min : float, optional
                Minimum physical spectral value to include.
            max : float, optional
                Maximum physical spectral value to include.
            weights : array-like, optional
                A 1-D array of weights (or a boolean mask) for the spectral channels. 
                Can be used to apply a filter transmission curve. Default is equal weights.
            method : {'sum', 'mean', 'median'}, optional
                Aggregation method for the spectral channels. Default is ``'sum'``.
                Note: 'median' is only supported if `weights` are purely boolean (0 or 1).

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

            try:
                image_wcs = self.wcs.celestial if self.wcs is not None else None
            except Exception:
                image_wcs = None

            # Handle weights and boolean masks natively
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                if w.shape != (self.shape[0],):
                    raise ValueError(f"weights must be a 1-D array of length {self.shape[0]}")
            else:
                w = np.ones(self.shape[0], dtype=float)

            # Handle min/max boundaries
            axis_mask = w > 0  # Only process channels with non-zero weight to save memory
            if min is not None:
                axis_mask &= (self.z >= min)
            if max is not None:
                axis_mask &= (self.z <= max)

            if not np.any(axis_mask):
                raise ValueError("The provided min, max, or weights selected zero spectral channels.")

            # Slice down to valid channels
            valid_values = self.values[axis_mask]
            valid_err = self.err[axis_mask]
            valid_w = w[axis_mask][:, None, None]  # Broadcast weights to 3D (z, y, x)
            pixel_mask = self.mask[axis_mask] if self.mask is not None else np.ones_like(valid_values, dtype=bool)

            # Mask invalid spatial pixels
            masked_values = np.where(pixel_mask, valid_values, np.nan)
            masked_var = np.where(pixel_mask, valid_err ** 2, np.nan)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                
                if method == 'sum':
                    image_values = np.nansum(masked_values * valid_w, axis=0)
                    image_err = np.sqrt(np.nansum(masked_var * (valid_w ** 2), axis=0))
                    
                elif method == 'mean':
                    weight_sum = np.nansum(np.where(pixel_mask, valid_w, np.nan), axis=0)
                    safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0) # Avoid division by zero
                    
                    image_values = np.nansum(masked_values * valid_w, axis=0) / safe_weight_sum
                    image_err = np.sqrt(np.nansum(masked_var * (valid_w ** 2), axis=0)) / safe_weight_sum
                    
                elif method == 'median':
                    if not np.all((valid_w == 1.0) | (valid_w == 0.0)):
                        raise ValueError("method='median' is only supported when weights are strictly boolean (0 or 1).")
                    
                    count = np.sum(pixel_mask, axis=0)
                    image_values = np.nanmedian(masked_values, axis=0)
                    image_err = np.sqrt(np.nansum(masked_var, axis=0) / np.maximum(count, 1) ** 2) * 1.2533
                    
                else:
                    raise ValueError("method must be one of 'sum', 'mean', or 'median'.")

            # Re-mask spatial pixels that were completely invalid across all sliced spectral channels
            image_mask = np.sum(pixel_mask, axis=0) > 0
            image_values[~image_mask] = 0.0
            image_err[~image_mask] = 0.0

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
            )