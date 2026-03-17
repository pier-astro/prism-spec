"""
prism.data.image — 2-D image container.
"""
import numpy as np
import warnings
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from datetime import datetime
import prism.config

class Image:
    """
    Container for a 2-D astronomical image (flux, error, mask).

    Designed to store spatially resolved maps of fitting results or
    collapsed cube slices.  Supports arithmetic operations between
    compatible ``Image`` objects and numpy arrays.

    Parameters
    ----------
    data : array-like, shape (N_y, N_x) or (N_spaxel,)
        The 2-D (or flattened 1-D) flux data array.
    err : array-like, same shape as data, optional
        Standard-deviation error array.  Mutually exclusive with ``var``.
    var : array-like, same shape as data, optional
        Variance array.  Mutually exclusive with ``err``.
    mask : array-like (bool), same shape as data, optional
        Boolean mask (``True`` = valid pixel).
    wcs : `~astropy.wcs.WCS`, optional
        Spatial WCS.
    header : dict, optional
        Mapping of extension names to `~astropy.io.fits.Header` objects.
    unit : `~astropy.units.Unit` or str, optional
        Physical unit of the data.
    is_var : bool, optional
        Whether the supplied error extension is variance (``True``) or
        standard deviation (``False``).
    """

    __array_priority__ = 1000
    def __init__(self, data, err=None, var=None, mask=None, wcs=None,
                 header=None, unit=None, is_var=None):
        data = np.asarray(data)
        
        # Determine shapes
        self.shape = data.shape
        if len(self.shape) not in [1, 2]:
            raise ValueError("Data must be either 2D (y, x) or 1D (spaxel).")

        self._data = data
        
        if err is not None and var is not None:
            raise ValueError("Cannot provide both 'err' and 'var'. They are mutually exclusive.")
        if var is not None:
            var_array = np.asarray(var)
            if var_array.shape != self.shape:
                raise ValueError(f"Variance shape mismatch: expected {self.shape}, got {var_array.shape}")
            if np.any(var_array < 0):
                raise ValueError("Variance must be non-negative.")
            self._err = np.sqrt(var_array)
            self.is_var = True if is_var is None else is_var
        elif err is not None:
            err_array = np.asarray(err)
            if err_array.shape != self.shape:
                raise ValueError(f"Error shape mismatch: expected {self.shape}, got {err_array.shape}")
            self._err = err_array
            self.is_var = False if is_var is None else is_var
        else:
            self._err = np.zeros_like(self._data)
            self.is_var = False if is_var is None else is_var
        
        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(self._data, dtype=bool)
        if self.mask.shape != self.shape:
            raise ValueError(f"Mask shape mismatch: expected {self.shape}, got {self.mask.shape}")
        self.wcs = wcs
        
        self.header = header if header is not None else {}
        if isinstance(unit, str):
            self.unit = u.Unit(unit)
        elif isinstance(unit, u.Quantity):
            self.unit = unit.unit
        else:
            self.unit = unit

    @property
    def data(self):
        """Return the flux data."""
        return self._data

    @data.setter
    def data(self, value):
        data_array = np.asarray(value)
        if data_array.shape != self.shape:
            raise ValueError(f"Data shape mismatch: expected {self.shape}, got {data_array.shape}")
        self._data = data_array
        
    @property
    def err(self):
        """Return the error (standard deviation) data."""
        return self._err

    @err.setter
    def err(self, value):
        err_array = np.asarray(value)
        if err_array.shape != self.shape:
            raise ValueError(f"Error shape mismatch: expected {self.shape}, got {err_array.shape}")
        self._err = err_array
        self.is_var = False
        
    @property
    def var(self):
        """Return the variance (error squared)."""
        return self._err ** 2

    @var.setter
    def var(self, value):
        var_array = np.asarray(value)
        if var_array.shape != self.shape:
            raise ValueError(f"Variance shape mismatch: expected {self.shape}, got {var_array.shape}")
        if np.any(var_array < 0):
            raise ValueError("Variance must be non-negative.")
        self._err = np.sqrt(var_array)
        self.is_var = True

    def _as_operand_array(self, other):
        """Normalize arithmetic operand type (scalar, ndarray, or ``Image``)."""
        if np.isscalar(other):
            return "scalar", other
        if isinstance(other, Image):
            return "image", other
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return "array", other
        raise TypeError(f"Operand must be an Image, scalar, or numpy.ndarray, got {type(other)}")

    def _check_compatibility(self, other):
        """Validate shape and WCS compatibility for image arithmetic."""
        kind, parsed_other = self._as_operand_array(other)
        if kind in {"scalar", "array"}:
            return True
        if self.shape != parsed_other.shape:
            if not prism.config.FORCE_OP_MISMATCH:
                raise ValueError(f"Shape mismatch: {self.shape} vs {parsed_other.shape}")
        
        # Simple WCS check
        if self.wcs is not None and parsed_other.wcs is not None:
            # We can compare WCS to_header string roughly or basic properties
            wcs1 = self.wcs.to_header().tostring()
            wcs2 = parsed_other.wcs.to_header().tostring()
            if wcs1 != wcs2:
                if not prism.config.FORCE_OP_MISMATCH:
                    raise ValueError("WCS mismatch. Set prism.config.FORCE_OP_MISMATCH = True to ignore.")
        return True

    def _apply_op(self, other, op_data, op_err):
        """Apply binary operation and propagate errors/mask to a new ``Image``."""
        self._check_compatibility(other)

        kind, parsed_other = self._as_operand_array(other)

        if kind == "scalar":
            new_data = op_data(self.data, parsed_other)
            new_err = op_err(self.err, 0, self.data, parsed_other)
            new_mask = self.mask.copy()
        elif kind == "array":
            new_data = op_data(self.data, parsed_other)
            new_err = op_err(self.err, 0, self.data, parsed_other)
            new_mask = self.mask.copy()
        else:
            new_data = op_data(self.data, parsed_other.data)
            new_err = op_err(self.err, parsed_other.err, self.data, parsed_other.data)
            new_mask = self.mask & parsed_other.mask
            
        return Image(data=new_data, err=new_err, mask=new_mask, wcs=self.wcs, 
                     header=self.header, unit=self.unit, is_var=False)

    def __add__(self, other):
        """Add scalar/array/image with standard error propagation."""
        return self._apply_op(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __sub__(self, other):
        """Subtract scalar/array/image with standard error propagation."""
        return self._apply_op(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __mul__(self, other):
        """Multiply by scalar/array/image with propagated uncertainties."""
        def err_mul(e1, e2, d1, d2):
            return np.sqrt((d2 * e1)**2 + (d1 * e2)**2)
        return self._apply_op(other, np.multiply, err_mul)

    def __truediv__(self, other):
        """Divide by scalar/array/image with propagated uncertainties."""
        def err_div(e1, e2, d1, d2):
            return np.sqrt((e1 / d2)**2 + ((d1 * e2) / d2**2)**2)
        return self._apply_op(other, np.divide, err_div)

    def _apply_rop(self, other, op_data, op_err):
        """Apply reversed binary op ``other op self`` for scalar/array operands."""
        kind, parsed_other = self._as_operand_array(other)
        if kind == "image":
            return NotImplemented

        if kind == "scalar":
            left_data = parsed_other
            left_err = 0
            new_data = op_data(left_data, self.data)
            new_err = op_err(left_err, self.err, left_data, self.data)
            new_mask = self.mask.copy()
        elif kind == "array":
            left_data = parsed_other
            left_err = 0
            new_data = op_data(left_data, self.data)
            new_err = op_err(left_err, self.err, left_data, self.data)
            new_mask = self.mask.copy()
        else:
            return NotImplemented

        return Image(data=new_data, err=new_err, mask=new_mask, wcs=self.wcs,
                     header=self.header, unit=self.unit, is_var=False)

    def __radd__(self, other):
        """Right-hand addition: ``other + self``."""
        return self._apply_rop(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __rsub__(self, other):
        """Right-hand subtraction: ``other - self``."""
        return self._apply_rop(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __rmul__(self, other):
        """Right-hand multiplication: ``other * self``."""
        def err_mul(e1, e2, d1, d2):
            return np.sqrt((d2 * e1)**2 + (d1 * e2)**2)
        return self._apply_rop(other, np.multiply, err_mul)

    def __rtruediv__(self, other):
        """Right-hand division: ``other / self``."""
        def err_div(e1, e2, d1, d2):
            return np.sqrt((e1 / d2)**2 + ((d1 * e2) / d2**2)**2)
        return self._apply_rop(other, np.divide, err_div)

    @classmethod
    def from_fits(cls, filename, ext_data=None, ext_err=None, ext_var=None, ext_mask=None, ext_wcs=None):
        """
        Build an ``Image`` from a FITS file.

        Parameters
        ----------
        filename : str
            Input FITS path.
        ext_data, ext_err, ext_var, ext_mask, ext_wcs : int or str, optional
            Extension selectors. If omitted, common extension names are
            auto-detected (DATA, ERR/STAT/VAR, MASK/DQ).
        """
        headers = {}
        with fits.open(filename) as hdul:
            ext_names = [hdu.name.upper() for hdu in hdul]
            
            # Auto-detect extensions if not provided
            if ext_data is None:
                if 'DATA' in ext_names:
                    ext_data = 'DATA'
                else:
                    ext_data = 0 # Default to primary
                    
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
            
            if ext_wcs is None:
                ext_wcs = ext_data

            # Load Data
            try:
                data = hdul[ext_data].data
                # Force to be 2D if it's not (some FITS might have degenerate dimensions)
                data = np.squeeze(data)
                headers['DATA'] = hdul[ext_data].header.copy()
            except KeyError:
                raise ValueError(f"Could not find data extension {ext_data} in {filename}")

            # Load WCS
            try:
                header_wcs = hdul[ext_wcs].header
                wcs = WCS(header_wcs)
                if wcs.naxis > 2:
                    wcs = wcs.celestial
                if ext_wcs != ext_data and isinstance(ext_wcs, str) and ext_wcs not in headers:
                    headers[ext_wcs] = hdul[ext_wcs].header.copy()
            except KeyError:
                raise ValueError(f"Could not find WCS extension {ext_wcs} in {filename}")
            
            # Unit
            bunit_str = headers['DATA'].get('BUNIT', None)
            unit = None
            if bunit_str:
                try:
                    unit = u.Unit(bunit_str)
                except ValueError:
                    warnings.warn(f"Could not parse BUNIT '{bunit_str}' into astropy.units.")
            
            # Load Err/Var
            err = None
            var = None
            if ext_err is not None:
                try:
                    if ext_err in hdul or (isinstance(ext_err, str) and ext_err in hdul):
                        err_ext_data = np.squeeze(hdul[ext_err].data)
                        if err_ext_data.shape == data.shape:
                            err = err_ext_data
                            headers[str(ext_err)] = hdul[ext_err].header.copy()
                except Exception as e:
                    warnings.warn(f"Failed to load error from extension {ext_err}: {e}")
            elif ext_var is not None:
                try:
                    if ext_var in hdul or (isinstance(ext_var, str) and ext_var in hdul):
                        var_ext_data = np.squeeze(hdul[ext_var].data)
                        if var_ext_data.shape == data.shape:
                            var = var_ext_data
                            key_name = 'VAR' if ext_var == 'STAT' else str(ext_var)
                            headers[key_name] = hdul[ext_var].header.copy()
                except Exception as e:
                    warnings.warn(f"Failed to load variance from extension {ext_var}: {e}")
            
            # Load Mask
            mask = None
            if ext_mask is not None:
                try:
                    if ext_mask in hdul or (isinstance(ext_mask, str) and ext_mask in hdul):
                        mask_data = np.squeeze(hdul[ext_mask].data)
                        if mask_data.shape == data.shape:
                            mask = mask_data.astype(bool)
                except Exception as e:
                    warnings.warn(f"Failed to load mask from extension {ext_mask}: {e}")

            if mask is None:
                mask = np.isfinite(data)
                if err is not None:
                    mask &= np.isfinite(err)
                elif var is not None:
                    mask &= np.isfinite(var)

        return cls(data=data, err=err, var=var, mask=mask, wcs=wcs, header=headers, unit=unit)

    def write(self, filename, overwrite=False, err=True, mask=False, cd_matrix=False, is_var=None):
        """
        Write image data to a FITS file.

        Parameters
        ----------
        filename : str
            Output FITS path.
        overwrite : bool
            Overwrite existing file.
        err : bool
            Write uncertainty extension (``ERR`` or ``STAT``).
        mask : bool
            Write mask extension (``DQ``).
        cd_matrix : bool
            Export WCS in legacy CD-matrix convention.
        is_var : bool, optional
            Force uncertainty extension to variance (``True``) or standard
            deviation (``False``). Defaults to ``self.is_var``.
        """
        from prism.data.cube import wcs_to_cd_matrix
        
        phdu_header = fits.Header()
        ext_header = fits.Header()
        err_header = fits.Header()
        mask_header = fits.Header()

        if self.wcs is not None:
            wcs_hdr = self.wcs.to_header()
            if cd_matrix:
                wcs_to_cd_matrix(wcs_hdr)
            ext_header.update(wcs_hdr)
            err_header.update(wcs_hdr)
            mask_header.update(wcs_hdr)

        if self.unit is not None:
            unit_str = self.unit.to_string(format='fits')
            ext_header['BUNIT'] = unit_str
        
        out_data = self._data.copy().astype(np.float32)
        if self.mask is not None:
            out_data[~self.mask] = np.nan
            
        hdus = [fits.PrimaryHDU(header=phdu_header)]
        hdus.append(fits.ImageHDU(data=out_data, header=ext_header, name='DATA'))
        
        if err:
            out_err = self._err.copy().astype(np.float32)
            if self.mask is not None:
                out_err[~self.mask] = np.nan
                
            write_var = self.is_var if is_var is None else is_var
            if write_var:
                out_err = out_err**2
                err_hdr = err_header.copy()
                if self.unit is not None:
                    err_hdr['BUNIT'] = (self.unit**2).to_string(format='fits')
                hdus.append(fits.ImageHDU(data=out_data, header=err_hdr, name='STAT'))
            else:
                err_hdr = err_header.copy()
                if self.unit is not None:
                    err_hdr['BUNIT'] = self.unit.to_string(format='fits')
                hdus.append(fits.ImageHDU(data=out_err, header=err_hdr, name='ERR'))

        if mask and self.mask is not None:
            out_mask = (~self.mask).astype(np.uint8)
            hdus.append(fits.ImageHDU(data=out_mask, header=mask_header, name='DQ'))
            
        phdu_header['AUTHOR'] = 'prism-spec'
        phdu_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        phdu_header['HISTORY'] = 'Processed and written by prism-spec'
        
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=overwrite)
