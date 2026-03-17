import numpy as np
import warnings
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import prism.config
from prism.data.image import Image


def wcs_to_cd_matrix(header):
    """
    Convert WCS PC matrix + CDELT to the legacy CD matrix format (ESO convention).
    Modifies the header in-place.
    """
    naxis = header.get('WCSAXES', 2)
    has_pc = any(f'PC{i}_{j}' in header
                 for i in range(1, naxis + 1) for j in range(1, naxis + 1))

    if has_pc:
        for i in range(1, naxis + 1):
            cdelt = header.get(f'CDELT{i}', 1.0)
            for j in range(1, naxis + 1):
                pc_key = f'PC{i}_{j}'
                if pc_key in header:
                    header[f'CD{i}_{j}'] = header[pc_key] * cdelt
                    del header[pc_key]
        for i in range(1, naxis + 1):
            if f'CDELT{i}' in header:
                del header[f'CDELT{i}']
    else:
        # No PC matrix, convert CDELTi to CDi_i
        for i in range(1, naxis + 1):
            cdelt_key = f'CDELT{i}'
            if cdelt_key in header:
                header[f'CD{i}_{i}'] = header[cdelt_key]
                del header[cdelt_key]
    return header


class Cube:
    __array_priority__ = 1000

    """
    A lightweight, Astropy-backed container for 3D integral field spectroscopic data,
    inspired by MPDAF and designed for robust spaxel-by-spaxel fitting.

    Parameters
    ----------
    data : array-like, shape (N_wave, N_y, N_x) or (N_wave, N_spaxel)
        The 3D (or flattened 2D) flux data array.
    wave : array-like, shape (N_wave,), optional
        The 1D wavelength array.
    err : array-like, same shape as data, optional
        The standard deviation (error) array. Mutually exclusive with `var`.
    var : array-like, same shape as data, optional
        The variance array. Mutually exclusive with `err`.
    mask : array-like (bool), same shape as data, optional
        Boolean mask array (True means valid, False means masked).
    wcs : `astropy.wcs.WCS`, optional
        The spatial/spectral WCS.
    header : dict, optional
        A dictionary mapping extension names (e.g. 'PRIMARY', 'DATA') to `astropy.io.fits.Header` objects.
    unit : `astropy.units.Unit` or str, optional
        The physical unit of the data (e.g. from BUNIT).
    is_var : bool, optional
        Flag indicating if the source error data was natively variance (True) or standard deviation (False).
    """
    def __init__(self, data, wave=None, err=None, var=None, mask=None, wcs=None,
                 header=None, unit=None, wave_unit=None, is_var=None):
        data = np.asarray(data)
        
        # Determine shapes
        self.shape = data.shape
        if len(self.shape) not in [2, 3]:
            raise ValueError("Data must be either 3D (wave, y, x) or 2D (wave, spaxel).")
            
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
            self._err = np.ones_like(self._data)
            self.is_var = False if is_var is None else is_var
        
        if wave is not None:
            self._wave = np.asarray(wave)
            if len(self._wave) != self.shape[0]:
                raise ValueError("Wavelength array length must match the first dimension of data.")
        else:
            self._wave = np.arange(self.shape[0], dtype=float)

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
        if isinstance(wave_unit, str):
            self.wave_unit = u.Unit(wave_unit)
        elif isinstance(wave_unit, u.Quantity):
            self.wave_unit = wave_unit.unit
        else:
            self.wave_unit = wave_unit if wave_unit is not None else u.AA
        
        # State tracking
        self._zcorrected = False
        self.z = None
        
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
        
    @property
    def wave(self):
        """Return the wavelength array."""
        return self._wave

    def _as_operand_array(self, other):
        if np.isscalar(other):
            return "scalar", other
        if isinstance(other, Cube):
            return "cube", other
        if isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return "array", other
        raise TypeError(f"Operand must be a Cube, scalar, or numpy.ndarray, got {type(other)}")

    def get_masked(self):
        """
        Return wavelength-consistent (wave, data, err) arrays with masked channels removed.

        A wavelength channel is excluded if **any** spaxel is masked at that channel,
        ensuring all returned arrays are free of NaN values and have matching shapes.

        Returns
        -------
        wave : ndarray, shape (N_valid,)
        data : ndarray, shape (N_valid, ...) or (N_valid, N_y, N_x)
        err  : ndarray, same shape as data
        """
        spatial_axes = tuple(range(1, self.mask.ndim))
        channel_valid = self.mask.all(axis=spatial_axes)  # (N_wave,)
        return self._wave[channel_valid], self._data[channel_valid], self._err[channel_valid]

    @classmethod
    def from_fits(cls, filename, ext_data=None, ext_err=None, ext_var=None, ext_mask=None,
                  wave=None, wave_ext=None, ext_wcs=None):
        """
        Load a Cube from a FITS file.
        Provides MPDAF-like flexibility for identifying extensions and
        automatically applying WCS fixes for known instruments (like MUSE).
        """
        with fits.open(filename) as hdul:
            headers = {'PRIMARY': hdul[0].header.copy()}
            
            # --- 1. Identify Extensions ---
            ext_names = [hdu.name.upper() for hdu in hdul]
            
            if ext_data is None:
                if 'DATA' in ext_names:
                    ext_data = 'DATA'
                elif 'SCI' in ext_names:
                    ext_data = 'SCI'
                else:
                    ext_data = 1 # Fallback
            
            if ext_wcs is None:
                ext_wcs = ext_data
                
            # Try to auto-detect error/variance
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

            # --- 2. Load Data ---
            try:
                data = hdul[ext_data].data
                headers['DATA'] = hdul[ext_data].header.copy() # Standardize naming to DATA in dict
            except KeyError:
                raise ValueError(f"Could not find data extension {ext_data} in {filename}")

            try:
                header_wcs = hdul[ext_wcs].header
                if ext_wcs != ext_data and isinstance(ext_wcs, str) and ext_wcs not in headers:
                    headers[ext_wcs] = hdul[ext_wcs].header.copy()
            except KeyError:
                 raise ValueError(f"Could not find WCS extension {ext_wcs} in {filename}")

            # Extract units
            bunit_str = headers['DATA'].get('BUNIT', None)
            unit = None
            if bunit_str:
                try:
                    unit = u.Unit(bunit_str)
                except ValueError:
                    warnings.warn(f"Could not parse BUNIT '{bunit_str}' into astropy.units.")

            # --- 3. Extract instrument-specific WCS fixes (e.g. MUSE) ---
            instrument = hdul[0].header.get('INSTRUME', '').strip().upper()
            if 'MUSE' in instrument:
                if 'CUNIT3' not in header_wcs:
                    header_wcs['CUNIT3'] = 'Angstrom'
                for key in list(header_wcs.keys()):
                    if key.endswith('A') or key.endswith('B'):
                        del header_wcs[key]

            # --- 4. Create WCS ---
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(header_wcs)
            
            # --- 5. Extract wavelength array ---
            if wave is None:
                if wave_ext is not None:
                    if isinstance(hdul[wave_ext].data, fits.FITS_rec):
                        col_names = [c.upper() for c in hdul[wave_ext].data.columns.names]
                        wave_col = [c for c in col_names if 'WAVE' in c]
                        if wave_col:
                            wave = hdul[wave_ext].data[wave_col[0]]
                        else:
                            wave = hdul[wave_ext].data
                    else:
                        wave = hdul[wave_ext].data
                else:
                    if wcs.naxis >= 3:
                        n_wave = data.shape[0]
                        pix_coords = np.zeros((n_wave, wcs.naxis))
                        pix_coords[:, wcs.naxis-1] = np.arange(n_wave)
                        world_coords = wcs.all_pix2world(pix_coords, 0)
                        wave = world_coords[:, wcs.naxis-1]
                        spectral_cunit = wcs.wcs.cunit[wcs.naxis - 1]
                        if spectral_cunit:
                            wave = (wave * u.Unit(spectral_cunit.to_string())).to(u.AA).value

            # --- 6. Load Error/Variance ---
            err = None
            var = None
            if ext_err is not None:
                try:
                    if ext_err in hdul or (isinstance(ext_err, str) and ext_err in hdul):
                        err_ext_data = hdul[ext_err].data
                        if err_ext_data.shape == data.shape:
                            err = err_ext_data
                            headers[str(ext_err)] = hdul[ext_err].header.copy()
                except Exception as e:
                    warnings.warn(f"Failed to load error from extension {ext_err}: {e}")
            elif ext_var is not None:
                try:
                    if ext_var in hdul or (isinstance(ext_var, str) and ext_var in hdul):
                        var_ext_data = hdul[ext_var].data
                        if var_ext_data.shape == data.shape:
                            var = var_ext_data
                            # Standardize key in dict if STAT was used
                            key_name = 'VAR' if ext_var == 'STAT' else str(ext_var)
                            headers[key_name] = hdul[ext_var].header.copy()
                except Exception as e:
                    warnings.warn(f"Failed to load variance from extension {ext_var}: {e}")
                    
            # --- 7. Load Mask ---
            mask = None
            if ext_mask is not None:
                try:
                    mask_data = hdul[ext_mask].data
                    if mask_data.shape == data.shape:
                        mask = mask_data.astype(bool)
                except Exception as e:
                    warnings.warn(f"Failed to load mask from extension {ext_mask}: {e}")

            # --- 8. Finite mask fallback ---
            # Only applied when no instrument mask was available.
            if mask is None:
                mask = np.isfinite(data)
                if err is not None:
                    mask &= np.isfinite(err)
                elif var is not None:
                    mask &= np.isfinite(var)

        return cls(data=data, wave=wave, err=err, var=var, mask=mask, wcs=wcs, header=headers, unit=unit, wave_unit=u.AA)

    def zcorrect(self, redshift=None):
        """Correct the cube spectra for redshift."""
        if self._zcorrected:
            warnings.warn("Cube is already redshift corrected. Skipping.", UserWarning)
            return
            
        if redshift is not None:
            self.z = redshift
            
        if self.z is None:
            raise ValueError("Cannot correct for redshift. Provide a z value.")

        self._wave /= (1 + self.z)
        self._data *= (1 + self.z)
        self._err *= (1 + self.z)
        self._zcorrected = True

    def undo_zcorrect(self):
        """Undo the redshift correction and return the cube to the observed frame."""
        if not self._zcorrected:
            warnings.warn("Cube is not redshift corrected. Skipping.", UserWarning)
            return

        if self.z is None:
            raise ValueError("Cannot undo redshift correction. No z value stored.")

        # Re-apply the (1+z) factor to restore the observed arrays
        self._wave *= (1 + self.z)
        self._data /= (1 + self.z)
        self._err /= (1 + self.z)
        self._zcorrected = False

    def write(self, filename, overwrite=False, err=True, mask=False, 
              cd_matrix=False, is_var=None, keep_keywords='default'):
        """Save the cube data to a FITS file. Masked values are set to np.nan."""
        from datetime import datetime
        wcs_header = self.wcs.to_header() if self.wcs else fits.Header()
        
        # Apply Redshift correction to the output WCS grid if modified
        if self._zcorrected and self.z is not None:
            z_scale = 1.0 / (1.0 + self.z)
            if 'CRVAL3' in wcs_header:
                wcs_header['CRVAL3'] *= z_scale
            if 'CDELT3' in wcs_header:
                wcs_header['CDELT3'] *= z_scale
            if 'CD3_3' in wcs_header:
                wcs_header['CD3_3'] *= z_scale
            if 'PC3_3' in wcs_header:
                wcs_header['PC3_3'] *= z_scale
        
        # WCS Legacy formatting
        if cd_matrix:
            # Convert spectral axis from SI (m) to Angstrom before building CD matrix
            if wcs_header.get('CUNIT3', '').strip() == 'm':
                wcs_header['CUNIT3'] = 'Angstrom'
                if 'CRVAL3' in wcs_header:
                    wcs_header['CRVAL3'] *= 1e10
                if 'CDELT3' in wcs_header:
                    wcs_header['CDELT3'] *= 1e10

            wcs_to_cd_matrix(wcs_header)

            # Explicitly zero the spatial–spectral cross-terms (ESO convention)
            for i, j in [(1, 3), (2, 3), (3, 1), (3, 2)]:
                if f'CD{i}_{j}' not in wcs_header:
                    wcs_header[f'CD{i}_{j}'] = 0.0
        
        # Base headers
        phdu_header = fits.Header()
        ext_header = wcs_header.copy()
        
        # Keyword filtering
        if keep_keywords:
            if keep_keywords == 'default' or keep_keywords is True:
                primary_keys = ['OBJECT', 'OBSERVER', 'PROG_ID', 'OBSTECH', 'TELESCOP', 'TELESCOPE', 
                                'INSTRUME', 'INSTRUMENT', 'DATE-OBS', 'MJD-OBS', 'MJD-END', 'EXPTIME', 
                                'TEXPTIME', 'DARKTIME', 'FLUXCAL', 'WAVELMIN', 'WAVELMAX']
                data_keys = ['BTYPE', 'SPEC_RES', 'SKY_RES', 'SKY_RERR', 'RA', 'DEC', 'EQUINOX', 
                             'RADESYS', 'EPOCH', 'FILTER', 'GRISM', 'GRATING', 'SLIT']
            else:
                primary_keys = data_keys = keep_keywords
            
            # Primary extraction
            if 'PRIMARY' in self.header:
                src_hdr = self.header['PRIMARY']
                if keep_keywords == 'all':
                    for k, v in src_hdr.items():
                        if k not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'COMMENT', 'HISTORY'] and not k.startswith('NAXIS'):
                            phdu_header[k] = (v, src_hdr.comments[k])
                else:
                    for k in primary_keys:
                        if k in src_hdr:
                            phdu_header[k] = (src_hdr[k], src_hdr.comments[k])
            
            # Data extraction (search across all other loaded headers for the keys if needed)
            src_hdrs = [v for k, v in self.header.items() if k != 'PRIMARY']
            if keep_keywords == 'all':
                for src_hdr in src_hdrs:
                    for k, v in src_hdr.items():
                        if k not in ['XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'] and not k.startswith('NAXIS') and k not in ext_header:
                            ext_header[k] = (v, src_hdr.comments[k])
            else:
                for src_hdr in src_hdrs:
                    for k in data_keys:
                        if k in src_hdr and k not in ext_header:
                            ext_header[k] = (src_hdr[k], src_hdr.comments[k])
        
        # Provenance
        phdu_header['AUTHOR'] = 'prism-spec'
        phdu_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        phdu_header['HISTORY'] = 'Processed and written by prism-spec'
        
        # Unit formatting
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
                err_hdu = fits.ImageHDU(data=out_err**2, header=ext_header.copy(), name='VAR')
            else:
                err_hdu = fits.ImageHDU(data=out_err, header=ext_header.copy(), name='ERR')
            hdus.append(err_hdu)
            
        if mask and self.mask is not None:
            mask_data = self.mask.astype(np.uint8)
            hdus.append(fits.ImageHDU(data=mask_data, header=wcs_header, name='DQ'))
            
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=overwrite)

    def _check_compatibility(self, other):
        kind, parsed_other = self._as_operand_array(other)
        if kind in {"scalar", "array"}:
            return True

        if self.shape != parsed_other.shape:
            if not prism.config.FORCE_OP_MISMATCH:
                raise ValueError(f"Shape mismatch: {self.shape} vs {parsed_other.shape}")
                
        if hasattr(self, 'wave') and hasattr(parsed_other, 'wave'):
            if not np.allclose(self.wave, parsed_other.wave, rtol=1e-5, atol=1e-5):
                if not prism.config.FORCE_OP_MISMATCH:
                    raise ValueError("Wavelength array mismatch. Set prism.config.FORCE_OP_MISMATCH = True to ignore.")
        
        if self.wcs is not None and parsed_other.wcs is not None:
            wcs1 = self.wcs.to_header().tostring()
            wcs2 = parsed_other.wcs.to_header().tostring()
            if wcs1 != wcs2:
                if not prism.config.FORCE_OP_MISMATCH:
                    raise ValueError("WCS mismatch. Set prism.config.FORCE_OP_MISMATCH = True to ignore.")
        return True

    def _apply_op(self, other, op_data, op_err):
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
            
        return Cube(data=new_data, wave=self.wave, err=new_err, mask=new_mask,
                    wcs=self.wcs, header=self.header, unit=self.unit,
                    wave_unit=self.wave_unit, is_var=False)

    def __add__(self, other):
        import numpy as np
        return self._apply_op(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __sub__(self, other):
        import numpy as np
        return self._apply_op(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __mul__(self, other):
        import numpy as np
        return self._apply_op(other, np.multiply, lambda e1, e2, d1, d2: np.sqrt((d2 * e1)**2 + (d1 * e2)**2))

    def __truediv__(self, other):
        import numpy as np
        return self._apply_op(other, np.divide, lambda e1, e2, d1, d2: np.sqrt((e1 / d2)**2 + ((d1 * e2) / d2**2)**2))

    def _apply_rop(self, other, op_data, op_err):
        """Helper to apply reversed operations: other (op) self."""
        kind, parsed_other = self._as_operand_array(other)
        if kind == "cube":
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

        return Cube(data=new_data, wave=self.wave, err=new_err, mask=new_mask,
                    wcs=self.wcs, header=self.header, unit=self.unit,
                    wave_unit=self.wave_unit, is_var=False)

    def __radd__(self, other):
        import numpy as np
        return self._apply_rop(other, np.add, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __rsub__(self, other):
        import numpy as np
        return self._apply_rop(other, np.subtract, lambda e1, e2, d1, d2: np.sqrt(e1**2 + e2**2))

    def __rmul__(self, other):
        import numpy as np
        return self._apply_rop(other, np.multiply, lambda e1, e2, d1, d2: np.sqrt((d2 * e1)**2 + (d1 * e2)**2))

    def __rtruediv__(self, other):
        import numpy as np
        return self._apply_rop(other, np.divide, lambda e1, e2, d1, d2: np.sqrt((e1 / d2)**2 + ((d1 * e2) / d2**2)**2))

    def to_image(self, wmask=None, wbounds=None, method='sum'):
        """
        Collapse the 3D cube along its spectral axis to create a 2D Image.

        Parameters
        ----------
        wmask : array-like (bool), optional
            1D mask along the spectral axis. True means to include the wavelength.
        wbounds : tuple of float, optional
            Tuple (wave_min, wave_max) to limit the wavelengths used.
        method : {'sum', 'mean', 'median'}
            The collapsing method to use.

        Returns
        -------
        Image
            The collapsed 2D Image.
        """
        import numpy as np
        import warnings
        from prism.data.image import Image

        try:
            image_wcs = self.wcs.celestial if self.wcs is not None else None
        except Exception:
            image_wcs = None

        wave_mask = np.ones(self.shape[0], dtype=bool)
        if wmask is not None:
            wave_mask &= np.asarray(wmask, dtype=bool)
        if wbounds is not None:
            wave_mask &= (self.wave >= wbounds[0]) & (self.wave <= wbounds[1])

        valid_data = self.data[wave_mask]
        valid_err = self.err[wave_mask]
        
        if self.mask is not None:
            pixel_mask = self.mask[wave_mask]
        else:
            pixel_mask = np.ones_like(valid_data, dtype=bool)

        masked_data = np.where(pixel_mask, valid_data, np.nan)
        masked_var = np.where(pixel_mask, valid_err**2, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method == 'sum':
                img_data = np.nansum(masked_data, axis=0)
                img_var = np.nansum(masked_var, axis=0)
                img_err = np.sqrt(img_var)
            elif method == 'mean':
                count = np.sum(pixel_mask, axis=0)
                img_data = np.nanmean(masked_data, axis=0)
                img_var = np.nansum(masked_var, axis=0) / np.maximum(count, 1)**2
                img_err = np.sqrt(img_var)
            elif method == 'median':
                count = np.sum(pixel_mask, axis=0)
                img_data = np.nanmedian(masked_data, axis=0)
                img_var = np.nansum(masked_var, axis=0) / np.maximum(count, 1)**2
                img_err = np.sqrt(img_var) * 1.2533
            else:
                raise ValueError(f"Method {method} not supported. Use 'sum', 'mean', or 'median'.")

        img_mask = np.sum(pixel_mask, axis=0) > 0
        img_data[~img_mask] = 0.0
        img_err[~img_mask] = 0.0

        return Image(data=img_data, err=img_err, mask=img_mask, wcs=image_wcs, 
                     header=self.header.get("PRIMARY", None), unit=self.unit, is_var=False)
