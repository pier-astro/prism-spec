import numpy as np
import warnings
import os
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

class Cube:
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
        The standard deviation (error) array.
    mask : array-like (bool), same shape as data, optional
        Boolean mask array (True means valid, False means masked).
    wcs : `astropy.wcs.WCS`, optional
        The spatial/spectral WCS.
    """
    def __init__(self, data, wave=None, err=None, mask=None, wcs=None):
        data = np.asarray(data)
        
        # Determine shapes
        self.shape = data.shape
        if len(self.shape) not in [2, 3]:
            raise ValueError("Data must be either 3D (wave, y, x) or 2D (wave, spaxel).")
            
        self._data = data
        self._err = np.asarray(err) if err is not None else np.ones_like(self._data)
        
        if wave is not None:
            self._wave = np.asarray(wave)
            if len(self._wave) != self.shape[0]:
                raise ValueError("Wavelength array length must match the first dimension of data.")
        else:
            self._wave = np.arange(self.shape[0], dtype=float)

        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones_like(self._data, dtype=bool)
        
        self.wcs = wcs
        
        # State tracking
        self._zcorrected = False
        self.z = None
        
    @property
    def data(self):
        """Return the flux data."""
        return self._data
        
    @property
    def err(self):
        """Return the error (standard deviation) data."""
        return self._err
        
    @property
    def var(self):
        """Return the variance (error squared)."""
        return self._err ** 2
        
    @property
    def wave(self):
        """Return the wavelength array."""
        return self._wave

    @classmethod
    def from_fits(cls, filename, ext=1, ext_wcs=1, ext_err=2, ext_mask=None,
                  wave=None, wave_ext=None):
        """
        Load a Cube from a FITS file.
        Provides MPDAF-like flexibility for identifying extensions and
        automatically applying WCS fixes for known instruments (like MUSE).
        """
        with fits.open(filename) as hdul:
            # 1. Load Data
            try:
                data = hdul[ext].data
                header_wcs = hdul[ext_wcs].header
            except KeyError:
                raise ValueError(f"Could not find data extension {ext} or WCS extension {ext_wcs} in {filename}")

            # 2. Extract instrument-specific WCS fixes (e.g. MUSE)
            instrument = hdul[0].header.get('INSTRUME', '').strip().upper()
            if 'MUSE' in instrument:
                # Common MUSE WCS fix: Sometimes CUNIT3 is missing or uses non-standard terms
                if 'CUNIT3' not in header_wcs:
                    header_wcs['CUNIT3'] = 'Angstrom'
                # Remove alternate WCS that astropy might fail on
                for key in list(header_wcs.keys()):
                    if key.endswith('A') or key.endswith('B'):
                        del header_wcs[key]

            # 3. Create WCS
            # Suppress astropy WCS warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(header_wcs)
            
            # 4. Extract wavelength array if not explicitly provided
            if wave is None:
                if wave_ext is not None:
                    # If wavelength is explicitly in another extension 
                    # (e.g. sometimes provided as an image plane or table)
                    if isinstance(hdul[wave_ext].data, fits.FITS_rec):
                        # Attempt to guess column name if it's a table
                        col_names = [c.upper() for c in hdul[wave_ext].data.columns.names]
                        wave_col = [c for c in col_names if 'WAVE' in c]
                        if wave_col:
                            wave = hdul[wave_ext].data[wave_col[0]]
                        else:
                            wave = hdul[wave_ext].data
                    else:
                        wave = hdul[wave_ext].data
                else:
                    # Generate from WCS
                    if wcs.naxis >= 3:
                        # naxis=3 corresponds to wavelength in standard MUSE/IFU
                        # astropy.wcs.WCS drops degenerate dimensions. 
                        # To reliably get the 3rd axis:
                        n_wave = data.shape[0]
                        # Create pixel coordinates for the spectral axis, fixed spatial 0,0
                        pix_coords = np.zeros((n_wave, wcs.naxis))
                        # Typically in FITS (NAXIS1=x, NAXIS2=y, NAXIS3=wave).
                        # astropy pixel_to_world takes it in x, y, wave order.
                        pix_coords[:, wcs.naxis-1] = np.arange(n_wave)
                        world_coords = wcs.all_pix2world(pix_coords, 0)
                        wave = world_coords[:, wcs.naxis-1]
                        
                        # Apply multiplier to convert to Angstrom if necessary
                        if 'CUNIT3' in header_wcs and header_wcs['CUNIT3'].strip() == 'm':
                            wave *= 1e10 # m to Angstrom

            # 5. Load Error
            err = None
            if ext_err is not None:
                try:
                    # Check if the error extension exists and is correctly shaped
                    if ext_err in hdul or (isinstance(ext_err, str) and ext_err in hdul):
                        err_ext_data = hdul[ext_err].data
                        if err_ext_data.shape == data.shape:
                            # It is variance in MUSE, we want err (stddev)
                            if 'MUSE' in instrument and hdul[ext_err].header.get('EXTNAME', '') == 'STAT':
                                err = np.sqrt(err_ext_data)
                            else:
                                # Assume it's error natively or user specified
                                err = err_ext_data
                except Exception as e:
                    warnings.warn(f"Failed to load error from extension {ext_err}: {e}")
                    
            # 6. Load Mask
            mask = None
            if ext_mask is not None:
                try:
                    mask_data = hdul[ext_mask].data
                    if mask_data.shape == data.shape:
                        # Assuming 0 is bad, 1 is good (or invert if it's a bitmask)
                        mask = mask_data.astype(bool)
                except Exception as e:
                    warnings.warn(f"Failed to load mask from extension {ext_mask}: {e}")
                    
        return cls(data=data, wave=wave, err=err, mask=mask, wcs=wcs)

    def zcorrect(self, redshift=None):
        """Correct the cube spectra for redshift."""
        if self._zcorrected:
            warnings.warn("Cube is already redshift corrected. Skipping.", UserWarning)
            return
            
        if redshift is not None:
            self.z = redshift
            
        if self.z is None:
            raise ValueError("Cannot correct for redshift. Provide a z value.")

        # Apply correction to the data
        self._wave /= (1 + self.z)
        self._data *= (1 + self.z)
        self._err *= (1 + self.z)
        self._zcorrected = True

    def write(self, filename, overwrite=False):
        """Save the cube data to a FITS file. Masked values are set to np.nan."""
        header = self.wcs.to_header() if self.wcs else fits.Header()
        
        # Create copies to safely modify
        out_data = self._data.copy().astype(np.float32)
        out_err = self._err.copy().astype(np.float32)
        
        # Apply mask
        if self.mask is not None:
            # We assume mask=False means invalid/masked
            invalid = ~self.mask
            out_data[invalid] = np.nan
            out_err[invalid] = np.nan
            
        hdus = [fits.PrimaryHDU()]  # Empty primary HDU
        
        # Data extension
        data_hdu = fits.ImageHDU(data=out_data, header=header, name='DATA')
        hdus.append(data_hdu)
        
        # Error extension
        var_hdu = fits.ImageHDU(data=out_err, header=header, name='ERR')
        hdus.append(var_hdu)
        
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(filename, overwrite=overwrite)
