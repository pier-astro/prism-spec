from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from sfdmap2 import sfdmap
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import os
import warnings

from .tools import resample_spectrum, downsample_wave, vac_to_air, air_to_vac

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
script_dir = os.path.dirname(__file__)
sfdpath = os.path.join(script_dir, "sfddata")
plt.rcParams['axes.xmargin'] = 0

class Spectrum():
    def __init__(self,
                 wave,
                 flux,
                 err=None,
                 ra=None,
                 dec=None,
                 z=None,
                 wunit=None,
                 unit=None,
                 _is_air_wave=True,
                 name='spectrum'):

        if wave is None or flux is None:
            raise ValueError("`wave` and `flux` are required and cannot be None.")

        # --- Original data (never modified after __init__) ---
        self._original_wave = np.asarray(wave)
        self._original_flux = np.asarray(flux)
        self._original_err = np.asarray(err) if err is not None else np.ones_like(self._original_wave)

        # --- Full data grid (unmasked, but with corrections applied) ---
        self._full_wave = self._original_wave.copy()
        self._full_flux = self._original_flux.copy()
        self._full_err = self._original_err.copy()

        # --- State flags ---
        self._zcorrected = False
        self._dereddened = False
        self._is_air_wave = _is_air_wave
        self._rebinned = False
        
        self.name = name
        self.ra = ra
        self.dec = dec
        self.z = z
        self.ebv = None

        # Accept astropy units or strings for wunit and unit
        if wunit is not None and not isinstance(wunit, u.Unit):
            try:
                wunit = u.Unit(wunit)
            except Exception:
                raise ValueError(f"Could not convert wunit '{wunit}' to astropy Unit.")
        if unit is not None and not isinstance(unit, u.Unit):
            try:
                unit = u.Unit(unit)
            except Exception:
                raise ValueError(f"Could not convert unit '{unit}' to astropy Unit.")
        self.wunit = wunit
        self.unit = unit

        # --- Mask and Working Arrays ---
        self.mask = np.ones_like(self._full_wave, dtype=bool)
        self._update_working_arrays()

    def _update_working_arrays(self):
        """Update working arrays (wave, flux, err) based on the current mask and corrected originals."""
        if self.mask is not None:
            self.wave = self._full_wave[self.mask]
            self.flux = self._full_flux[self.mask]
            self.err = self._full_err[self.mask]
        else: # This happens after rebinning
            self.wave = self._full_wave
            self.flux = self._full_flux
            self.err = self._full_err

        # Compute derived quantities
        if self.wave is not None and len(self.wave) > 1:
            # Use mean step for non-uniform grids
            log_wave = np.log(self.wave)
            mean_log_step = np.mean(np.diff(log_wave))
            self.velscale = mean_log_step * c_kms
            self.fwhm = 2.355 * (np.exp(mean_log_step) - 1) * self.wave

    @property
    def orig_wave(self):
        """Return the original, unmodified wavelength array."""
        return self._original_wave

    @property
    def orig_flux(self):
        """Return the original, unmodified flux array."""
        return self._original_flux

    @property
    def orig_err(self):
        """Return the original, unmodified error array."""
        return self._original_err
    
    def crop(self, wbounds=None, wmask=None):
        """
        Crop the spectrum. Operates on the CURRENT working wavelength.
        The underlying mask is always on the original data grid.
        """
        if self._rebinned:
            raise RuntimeError("Cannot crop after rebinning.")

        orig_indices = np.flatnonzero(self.mask)
        new_mask = np.zeros_like(self.mask, dtype=bool)

        if wbounds is not None:
            wmin, wmax = wbounds
            current_mask = (self.wave > wmin) & (self.wave < wmax)
            new_mask[orig_indices] = current_mask
        elif wmask is not None:
            wmask = np.asarray(wmask)
            if wmask.shape != self.wave.shape:
                raise ValueError("wmask must have the same shape as the current (masked) wavelength array.")
            new_mask[orig_indices] = wmask
        else:
            raise ValueError("Either wbounds or wmask must be provided.")

        self.mask &= new_mask
        self._update_working_arrays()

    def rebin(self, factor:int=None, new_wave=None, fill=np.nan, method='flux-conserving'):
        """
        Rebin the spectrum. This is a permanent change that prevents further cropping or resets.
        """
        if new_wave is not None:
            new_flux, new_err = resample_spectrum(self.wave, self.flux, self.err, new_wave=new_wave, fill=fill, method=method)
            self._full_wave = new_wave
            self._full_flux = new_flux
            self._full_err = new_err
        elif factor is not None and factor > 1:
            new_wave = downsample_wave(self.wave, factor)
            new_flux, new_err = resample_spectrum(self.wave, self.flux, self.err, new_wave=new_wave, fill=fill, method=method)
            self._full_wave = new_wave
            self._full_flux = new_flux
            self._full_err = new_err
        else:
            raise ValueError("Either factor > 1 or new_wave must be provided.")
        
        # If the mask is not all True, advise the use that the rebinning removes corrections
        if self.mask is not None and not np.all(self.mask):
            warnings.warn("Rebinning removes all corrections. Consider reapplying them after rebinning.", UserWarning)

        self.mask = None
        self._rebinned = True
        self._update_working_arrays()

    def reset(self):
        """
        Reset the spectrum to its pristine state, undoing all corrections and cropping.
        """
        if self._rebinned:
            warnings.warn("Cannot reset a rebinned spectrum. Re-initialize the object instead.", UserWarning)
            return

        self._full_wave = self._original_wave.copy()
        self._full_flux = self._original_flux.copy()
        self._full_err = self._original_err.copy()
        
        self.mask = np.ones_like(self._full_wave, dtype=bool)
        self._zcorrected = False
        self._dereddened = False
        self._update_working_arrays()

    def require_original_grid(self):
        """Raise an error if not on the original wavelength grid."""
        if self._rebinned:
            raise RuntimeError("Operation requires the original wavelength grid, but spectrum has been rebinned.")

    @classmethod
    def from_txt(cls, filename, ra=None, dec=None, z=None, wunit=None, unit=None, name=None):
        """Create a Spectrum from a text file."""
        try:
            try:
                wave, flux, err = np.genfromtxt(filename, unpack=True)
            except ValueError:
                wave, flux = np.genfromtxt(filename, unpack=True)
                err = None
        except Exception as e:
            raise IOError(f"Error reading file {filename}. Check format.") from e
    
        if name is None:
            name = os.path.splitext(os.path.basename(filename))[0]

        return cls(wave=wave, flux=flux, err=err, ra=ra, dec=dec, z=z, wunit=wunit, unit=unit, name=name)

    @classmethod
    def from_fits(cls, filename, ext=1,
                  wave_col='WAVELENGTH', flux_col='FLUX', err_col='ERROR',
                  ra=None, dec=None, z=None, name=None):
        """Load a spectrum from a FITS file."""
        with fits.open(filename) as hdul:
            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header

            colnames = data.columns.names
            wave = data[wave_col]
            flux = data[flux_col]
            err = data[err_col] if err_col in colnames else None

            wunit = header.get(f'TUNIT{colnames.index(wave_col)+1}', None)
            unit = header.get(f'TUNIT{colnames.index(flux_col)+1}', None)

            ra = ra if ra is not None else header.get('RA')
            dec = dec if dec is not None else header.get('DEC')
            z = z if z is not None else header.get('REDSHIFT')
            name = name if name is not None else os.path.splitext(os.path.basename(filename))[0]

            return cls(wave=wave, flux=flux, err=err, ra=ra, dec=dec, z=z,
                       wunit=wunit, unit=unit, name=name)
        
    def deredden(self, ebv=None):
        """Deredden the flux using Fitzpatrick (1999) law."""
        if self._dereddened:
            warnings.warn("Spectrum is already dereddened. Skipping.", UserWarning)
            return
        if ebv is not None:
            self.ebv = ebv
        elif self.ra is not None and self.dec is not None:
            m = sfdmap.SFDMap(sfdpath)
            self.ebv = m.ebv(self.ra, self.dec)
        else:
            raise ValueError("Cannot determine E(B-V). Provide ra/dec or an ebv value.")
        
        # Apply correction to the full, unmasked data
        self._full_flux = pyasl.unred(self._full_wave, self._full_flux, self.ebv, R_V=3.1)
        self._full_err = pyasl.unred(self._full_wave, self._full_err, self.ebv, R_V=3.1)
        
        self._dereddened = True
        self._update_working_arrays()

    def zcorrect(self, redshift=None):
        """Correct the spectrum for redshift."""
        if self._zcorrected:
            warnings.warn("Spectrum is already redshift corrected. Skipping.", UserWarning)
            return
        if redshift is not None:
            self.z = redshift
        if self.z is None:
            raise ValueError("Cannot correct for redshift. Provide a z value.")

        # Apply correction to the full, unmasked data
        self._full_wave /= (1 + self.z)
        self._full_flux *= (1 + self.z)
        self._full_err *= (1 + self.z)
        
        self._zcorrected = True
        self._update_working_arrays()

    def vac_to_air(self):
        """Convert vacuum to air wavelengths."""
        if self._is_air_wave:
            warnings.warn("Spectrum is already in air wavelengths. Skipping.", UserWarning)
            return
        self._full_wave = vac_to_air(self._full_wave)
        self._is_air_wave = True
        self._update_working_arrays()

    def air_to_vac(self):
        """Convert air to vacuum wavelengths."""
        if not self._is_air_wave:
            warnings.warn("Spectrum is already in vacuum wavelengths. Skipping.", UserWarning)
            return
        self._full_wave = air_to_vac(self._full_wave)
        self._is_air_wave = False
        self._update_working_arrays()

    def plot_spectrum(self, ax=None):
        """Plot the spectrum using matplotlib."""
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        if not np.all(self.err == 1):
            ax.errorbar(self.wave, self.flux, yerr=self.err, color="black", markersize=0, ls='none', alpha=0.5)
        ax.plot(self.wave, self.flux, label=self.name, color='black', drawstyle='steps-mid')
        ax.set_xlabel(f'Wavelength [{self.wunit or "Unknown"}]')
        ax.set_ylabel(f'Flux [{self.unit or "Unknown"}]')
        ax.legend(frameon=False)
        if created_fig:
            plt.show()