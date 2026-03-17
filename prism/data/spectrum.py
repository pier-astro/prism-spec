"""
prism.data.spectrum — 1-D spectrum container.
"""
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
from sfdmap2 import sfdmap
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import os
import warnings

from ..utils.tools import resample_spectrum, downsample_wave, vac_to_air, air_to_vac

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
script_dir = os.path.dirname(__file__)
dustpath = os.path.join(script_dir, "..", "..", "resources", "dust")

class Spectrum():
    """
    Container for a 1-D astronomical spectrum.

    Internally maintains three data layers:

    * **original** — immutable copy of the input arrays.
    * **full** — the working arrays after corrections (z, dust, vac/air)
      applied on the full grid.
    * **working** — the masked view used for plotting and fitting
      (``wave``, ``flux``, ``err`` public attributes).

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    flux : array-like
        Flux array.
    err : array-like, optional
        1-sigma uncertainties.  If not provided, a unity array is used.
    ra, dec : float, optional
        Sky coordinates (degrees), used for dust-map queries.
    z : float, optional
        Source redshift.
    wave_unit : `~astropy.units.Unit` or str, optional
        Physical unit of the wavelength array.
    unit : `~astropy.units.Unit` or str, optional
        Physical unit of the flux array.
    _is_air_wave : bool
        Set to ``False`` if the wavelengths are in vacuum (default ``True``).
    name : str
        Label used in plots and output (default ``'spectrum'``).

    Attributes
    ----------
    wave, flux, err : ndarray
        Masked working arrays (update after every crop/correction).
    velscale : float
        Velocity scale per pixel [km s⁻¹] for a log-spaced grid.
    fwhm : ndarray
        FWHM at each pixel for a log-spaced grid [Å].
    """
    def __init__(self,
                 wave,
                 flux,
                 err=None,
                 ra=None,
                 dec=None,
                 z=None,
                 wave_unit=None,
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

        # Accept astropy units or strings for wave_unit and unit
        if wave_unit is not None and not isinstance(wave_unit, u.Unit):
            if isinstance(wave_unit, u.Quantity):
                wave_unit = wave_unit.unit
            else:
                try:
                    wave_unit = u.Unit(wave_unit)
                except Exception:
                    raise ValueError(f"Could not convert wave_unit '{wave_unit}' to astropy Unit.")
        if unit is not None and not isinstance(unit, u.Unit):
            if isinstance(unit, u.Quantity):
                unit = unit.unit
            else:
                try:
                    unit = u.Unit(unit)
                except Exception:
                    raise ValueError(f"Could not convert unit '{unit}' to astropy Unit.")
        self.wave_unit = wave_unit
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
        Crop the current working spectrum by wavelength bounds or mask.

        Parameters
        ----------
        wbounds : tuple(float, float), optional
            Inclusive wavelength range ``(wmin, wmax)`` in the current
            working frame.
        wmask : array-like of bool, optional
            Boolean mask on the current working arrays.

        Notes
        -----
        Cropping updates the internal mask on the original grid and is not
        allowed after ``rebin()``.
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
        Rebin the spectrum onto a new wavelength grid.

        Parameters
        ----------
        factor : int, optional
            Downsampling factor for regular grouping of current pixels.
        new_wave : array-like, optional
            Target wavelength grid.
        fill : float
            Fill value used for out-of-range bins.
        method : {'flux-conserving', 'mean', 'median'}
            Resampling strategy.

        Notes
        -----
        Rebinning is treated as permanent for this instance: subsequent
        ``crop()`` and ``reset()`` operations are disabled.
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
        """Raise ``RuntimeError`` if the spectrum has been rebinned."""
        if self._rebinned:
            raise RuntimeError("Operation requires the original wavelength grid, but spectrum has been rebinned.")

    @classmethod
    def from_txt(cls, filename, ra=None, dec=None, z=None, wave_unit=None, unit=None, name=None):
        """
        Build a ``Spectrum`` from a plain-text file.

        Expects two or three columns: ``wave flux [err]``.
        """
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

        return cls(wave=wave, flux=flux, err=err, ra=ra, dec=dec, z=z, wave_unit=wave_unit, unit=unit, name=name)

    @classmethod
    def from_fits(cls, filename, ext=1,
                  wave_col='WAVELENGTH', flux_col='FLUX', err_col='ERROR',
                  ra=None, dec=None, z=None, name=None):
        """
        Build a ``Spectrum`` from a table-like FITS extension.

        Parameters
        ----------
        ext : int or str
            FITS extension index/name containing tabular spectral columns.
        wave_col, flux_col, err_col : str
            Column names used to load wavelength, flux, and uncertainty.
        """
        with fits.open(filename) as hdul:
            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header

            colnames = data.columns.names
            wave = data[wave_col]
            flux = data[flux_col]
            err = data[err_col] if err_col in colnames else None

            wave_unit = header.get(f'TUNIT{colnames.index(wave_col)+1}', None)
            unit = header.get(f'TUNIT{colnames.index(flux_col)+1}', None)

            ra = ra if ra is not None else header.get('RA')
            dec = dec if dec is not None else header.get('DEC')
            z = z if z is not None else header.get('REDSHIFT')
            name = name if name is not None else os.path.splitext(os.path.basename(filename))[0]

            return cls(wave=wave, flux=flux, err=err, ra=ra, dec=dec, z=z,
                       wave_unit=wave_unit, unit=unit, name=name)
        
    def deredden(self, ebv=None):
        """
        Apply Galactic dereddening to flux and error arrays.

        If ``ebv`` is not provided, ``ra/dec`` must be available and an
        ``SFDMap`` query is performed.
        """
        if self._dereddened:
            warnings.warn("Spectrum is already dereddened. Skipping.", UserWarning)
            return
        if ebv is not None:
            self.ebv = ebv
        elif self.ra is not None and self.dec is not None:
            m = sfdmap.SFDMap(dustpath)
            self.ebv = m.ebv(self.ra, self.dec)
        else:
            raise ValueError("Cannot determine E(B-V). Provide ra/dec or an ebv value.")
        
        # Apply correction to the full, unmasked data
        self._full_flux = pyasl.unred(self._full_wave, self._full_flux, self.ebv, R_V=3.1)
        self._full_err = pyasl.unred(self._full_wave, self._full_err, self.ebv, R_V=3.1)
        
        self._dereddened = True
        self._update_working_arrays()

    def zcorrect(self, redshift=None):
        """
        Shift spectrum from observed to rest frame using ``1 + z``.

        Wavelength is divided by ``(1+z)`` while flux/error are multiplied
        by ``(1+z)``.
        """
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
        """Convert wavelength array from vacuum to air scale in-place."""
        if self._is_air_wave:
            warnings.warn("Spectrum is already in air wavelengths. Skipping.", UserWarning)
            return
        self._full_wave = vac_to_air(self._full_wave)
        self._is_air_wave = True
        self._update_working_arrays()

    def air_to_vac(self):
        """Convert wavelength array from air to vacuum scale in-place."""
        if not self._is_air_wave:
            warnings.warn("Spectrum is already in vacuum wavelengths. Skipping.", UserWarning)
            return
        self._full_wave = air_to_vac(self._full_wave)
        self._is_air_wave = False
        self._update_working_arrays()

    def plot_spectrum(self, ax=None):
        """
        Plot the working spectrum using Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axis. If omitted, a new figure/axis is created.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        if not np.all(self.err == 1):
            ax.errorbar(self.wave, self.flux, yerr=self.err, color="black", markersize=0, ls='none', alpha=0.5)
        ax.plot(self.wave, self.flux, label=self.name, color='black', drawstyle='steps-mid')
        ax.set_xlabel(f'Wavelength [{self.wave_unit or "Unknown"}]')
        ax.set_ylabel(f'Flux [{self.unit or "Unknown"}]')
        ax.legend(frameon=False)
        if created_fig:
            plt.show()