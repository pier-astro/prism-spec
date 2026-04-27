"""Generic 1-D spectral data container."""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
import astropy.units as u
from .core import (
    Data1D,
    convert_spectral_axis,
    is_wavelength_like,
    normalize_unit,
    parse_celestial_coord,
    parse_err,
    subset_axis_err,
)
from ..utils.tools import downsample_wave, resample_spectrum


c_kms = const.c.to_value(u.km / u.s)
script_dir = os.path.dirname(__file__)
dustpath = os.path.join(script_dir, '..', '..', 'resources', 'dust')


class Spectrum(Data1D):
    """Generic spectral container using coordinate/value naming."""

    def __init__(
        self,
        x,
        y,
        *,
        yerr=None,
        xerr=None,
        ra=None,
        dec=None,
        redshift=None,
        xunit=None,
        yunit=None,
        xtype=None,
        ytype=None,
        name='spectrum',
        meta=None,
    ):
        super().__init__(
            x=x,
            y=y,
            xerr=xerr,
            yerr=yerr,
            xunit=xunit,
            yunit=yunit,
            xtype=xtype,
            ytype=ytype,
            name=name,
            meta=meta,
        )

        self.ra = parse_celestial_coord(ra)
        self.dec = parse_celestial_coord(dec)
        self.redshift = redshift

        self._original_x = self.x.copy()
        self._original_y = self.y.copy()
        self._original_xerr = None if self.xerr is None else self.xerr.copy()
        self._original_yerr = parse_err(self.yerr, shape=self.y.shape, name='yerr')

        self._full_x = self._original_x.copy()
        self._full_y = self._original_y.copy()
        self._full_xerr = None if self._original_xerr is None else self._original_xerr.copy()
        self._full_yerr = self._original_yerr.copy()

        self.mask = np.ones_like(self._full_x, dtype=bool)
        self._rebinned = False
        self._update_working_arrays()

    def _update_working_arrays(self):
        if self.mask is None:
            self.x = self._full_x
            self.y = self._full_y
            self.xerr = self._full_xerr
            self.yerr = self._full_yerr
        else:
            self.x = self._full_x[self.mask]
            self.y = self._full_y[self.mask]
            self.xerr = subset_axis_err(self._full_xerr, self.mask)
            self.yerr = subset_axis_err(self._full_yerr, self.mask)

        if self.x.size > 1 and (self.xtype or '').lower() in {'wavelength', 'spectral'} and np.all(self.x > 0):
            log_x = np.log(self.x)
            mean_log_step = np.mean(np.diff(log_x))
            self.velscale = mean_log_step * c_kms
            self.fwhm = 2.355 * (np.exp(mean_log_step) - 1.0) * self.x
        else:
            self.velscale = np.nan
            self.fwhm = np.full_like(self.x, np.nan, dtype=float)

    @property
    def orig_x(self):
        return self._original_x

    @property
    def orig_y(self):
        return self._original_y

    @property
    def orig_xerr(self):
        return self._original_xerr

    @property
    def orig_yerr(self):
        return self._original_yerr

    @property
    def var(self):
        if self.yerr is None:
            return None
        if self.yerr.ndim != 1:
            raise ValueError('Variance view is only defined for symmetric yerr.')
        return self.yerr ** 2

    def copy(self):
        copied = self.__class__(
            x=self._original_x.copy(),
            y=self._original_y.copy(),
            xerr=None if self._original_xerr is None else self._original_xerr.copy(),
            yerr=self._original_yerr.copy(),
            ra=self.ra,
            dec=self.dec,
            redshift=self.redshift,
            xunit=self.xunit,
            yunit=self.yunit,
            xtype=self.xtype,
            ytype=self.ytype,
            name=self.name,
            meta=self.meta.copy(),
        )
        copied._full_x = self._full_x.copy()
        copied._full_y = self._full_y.copy()
        copied._full_xerr = None if self._full_xerr is None else self._full_xerr.copy()
        copied._full_yerr = self._full_yerr.copy()
        copied.mask = None if self.mask is None else self.mask.copy()
        copied._rebinned = self._rebinned
        copied._update_working_arrays()
        return copied

    def _require_symmetric_yerr(self):
        if self.yerr is None:
            return np.ones_like(self.y)
        if self.yerr.ndim != 1:
            raise ValueError('Operation requires symmetric yerr; asymmetric yerr is not supported here.')
        return self.yerr

    def _require_wavelength_axis(self, *, operation):
        if not is_wavelength_like(self.xtype, self.xunit):
            raise ValueError(f"{operation} is only defined for wavelength-like axes.")

    def crop(self, bounds=None, mask=None):
        if self._rebinned:
            raise RuntimeError('Cannot crop after rebinning.')

        original_indices = np.flatnonzero(self.mask)
        new_mask = np.zeros_like(self.mask, dtype=bool)

        if bounds is not None:
            xmin, xmax = bounds
            current_mask = (self.x > xmin) & (self.x < xmax)
            new_mask[original_indices] = current_mask
        elif mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.x.shape:
                raise ValueError('mask must match the current working x shape.')
            new_mask[original_indices] = mask
        else:
            raise ValueError('Either bounds or mask must be provided.')

        self.mask &= new_mask
        self._update_working_arrays()

    def rebin(self, factor=None, new_x=None, fill=np.nan, method='flux-conserving'):
        if self.yerr is not None and self.yerr.ndim != 1:
            raise ValueError('Rebinning does not support asymmetric yerr.')

        symmetric_yerr = self._require_symmetric_yerr()

        if new_x is not None:
            new_y, new_yerr = resample_spectrum(self.x, self.y, symmetric_yerr, new_wave=new_x, fill=fill, method=method)
            self._full_x = np.asarray(new_x, dtype=float)
        elif factor is not None and factor > 1:
            rebinned_x = downsample_wave(self.x, factor)
            new_y, new_yerr = resample_spectrum(self.x, self.y, symmetric_yerr, new_wave=rebinned_x, fill=fill, method=method)
            self._full_x = np.asarray(rebinned_x, dtype=float)
        else:
            raise ValueError('Either factor > 1 or new_x must be provided.')

        if self.mask is not None and not np.all(self.mask):
            warnings.warn('Rebinning removes all corrections. Reapply them after rebinning if needed.', UserWarning)

        self._full_y = new_y
        self._full_yerr = new_yerr
        self._full_xerr = None
        self.mask = None
        self._rebinned = True
        self._update_working_arrays()

    def reset(self):
        if self._rebinned:
            warnings.warn('Cannot reset a rebinned spectrum. Re-initialize the object instead.', UserWarning)
            return

        self._full_x = self._original_x.copy()
        self._full_y = self._original_y.copy()
        self._full_xerr = None if self._original_xerr is None else self._original_xerr.copy()
        self._full_yerr = self._original_yerr.copy()
        self.mask = np.ones_like(self._full_x, dtype=bool)
        self._update_working_arrays()

    def require_original_grid(self):
        if self._rebinned:
            raise RuntimeError('Operation requires the original coordinate grid, but the spectrum has been rebinned.')

    @classmethod
    def from_txt(cls, filename, ra=None, dec=None, redshift=None, xunit=None, yunit=None, name=None, xtype=None, ytype=None):
        try:
            try:
                x, y, yerr = np.genfromtxt(filename, unpack=True)
            except ValueError:
                x, y = np.genfromtxt(filename, unpack=True)
                yerr = None
        except Exception as exc:
            raise IOError(f'Error reading file {filename}. Check format.') from exc

        if name is None:
            name = os.path.splitext(os.path.basename(filename))[0]

        return cls(
            x=x,
            y=y,
            yerr=yerr,
            ra=ra,
            dec=dec,
            redshift=redshift,
            xunit=xunit,
            yunit=yunit,
            xtype=xtype,
            ytype=ytype,
            name=name,
            meta={'source': filename},
        )

    @classmethod
    def from_fits(
        cls,
        filename,
        ext=1,
        x_col='WAVELENGTH',
        y_col='FLUX',
        yerr_col='ERROR',
        ra=None,
        dec=None,
        redshift=None,
        name=None,
        xtype=None,
        ytype=None,
    ):
        with fits.open(filename) as hdul:
            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header

            colnames = data.columns.names
            x = data[x_col]
            y = data[y_col]
            yerr = data[yerr_col] if yerr_col in colnames else None

            xunit = header.get(f'TUNIT{colnames.index(x_col) + 1}', None)
            yunit = header.get(f'TUNIT{colnames.index(y_col) + 1}', None)

            ra = ra if ra is not None else header.get('RA')
            dec = dec if dec is not None else header.get('DEC')
            redshift = redshift if redshift is not None else header.get('REDSHIFT')
            name = name if name is not None else os.path.splitext(os.path.basename(filename))[0]

        return cls(
            x=x,
            y=y,
            yerr=yerr,
            ra=ra,
            dec=dec,
            redshift=redshift,
            xunit=xunit,
            yunit=yunit,
            xtype=xtype,
            ytype=ytype,
            name=name,
            meta={'source': filename},
        )

    def wavelengths(self, unit=None):
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'wavelength', unit)
        return values

    def frequencies(self, unit=None):
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'frequency', unit)
        return values

    def energies(self, unit=None):
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'energy', unit)
        return values

    def velocity(self, rest=None, unit=None, convention='doppler'):
        values, _ = convert_spectral_axis(
            self.x,
            self.xtype,
            self.xunit,
            'velocity',
            unit,
            rest=rest,
            convention=convention,
        )
        return values

    def plot_spectrum(self, ax=None):
        created_fig = False
        if ax is None:
            _, ax = plt.subplots()
            created_fig = True

        if self.yerr is not None and self.yerr.ndim == 1 and not np.allclose(self.yerr, 1.0):
            ax.errorbar(self.x, self.y, yerr=self.yerr, color='black', markersize=0, ls='none', alpha=0.5)
        ax.plot(self.x, self.y, label=self.name, color='black', drawstyle='steps-mid')
        ax.set_xlabel(f'{self.xtype} [{self.xunit or "Unknown"}]')
        ax.set_ylabel(f'{self.ytype} [{self.yunit or "Unknown"}]')
        ax.legend(frameon=False)
        if created_fig:
            plt.show()