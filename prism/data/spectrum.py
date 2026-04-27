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
    """One-dimensional spectral container with reversible masking and rebinning.

    Parameters
    ----------
    x, y : array-like
        Spectral coordinate array and sampled values.
    yerr : array-like, optional
        Symmetric 1-D uncertainties on ``y``. Default is ``None``.
    xerr : array-like, optional
        Optional coordinate uncertainties. Default is ``None``.
    ra, dec : float, str, or astropy.units.Quantity, optional
        Sky coordinates associated with the spectrum. Default is ``None``.
    redshift : float, optional
        Source redshift stored with the spectrum. Default is ``None``.
    xunit, yunit : str or astropy.units.Unit, optional
        Units of the coordinate and value arrays. Default is ``None``.
    xtype, ytype : str, optional
        Explicit semantic labels for the axis and values. Default is inferred
        from the units when possible.
    name : str, optional
        Spectrum label. Default is ``'spectrum'``.
    meta : dict, optional
        Arbitrary metadata copied with the object. Default is ``None``.

    Notes
    -----
    ``Spectrum`` keeps both the original arrays and the currently active working
    arrays. :meth:`crop` updates a boolean mask on the original grid, so
    :meth:`reset` can restore the initial sampling. :meth:`rebin` instead builds a
    new spectral grid and therefore invalidates ``reset`` because the original
    sampling has been replaced. For wavelength-like logarithmic axes Prism also
    derives the approximate velocity scale and per-pixel FWHM implied by the grid.

    Examples
    --------
    >>> spec = Spectrum.from_txt(
    ...     'examples/data/agnspec.txt',
    ...     redshift=0.043,
    ...     xunit='AA',
    ... )
    >>> spec.crop(bounds=(4300.0, 7000.0))
    >>> spec.rebin(factor=2)
    """

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
        """Restrict the active spectrum to a subset of the original grid.

        Parameters
        ----------
        bounds : tuple of float, optional
            Open interval ``(xmin, xmax)`` applied to the current working axis.
            Default is ``None``.
        mask : array-like of bool, optional
            Boolean mask with the same shape as the current working axis. Default
            is ``None``.

        Returns
        -------
        None
            The object is updated in place.
        """
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
        """Resample the spectrum onto a coarser or explicit spectral grid.

        Parameters
        ----------
        factor : int, optional
            Integer downsampling factor applied to the current grid. Default is
            ``None``.
        new_x : array-like, optional
            Explicit target spectral grid. Default is ``None``.
        fill : float, optional
            Fill value used outside the overlap region. Default is ``numpy.nan``.
        method : str, optional
            Resampling scheme forwarded to :func:`resample_spectrum`. Default is
            ``'flux-conserving'``.

        Returns
        -------
        None
            The object is updated in place.

        Notes
        -----
        Rebinning replaces the stored full-resolution arrays. Because the new grid
        is no longer a masked view of the original sampling, :meth:`reset` cannot
        recover the pre-rebinned state.
        """
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
        """Build a spectrum from a whitespace-delimited text file.

        Parameters
        ----------
        filename : str or path-like
            Text file containing either ``x y`` or ``x y yerr`` columns.
        ra, dec, redshift : optional
            Metadata stored on the created spectrum. Defaults are ``None``.
        xunit, yunit : str or astropy.units.Unit, optional
            Units attached to the coordinate and values. Default is ``None``.
        name : str, optional
            Spectrum name. Defaults to the filename stem.
        xtype, ytype : str, optional
            Explicit semantic labels for the axis and values. Default is ``None``.

        Returns
        -------
        Spectrum
            Loaded spectrum with provenance stored in ``meta['source']``.
        """
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
        """Build a spectrum from a binary-table FITS extension.

        Parameters
        ----------
        filename : str or path-like
            FITS file containing spectral columns.
        ext : int, optional
            FITS extension index. Default is ``1``.
        x_col, y_col, yerr_col : str, optional
            Column names for the spectral axis, values, and optional
            uncertainties. Defaults are ``'WAVELENGTH'``, ``'FLUX'``, and
            ``'ERROR'``.
        ra, dec, redshift : optional
            Metadata stored on the created spectrum. Defaults fall back to the
            FITS header when present.
        name : str, optional
            Spectrum name. Defaults to the filename stem.
        xtype, ytype : str, optional
            Explicit semantic labels for the axis and values. Default is ``None``.

        Returns
        -------
        Spectrum
            Loaded spectrum with provenance stored in ``meta['source']``.
        """
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
        """Return the spectral axis converted to wavelength values.

        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            Output wavelength unit. Default uses the native unit.

        Returns
        -------
        numpy.ndarray
            Wavelength values on the current working grid.
        """
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'wavelength', unit)
        return values

    def frequencies(self, unit=None):
        """Return the spectral axis converted to frequency values.

        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            Output frequency unit. Default uses the native unit.

        Returns
        -------
        numpy.ndarray
            Frequency values on the current working grid.
        """
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'frequency', unit)
        return values

    def energies(self, unit=None):
        """Return the spectral axis converted to energy values.

        Parameters
        ----------
        unit : str or astropy.units.Unit, optional
            Output energy unit. Default uses the native unit.

        Returns
        -------
        numpy.ndarray
            Energy values on the current working grid.
        """
        values, _ = convert_spectral_axis(self.x, self.xtype, self.xunit, 'energy', unit)
        return values

    def velocity(self, rest=None, unit=None, convention='doppler'):
        """Return the spectral axis converted to velocity offsets.

        Parameters
        ----------
        rest : float or astropy.units.Quantity, optional
            Rest spectral coordinate used as the zero-velocity reference.
            Default is ``None``.
        unit : str or astropy.units.Unit, optional
            Output velocity unit. Default uses the native conversion target.
        convention : str, optional
            Velocity convention forwarded to :func:`convert_spectral_axis`.
            Default is ``'doppler'``.

        Returns
        -------
        numpy.ndarray
            Velocity values on the current working grid.
        """
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