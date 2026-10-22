"""Generic 1-D spectral data container."""

from __future__ import annotations

import os
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy.io import fits
import astropy.units as u
from .core import (
    Data1D,
    convert_spectral_axis,
    is_wavelength_like,
    parse_celestial_coord,
    parse_err,
    subset_axis_err,
)
from ..utils.tools import downsample_wave, resample_spectrum


c_kms = const.c.to_value(u.km / u.s)
script_dir = os.path.dirname(__file__)
dustpath = os.path.join(script_dir, '..', '..', 'resources', 'dust')


class Spectrum(Data1D):
    """One-dimensional spectral container with explicit cutouts and rebinning.

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
    frame : {None, 'observed', 'rest'}, optional
        Descriptive reference-frame metadata; default unknown. The caller applies
        physical transformations and passes redshift explicitly to a response.
    xunit, yunit : str or astropy.units.Unit, optional
        Units of the coordinate and value arrays. Default is ``None``.
    xtype, ytype : str, optional
        Explicit semantic labels for the axis and values. Default is inferred
        from the units when possible.
    name : str, optional
        Spectrum label. Default is ``'spectrum'``.
    meta : dict, optional
        Arbitrary metadata copied with the object. Default is ``None``.
    dtype : numpy dtype, optional
        Floating-point dtype used for values and value uncertainties. Default is
        ``numpy.float32``; pass ``numpy.float64`` for higher precision.

    Notes
    -----
    Public arrays contain the current data. Assignments and in-place arithmetic
    survive copy, cutout and rebin operations. cutout() selects coordinates,
    values and errors together; it does not retain excluded samples or a mask.
    Keep an explicit copy if an earlier state is needed. Sampling estimates
    follow the current coordinate grid and do not describe an instrumental LSF.

    Examples
    --------
    >>> spec = Spectrum.from_txt(
    ...     'examples/data/agnspec.txt',
    ...     redshift=0.043,
    ...     xunit='AA',
    ... )
    >>> spec = spec.cutout(min=4300.0, max=7000.0)
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
        frame=None,
        xunit=None,
        yunit=None,
        xtype=None,
        ytype=None,
        name='spectrum',
        meta=None,
        dtype=np.float32,
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
            dtype=dtype,
        )

        self.ra = parse_celestial_coord(ra)
        self.dec = parse_celestial_coord(dec)
        self.redshift = redshift
        self.frame = frame if frame is not None else self.meta.get('frame')
        self.x = self.x.copy()
        self.y = self.y.copy()
        self.xerr = None if self.xerr is None else self.xerr.copy()
        self.yerr = parse_err(self.yerr, shape=self.y.shape, name='yerr', dtype=self.dtype).copy()

    @property
    def frame(self):
        """Coordinate reference frame: None (unknown), 'observed', or 'rest'.

        Descriptive metadata only; assignment never transforms data or configures
        an instrumental response. Velocity coordinates remain a separate axis choice.
        """
        return self.meta.get('frame')

    @frame.setter
    def frame(self, value):
        if value not in (None, 'observed', 'rest'):
            raise ValueError("frame must be None, 'observed', or 'rest'.")
        if value is None:
            self.meta.pop('frame', None)
        else:
            self.meta['frame'] = value

    @property
    def velscale(self):
        """Approximate km/s step derived from the current wavelength grid."""
        if self.x.size > 1 and (self.xtype or '').lower() in {'wavelength', 'spectral'} and np.all(self.x > 0):
            return float(np.mean(np.diff(np.log(self.x)))) * c_kms
        return np.nan

    @property
    def fwhm(self):
        """Approximate pixel FWHM in coordinate units, not an instrumental LSF."""
        return 2.355 * (np.exp(self.velscale/c_kms) - 1.) * self.x

    @property
    def var(self):
        """numpy.ndarray or None: The variance array computed from symmetric `yerr`.

        Raises
        ------
        ValueError
            If `yerr` is asymmetric (not a 1-D array).
        """
        if self.yerr is None:
            return None
        if self.yerr.ndim != 1:
            raise ValueError('Variance view is only defined for symmetric yerr.')
        return self.yerr ** 2

    def copy(self):
        """Create an independent deep copy of the spectrum.

        Returns
        -------
        Spectrum
            A new instance with independent arrays and metadata.
        """
        self._validate_arrays()
        return copy.deepcopy(self)

    def _validate_arrays(self):
        """Reject incompatible direct assignments before using the current data."""
        shape = np.shape(self.x)
        if len(shape) != 1 or np.shape(self.y) != shape:
            raise ValueError('Spectrum x and y must be one-dimensional arrays of the same shape.')
        if self.yerr is not None and np.shape(self.yerr) != shape:
            raise ValueError('Spectrum yerr must match the current y shape.')
        if self.xerr is not None and np.shape(self.xerr) not in (shape, (2, shape[0])):
            raise ValueError('Spectrum xerr must have shape (N,) or (2, N), matching x.')

    def _require_symmetric_yerr(self):
        if self.yerr is None:
            return np.ones_like(self.y)
        if self.yerr.ndim != 1:
            raise ValueError('Operation requires symmetric yerr; asymmetric yerr is not supported here.')
        return self.yerr

    def _require_wavelength_axis(self, *, operation):
        if not is_wavelength_like(self.xtype, self.xunit):
            raise ValueError(f"{operation} is only defined for wavelength-like axes.")

    def cutout(self, min=None, max=None, mask=None, inplace=False):
        """Extract selected samples from the current spectrum.

        Parameters
        ----------
        min, max : float, optional
            Inclusive bounds applied to the current working axis. Default is
            ``None`` for either side.
        mask : array-like of bool, optional
            Boolean mask with the same shape as the current working axis. Default
            is ``None``.
        inplace : bool, optional
            Modify the spectrum in place. Default is ``False``.

        Returns
        -------
        Spectrum
            The cutout spectrum object.
            
        Examples
        --------
        >>> sub = spec.cutout(min=4000.0, max=5000.0)
        """
        self._validate_arrays()
        if min is None and max is None and mask is None:
            raise ValueError('At least one of min, max, or mask must be provided.')

        target = self if inplace else self.copy()
        current_mask = np.ones_like(target.x, dtype=bool)

        if min is not None:
            current_mask &= target.x >= min
        if max is not None:
            current_mask &= target.x <= max

        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool:
                raise TypeError('mask must be a boolean mask.')
            if mask.shape != target.x.shape:
                raise ValueError('mask must match the current working x shape.')
            current_mask &= mask

        target.x = target.x[current_mask].copy()
        target.y = target.y[current_mask].copy()
        target.xerr = subset_axis_err(target.xerr, current_mask)
        target.yerr = subset_axis_err(target.yerr, current_mask)

        return target

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
        Rebinning replaces the current arrays. Retain an explicit copy if the
        previous sampling is needed. A further cutout is allowed.
        
        Examples
        --------
        >>> spec.rebin(factor=2)
        >>> # or rebin to a new specific wavelength grid
        >>> spec.rebin(new_x=np.linspace(4000, 5000, 100))
        """
        self._validate_arrays()

        symmetric_yerr = self._require_symmetric_yerr()

        if new_x is not None:
            new_y, new_yerr = resample_spectrum(self.x, self.y, symmetric_yerr, new_wave=new_x, fill=fill, method=method)
            self.x = np.asarray(new_x, dtype=float).copy()
        elif factor is not None and factor > 1:
            rebinned_x = downsample_wave(self.x, factor)
            new_y, new_yerr = resample_spectrum(self.x, self.y, symmetric_yerr, new_wave=rebinned_x, fill=fill, method=method)
            self.x = np.asarray(rebinned_x, dtype=float).copy()
        else:
            raise ValueError('Either factor > 1 or new_x must be provided.')

        self.y = new_y
        self.yerr = new_yerr
        self.xerr = None

    @classmethod
    def from_txt(cls, filename, ra=None, dec=None, redshift=None, xunit=None, yunit=None, name=None, xtype=None, ytype=None, dtype=np.float32, frame=None):
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
        frame : {None, 'observed', 'rest'}, optional
            Descriptive frame metadata; default unknown.
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
            frame=frame,
            xunit=xunit,
            yunit=yunit,
            xtype=xtype,
            ytype=ytype,
            name=name,
            meta={'source': filename},
            dtype=dtype,
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
        frame=None,
        name=None,
        xtype=None,
        ytype=None,
        dtype=np.float32,
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
            Metadata stored on the created spectrum. Coordinates default to
            the spectral-extension header, then PRIMARY when absent there.
            Explicit ra/dec values take precedence.
        name : str, optional
            Spectrum name. Defaults to the filename stem.
        frame : {None, 'observed', 'rest'}, optional
            Descriptive frame metadata; default unknown.
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

            ra = ra if ra is not None else header.get('RA', hdul[0].header.get('RA'))
            dec = dec if dec is not None else header.get('DEC', hdul[0].header.get('DEC'))
            redshift = redshift if redshift is not None else header.get('REDSHIFT')
            name = name if name is not None else os.path.splitext(os.path.basename(filename))[0]

        return cls(
            x=x,
            y=y,
            yerr=yerr,
            ra=ra,
            dec=dec,
            redshift=redshift,
            frame=frame,
            xunit=xunit,
            yunit=yunit,
            xtype=xtype,
            ytype=ytype,
            name=name,
            meta={'source': filename},
            dtype=dtype,
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

    def save(self, filename, overwrite=False):
        """Save active arrays, units and metadata as a FITS spectrum snapshot.

        Parameters
        ----------
        filename : str or path-like
            Output FITS snapshot.
        overwrite : bool, optional
            Replace an existing file. Default is False.

        Notes
        -----
        Coordinate errors may have lower/upper rows; flux errors are symmetric.
        Only the current samples are stored; no selection history is retained.
        Metadata must be JSON-compatible; paths and NumPy scalars are converted
        explicitly. Use :meth:`load` for this format and :meth:`from_fits` for
        external files.
        """
        self._validate_arrays()

        def encode(value):
            if isinstance(value, os.PathLike):
                return os.fspath(value)
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f'Unsupported spectrum metadata: {type(value).__name__}')

        state = dict(name=self.name, ra=self.ra, dec=self.dec, redshift=self.redshift,
                     xtype=self.xtype, ytype=self.ytype, dtype=self.dtype.str, meta=self.meta)
        payload = json.dumps(state, default=encode, allow_nan=False).encode('utf-8')
        columns = []
        for name, values, unit in (('X', self.x, self.xunit), ('Y', self.y, self.yunit),
                                   ('XERR', self.xerr, self.xunit), ('YERR', self.yerr, self.yunit)):
            if values is None:
                continue
            values = np.asarray(values)
            repeat = 2 if values.ndim == 2 else 1
            columns.append(fits.Column(name=name, format=f'{repeat}D',
                array=values.T if repeat == 2 else values,
                unit=None if unit is None else unit.to_string('fits')))
        primary = fits.PrimaryHDU()
        primary.header['PRISMSP'] = 1
        fits.HDUList([primary, fits.BinTableHDU.from_columns(columns, name='SPECTRUM'),
            fits.ImageHDU(np.frombuffer(payload, dtype=np.uint8), name='METADATA')]).writeto(
                filename, overwrite=overwrite)

    @classmethod
    def load(cls, filename):
        """Restore a snapshot written by :meth:`save`, including units and errors.

        Parameters
        ----------
        filename : str or path-like
            Snapshot written by :meth:`save`.

        Returns
        -------
        Spectrum
            Independent active spectrum with its saved units, errors and metadata.

        Notes
        -----
        Use :meth:`from_fits` for external spectra.
        """
        with fits.open(filename) as hdus:
            if hdus[0].header.get('PRISMSP') != 1:
                raise ValueError('Not a Prism spectrum snapshot; use Spectrum.from_fits for external files.')
            state = json.loads(hdus['METADATA'].data.tobytes().decode('utf-8'))
            table = hdus['SPECTRUM']
            def values(name):
                if name not in table.columns.names:
                    return None
                array = np.array(table.data[name], copy=True)
                return array.T if array.ndim == 2 else array
            return cls(values('X'), values('Y'), xerr=values('XERR'), yerr=values('YERR'),
                       xunit=table.columns['X'].unit, yunit=table.columns['Y'].unit, **state)

    def plot(self, ax=None):
        """Render the 1-D spectrum using Matplotlib.

        Plots the spectral flux values as a mid-step function with errorbars.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes for the plot. If ``None``, a new figure and axes
            are created and immediately displayed.

        Returns
        -------
        None
        
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> spec.plot(ax=ax)
        >>> plt.show()
        """
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
