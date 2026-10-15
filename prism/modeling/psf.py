"""Field-star PSF fitting and compact wavelength-dependent spatial models.

Spatial parameters follow Astropy conventions in local, zero-based pixels;
angles are in radians. Supports native Gaussian2D/Moffat2D and FWHM-parameterized Prism Moffats.
Band fits and empirical chromatic trends are separate operations. No data
resampling, automatic outlier rejection, or host decomposition is performed.
"""

import json
import warnings
from numbers import Integral

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.modeling.models import Const2D, Gaussian2D, Moffat2D, Planar2D
from astropy.table import Table
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares, lsq_linear
from scipy.interpolate import BSpline

from .models.spatial import MoffatPSF, EllipticalMoffatPSF

from .._version import __version__

from .extraction import aperture, optimal

__all__ = ['fitimage', 'fitbands', 'fittrends', 'ChromaticPSF', 'aperture', 'optimal']

_MODELS = {cls.__name__: cls for cls in (Gaussian2D, Moffat2D, MoffatPSF, EllipticalMoffatPSF)}
_CENTERS = {Gaussian2D: ('x_mean', 'y_mean'), Moffat2D: ('x_0', 'y_0'),
            MoffatPSF: ('x_0', 'y_0'), EllipticalMoffatPSF: ('x_0', 'y_0')}


def _profile(model):
    """Check the supported profile and return its analytic total flux."""
    if type(model) not in _CENTERS:
        raise TypeError('Use Gaussian2D, Moffat2D, MoffatPSF or EllipticalMoffatPSF.')
    if model.parameters.size != len(model.param_names):
        raise ValueError('Model sets are not supported.')
    if any(getattr(model, name).unit is not None for name in model.param_names):
        raise ValueError('Use unitless model parameters: pixels and radians.')
    if not np.all(np.isfinite(model.parameters)):
        raise ValueError('PSF parameters must be finite.')
    if model.amplitude.value <= 0:
        raise ValueError('PSF amplitude must be positive.')
    if type(model) is Gaussian2D:
        if model.x_stddev.value <= 0 or model.y_stddev.value <= 0:
            raise ValueError('Gaussian widths must be positive.')
        return 2 * np.pi * model.amplitude.value * model.x_stddev.value * model.y_stddev.value
    if type(model) is Moffat2D:
        if model.gamma.value <= 0 or model.alpha.value <= 1:
            raise ValueError('A normalized Moffat requires gamma > 0 and alpha > 1.')
        return np.pi * model.amplitude.value * model.gamma.value**2 / (model.alpha.value - 1)
    widths = ([model.fwhm.value]*2 if type(model) is MoffatPSF
              else [model.fwhm_x.value, model.fwhm_y.value])
    if min(widths) <= 0 or model.beta.value <= 1:
        raise ValueError('A normalized Moffat requires positive FWHM and beta > 1.')
    scales = np.asarray(widths) / (2*np.sqrt(np.expm1(np.log(2)/model.beta.value)))
    return np.pi * model.amplitude.value * np.prod(scales) / (model.beta.value - 1)


def _state(model):
    """Record explicit scalar model state without executable serialization."""
    if any(model.tied.values()):
        raise ValueError('Tied parameters are not supported in the initial PSF workflow.')
    return {
        'class': type(model).__name__,
        'parameters': {
            name: {'value': float(getattr(model, name).value),
                   'fixed': bool(model.fixed[name]),
                   'bounds': [None if b is None or np.isinf(b) else float(b)
                              for b in model.bounds[name]]}
            for name in model.param_names
        },
    }


def _restore(state):
    name = state['class']
    if name not in _MODELS:
        raise ValueError(f'Unsupported saved PSF profile: {name}.')
    model = _MODELS[name]()
    if set(state['parameters']) != set(model.param_names):
        raise ValueError('Saved parameters do not match the profile.')
    for name, parameter in state['parameters'].items():
        getattr(model, name).value = parameter['value']
        model.fixed[name] = parameter['fixed']
        model.bounds[name] = tuple(parameter['bounds'])
    _profile(model)
    return model


def fittrends(bands, trends, *, basis='bspline', display=True, save=None):
    """Fit explicit-complexity smooth trends to independent PSF band measurements.

    Parameters
    ----------
    bands : astropy.table.Table
        Output of `fitbands`. Slice rows before calling to exclude bands
        manually. Failed rows are retained as diagnostics but never fitted.
    trends : dict of str to int
        Exact free shape/centroid parameter names. With basis='bspline', values
        are coefficient counts (>=1); with basis='polynomial', they are degrees
        (>=0). Thus a 5-coefficient spline and degree-4 polynomial each have five
        free parameters. Omit amplitude and parameters fixed in the input model.
    basis : {'bspline', 'polynomial'}, optional
        Default 'bspline': piecewise degree min(3, count-1), with uniformly
        spaced interior knots across the calibrated channel domain. One coefficient is a
        constant; up to four coefficients give a single polynomial segment.
        Finite physical/user bounds constrain spline coefficients, bounding
        the entire spline on its base domain by the convex-hull property.
        The fitted spline extends smoothly to the calibrated channel limits;
        band centers remain the measurement coordinates. These edge portions
        are less constrained by measurements. Evaluation outside the
        calibrated domain is rejected. 'polynomial' retains the global power-basis fit
        and prior degree syntax.
    display : bool, optional
        Show parameter measurements and adopted trends. Default is True.
    save : str or pathlib.Path, optional
        Save the diagnostic figure to this path (format from extension).
        None saves nothing. With display=False and save=None no figure is made.
        This does not save the numerical PSF model.

    Returns
    -------
    ChromaticPSF
        Normalized spatial profile and explicit trend laws, with band diagnostics.
        The valid domain spans the actual channels contributing to successful
        bands (or their representative wavelengths if channel metadata is
        absent). Within-band evaluation uses the fitted empirical trend; gaps
        between accepted bands are interpolated. Evaluation outside the
        calibrated domain is rejected.

    Notes
    -----
    Each trend uses inverse standard deviations when all contributing band
    errors are positive and finite; otherwise it is unweighted, with a warning.
    Trend covariance assumes independent band estimates. Band covariances
    are retained, but cross-parameter and cross-band correlations are not
    propagated through the trends. No clipping or joint image fit is applied.
    Rotation is unwrapped with period pi; it remains ill-conditioned
    for nearly circular profiles. Broad-band averages approximate a PSF at the
    band's representative wavelength.

    Examples
    --------
    >>> psf = fittrends(bands, {'fwhm': 5, 'beta': 5,
    ...                         'x_0': 5, 'y_0': 5}, display=False)
    """
    model = _restore(bands.meta['profile'])
    required = {n for n in model.param_names if n != 'amplitude' and not model.fixed[n]}
    if set(trends) != required:
        raise ValueError(f'trends must contain exactly these free parameters: {sorted(required)}.')
    if basis not in ('bspline', 'polynomial'):
        raise ValueError("basis must be 'bspline' or 'polynomial'.")
    for name, count in trends.items():
        minimum = 1 if basis == 'bspline' else 0
        if isinstance(count, bool) or not isinstance(count, Integral) or count < minimum:
            raise ValueError(f'{name}: expected an integer >= {minimum} for {basis}.')
    good = np.asarray(bands['success'], dtype=bool)
    wave = np.asarray(bands['wave'], dtype=float)
    if not np.any(good) or not np.all(np.isfinite(wave[good])):
        raise ValueError('No successful bands with finite wavelengths.')
    if np.unique(wave[good]).size < 2:
        raise ValueError('At least two distinct successful band wavelengths are required.')
    domain = [float(wave[good].min()), float(wave[good].max())]
    if 'channels' in bands.colnames and 'wavelengths' in bands.meta:
        channels = np.any(np.asarray(bands['channels'], dtype=bool)[good], axis=0)
        calibrated = np.asarray(bands.meta['wavelengths'], dtype=float)[channels]
        domain = [float(calibrated.min()), float(calibrated.max())]
    laws = {}
    for name, count in trends.items():
        ncoeff = count if basis == 'bspline' else count + 1
        values = np.asarray(bands[name], dtype=float)[good]
        errors = np.asarray(bands[name + '_err'], dtype=float)[good]
        if np.unique(wave[good]).size < ncoeff or not np.all(np.isfinite(values)):
            raise ValueError(f'Not enough finite, distinct band measurements for {name}, coefficient count {ncoeff}.')
        order = np.argsort(wave[good])
        if name == 'theta':
            values = values.copy()
            values[order] = np.unwrap(2 * values[order]) / 2
        weighted = np.all(np.isfinite(errors) & (errors > 0))
        if not weighted:
            warnings.warn(f'{name}: unavailable band errors; fitting an unweighted trend.', UserWarning)
        weights = 1 / errors if weighted else np.ones_like(errors)
        # Fit in [-1, 1], avoiding ill-conditioned powers of wavelength in Angstrom.
        x = 2 * (wave[good] - domain[0]) / (domain[1] - domain[0]) - 1
        law = {'basis': basis, 'weighted': bool(weighted)}
        if basis == 'polynomial':
            design = np.polynomial.polynomial.polyvander(x, count)
        else:
            degree = min(3, ncoeff - 1)
            left, right = -1., 1.
            interior = np.linspace(left, right, ncoeff - degree + 1)[1:-1]
            knots = np.r_[np.repeat(left, degree+1), interior, np.repeat(right, degree+1)]
            design = BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()
            law.update(knots=knots.tolist(), degree=int(degree))
        matrix = design * weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, values * weights, rcond=None)
        if rank != ncoeff:
            raise ValueError(f'Rank-deficient trend for {name}; reduce its complexity or revise the bands.')
        active = False
        if basis == 'bspline':
            lo, hi = model.bounds[name]
            lo = -np.inf if lo is None else lo
            hi = np.inf if hi is None else hi
            if name in ('gamma', 'x_stddev', 'y_stddev', 'fwhm', 'fwhm_x', 'fwhm_y'):
                lo = max(lo, np.finfo(float).eps)
            elif name in ('alpha', 'beta'):
                lo = max(lo, np.nextafter(1., np.inf))
            if np.any(coefficients < lo) or np.any(coefficients > hi):
                result = lsq_linear(matrix, values * weights, bounds=(lo, hi))
                if not result.success:
                    raise ValueError(f'Bounded spline fit failed for {name}: {result.message}')
                coefficients = result.x
                active = bool(np.any(result.active_mask))
        residuals = values - design @ coefficients
        covariance = None
        if active:
            warnings.warn(f'{name}: spline coefficient bound active; covariance unavailable.', UserWarning)
        elif weighted or len(values) > ncoeff:
            # QR avoids squaring the matrix condition number in normal equations.
            _, r = np.linalg.qr(matrix, mode='reduced')
            inverse = np.linalg.solve(r, np.eye(ncoeff))
            covariance = inverse @ inverse.T
            if not weighted:
                covariance *= np.sum(residuals**2) / (len(values) - ncoeff)
        law.update(coefficients=coefficients.tolist(),
                   covariance=None if covariance is None else covariance.tolist())
        laws[name] = law
    psf = ChromaticPSF(model, laws, domain, unit=bands.meta['waveunit'],
                       bands=bands.copy(), factor=bands.meta.get('factor', 1))
    if display or save is not None:
        import matplotlib.pyplot as plt
        names = list(trends)
        if names:
            fig, axes = plt.subplots(1, len(names), squeeze=False, figsize=(4 * len(names), 3.5))
            grid = np.linspace(*domain, 200)
            x = 2 * (grid - domain[0]) / (domain[1] - domain[0]) - 1
            for ax, name in zip(axes.flat, names):
                values = np.asarray(bands[name], dtype=float).copy()
                if name == 'theta':
                    order = np.argsort(wave[good])
                    unwrapped = values[good].copy()
                    unwrapped[order] = np.unwrap(2 * unwrapped[order]) / 2
                    values[good] = unwrapped
                errors = np.asarray(bands[name + '_err'], dtype=float)
                validerr = good & np.isfinite(errors) & (errors > 0)
                ax.plot(wave[good], values[good], 'ko', label='Bands')
                ax.errorbar(wave[validerr], values[validerr], yerr=errors[validerr], fmt='none', color='k')
                ax.plot(grid, _law(laws[name], x), 'r-', label='Trend')
                failed = ~good & np.isfinite(values)
                ax.plot(wave[failed], values[failed], 'x', color='gray', label='Failed')
                unit = str(bands[name].unit or '')
                ax.set(xlabel=f'Wavelength [{psf.unit}]',
                       ylabel=f'{name} [{unit}]' if unit else name)
            axes.flat[0].legend()
            fig.tight_layout()
            try:
                if save is not None:
                    fig.savefig(save)
                if display:
                    plt.show()
            except Exception:
                plt.close(fig)
                raise
            if not display:
                plt.close(fig)
    return psf


def _law(law, x):
    if law.get('basis', 'polynomial') == 'polynomial':
        return Polynomial(law['coefficients'])(x)
    if law.get('extension') == 'constant':
        x = np.clip(x, law['knots'][law['degree']], law['knots'][-law['degree']-1])
    return BSpline(law['knots'], law['coefficients'], law['degree'], extrapolate=False)(x)


class ChromaticPSF:
    """An analytic PSF profile with empirical wavelength-dependent parameters.

    Parameters
    ----------
    model : Gaussian2D, Moffat2D, MoffatPSF or EllipticalMoffatPSF
        Spatial template. Coordinates and widths are in local zero-based pixels;
        angles are radians. Amplitude is discarded on normalized evaluation.
    trends : dict
        Trend definitions produced by `fittrends`: basis, coefficients, optional
        covariance, and for B-splines the knot vector and local degree. Both
        bases use wavelengths scaled to [-1, 1]. Fixed parameters remain in the spatial template.
    domain : pair of float
        Inclusive wavelength validity limits, in ``unit``.
    unit : unit-like
        Wavelength unit. Must be a length unit.
    bands : astropy.table.Table, optional
        Independent band-fit diagnostics. No data cube or live fitter is stored.
    factor : int, optional
        Default midpoint samples per pixel axis for reconstruction. Default 1.

    Notes
    -----
    Evaluation returns a normalized Astropy model; it does not renormalize a
    finite image. Gaussian integrals are analytic on the infinite plane;
    a Moffat requires alpha > 1 (native) or beta > 1 (Prism). Physical validity is checked at every requested
    wavelength. Trend uncertainty is stored, not propagated into images.
    """

    def __init__(self, model, trends, domain, *, unit, bands=None, factor=1):
        _profile(model)
        _state(model)
        self.model = model.copy()
        self.trends = json.loads(json.dumps(trends, allow_nan=False))
        self.domain = np.asarray(domain, dtype=float).copy()
        if self.domain.shape != (2,) or not np.all(np.isfinite(self.domain)) or self.domain[0] >= self.domain[1]:
            raise ValueError('domain must contain two finite increasing wavelengths.')
        self.unit = u.Unit(unit)
        if not self.unit.is_equivalent(u.m):
            raise ValueError('PSF wavelength unit must be a length unit.')
        required = {n for n in model.param_names if n != 'amplitude' and not model.fixed[n]}
        if set(self.trends) != required:
            raise ValueError(f'Expected trends for {sorted(required)}.')
        for name, law in self.trends.items():
            coefficients = np.asarray(law['coefficients'], dtype=float)
            if coefficients.ndim != 1 or coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
                raise ValueError(f'Invalid coefficients for {name}.')
            basis = law.get('basis', 'polynomial')
            if basis not in ('polynomial', 'bspline'):
                raise ValueError(f'Unknown trend basis for {name}.')
            if basis == 'bspline':
                degree = law['degree']
                knots = np.asarray(law['knots'], dtype=float)
                if (isinstance(degree, bool) or not isinstance(degree, Integral)
                        or not 0 <= degree <= 3 or knots.ndim != 1
                        or len(knots) != len(coefficients) + degree + 1
                        or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) < 0)
                        or not -1 <= knots[degree] < knots[-degree-1] <= 1
                        or law.get('extension') not in (None, 'constant')
                        or (law.get('extension') is None
                            and (knots[degree] != -1 or knots[-degree-1] != 1))):
                    raise ValueError(f'Invalid spline knots or degree for {name}.')
            covariance = law.get('covariance')
            if covariance is not None:
                covariance = np.asarray(covariance, dtype=float)
                if covariance.shape != (coefficients.size, coefficients.size) or not np.all(np.isfinite(covariance)):
                    raise ValueError(f'Invalid trend covariance for {name}.')
        if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
            raise ValueError('factor must be a positive integer.')
        self.factor = int(factor)
        self.bands = None if bands is None else bands.copy()

    def __call__(self, wave):
        """Return the unit-integral Astropy profile at one wavelength.

        Parameters
        ----------
        wave : float or astropy.units.Quantity
            Scalar wavelength. Bare numbers use the stored wavelength unit.

        Returns
        -------
        Astropy Fittable2DModel
            A fresh normalized model in local pixel coordinates.

        Raises
        ------
        ValueError
            For extrapolation, non-scalar wavelengths, or unphysical profiles.

        Examples
        --------
        >>> profile = psf(6000 * u.AA)
        >>> density = profile(10.0, 12.0)
        """
        wave = u.Quantity(wave, self.unit).to_value(self.unit)
        if np.ndim(wave) != 0 or not np.isfinite(wave):
            raise ValueError('Provide one finite wavelength.')
        if wave < self.domain[0] or wave > self.domain[1]:
            edge = self.domain[0] if wave < self.domain[0] else self.domain[1]
            # Unit conversion can move a calibrated endpoint by a few ulps.
            if not np.isclose(wave, edge, rtol=8*np.finfo(float).eps, atol=0):
                raise ValueError(f'Wavelength outside PSF domain {self.domain} {self.unit}.')
            wave = edge
        x = 2 * (wave - self.domain[0]) / (self.domain[1] - self.domain[0]) - 1
        model = self.model.copy()
        model.amplitude = 1.0
        for name, law in self.trends.items():
            value = _law(law, x)
            lo, hi = model.bounds[name]
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(f'Trend for {name} violates the declared parameter bounds.')
            getattr(model, name).value = value
        model.amplitude = 1.0 / _profile(model)
        return model

    def fwhm(self, wave):
        """Return the two principal-axis full widths at requested wavelengths.

        Parameters
        ----------
        wave : float, array-like or Quantity
            Wavelengths within the fitted domain; bare numbers use self.unit.

        Returns
        -------
        astropy.units.Quantity
            Shape ``wave.shape + (2,)`` in pixels, in principal-axis (x, y)
            order, independent of theta. Circular profiles repeat the width.
            These are intrinsic profile widths, not pixel-convolved widths.
            Conversion to arcseconds is external; a common scalar scale applies
            directly only for square pixels. No redundant curve is saved.

        Examples
        --------
        >>> widths = psf.fwhm([5000, 6000, 7000])
        """
        wave = u.Quantity(wave, self.unit).to_value(self.unit)
        widths = []
        for w in np.asarray(wave).reshape(-1):
            model = self(w)
            if type(model) is Gaussian2D:
                pair = [model.x_fwhm, model.y_fwhm]
            elif type(model) is Moffat2D:
                width = 2*model.gamma.value*np.sqrt(np.expm1(np.log(2)/model.alpha.value))
                pair = [width, width]
            elif type(model) is MoffatPSF:
                pair = [model.fwhm.value]*2
            else:
                pair = [model.fwhm_x.value, model.fwhm_y.value]
            widths.append(pair)
        return np.asarray(widths).reshape(np.shape(wave) + (2,)) * u.pix

    def get_image(self, wave, shape, *, center=None, factor=None):
        """Evaluate the normalized PSF on a local pixel grid.

        Parameters
        ----------
        wave : float or Quantity
            One wavelength within the fitted domain, in the stored unit.
        shape : pair of int
            Output dimensions (ny, nx). Pixel centers run from zero.
        center : pair of float, optional
            Override the fitted center with (y, x), in output pixels. Widths
            remain in the original pixel scale; no WCS reprojection is done.
        factor : int, optional
            Midpoint quadrature samples per pixel axis. None uses the factor
            stored by fitbands; 1 evaluates only pixel centers.

        Returns
        -------
        numpy.ndarray
            Float64 fractions of the infinite-plane flux in each pixel,
            approximated by midpoint integration. The finite image is not
            rescaled to sum to one. Increasing factor tests convergence.

        Examples
        --------
        >>> image = psf.get_image(6000, (31, 31), center=(15, 15), factor=5)
        """
        if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape):
            raise ValueError('shape must be a pair of positive integers (ny, nx).')
        model = self(wave)
        if center is not None:
            center = np.asarray(center, dtype=float)
            if center.shape != (2,) or not np.all(np.isfinite(center)):
                raise ValueError('center must be a finite (y, x) pair.')
            xname, yname = _CENTERS[type(model)]
            getattr(model, xname).value = center[1]
            getattr(model, yname).value = center[0]
        y, x = np.indices(shape, dtype=float)
        return _pixels(model, x, y, self.factor if factor is None else factor)

    def get_cube(self, wave, shape, *, center=None, factor=None):
        """Evaluate a PSF cube in (wavelength, y, x) order.

        Parameters
        ----------
        wave : one-dimensional array or Quantity
            Wavelengths in the stored unit, all inside the fitted domain.
        shape : pair of int
            Spatial dimensions (ny, nx), as in `get_image`.
        center : pair of float, optional
            Fixed output center (y, x); None retains the fitted centroid trend.
        factor : int, optional
            Sampling per pixel axis; None uses the stored value.

        Returns
        -------
        numpy.ndarray
            A full float64 cube. Prefer get_image in a loop if only aperture
            fractions are needed. Use FITS directly for an optional cube export;
            `write` saves the compact model rather than this derived array.
        """
        wave = u.Quantity(wave, self.unit).to_value(self.unit)
        if wave.ndim != 1 or wave.size == 0:
            raise ValueError('wave must be a nonempty one-dimensional array.')
        return np.stack([self.get_image(w, shape, center=center, factor=factor) for w in wave])

    def write(self, filename, *, overwrite=False):
        """Save the compact model and available band diagnostics to FITS.

        Parameters
        ----------
        filename : str or pathlib.Path
            Destination FITS file; no PSF cube is materialized.
        overwrite : bool, optional
            Replace an existing file. Default is False.

        Notes
        -----
        MODEL_DEF stores plain JSON: a supported Astropy profile identifier,
        scalar parameter states, trend coefficients/covariances and knots, wavelength
        domain and unit, pixel convention, normalization, and sampling factor.
        BANDS optionally stores measurements and their local covariances. No
        pickle, executable expression, dynamic import, or live fitter is saved.
        """
        state = {'profile': _state(self.model), 'trends': self.trends,
                 'domain': self.domain.tolist(), 'waveunit': str(self.unit),
                 'factor': self.factor, 'normalization': 'infinite-plane',
                 'basis': 'per-parameter-on-minus1-plus1',
                 'coordinates': 'local-zero-based-pixels',
                 'bandmeta': {} if self.bands is None else self.bands.meta}
        payload = json.dumps(state, allow_nan=False)
        header = fits.Header({'PSFVER': 2, 'PRISMVER': __version__})
        definition = fits.BinTableHDU.from_columns([
            fits.Column(name='JSON', format=f'{len(payload)}A', array=[payload])], name='MODEL_DEF')
        hdus = [fits.PrimaryHDU(header=header), definition]
        if self.bands is not None:
            table = self.bands.copy()
            table.meta.clear()  # Metadata belongs in MODEL_DEF, not lossy FITS cards.
            hdus.append(fits.BinTableHDU(table, name='BANDS'))
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)

    @classmethod
    def from_fits(cls, filename):
        """Reload a compact PSF saved by `write`, without the original cube.

        Parameters
        ----------
        filename : str or pathlib.Path
            A Prism PSF FITS file with a supported format version.

        Returns
        -------
        ChromaticPSF
            A reconstructed model with optional band diagnostics.

        Examples
        --------
        >>> psf.write('psf.fits')
        >>> loaded = ChromaticPSF.from_fits('psf.fits')
        """
        with fits.open(filename, memmap=False) as hdus:
            version = hdus[0].header.get('PSFVER')
            if version not in (1, 2):
                raise ValueError('Unsupported or missing PSF format version.')
            state = json.loads(hdus['MODEL_DEF'].data['JSON'][0])
            if (state['normalization'] != 'infinite-plane'
                    or state['coordinates'] != 'local-zero-based-pixels'
                    or state['basis'] != ('power-ascending-on-minus1-plus1' if version == 1
                                          else 'per-parameter-on-minus1-plus1')):
                raise ValueError('Unsupported PSF normalization, coordinates, or polynomial basis.')
            bands = Table.read(hdus['BANDS'], format='fits') if 'BANDS' in hdus else None
            if bands is not None:
                bands.meta = state['bandmeta']
                bands['wave'].unit = u.Unit(state['waveunit'])
        result = cls(_restore(state['profile']), state['trends'], state['domain'],
                     unit=state['waveunit'], bands=bands, factor=state['factor'])
        return result


def _pixels(model, x, y, factor):
    """Integrate over unit pixels by midpoint quadrature without a subpixel cube."""
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError('factor must be a positive integer.')
    if factor == 1:
        return np.asarray(model(x, y), dtype=float)
    offsets = (np.arange(factor) + 0.5) / factor - 0.5
    values = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)
    for dy in offsets:
        for dx in offsets:
            values += model(x + dx, y + dy)
    return values / factor**2


def _fitsetup(model, background, factor):
    """Validate shared spatial-fit constraints and assemble parameter vectors."""
    _profile(model)
    _state(model)
    if model.parameters.size != len(model.param_names):
        raise ValueError('Model sets are not supported.')
    if background is not None:
        if (type(background) not in (Const2D, Planar2D)
                or background.parameters.size != len(background.param_names)
                or any(getattr(background, n).unit is not None for n in background.param_names)):
            raise TypeError('background must be a unitless Const2D, Planar2D or None.')
        if not np.all(np.isfinite(background.parameters)):
            raise ValueError('Background must be finite.')
        _state(background)
    # Validate sampling once before any fits, including empty/failed bands.
    _pixels(model, np.array([0.]), np.array([0.]), factor)
    names = list(model.param_names) + ([] if background is None else
        ['background'] if type(background) is Const2D else list(background.param_names))
    initial = np.r_[model.parameters, [] if background is None else background.parameters]
    fixed = np.array([model.fixed[n] for n in model.param_names] +
                     ([] if background is None else [background.fixed[n] for n in background.param_names]))
    bounds = ([model.bounds[n] for n in model.param_names] +
              ([] if background is None else [background.bounds[n] for n in background.param_names]))
    lower = np.array([-np.inf if lo is None else lo for lo, hi in bounds])
    upper = np.array([np.inf if hi is None else hi for lo, hi in bounds])
    if type(model) is Gaussian2D:
        positive = ['amplitude', 'x_stddev', 'y_stddev']
    elif type(model) is Moffat2D:
        positive = ['amplitude', 'gamma', 'alpha']
    elif type(model) is MoffatPSF:
        positive = ['amplitude', 'fwhm', 'beta']
    else:
        positive = ['amplitude', 'fwhm_x', 'fwhm_y', 'beta']
    for name in positive:
        index = names.index(name)
        lower[index] = max(lower[index], np.nextafter(1., np.inf)
                           if name in ('alpha', 'beta') else np.finfo(float).eps)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError('Parameter bounds must not contain NaN.')
    if np.any(lower > upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError('Initial parameters must satisfy physical and user bounds.')
    free = ~fixed
    if np.any(lower[free] >= upper[free]):
        raise ValueError('Free parameter bounds must have nonzero width; use fixed=True instead.')
    return names, initial, free, lower, upper


def _fitunits(table, names, unit):
    """Attach the same parameter/error units to image and band results."""
    units = {'amplitude': u.Unit(unit or u.dimensionless_unscaled),
             'background': u.Unit(unit or u.dimensionless_unscaled),
             'intercept': u.Unit(unit or u.dimensionless_unscaled),
             'slope_x': u.Unit(unit or u.dimensionless_unscaled) / u.pix,
             'slope_y': u.Unit(unit or u.dimensionless_unscaled) / u.pix,
             'x_mean': u.pix, 'y_mean': u.pix, 'x_0': u.pix, 'y_0': u.pix,
             'x_stddev': u.pix, 'y_stddev': u.pix, 'gamma': u.pix, 'theta': u.rad,
             'fwhm': u.pix, 'fwhm_x': u.pix, 'fwhm_y': u.pix}
    for name in names:
        table[name].unit = units.get(name, u.dimensionless_unscaled)
        table[name + '_err'].unit = table[name].unit


def fitimage(image, model, *, mask=None, background=None, factor=1, init=None):
    """Fit a spatial profile and optional background to one Image.

    Parameters
    ----------
    image : prism.data.Image
        Two-dimensional values, standard errors and valid-data mask. Positive
        finite errors are required; invalid pixels are excluded. Model
        coordinates are local zero-based array pixels, irrespective of WCS
        or image.x/image.y offsets. Errors assume independent pixels.
    model : Gaussian2D, Moffat2D, MoffatPSF or EllipticalMoffatPSF
        Initial spatial profile, in pixels and radians. Astropy fixed flags
        and bounds are honored, together with positive amplitude/widths and
        Moffat alpha/beta > 1. Inputs are not modified. Tied parameters and
        model sets are unsupported.
    mask : array-like of bool, optional
        Additional selection, with the image shape; True selects pixels.
        Intersected with image.mask and finite positive-error support.
    background : Const2D or Planar2D, optional
        Separate background, with its own initial values, fixed flags and
        bounds. None means zero. It is excluded from PSF normalization.
    factor : int, optional
        Midpoint samples per pixel axis; default 1 samples pixel centers.
        Integrates the profile without resampling the observations.
    init : {None, 'peak'}, optional
        Default None preserves all supplied initial values. 'peak' seeds free
        centroid coordinates and peak amplitude from the brightest valid
        background-subtracted pixel in the selection, clipped to their bounds.
        Fixed parameters, widths, shape and background are never reseeded.
        This is only an initialization, not a centroid measurement.

    Returns
    -------
    astropy.table.Row
        One spatial-fit result: npix, success, message, chi2, dof, redchi2,
        fitted parameters, *_err and covariance. Same fields and conventions
        as fitbands, without spectral-band columns. Units and covariance
        ordering are available through the parent table and row.meta.
        Failed fits have NaN measurements. No new result class is required.

    Notes
    -----
    Local covariance uses absolute input errors, without reduced-chi-square
    rescaling. Active bounds or rank deficiency leave covariance unavailable
    (NaN), with a warning. Fixed parameters have zero conditional variance
    when covariance is available. Convergence does not establish model adequacy.

    Examples
    --------
    >>> initial = fitimage(image, model, mask=selection, background=Const2D())
    >>> assert initial['success'], initial['message']
    >>> center = (float(initial['y_0']), float(initial['x_0']))
    """
    from ..data import Image

    if not isinstance(image, Image) or image.values.ndim != 2:
        raise TypeError('fitimage requires a two-dimensional prism.data.Image.')
    if init not in (None, 'peak'):
        raise ValueError("init must be None or 'peak'.")
    good = (image.mask & np.isfinite(image.values) & np.isfinite(image.err)
            & (image.err > 0))
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != image.shape or mask.dtype.kind != 'b':
            raise ValueError('mask must be boolean and have the image shape.')
        good &= mask
    model = model.copy()
    background = None if background is None else background.copy()
    # Validate user constraints before applying any optional initial guesses.
    names, initial, free, lower, upper = _fitsetup(model, background, factor)
    if init == 'peak' and good.any():
        y, x = np.nonzero(good)
        values = np.asarray(image.values[good], dtype=float)
        if background is not None:
            values = values - background(x, y)
        peak = int(np.argmax(values))
        seeds = dict(zip(_CENTERS[type(model)], (x[peak], y[peak])))
        seeds['amplitude'] = max(float(values[peak]), np.finfo(float).eps)
        for name, value in seeds.items():
            index = names.index(name)
            if free[index]:
                getattr(model, name).value = np.clip(value, lower[index], upper[index])
        names, initial, free, lower, upper = _fitsetup(model, background, factor)
    row = dict(npix=int(good.sum()), dof=int(good.sum()-free.sum()),
               success=False, message='Insufficient valid pixels for free parameters',
               chi2=np.nan, redchi2=np.nan)
    for name in names:
        row[name] = np.nan
        row[name + '_err'] = np.nan
    row['covariance'] = np.full((len(names), len(names)), np.nan)
    if row['dof'] > 0:
        y, x = np.nonzero(good)
        fitted = model.copy()
        fitted_background = None if background is None else background.copy()
        parameters = initial.copy()

        def residual(p):
            parameters[free] = p
            fitted.parameters = parameters[:len(model.param_names)]
            prediction = _pixels(fitted, x, y, factor)
            if background is not None:
                fitted_background.parameters = parameters[len(model.param_names):]
                prediction = prediction + fitted_background(x, y)
            return (prediction - image.values[good]) / image.err[good]

        try:
            if free.any():
                result = least_squares(residual, initial[free], bounds=(lower[free], upper[free]), x_scale='jac')
                residuals = residual(result.x)
                success = bool(result.success and np.all(np.isfinite(residuals)))
                message = result.message
                covariance = np.full((len(names), len(names)), np.nan)
                _, singular, vectors = np.linalg.svd(result.jac, full_matrices=False)
                rank = np.sum(singular > np.finfo(float).eps * max(result.jac.shape) * singular[0]) if singular.size else 0
                if success and rank == free.sum() and not np.any(result.active_mask):
                    covariance = np.zeros((len(names), len(names)))
                    covariance[np.ix_(free, free)] = (vectors.T / singular**2) @ vectors
                elif success:
                    warnings.warn('Image fit: covariance unavailable (rank deficiency or active bounds).', UserWarning)
            else:
                residuals = residual(np.array([]))
                success, message = bool(np.all(np.isfinite(residuals))), 'All parameters fixed'
                covariance = np.zeros((len(names), len(names)))
            if success:
                _profile(fitted)
                row.update(success=True, message=str(message), chi2=float(np.sum(residuals**2)),
                           redchi2=float(np.sum(residuals**2) / row['dof']), covariance=covariance)
                for n, name in enumerate(names):
                    row[name] = float(parameters[n])
                    row[name + '_err'] = float(np.sqrt(covariance[n, n]))
            else:
                row['message'] = str(message)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row['message'] = f'{type(exc).__name__}: {exc}'
    table = Table(rows=[row])
    _fitunits(table, names, image.unit)
    table.meta = {'profile': _state(model),
                  'background': None if background is None else _state(background),
                  'factor': int(factor), 'shape': list(image.shape),
                  'covnames': names, 'init': init}
    return table[0]


def fitbands(cube, model, bands=20, *, background=None, factor=1):
    """Fit a spatial Astropy profile independently in wavelength-band images.

    Parameters
    ----------
    cube : prism.data.Cube
        A 3-D wavelength cube. Pre-mask contaminants/laser channels using
        cube.mask (True means valid). Model coordinates use local array pixels,
        even when the cube retains offset coordinates after a cutout. Errors
        must be supplied for meaningful weighted fits and parameter errors.
    model : Gaussian2D, Moffat2D, MoffatPSF or EllipticalMoffatPSF
        Starting spatial profile in pixels and radians. Astropy parameter
        ``fixed`` flags and ``bounds`` are honored. Set initial positions,
        widths, and amplitude explicitly. Tied parameters and model sets are
        not supported. Positive amplitude/widths and Moffat alpha/beta > 1 are
        enforced in addition to user bounds. Inputs are not modified.
    bands : int or array-like of shape (n, 2), optional
        Default 20 equal-width intervals. Alternatively explicit increasing,
        non-overlapping wavelength pairs, with optional wavelength units.
        Intervals are [lower, upper), except the last includes its upper edge.
        Gaps are allowed. Images use channel means; spectral sampling must be
        increasing, but need not be uniform.
    background : Const2D or Planar2D, optional
        Separate background with Astropy initial parameters, bounds and fixed
        flags. None means zero. Const2D fits a constant; Planar2D fits
        slope_x*x + slope_y*y + intercept in local pixels. The background
        never contributes to the PSF flux normalization.
    factor : int, optional
        Midpoint integration samples per pixel axis, used during fitting and
        stored for reconstruction. Default 1 samples pixel centers. A value
        such as 5 approximates pixel integrals; check convergence for your PSF.
        This integrates the model without resampling the observed data.

    Returns
    -------
    astropy.table.Table
        One row per band: zero-based original band ID (nband), bounds, representative wave (mean of channels with
        any valid pixel), channel selection, pixel count, success/message,
        chi-square (chi2), degrees of freedom (dof), reduced chi-square
        (redchi2 = chi2/dof), fitted parameters and *_err columns,
        background parameters and their *_err columns, and covariance.
        Const2D uses the column background; Planar2D uses its native names
        slope_x, slope_y, intercept. Covariance ordering is
        in meta['covnames']; fixed parameters have zero conditional variance.
        Failed rows have NaN measurements, including redchi2. meta also stores initial profile,
        background, sampling factor, input wavelengths/unit and spatial shape.
        Slice rows to select bands before passing the table to `fittrends`.

    Notes
    -----
    Band cutouts use Cube.get_image(method='mean') in the cube dtype.
    Nonfinite and nonpositive-error samples are masked on the cutout only.
    Images are unweighted means of valid samples; independent channel errors
    are propagated in quadrature. Resampling covariance is not modeled.
    Parameter covariance is a local linear estimate with absolute input errors,
    not multiplied by reduced chi-square. Rank-deficient fits or active bounds
    leave errors unavailable (NaN) and emit a warning. Convergence alone does
    not establish that a profile is an adequate description of the data.
    Each image approximates a PSF at one representative wavelength. Partial
    spatial masking can make spectral support differ across pixels; this is
    warned about, not silently interpreted as a precise monochromatic PSF.

    Examples
    --------
    >>> from astropy.modeling.models import Moffat2D, Const2D
    >>> model = Moffat2D(amplitude=100, x_0=15, y_0=15, gamma=2, alpha=2.5)
    >>> bands = fitbands(cube, model, bands=20,
    ...                  background=Const2D(0), factor=5)
    """
    names, _, _, _, _ = _fitsetup(model, background, factor)
    profile = _state(model)
    if cube.values.ndim != 3:
        raise ValueError('fitbands requires a 3-D (wavelength, y, x) cube.')
    unit = u.Unit(cube.zunit)
    if not unit.is_equivalent(u.m):
        raise ValueError('Cube spectral coordinates must have a wavelength unit.')
    wave = np.asarray(cube.z, dtype=float)
    if wave.size < 2 or not np.all(np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
        raise ValueError('Cube wavelengths must be finite and strictly increasing.')
    if isinstance(bands, Integral) and not isinstance(bands, bool):
        if bands < 1:
            raise ValueError('bands must be positive.')
        edges = np.linspace(wave[0], wave[-1], bands + 1)
        intervals = np.column_stack((edges[:-1], edges[1:]))
    else:
        intervals = u.Quantity(bands, unit).to_value(unit)
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) == 0
            or not np.all(np.isfinite(intervals)) or np.any(intervals[:, 0] >= intervals[:, 1])
            or np.any(intervals[1:, 0] < intervals[:-1, 1])):
        raise ValueError('bands must be increasing, non-overlapping (lower, upper) pairs.')
    # Scan one plane at a time rather than allocating full-cube validity copies.
    supported = np.array([np.any(mask & np.isfinite(data) & np.isfinite(error) & (error > 0))
                          for data, error, mask in zip(cube.values, cube.err, cube.mask)])
    rows = []
    for index, (lo, hi) in enumerate(intervals):
        # Unit conversion can move an exactly shared edge by a few ulps.
        atlower = np.isclose(wave, lo, rtol=8*np.finfo(float).eps, atol=0)
        atupper = np.isclose(wave, hi, rtol=8*np.finfo(float).eps, atol=0)
        selected = ((wave > lo) | atlower) & (
            ((wave < hi) | atupper) if index == len(intervals)-1
            else ((wave < hi) & ~atupper))
        selected &= supported
        row = dict(nband=index, lower=float(lo), upper=float(hi), wave=float(np.mean(wave[selected])) if selected.any() else np.nan,
                   channels=selected.copy(), nchan=int(selected.sum()), npix=0,
                   success=False, message='No valid channels', chi2=np.nan, dof=0, redchi2=np.nan)
        for name in names:
            row[name] = np.nan
            row[name + '_err'] = np.nan
        row['covariance'] = np.full((len(names), len(names)), np.nan)
        if not selected.any():
            rows.append(row)
            continue
        channels = np.flatnonzero(selected)
        first, last = int(channels[0]), int(channels[-1])+1
        bandcube = cube.cutout_slices(z=slice(first, last))
        bandcube.mask &= (np.isfinite(bandcube.values) & np.isfinite(bandcube.err)
                          & (bandcube.err > 0))
        image = bandcube.get_image(weights=selected[first:last], method='mean')
        count = np.count_nonzero(bandcube.mask & selected[first:last, None, None], axis=0)
        good = image.mask & np.isfinite(image.values) & np.isfinite(image.err) & (image.err > 0)
        if np.any(count[good] != selected.sum()):
            warnings.warn(f'Band {index}: spectral support differs across spatial pixels.', UserWarning)
        fitted = fitimage(image, model, background=background, factor=factor)
        for name in fitted.colnames:
            row[name] = fitted[name]
        rows.append(row)
    table = Table(rows=rows)
    table['wave'].unit = unit
    table['lower'].unit = unit
    table['upper'].unit = unit
    _fitunits(table, names, cube.unit)
    table.meta = {'profile': profile, 'background': None if background is None else _state(background),
                  'waveunit': str(unit), 'wavelengths': wave.tolist(), 'factor': int(factor),
                  'shape': list(cube.shape[1:]), 'covnames': names, 'collapse': 'mean'}
    failed = int(np.sum(~np.asarray(table['success'])))
    if failed:
        warnings.warn(f'{failed} of {len(table)} PSF band fits failed; inspect success and message.', UserWarning)
    return table
