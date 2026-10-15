"""PSF-based spectral extraction without iterative host–nucleus deblending.

Both methods reuse a calibrated ChromaticPSF at a fixed, explicit target center.
They return a Spectrum and an inspectable table, with errors conditional on that
PSF and independent voxel errors. Spatial units are local zero-based pixels.
"""

import warnings
from numbers import Integral

import numpy as np
from astropy import units as u
from astropy.modeling.models import Const2D
from astropy.table import Table

from ..data import Spectrum
from ..data.core import world_from_pixel_barycenter

__all__ = ['aperture', 'optimal']


def _inputs(cube, psf, mask, center, factor):
    """Validate the shared extraction geometry and retain only selected pixels."""
    if cube.values.ndim != 3:
        raise ValueError('Extraction requires a (wavelength, y, x) cube.')
    weights = np.asarray(mask, dtype=float)
    if (weights.shape != cube.shape[1:] or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights > 1) or not np.any(weights > 0)):
        raise ValueError('mask must be a nonempty spatial weight array in [0, 1].')
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('center must be a finite (y, x) pixel pair.')
    factor = psf.factor if factor is None else factor
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError('factor must be a positive integer.')
    wave = u.Quantity(cube.z, cube.zunit).to_value(psf.unit)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError('Cube wavelengths must be finite and compatible with the PSF.')
    selected = weights > 0
    y, x = np.nonzero(selected)
    shape = (int(y.max()-y.min()+1), int(x.max()-x.min()+1))
    local = (y-y.min(), x-x.min())
    shifted = (float(center[0]-y.min()), float(center[1]-x.min()))
    geometry = {'shape': shape, 'center': shifted, 'factor': int(factor)}
    data = np.asarray(cube.values[:, selected], dtype=float)
    errors = np.asarray(cube.err[:, selected], dtype=float)
    valid = cube.mask[:, selected] & np.isfinite(data) & np.isfinite(errors) & (errors > 0)
    return weights[selected], wave, geometry, local, data, errors, valid


def _result(cube, psf, rows, center, factor, method, mask, background=None):
    """Attach units and provenance to the spectrum and its diagnostics."""
    table = Table(rows=rows)
    table['wave'].unit = cube.zunit
    for name in ['flux', 'flux_err', 'rawflux', 'rawerr', 'background', 'background_err']:
        if name in table.colnames:
            table[name].unit = cube.unit
    if 'covariance' in table.colnames:
        table['covariance'].unit = u.Unit(cube.unit or u.dimensionless_unscaled)**2
    table.meta = {'method': method, 'center': [float(v) for v in center],
                  'factor': int(factor), 'psf_domain': psf.domain.tolist(),
                  'psf_waveunit': str(psf.unit), 'mask': np.asarray(mask).tolist(),
                  'uncertainty': 'conditional on fixed PSF; independent voxel errors'}
    if method == 'optimal':
        table.meta['covnames'] = ['flux', 'background']
        table.meta['background'] = ('none' if background is None else
                                    'fixed constant' if background.amplitude.fixed else 'fitted constant')
    ra, dec = world_from_pixel_barycenter(cube.wcs, float(center[1]), float(center[0]))
    spectrum = Spectrum(cube.z.copy(), np.asarray(table['flux']),
                        yerr=np.asarray(table['flux_err']), xunit=cube.zunit,
                        yunit=cube.unit, xtype=cube.ztype, ytype=cube.valuetype,
                        ra=ra, dec=dec, redshift=cube.redshift, dtype=float,
                        meta={'method': method, 'uncertainty': table.meta['uncertainty']})
    failed = np.count_nonzero(~np.asarray(table['success']))
    if failed:
        warnings.warn(f'{failed} of {len(table)} {method} channels failed; inspect success and message.', UserWarning)
    return spectrum, table


def aperture(cube, psf, mask, *, center, factor=None):
    """Sum a nuclear aperture and correct its wavelength-dependent PSF losses.

    Parameters
    ----------
    cube : prism.data.Cube
        Flux-per-spaxel wavelength cube with calibrated standard deviations or
        variances. Surface-brightness data must first be converted consistently
        using pixel solid angle. Sky/instrument background, if needed, should
        be handled explicitly before this extraction.
    psf : ChromaticPSF
        Field-star calibration. Its spatial pixel scale and orientation must
        match the target cube; its centroid is overridden, not its shape.
    mask : two-dimensional array-like
        Fixed extraction weights in [0, 1], with the cube spatial shape. True
        selects a full pixel. The identical weights are used for data and PSF.
    center : pair of float
        Explicit target centroid (y, x), in local zero-based cube pixels. It is
        held fixed with wavelength. Estimate it independently from a suitable
        nuclear image; this function does not refit a centroid or PSF shape.
    factor : int, optional
        Midpoint samples per pixel axis; None retains the calibrated PSF value.
        Use convergence checks before changing it from the fitted value.

    Returns
    -------
    spectrum : prism.data.Spectrum
        PSF-aperture-corrected flux with conditional statistical errors.
    diagnostics : astropy.table.Table
        wave, flux, flux_err, rawflux, rawerr, fraction, npix, success, message.
        Units and extraction geometry are retained; the table can be exported.
        success marks a valid extraction, not optimizer convergence.
        Failed channels have NaN output flux/error. The extraction does not
        shrink around masked/invalid pixels: any invalid selected pixel makes
        that channel invalid. Out-of-domain/unphysical PSF evaluations are
        also reported as failed channels rather than extrapolated.

    Notes
    -----
    For normalized pixel-integrated profile P and weights w:
    fraction = sum(w*P); flux = sum(w*D)/fraction;
    variance = sum((w*sigma)**2)/fraction**2. PSF, centering and calibration
    uncertainty, voxel covariance and host-contamination systematics are not
    included. No oversubtraction factor or implicit background removal is used.

    Prefer this method when the desired product is a composite nuclear spectrum
    or when spatially structured, asymmetric, compact or chromatically varying
    extended emission is unknown. It imposes no flat-host model, but it rescales
    all aperture light: host emission H becomes H/fraction. It is not a total
    host spectrum, nor an unbiased isolated AGN spectrum. Line ratios can also
    be affected if their components have different spatial distributions.

    Prefer `optimal` when the target is the PSF-shaped component, the PSF is
    transferable, and any residual background is negligible, known, or locally
    well described by its fitted constant. Optimal weighting can improve
    precision in heterogeneous/background-dominated noise. Compare apertures
    and spatial residuals before treating either product as a physical separation.

    Examples
    --------
    >>> spectrum, diagnostics = aperture(cube, psf, mask, center=(13.1, 12.2))
    >>> diagnostics.write('aperture.ecsv', overwrite=True)
    """
    weights, wave, geometry, local, data, errors, valid = _inputs(cube, psf, mask, center, factor)
    rows = []
    for i, w in enumerate(wave):
        row = dict(wave=float(cube.z[i]), flux=np.nan, flux_err=np.nan,
                   rawflux=np.nan, rawerr=np.nan, fraction=np.nan,
                   npix=int(np.count_nonzero(valid[i])), success=False, message='Invalid aperture pixels')
        if not valid[i].all():
            rows.append(row)
            continue
        try:
            profile = psf.get_image(w, **geometry)[local]
            fraction = float(np.sum(weights*profile))
            if not np.isfinite(fraction) or not 0 < fraction <= 1 + 1e-12:
                raise ValueError('Invalid PSF fraction; inspect the model and pixel-integration convergence.')
            rawflux = float(np.sum(weights*data[i]))
            rawerr = float(np.sqrt(np.sum((weights*errors[i])**2)))
            row.update(rawflux=rawflux, rawerr=rawerr, fraction=fraction,
                       flux=rawflux/fraction, flux_err=rawerr/fraction,
                       success=True, message='Aperture corrected')
        except ValueError as exc:
            row['message'] = str(exc)
        rows.append(row)
    return _result(cube, psf, rows, center, geometry['factor'], 'aperture', mask)


def optimal(cube, psf, mask, *, center, background=None, factor=None):
    """Estimate PSF amplitude by variance-weighted linear spectral extraction.

    Parameters
    ----------
    cube, psf, center, factor
        As in `aperture`. No PSF width, wing exponent, ellipticity or centroid
        is refitted here. Calibrated voxel errors are required for meaningful
        absolute uncertainties.
    mask : two-dimensional array-like
        Binary spatial selection, with the cube spatial shape. Fractional pixel
        weights are not accepted: the regression uses whole-pixel measurements.
        Masked/nonfinite samples or nonpositive errors are excluded explicitly
        per wavelength; npix and fraction record the remaining spatial support.
    background : astropy.modeling.models.Const2D, optional
        None means zero background. A free Const2D fits a separate constant at
        each wavelength. A fixed Const2D imposes its supplied value; uncertainty
        in an imposed value is not propagated. Tied or finite-bounded background
        parameters are not supported by this unconstrained linear estimator.

    Returns
    -------
    spectrum : prism.data.Spectrum
        Integrated PSF amplitude, already on the infinite-plane normalization;
        do not apply another aperture correction. Amplitudes may be negative
        in noisy channels: they are not clipped.
    diagnostics : astropy.table.Table
        wave, flux, flux_err, background, background_err, covariance, fraction,
        npix, success, message, chi2, dof, redchi2. Covariance order is
        (flux, background), in squared input flux units. Background is per
        pixel. Fixed/absent background has zero covariance and uncertainty.
        Rank-deficient or empty fits yield failed channels with NaN flux/error.
        A determined fit with dof=0 has no defined redchi2.

    Notes
    -----
    WHY "OPTIMAL": this is generalized least squares with diagonal noise
    covariance. For a correct fixed PSF/background model and known independent
    variances, it is the best linear unbiased amplitude estimator (minimum
    variance within linear unbiased estimators); with Gaussian noise it is also
    the maximum-likelihood estimator. This statistical qualification does not
    mean universally optimal, physically deblended, or robust to model mismatch.
    Source-dominated noise does not guarantee an advantage over a simple sum.

    Fit D_i = F*P_i + B, solving F and optionally B jointly. The covariance is
    (X.T W X)^-1, with absolute input errors: it is not rescaled to force reduced
    chi-square to one. Correlation with the fitted background is included. PSF,
    centering, correlated-voxel, model-mismatch and host systematic uncertainties
    are not included. A tiny region can make the PSF and constant degenerate.

    PREFER `aperture` when extended emission can have unknown compact peaks,
    asymmetric spatial structure, or wavelength-dependent morphology. Such light
    can project onto the PSF and bias F; a fitted flat background does not protect
    against it. PSF-subtracted residuals therefore need not retain all host light.
    This routine is not an iterative deblender or a guaranteed host-preserving
    subtraction. Use it for a PSF-shaped flux estimate under explicit assumptions,
    and compare with aperture extraction and spatial residuals.

    For known background, the solution reduces to
    F = sum(P*(D-B)/variance) / sum(P**2/variance).
    The PSF is never renormalized over the selected or remaining valid pixels.

    Examples
    --------
    >>> from astropy.modeling.models import Const2D
    >>> spectrum, diagnostics = optimal(cube, psf, mask, center=(13.1, 12.2),
    ...                                 background=Const2D())
    """
    weights, wave, geometry, local, data, errors, valid = _inputs(cube, psf, mask, center, factor)
    if not np.all(weights == 1):
        raise ValueError('optimal requires a binary spatial mask.')
    if background is not None:
        if type(background) is not Const2D or background.parameters.size != 1 or background.amplitude.unit is not None:
            raise TypeError('background must be a unitless scalar Const2D or None.')
        lo, hi = background.amplitude.bounds
        if (background.amplitude.tied or (lo is not None and lo != -np.inf)
                or (hi is not None and hi != np.inf)):
            raise ValueError('optimal uses an unconstrained constant background: no ties or finite bounds.')
        if not np.isfinite(background.amplitude.value):
            raise ValueError('Background must be finite.')
    free_background = background is not None and not background.amplitude.fixed
    imposed = 0. if background is None or free_background else float(background.amplitude.value)
    rows = []
    for i, w in enumerate(wave):
        good = valid[i]
        row = dict(wave=float(cube.z[i]), flux=np.nan, flux_err=np.nan,
                   background=np.nan, background_err=np.nan, covariance=np.full((2, 2), np.nan),
                   fraction=np.nan, npix=int(good.sum()), success=False,
                   message='No valid pixels', chi2=np.nan, dof=int(good.sum() - 1 - free_background), redchi2=np.nan)
        if not good.any():
            rows.append(row)
            continue
        try:
            profile = psf.get_image(w, **geometry)[local][good]
            if (not np.isfinite(profile).all() or np.any(profile < 0)
                    or not 0 < profile.sum() <= 1 + 1e-12):
                raise ValueError('Invalid PSF fraction; inspect the model and pixel-integration convergence.')
            design = np.column_stack((profile, np.ones_like(profile))) if free_background else profile[:, None]
            matrix = design / errors[i, good, None]
            rhs = (data[i, good] - imposed) / errors[i, good]
            left, singular, right = np.linalg.svd(matrix, full_matrices=False)
            rank = np.sum(singular > np.finfo(float).eps * max(matrix.shape)*singular[0]) if singular.size else 0
            if rank < design.shape[1]:
                raise ValueError('Rank-deficient PSF/background fit: use more spatial support or a fixed background.')
            parameters = right.T @ ((left.T @ rhs) / singular)
            fitted_cov = (right.T / singular**2) @ right
            covariance = np.zeros((2, 2))
            covariance[:len(parameters), :len(parameters)] = fitted_cov
            residual = (data[i, good] - imposed - design @ parameters) / errors[i, good]
            chi2 = float(np.sum(residual**2))
            row.update(flux=float(parameters[0]), flux_err=float(np.sqrt(covariance[0, 0])),
                       background=float(parameters[1]) if free_background else imposed,
                       background_err=float(np.sqrt(covariance[1, 1])), covariance=covariance,
                       fraction=float(profile.sum()), chi2=chi2,
                       redchi2=chi2/row['dof'] if row['dof'] > 0 else np.nan,
                       success=True, message='Linear PSF fit')
        except ValueError as exc:
            row['message'] = str(exc)
        rows.append(row)
    return _result(cube, psf, rows, center, geometry['factor'], 'optimal', mask, background)
