import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import NDDataArray, VarianceUncertainty
from astropy.wcs import WCS

warnings.filterwarnings('ignore', message=r'scipy\.misc is deprecated.*', category=DeprecationWarning)

from prism.data import Cube, Image, RegionMeasurement, Spectrum
from prism.modeling.fitting import ScipyTRF
from prism.modeling.fitting.uncertainty.resample import ResampleError, bootstrap


def test_spectrum_generic_axis_workflow():
    x = np.linspace(5000.0, 5100.0, 12)
    y = 1.0 + 0.05 * np.sin(np.linspace(0.0, np.pi, x.size))
    yerr = np.full_like(x, 0.02)
    xerr = np.full_like(x, 0.5)

    spec = Spectrum(
        x=x,
        y=y,
        yerr=yerr,
        xerr=xerr,
        xunit='AA',
        yunit='Jy',
    )
    assert spec.xtype == 'wavelength'
    assert spec.ytype == 'flux-density'
    assert spec.medium is None

    freq = spec.frequencies(unit=u.THz)
    assert freq.shape == x.shape
    assert np.all(np.isfinite(freq))

    spec.crop(bounds=(5020.0, 5080.0))
    assert spec.x.size < x.size

    spec.reset()
    np.testing.assert_allclose(spec.x, x)
    np.testing.assert_allclose(spec.yerr, yerr)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        spec.rebin(factor=2)
    assert spec.x.size < x.size
    assert np.isfinite(spec.y[1:-1]).all()

    freq_spec = Spectrum(x=np.linspace(1.0, 2.0, 5), y=np.ones(5), xunit='THz')
    assert freq_spec.xtype == 'frequency'
    with pytest.raises(ValueError, match='wavelength-like'):
        freq_spec.air_to_vac()


def test_single_spectrum_fit_workflow():
    x = np.linspace(4990.0, 5010.0, 121)
    true_model = models.Const1D(0.2, name='cont') + models.Gaussian1D(1.5, 5000.0, 1.2, name='line')
    yerr = np.full_like(x, 0.03)

    rng = np.random.default_rng(7)
    spec = Spectrum(
        x=x,
        y=true_model(x) + rng.normal(0.0, yerr, x.size),
        yerr=yerr,
        xunit='AA',
    )

    init = models.Const1D(0.1, name='cont') + models.Gaussian1D(1.0, 4999.0, 1.8, name='line')
    fitter = ScipyTRF(calc_uncertainties=True)
    fitted = fitter(init, x=spec.x, y=spec.y, yerr=spec.yerr, inplace=False)

    assert np.isfinite(fitted.amplitude_1.value)
    assert abs(fitted.mean_1.value - 5000.0) < 0.5
    assert fitted.stddev_1.value > 0.0


def test_cube_image_workflow_and_native_fits_units(tmp_path):
    z = np.array([500.0, 501.0, 502.0, 503.0])
    values = np.arange(24, dtype=float).reshape(4, 2, 3)
    err = np.full_like(values, 0.1)

    cube = Cube(values=values, z=z, err=err, zunit='nm', unit='Jy')
    assert cube.ztype == 'wavelength'
    assert cube.medium is None
    image = cube.to_image(method='sum')

    assert isinstance(image, Image)
    assert image.values.shape == (2, 3)
    np.testing.assert_allclose(image.values, values.sum(axis=0))

    path = tmp_path / 'cube_native_units.fits'
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = 3
    header['NAXIS2'] = 2
    header['NAXIS3'] = 4
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'WAVE'
    header['CRPIX1'] = 1.0
    header['CRPIX2'] = 1.0
    header['CRPIX3'] = 1.0
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    header['CRVAL3'] = 500.0
    header['CDELT1'] = 1.0
    header['CDELT2'] = 1.0
    header['CDELT3'] = 1.0
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = 'nm'

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=values.astype(np.float32), header=header, name='DATA'),
        fits.ImageHDU(data=err.astype(np.float32), name='ERR'),
    ]).writeto(path)

    loaded = Cube.from_fits(path)
    assert loaded.zunit == u.nm
    assert loaded.ztype == 'wavelength'
    np.testing.assert_allclose(loaded.z, z)
    np.testing.assert_allclose(loaded.wavelengths(unit=u.AA), z * 10.0)
    np.testing.assert_allclose(loaded.frequencies(unit=u.Hz), (z * u.nm).to_value(u.Hz, equivalencies=u.spectral()))
    with pytest.raises(AttributeError):
        _ = loaded.wave

    cube_nd = cube.to_nddata()
    cube_roundtrip = Cube.from_nddata(cube_nd)
    np.testing.assert_allclose(cube_roundtrip.values, cube.values)
    np.testing.assert_allclose(cube_roundtrip.err, cube.err)
    np.testing.assert_allclose(cube_roundtrip.z, cube.z)

    image_nd = image.to_nddata()
    image_roundtrip = Image.from_nddata(image_nd)
    np.testing.assert_allclose(image_roundtrip.values, image.values)
    np.testing.assert_allclose(image_roundtrip.err, image.err)


def test_cube_wavelength_domain_processing():
    z = np.linspace(5000.0, 5010.0, 5)
    values = np.ones((5, 2, 2), dtype=float)
    err = np.full_like(values, 0.1)

    cube = Cube(
        values=values,
        z=z,
        err=err,
        zunit='AA',
    )
    assert cube.ztype == 'wavelength'
    assert cube.medium is None

    vac_cube = cube.air_to_vac(inplace=False)
    assert vac_cube.medium == 'vacuum'
    assert not np.allclose(vac_cube.z, cube.z)
    assert cube.medium is None

    air_cube = vac_cube.vac_to_air(inplace=False)
    np.testing.assert_allclose(air_cube.z, cube.z, rtol=0, atol=5e-4)

    red_cube = cube.deredden(ebv=np.array([[0.02, 0.03], [0.04, 0.05]]), inplace=False)
    assert not hasattr(red_cube, 'ebv')
    assert np.any(np.abs(red_cube.values - cube.values) > 0.0)
    assert np.any(np.abs(red_cube.err - cube.err) > 0.0)


def test_cube_and_image_crop_propagate_wcs_and_warn_on_direct_shape_change():
    cube_header = fits.Header()
    cube_header['NAXIS'] = 3
    cube_header['NAXIS1'] = 4
    cube_header['NAXIS2'] = 3
    cube_header['NAXIS3'] = 5
    cube_header['CTYPE1'] = 'RA---TAN'
    cube_header['CTYPE2'] = 'DEC--TAN'
    cube_header['CTYPE3'] = 'WAVE'
    cube_header['CRPIX1'] = 1.0
    cube_header['CRPIX2'] = 1.0
    cube_header['CRPIX3'] = 1.0
    cube_header['CRVAL1'] = 10.0
    cube_header['CRVAL2'] = 20.0
    cube_header['CRVAL3'] = 5000.0
    cube_header['CDELT1'] = 1.0
    cube_header['CDELT2'] = 2.0
    cube_header['CDELT3'] = 3.0

    cube = Cube(
        values=np.arange(60, dtype=float).reshape(5, 3, 4),
        z=np.arange(5, dtype=float),
        err=np.full((5, 3, 4), 0.1),
        wcs=WCS(cube_header),
        zunit='AA',
    )
    cropped_cube = cube.crop(zslice=slice(1, 4), yslice=slice(1, 3), xslice=slice(1, 4), inplace=False)
    assert cropped_cube.shape == (3, 2, 3)
    assert cropped_cube.wcs is not None
    np.testing.assert_allclose(cropped_cube.z, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cropped_cube.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cropped_cube.y, [1.0, 2.0])

    with pytest.warns(UserWarning, match=r'Prefer crop\(\)/crop_spectral\(\)'):
        cube.values = cube.values[:3]
    assert cube.shape == (3, 3, 4)

    image_header = fits.Header()
    image_header['NAXIS'] = 2
    image_header['NAXIS1'] = 4
    image_header['NAXIS2'] = 3
    image_header['CTYPE1'] = 'RA---TAN'
    image_header['CTYPE2'] = 'DEC--TAN'
    image_header['CRPIX1'] = 1.0
    image_header['CRPIX2'] = 1.0
    image_header['CRVAL1'] = 10.0
    image_header['CRVAL2'] = 20.0
    image_header['CDELT1'] = 1.0
    image_header['CDELT2'] = 2.0

    image = Image(
        values=np.arange(12, dtype=float).reshape(3, 4),
        err=np.full((3, 4), 0.1),
        wcs=WCS(image_header),
    )
    cropped_image = image.crop(yslice=slice(1, 3), xslice=slice(1, 4), inplace=False)
    assert cropped_image.shape == (2, 3)
    assert cropped_image.wcs is not None
    np.testing.assert_allclose(cropped_image.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cropped_image.y, [1.0, 2.0])

    with pytest.warns(UserWarning, match=r'Prefer crop\(\)'):
        image.values = image.values[:2]
    assert image.shape == (2, 4)


def test_cube_binmap_extraction_and_spatial_scales():
    cube_header = fits.Header()
    cube_header['NAXIS'] = 3
    cube_header['NAXIS1'] = 2
    cube_header['NAXIS2'] = 2
    cube_header['NAXIS3'] = 3
    cube_header['CTYPE1'] = 'RA---TAN'
    cube_header['CTYPE2'] = 'DEC--TAN'
    cube_header['CTYPE3'] = 'WAVE'
    cube_header['CRPIX1'] = 1.0
    cube_header['CRPIX2'] = 1.0
    cube_header['CRPIX3'] = 1.0
    cube_header['CRVAL1'] = 10.0
    cube_header['CRVAL2'] = 20.0
    cube_header['CRVAL3'] = 5000.0
    cube_header['CDELT1'] = 1 / 3600
    cube_header['CDELT2'] = 2 / 3600
    cube_header['CDELT3'] = 1.0
    cube_header['CUNIT1'] = 'deg'
    cube_header['CUNIT2'] = 'deg'
    cube_header['CUNIT3'] = 'Angstrom'

    values = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[3.0, 5.0], [7.0, 9.0]],
        ]
    )
    err = np.full_like(values, 0.2)
    binmap = np.array([[0, 0], [1, 1]])
    cube = Cube(
        values=values,
        z=np.array([5000.0, 5001.0, 5002.0]),
        err=err,
        wcs=WCS(cube_header),
        zunit='AA',
        unit='Jy',
        binmap=binmap,
    )

    binned = cube.apply_binmap(method='mean', inplace=False)
    np.testing.assert_allclose(binned.values[:, 0, 0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(binned.values[:, 0, 1], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(binned.values[:, 1, 0], [6.0, 7.0, 8.0])
    assert set(binned.unique_bins) == {0, 1}

    extracted = cube.extract_spectrum(np.array([[1.0, 0.5], [0.0, 0.0]]), method='sum')
    assert isinstance(extracted, Spectrum)
    np.testing.assert_allclose(extracted.x, cube.z)
    np.testing.assert_allclose(extracted.y, values[:, 0, 0] + 0.5 * values[:, 0, 1])
    scales = cube.pixel_scales(unit='arcsec')
    np.testing.assert_allclose(scales.value, [1.0, 2.0], atol=1e-6)


def test_image_binmap_region_measurement_and_spatial_scales():
    image_header = fits.Header()
    image_header['NAXIS'] = 2
    image_header['NAXIS1'] = 2
    image_header['NAXIS2'] = 2
    image_header['CTYPE1'] = 'RA---TAN'
    image_header['CTYPE2'] = 'DEC--TAN'
    image_header['CRPIX1'] = 1.0
    image_header['CRPIX2'] = 1.0
    image_header['CRVAL1'] = 10.0
    image_header['CRVAL2'] = 20.0
    image_header['CDELT1'] = 1 / 3600
    image_header['CDELT2'] = 2 / 3600
    image_header['CUNIT1'] = 'deg'
    image_header['CUNIT2'] = 'deg'

    image = Image(
        values=np.array([[1.0, 3.0], [5.0, 7.0]]),
        err=np.full((2, 2), 0.2),
        wcs=WCS(image_header),
        unit='Jy',
        binmap=np.array([[0, 0], [1, 1]]),
    )
    binned = image.apply_binmap(method='sum', inplace=False)
    np.testing.assert_allclose(binned.values, [[4.0, 4.0], [12.0, 12.0]])

    measurement = image.measure_region(np.array([[1.0, 0.5], [0.0, 0.0]]), method='sum')
    assert isinstance(measurement, RegionMeasurement)
    assert np.isclose(measurement.value, 2.5)
    assert measurement.n_pixels == 2
    assert measurement.area is not None

    scales = image.pixel_scales(unit='arcsec')
    np.testing.assert_allclose(scales.value, [1.0, 2.0], atol=1e-6)

    with pytest.raises(ValueError, match='Fractional masks are not supported'):
        image.measure_region(np.array([[1.0, 0.5], [0.0, 0.0]]), method='median')


def test_nddata_interoperability_from_external_objects():
    cube_nd = NDDataArray(
        data=np.arange(12, dtype=float).reshape(3, 2, 2),
        unit=u.Jy,
        uncertainty=VarianceUncertainty(np.full((3, 2, 2), 0.04)),
        mask=np.array(
            [
                [[False, False], [False, True]],
                [[False, False], [False, True]],
                [[False, False], [False, True]],
            ]
        ),
        meta={'prism_z': np.array([5000.0, 5001.0, 5002.0]), 'prism_zunit': 'AA'},
    )
    cube = Cube.from_nddata(cube_nd)
    assert cube.is_var is True
    assert cube.ztype == 'wavelength'
    assert not cube.mask[0, 1, 1]

    image_nd = NDDataArray(
        data=np.arange(4, dtype=float).reshape(2, 2),
        unit=u.Jy,
        uncertainty=VarianceUncertainty(np.full((2, 2), 0.09)),
        mask=np.array([[False, True], [False, False]]),
        meta={'prism_binmap': np.array([[0, 0], [1, 1]])},
    )
    image = Image.from_nddata(image_nd)
    assert image.is_var is True
    np.testing.assert_allclose(image.err, 0.3)
    assert image.binmap[1, 0] == 1


def test_multifit_uses_binmap_representatives():
    x = np.linspace(-1.0, 1.0, 21)
    true_cube = np.empty((2, 2, x.size), dtype=float)
    binmap = np.array([[0, 0], [1, 1]])
    params = {
        0: (1.0, 0.1),
        1: (2.0, -0.2),
    }
    for j in range(2):
        for i in range(2):
            slope, intercept = params[int(binmap[j, i])]
            true_cube[j, i] = intercept + slope * x

    fitter = ScipyTRF()
    init = models.Linear1D(slope=0.0, intercept=0.0)
    result = fitter.multifit(init, x, true_cube, binmap=binmap, nproc=1, progress=False)

    slope_map = result.slope.value
    intercept_map = result.intercept.value
    np.testing.assert_allclose(slope_map[0, 0], slope_map[0, 1])
    np.testing.assert_allclose(slope_map[1, 0], slope_map[1, 1])
    np.testing.assert_allclose(intercept_map[0, 0], intercept_map[0, 1])
    np.testing.assert_allclose(intercept_map[1, 0], intercept_map[1, 1])
    np.testing.assert_allclose(slope_map[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(slope_map[1, 0], 2.0, atol=1e-6)


def test_bootstrap_uncertainty_boundaries():
    x = np.linspace(-1.0, 1.0, 9)
    y = 2.0 + 0.5 * x
    sym = np.full_like(x, 0.1)
    asym = np.vstack([np.full_like(x, 0.08), np.full_like(x, 0.12)])

    fitter = ScipyTRF()
    model = models.Linear1D(slope=0.3, intercept=1.8)

    with pytest.raises(ValueError, match='Asymmetric yerr'):
        fitter(model, x, y, yerr=asym, inplace=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        samples = bootstrap(
            model=model,
            fitter=fitter,
            x=x,
            y=y,
            yerr=asym,
            n_samples=4,
            seed=0,
            verbose=False,
            noise_dist='uniform',
        )
    assert set(samples) == {'slope', 'intercept'}
    assert all(np.isfinite(values).sum() == 4 for values in samples.values())

    with pytest.raises(ResampleError, match='Asymmetric yerr is ambiguous'):
        bootstrap(
            model=model,
            fitter=fitter,
            x=x,
            y=y,
            yerr=asym,
            n_samples=2,
            seed=0,
            verbose=False,
            noise_dist='gauss',
        )

    perturbed = bootstrap(
        model=model,
        fitter=fitter,
        x=x,
        y=y,
        yerr=sym,
        xerr=np.full_like(x, 0.02),
        perturb_x=True,
        n_samples=3,
        seed=1,
        verbose=False,
        noise_dist='uniform',
    )
    assert all(np.isfinite(values).sum() == 3 for values in perturbed.values())