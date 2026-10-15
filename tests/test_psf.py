"""Scientific checks for spatial PSF normalization, recovery and persistence."""
import numpy as np
import pytest
import astropy.units as u
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.table import Table
from scipy.integrate import quad
from scipy.special import erf

from prism.modeling.psf import ChromaticPSF, fitbands, fittrends


def test_profile_normalization_and_domain():
    for model in [Gaussian2D(x_stddev=0.7, y_stddev=1.2), Moffat2D(gamma=1.3, alpha=2.4)]:
        for name in model.param_names:
            if name != 'amplitude':
                model.fixed[name] = True
        psf = ChromaticPSF(model, {}, [5000, 7000], unit=u.AA)
        profile = psf(600 * u.nm)
        if isinstance(profile, Moffat2D):
            total = quad(lambda r: 2*np.pi*r*profile(r, 0), 0, np.inf)[0]
        else:
            total = quad(lambda x: profile(x, 0), -np.inf, np.inf)[0]
            total *= np.sqrt(2*np.pi)*profile.y_stddev.value
        assert total == pytest.approx(1, rel=1e-8)
        with pytest.raises(ValueError, match='outside'):
            psf(4000)
        with pytest.raises(ValueError, match='one finite'):
            psf([5000, 6000])


def test_explicit_trends_and_unphysical_predictions():
    model = Moffat2D(gamma=2, alpha=2.5, fixed={'x_0': True, 'y_0': True})
    from prism.modeling.psf import _state
    bands = Table({'wave': [5000., 6000., 7000.], 'success': [True]*3,
                   'gamma': [1., 2., 3.], 'gamma_err': [.1]*3,
                   'alpha': [2.5]*3, 'alpha_err': [.1]*3})
    bands.meta = {'profile': _state(model), 'waveunit': 'Angstrom'}
    psf = fittrends(bands, {'gamma': 1, 'alpha': 0}, basis='polynomial', display=False)
    assert psf(5500).gamma.value == pytest.approx(1.5)
    with pytest.raises(ValueError, match='exactly'):
        fittrends(bands, {'gammma': 1}, basis='polynomial', display=False)
    with pytest.raises(ValueError, match='enough'):
        fittrends(bands, {'gamma': 3, 'alpha': 0}, basis='polynomial', display=False)
    psf.trends['alpha']['coefficients'] = [0.5]
    with pytest.raises(ValueError, match='alpha > 1'):
        psf(6000)


def _gaussian_cube():
    """Generate exact pixel integrals independently of PSF quadrature."""
    from prism.data import Cube
    wave = np.linspace(5000, 7000, 9)
    y, x = np.indices((13, 13))
    values = []
    for w in wave:
        width = 1.2 + 0.2*(w-6000)/1000
        px = .5*(erf((x+.5-6.2)/(np.sqrt(2)*width))-erf((x-.5-6.2)/(np.sqrt(2)*width)))
        py = .5*(erf((y+.5-5.8)/(np.sqrt(2)*1.4))-erf((y-.5-5.8)/(np.sqrt(2)*1.4)))
        values.append(1000*px*py+2)
    return Cube(values, z=wave, zunit=u.AA, err=np.full((9, 13, 13), .1), dtype=float)


def test_gaussian_recovery_masks_and_fits_roundtrip(tmp_path):
    from astropy.modeling.models import Const2D
    cube = _gaussian_cube()
    cube.mask[3] = False  # Simulate a fully masked laser channel.
    model = Gaussian2D(amplitude=100, x_mean=6, y_mean=6, x_stddev=1.3,
                       y_stddev=1.5, theta=0, fixed={'theta': True})
    with pytest.warns(UserWarning, match='1 of 9'):
        bands = fitbands(cube, model, bands=9, background=Const2D(0), factor=9)
    assert np.count_nonzero(bands['success']) == 8
    np.testing.assert_array_equal(bands['nband'], np.arange(9))
    np.testing.assert_array_equal(bands[[2, 5, 8]]['nband'], [2, 5, 8])
    assert not np.any(bands['channels'][:, 3])
    good = np.asarray(bands['success'])
    np.testing.assert_allclose(bands['redchi2'][good], bands['chi2'][good] / bands['dof'][good])
    assert np.isnan(bands['redchi2'][~good]).all()
    good = np.asarray(bands['success'])
    expected = 1.2+.2*(np.asarray(bands['wave'])[good]-6000)/1000
    np.testing.assert_allclose(bands['x_stddev'][good], expected, atol=.002)
    np.testing.assert_allclose(bands['y_stddev'][good], 1.4, atol=.002)
    np.testing.assert_allclose(bands['background'][good], 2, atol=.001)
    np.testing.assert_allclose(bands['x_mean'][good], 6.2, atol=.001)
    assert model.x_mean.value == 6  # Caller model unchanged.
    psf = fittrends(bands, {'x_mean': 0, 'y_mean': 0, 'x_stddev': 1,
                            'y_stddev': 0}, basis='polynomial', display=False, save=tmp_path/'trends.pdf')
    assert (tmp_path/'trends.pdf').stat().st_size > 0
    path = tmp_path/'psf.fits'
    psf.write(path)
    with pytest.raises(OSError):
        psf.write(path)
    loaded = ChromaticPSF.from_fits(path)
    np.testing.assert_array_equal(loaded.bands['channels'], bands['channels'])
    np.testing.assert_allclose(loaded.bands['covariance'], bands['covariance'], equal_nan=True)
    assert loaded.bands.meta == bands.meta
    np.testing.assert_allclose(loaded.bands['redchi2'], bands['redchi2'], equal_nan=True)
    np.testing.assert_array_equal(loaded.bands['nband'], bands['nband'])
    assert loaded.factor == 9
    assert loaded.bands['lower'].unit == u.AA
    assert loaded.bands['x_stddev'].unit == u.pix
    assert loaded.bands['wave'].unit == u.AA
    np.testing.assert_array_equal(psf.get_cube([5100, 6000, 6900], (13, 13)),
                                  loaded.get_cube([5100, 6000, 6900], (13, 13)))
    # Full image flux and a central-pixel aperture correction recover the source.
    image = loaded.get_image(6000, (13, 13))
    assert image.sum() == pytest.approx(1, abs=2e-5)
    observed = cube.values[4, 6, 6]-2
    assert observed/image[6, 6] == pytest.approx(1000, rel=.002)


def test_pixel_integration_convergence_and_finite_field():
    model = Gaussian2D(x_mean=0.23, y_mean=-.11, x_stddev=.4, y_stddev=.55)
    for name in model.param_names:
        if name != 'amplitude':
            model.fixed[name] = True
    psf = ChromaticPSF(model, {}, [5000, 7000], unit='Angstrom')
    exact = .5*(erf((.5-.23)/(np.sqrt(2)*.4))-erf((-.5-.23)/(np.sqrt(2)*.4)))
    exact *= .5*(erf((.5+.11)/(np.sqrt(2)*.55))-erf((-.5+.11)/(np.sqrt(2)*.55)))
    errors = [abs(psf.get_image(6000, (1, 1), factor=f)[0, 0]-exact) for f in [1, 5, 25]]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-4
    assert psf.get_image(6000, (1, 1), factor=25).sum() < 1
    assert psf.get_image(6000, (1, 1), center=(0, 0), factor=25)[0, 0] > exact
    with pytest.raises(ValueError, match='factor'):
        psf.get_image(6000, (2, 2), factor=0)


def test_moffat_recovery_and_fixed_background(tmp_path):
    from astropy.modeling.models import Const2D
    from prism.data import Cube
    y, x = np.indices((17, 17))
    # Independent continuous profile at centers, to validate model conventions.
    truth = 60*(1+((x-8.1)**2+(y-7.9)**2)/1.8**2)**(-2.6)+3
    cube = Cube(np.repeat(truth[None], 6, axis=0), z=np.arange(6)*100+5000,
                zunit='Angstrom', err=np.ones((6, 17, 17))*.1, dtype=float)
    background = Const2D(3, fixed={'amplitude': True})
    model = Moffat2D(50, 8, 8, 2, 2.5, fixed={'alpha': True})
    # Fixed means imposed, not estimated: use the correct imposed exponent.
    model.alpha = 2.6
    bands = fitbands(cube, model, bands=3, background=background)
    assert np.all(bands['success'])
    np.testing.assert_allclose(bands['gamma'], 1.8, atol=1e-6)
    np.testing.assert_allclose(bands['alpha'], 2.6)
    np.testing.assert_allclose(bands['alpha_err'], 0)
    np.testing.assert_allclose(bands['background_err'], 0)
    psf = fittrends(bands, {'gamma': 0, 'x_0': 0, 'y_0': 0}, basis='polynomial', display=False)
    assert psf(5200).alpha.value == 2.6
    psf.write(tmp_path/'moffat.fits')
    loaded = ChromaticPSF.from_fits(tmp_path/'moffat.fits')
    assert loaded.model.alpha.fixed
    assert loaded.model.alpha.bounds == psf.model.alpha.bounds
    np.testing.assert_array_equal(loaded.get_image(5200, (17, 17)), psf.get_image(5200, (17, 17)))


def test_invalid_bands_models_and_constraints():
    cube = _gaussian_cube()
    with pytest.raises(ValueError, match='non-overlapping'):
        fitbands(cube, Gaussian2D(), [[5000, 6000], [5500, 7000]])
    with pytest.raises(ValueError, match='alpha > 1'):
        fitbands(cube, Moffat2D(alpha=.5))
    model = Gaussian2D(tied={'x_mean': lambda m: m.y_mean})
    with pytest.raises(ValueError, match='Tied'):
        fitbands(cube, model)
    with pytest.raises(ValueError, match='factor'):
        fitbands(cube, Gaussian2D(), factor=2.5)
    cube.zunit = u.Hz
    with pytest.raises(ValueError, match='wavelength'):
        fitbands(cube, Gaussian2D())


def test_manual_band_selection_and_unweighted_trends():
    from prism.modeling.psf import _state
    model = Moffat2D(fixed={'alpha': True, 'x_0': True, 'y_0': True})
    model.alpha = 2.5
    bands = Table({'wave': [5000., 6000., 7000.], 'success': [True]*3,
                   'gamma': [1., 99., 3.], 'gamma_err': [np.nan]*3})
    bands.meta = {'profile': _state(model), 'waveunit': 'Angstrom'}
    with pytest.warns(UserWarning, match='unweighted'):
        psf = fittrends(bands[[0, 2]], {'gamma': 1}, basis='polynomial', display=False)
    assert psf(6000).gamma.value == pytest.approx(2)
    assert psf.trends['gamma']['covariance'] is None


def test_diagnostic_display_controls(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from prism.modeling.psf import _state
    model = Moffat2D(alpha=2.5, fixed={'alpha': True, 'x_0': True, 'y_0': True})
    bands = Table({'wave': [5000., 6000., 7000.], 'success': [True]*3,
                   'gamma': [1., 2., 3.], 'gamma_err': [.1]*3})
    bands.meta = {'profile': _state(model), 'waveunit': 'Angstrom'}
    calls = []
    monkeypatch.setattr(plt, 'show', lambda: calls.append(True))
    fittrends(bands, {'gamma': 1}, basis='polynomial', display=False)
    assert calls == [] and plt.get_fignums() == []
    fittrends(bands, {'gamma': 1}, basis='polynomial', save=tmp_path/'trends.png')
    assert calls == [True] and len(plt.get_fignums()) == 1
    plt.close('all')
    assert (tmp_path/'trends.png').stat().st_size > 0


def test_band_edges_units_and_partial_masks():
    from astropy.modeling.models import Const2D
    cube = _gaussian_cube()
    cube.mask[1, 0, 0] = False
    model = Gaussian2D(amplitude=100, x_mean=6.2, y_mean=5.8,
                       x_stddev=1.2, y_stddev=1.4, fixed={'theta': True})
    with pytest.warns(UserWarning, match='support differs'):
        bands = fitbands(cube, model, [[500, 600], [600, 700]]*u.nm,
                         background=Const2D(2), factor=3)
    assert np.all(bands['success'])
    assert not np.any(np.sum(bands['channels'], axis=0) > 1)
    assert bands['channels'][-1, -1]
    assert bands.meta['waveunit'] == 'Angstrom'
    psf = fittrends(bands, {'x_mean': 0, 'y_mean': 0, 'x_stddev': 1,
                            'y_stddev': 0}, basis='polynomial', display=False)
    np.testing.assert_array_equal(psf.domain, [5000, 7000])
    assert psf.get_cube(cube.z, cube.shape[1:]).shape == cube.shape


def test_all_fixed_profile_and_bounds(tmp_path):
    from prism.modeling.psf import _state
    model = Moffat2D(alpha=2.5)
    for name in model.param_names:
        model.fixed[name] = True
    bands = Table({'wave': [5000., 7000.], 'success': [True, True]})
    bands.meta = {'profile': _state(model), 'waveunit': 'Angstrom'}
    psf = fittrends(bands, {}, basis='polynomial', display=False)
    psf.write(tmp_path/'fixed.fits')
    loaded = ChromaticPSF.from_fits(tmp_path/'fixed.fits')
    np.testing.assert_array_equal(loaded.get_image(6000, (5, 5)), psf.get_image(6000, (5, 5)))
    with pytest.raises(ValueError, match='bounds'):
        fitbands(_gaussian_cube(), Gaussian2D(x_mean=20, bounds={'x_mean': (0, 10)}))


def test_planar_background_recovery_and_roundtrip(tmp_path):
    from astropy.modeling.models import Planar2D
    from prism.data import Cube
    y, x = np.indices((17, 17))
    truth = 70*(1+((x-8.2)**2+(y-7.8)**2)/1.7**2)**(-2.4)
    truth += .2*x - .3*y + 4
    cube = Cube(np.repeat(truth[None], 6, axis=0), z=np.arange(6)*100+5000,
                zunit='Angstrom', err=np.ones((6, 17, 17))*.1, dtype=float)
    model = Moffat2D(60, 8, 8, 2, 2.5)
    background = Planar2D(0, -.3, 3, fixed={'slope_y': True})
    bands = fitbands(cube, model, bands=3, background=background)
    assert np.all(bands['success'])
    for name, value in [('gamma', 1.7), ('alpha', 2.4), ('slope_x', .2),
                        ('slope_y', -.3), ('intercept', 4)]:
        np.testing.assert_allclose(bands[name], value, atol=1e-6)
    assert background.slope_x.value == 0
    np.testing.assert_allclose(bands['slope_y_err'], 0)
    assert bands['covariance'].shape == (3, 8, 8)
    psf = fittrends(bands, {'x_0': 0, 'y_0': 0, 'gamma': 0, 'alpha': 0}, basis='polynomial', display=False)
    psf.write(tmp_path/'plane.fits')
    loaded = ChromaticPSF.from_fits(tmp_path/'plane.fits')
    assert loaded.bands.meta['background']['class'] == 'Planar2D'
    assert loaded.bands['slope_x'].unit == u.pix**-1
    np.testing.assert_allclose(loaded.bands['intercept'], 4, atol=1e-6)
    np.testing.assert_array_equal(loaded.get_image(5200, (17, 17)), psf.get_image(5200, (17, 17)))


def test_fwhm_moffat_models_are_astropy_models():
    from astropy.modeling import Fittable2DModel
    from astropy.modeling.models import Gaussian2D
    from prism.modeling.models.spatial import MoffatPSF, EllipticalMoffatPSF
    beta, width = 2.7, 3.1
    circular = MoffatPSF(amplitude=5, fwhm=width, beta=beta)
    assert isinstance(circular, Fittable2DModel)
    assert circular(width/2, 0) == pytest.approx(2.5)
    native = Moffat2D(amplitude=5, gamma=width/(2*np.sqrt(2**(1/beta)-1)), alpha=beta)
    x, y = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-3, 3, 11))
    np.testing.assert_allclose(circular(x, y), native(x, y))
    ellipse = EllipticalMoffatPSF(amplitude=5, fwhm_x=width, fwhm_y=2, beta=beta, theta=.3)
    assert ellipse(width/2*np.cos(.3), width/2*np.sin(.3)) == pytest.approx(2.5)
    assert ellipse(-np.sin(.3), np.cos(.3)) == pytest.approx(2.5)
    # Ordinary Astropy fitter interoperability, not only the PSF utility.
    from astropy.modeling.fitting import TRFLSQFitter
    fitted = TRFLSQFitter()(MoffatPSF(amplitude=4, fwhm=3, beta=2.5), x, y, circular(x, y))
    assert fitted.fwhm.value == pytest.approx(width, rel=1e-5)


def test_elliptical_moffat_recovery_and_resolution(tmp_path):
    from prism.modeling.models.spatial import MoffatPSF, EllipticalMoffatPSF
    from prism.data import Cube
    y, x = np.indices((19, 19))
    dx, dy = x-9.2, y-8.8
    xp, yp = np.cos(.25)*dx+np.sin(.25)*dy, -np.sin(.25)*dx+np.cos(.25)*dy
    scale = 2*np.sqrt(2**(1/2.6)-1)
    truth = 60*(1+(xp/(3.2/scale))**2+(yp/(2.5/scale))**2)**(-2.6)
    cube = Cube(np.repeat(truth[None], 8, axis=0), z=np.linspace(5000, 7000, 8),
                zunit='Angstrom', err=np.ones((8, 19, 19))*.1, dtype=float)
    model = EllipticalMoffatPSF(50, 9, 9, 3, 2.4, 2.5, .2)
    bands = fitbands(cube, model, bands=4)
    assert np.all(bands['success'])
    for name,value in [('fwhm_x',3.2),('fwhm_y',2.5),('theta',.25),('beta',2.6)]:
        np.testing.assert_allclose(bands[name], value, atol=1e-5)
    psf = fittrends(bands, {name:1 for name in model.param_names if name!='amplitude'}, display=False)
    np.testing.assert_allclose(psf.fwhm([5000,6000,7000]).value, [[3.2,2.5]]*3, atol=1e-5)
    assert psf.fwhm(6000).unit == u.pix
    path = tmp_path/'ellipse.fits'
    psf.write(path)
    loaded = ChromaticPSF.from_fits(path)
    assert isinstance(loaded.model, EllipticalMoffatPSF)
    np.testing.assert_array_equal(loaded.get_image(6000,(19,19)), psf.get_image(6000,(19,19)))
    assert psf.get_image(6000,(101,101),center=(50,50),factor=9).sum() == pytest.approx(1,abs=3e-3)
    for native in [Gaussian2D(x_stddev=1.2,y_stddev=1.7),Moffat2D(gamma=2,alpha=2.5),MoffatPSF(fwhm=3)]:
        for name in native.param_names:
            if name!='amplitude':native.fixed[name]=True
        fixed=ChromaticPSF(native,{},[5000,7000],unit='Angstrom')
        widths=fixed.fwhm(6000).value
        if isinstance(native,Gaussian2D):np.testing.assert_allclose(widths,np.array([1.2,1.7])*np.sqrt(8*np.log(2)))
        elif isinstance(native,MoffatPSF):np.testing.assert_allclose(widths,[3,3])
        else:np.testing.assert_allclose(widths,[4*np.sqrt(2**.4-1)]*2)


def test_bspline_exact_recovery_and_equal_parameter_budget(tmp_path):
    from prism.modeling.models.spatial import MoffatPSF
    from prism.modeling.psf import _state
    model=MoffatPSF(fwhm=2,fixed={'beta':True,'x_0':True,'y_0':True})
    x=np.linspace(-1,1,25)
    values=2+.2*x+.3*np.maximum(0,x)**3
    bands=Table({'wave':6000+1000*x,'success':np.ones(25,dtype=bool),
                 'fwhm':values,'fwhm_err':np.full(25,.02)})
    bands.meta={'profile':_state(model),'waveunit':'Angstrom'}
    spline=fittrends(bands,{'fwhm':5},display=False,save=tmp_path/'spline.pdf')
    polynomial=fittrends(bands,{'fwhm':4},basis='polynomial',display=False)
    assert len(spline.trends['fwhm']['coefficients'])==len(polynomial.trends['fwhm']['coefficients'])==5
    assert spline.trends['fwhm']['degree']==3
    grid=np.linspace(-1,1,101)
    expected=2+.2*grid+.3*np.maximum(0,grid)**3
    np.testing.assert_allclose(spline.fwhm(6000+1000*grid).value[:,0],expected,atol=1e-12)
    spline.write(tmp_path/'spline.fits')
    loaded=ChromaticPSF.from_fits(tmp_path/'spline.fits')
    np.testing.assert_array_equal(loaded.fwhm(6000+1000*grid),spline.fwhm(6000+1000*grid))
    with pytest.raises(ValueError,match='integer'):
        fittrends(bands,{'fwhm':0},display=False)


def test_bspline_physical_bounds_and_legacy_fits(tmp_path):
    import json,warnings
    from astropy.io import fits
    from prism.modeling.models.spatial import MoffatPSF
    from prism.modeling.psf import _state
    model=MoffatPSF(fwhm=.8,bounds={'fwhm':(.1,1.5)},fixed={'beta':True,'x_0':True,'y_0':True})
    x=np.linspace(-1,1,25)
    bands=Table({'wave':6000+1000*x,'success':np.ones(25,dtype=bool),
                 'fwhm':.8+.5*np.sin(np.pi*x),'fwhm_err':np.full(25,.02)})
    bands.meta={'profile':_state(model),'waveunit':'Angstrom'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        psf=fittrends(bands,{'fwhm':5},display=False)
    widths=psf.fwhm(np.linspace(5000,7000,501)).value
    assert widths.min()>=.1 and widths.max()<=1.5
    # An independent v1 fixture verifies loading old global polynomial models.
    native=Moffat2D(gamma=2,alpha=2.5,fixed={'alpha':True,'x_0':True,'y_0':True})
    payload=json.dumps({'profile':_state(native),'trends':{'gamma':{'coefficients':[2,.2],
                        'covariance':None,'weighted':False}},'domain':[5000,7000],
                        'waveunit':'Angstrom','factor':1,'normalization':'infinite-plane',
                        'coordinates':'local-zero-based-pixels','basis':'power-ascending-on-minus1-plus1','bandmeta':{}})
    hdus=fits.HDUList([fits.PrimaryHDU(header=fits.Header({'PSFVER':1})),
          fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(payload)}A',array=[payload])],name='MODEL_DEF')])
    hdus.writeto(tmp_path/'legacy.fits')
    loaded=ChromaticPSF.from_fits(tmp_path/'legacy.fits')
    assert loaded(5500).gamma.value==pytest.approx(1.9)



def test_bspline_domain_reaches_band_edges_smoothly(tmp_path):
    from prism.modeling.models.spatial import MoffatPSF
    from prism.modeling.psf import _state
    model=MoffatPSF(fwhm=2,fixed={'beta':True,'x_0':True,'y_0':True})
    bands=Table({'wave':[5100.,6000.,6900.], 'success':[True]*3,
                 'fwhm':[1.,2.,3.], 'fwhm_err':[.1]*3,
                 'channels':[[True,True,False,False,False],
                             [False,False,True,False,False],
                             [False,False,False,True,True]]})
    bands.meta={'profile':_state(model),'waveunit':'Angstrom',
                'wavelengths':[5000.,5100.,6000.,6900.,7000.]}
    psf=fittrends(bands,{'fwhm':2},display=False)
    assert psf.trends['fwhm'].get('extension') is None
    wave=np.array([5000., 5099., 5100., 5101., 6000., 6899., 6900., 6901., 7000.])
    expected=2+(wave-6000)/900
    np.testing.assert_allclose(psf.fwhm(wave).value[:,0],expected,atol=1e-12)
    psf.write(tmp_path/'edges.fits')
    loaded=ChromaticPSF.from_fits(tmp_path/'edges.fits')
    np.testing.assert_allclose(loaded.fwhm(wave).value[:,0],expected,atol=1e-12)
    # Existing saved laws with constant padding keep their original meaning.
    legacy=dict(psf.trends['fwhm'], knots=[-.9,-.9,.9,.9],
                coefficients=[1.,3.], extension='constant')
    old=ChromaticPSF(model, {'fwhm':legacy}, psf.domain, unit=psf.unit)
    old.write(tmp_path/'legacy-padding.fits')
    old=ChromaticPSF.from_fits(tmp_path/'legacy-padding.fits')
    np.testing.assert_allclose(old.fwhm(5000),old.fwhm(5100))
    np.testing.assert_allclose(old.fwhm(7000),old.fwhm(6900))
    with pytest.raises(ValueError,match='outside'):
        psf(4999)


def test_fitimage_recovery_masks_and_local_coordinates():
    from prism.data import Image
    from prism.modeling.psf import fitimage
    from astropy.modeling.models import Planar2D
    cube = _gaussian_cube()
    y, x = np.indices(cube.shape[1:])
    values = cube.values[4].copy() + .03*x - .02*y
    values[0, 0] = 1e8
    image = Image(values, err=np.full(values.shape, .1), dtype=float,
                  x=np.arange(13)+100, y=np.arange(13)+200, unit=u.Jy)
    image.mask[0, 0] = False
    mask = (x-6)**2 + (y-6)**2 <= 5**2
    model = Gaussian2D(amplitude=100, x_mean=6, y_mean=6, x_stddev=1.3,
                       y_stddev=1.5, fixed={'theta': True})
    background = Planar2D()
    original_background = background.parameters.copy()
    original = model.parameters.copy()
    result = fitimage(image, model, mask=mask, background=background, factor=9)
    assert result['success'], result['message']
    np.testing.assert_allclose([result['x_mean'],result['y_mean']], [6.2,5.8], atol=1e-4)
    np.testing.assert_allclose([result['x_stddev'],result['y_stddev']], [1.2,1.4], atol=.001)
    np.testing.assert_allclose([result['slope_x'],result['slope_y'],result['intercept']], [.03,-.02,2], atol=1e-5)
    assert result['npix'] == np.count_nonzero(mask & image.mask)
    assert result['redchi2'] == pytest.approx(result['chi2']/result['dof'])
    assert result.table['amplitude'].unit == u.Jy
    assert result.table['slope_x'].unit == u.Jy/u.pix
    np.testing.assert_array_equal(model.parameters,original)
    np.testing.assert_array_equal(background.parameters,original_background)


def test_fitimage_absolute_covariance_matches_linear_solution():
    from prism.data import Image
    from prism.modeling.psf import fitimage
    y,x=np.indices((11,11))
    profile=Gaussian2D(amplitude=1,x_mean=5.2,y_mean=4.8,x_stddev=1.2,y_stddev=1.4)
    shape=profile(x,y)
    image=Image(25*shape,err=np.full(shape.shape,.2),dtype=float)
    for name in profile.param_names:
        profile.fixed[name]=name!='amplitude'
    result=fitimage(image,profile)
    assert result['success']
    assert result['amplitude']==pytest.approx(25)
    assert result['amplitude_err']==pytest.approx(.2/np.sqrt(np.sum(shape**2)),rel=1e-6)
    assert result['x_mean_err']==0


def test_fitimage_peak_is_optional_masked_and_respects_fixed_bounds():
    from prism.data import Image
    from prism.modeling.psf import fitimage
    y,x=np.indices((11,11))
    truth=Gaussian2D(amplitude=30,x_mean=6,y_mean=5,x_stddev=1,y_stddev=1)
    values=truth(x,y)+2
    values[0,0]=1e8
    image=Image(values,err=np.ones(values.shape),dtype=float)
    mask=np.ones(values.shape,dtype=bool);mask[0,0]=False
    model=Gaussian2D(amplitude=5,x_mean=4,y_mean=5,x_stddev=1,y_stddev=1,
                     fixed={'y_mean':True,'x_stddev':True,'y_stddev':True,'theta':True})
    model.x_mean.bounds=(3,6.5)
    from astropy.modeling.models import Const2D
    background=Const2D(2,fixed={'amplitude':True})
    default=fitimage(image,model,mask=mask,background=background)
    seeded=fitimage(image,model,mask=mask,background=background,init='peak')
    assert default['success'] and seeded['success']
    assert default.meta['profile']['parameters']['x_mean']['value']==4
    assert seeded.meta['profile']['parameters']['x_mean']['value']==6
    assert seeded.meta['profile']['parameters']['amplitude']['value']==30
    assert seeded['y_mean']==5 and model.x_mean.value==4
    np.testing.assert_allclose([seeded['amplitude'],seeded['x_mean']],[30,6],atol=1e-8)
    model.x_mean.bounds=(3,5.5)
    with pytest.warns(UserWarning,match='covariance unavailable'):
        bounded=fitimage(image,model,mask=mask,background=background,init='peak')
    assert bounded.meta['profile']['parameters']['x_mean']['value']==5.5
    assert bounded['x_mean']<=5.5


def test_fitimage_empty_support_and_validation():
    from prism.data import Image
    from prism.modeling.psf import fitimage
    image=Image(np.ones((5,5)),err=np.ones((5,5)))
    result=fitimage(image,Gaussian2D(),mask=np.zeros((5,5),bool),init='peak')
    assert not result['success'] and result['npix']==0
    assert np.isnan(result['amplitude']) and np.isnan(result['chi2'])
    with pytest.raises(ValueError,match='boolean'):
        fitimage(image,Gaussian2D(),mask=np.ones((5,5)))
    with pytest.raises(ValueError,match='image shape'):
        fitimage(image,Gaussian2D(),mask=np.ones((2,2),bool))
    with pytest.raises(ValueError,match='init'):
        fitimage(image,Gaussian2D(),init='auto')
    with pytest.raises(TypeError,match='two-dimensional'):
        fitimage(np.ones((5,5)),Gaussian2D())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_band_image_reuse_preserves_precision_support_and_input(dtype):
    from prism.data import Cube
    y,x=np.indices((5,5))
    profile=Gaussian2D(amplitude=1,x_mean=2,y_mean=2,x_stddev=1,y_stddev=1)
    shape=profile(x,y)
    values=np.array([2*shape,1000*shape,6*shape],dtype=dtype)
    errors=np.ones_like(values)
    errors[1]=0  # This channel has no support for a weighted spatial fit.
    errors[0,0,0]=0
    values[2,0,1]=np.nan
    cube=Cube(values,z=[5000,5001,5002],zunit=u.AA,err=errors,
              mask=np.ones(values.shape,bool),dtype=dtype)
    original_values=cube.values.copy();original_mask=cube.mask.copy()
    for name in profile.param_names:profile.fixed[name]=name!='amplitude'
    with pytest.warns(UserWarning,match='support differs'):
        result=fitbands(cube,profile,bands=1)
    assert result['success'][0]
    assert result['nchan'][0]==2
    np.testing.assert_array_equal(result['channels'][0],[True,False,True])
    # Independent weighted linear solution for the per-pixel two-channel mean.
    mean=4*shape
    mean[0,0]=6*shape[0,0];mean[0,1]=2*shape[0,1]
    variance=np.full(shape.shape,.5);variance[0,:2]=1
    expected=np.sum(shape*mean/variance)/np.sum(shape**2/variance)
    assert result['amplitude'][0]==pytest.approx(expected,rel=2e-6)
    assert result['amplitude_err'][0]==pytest.approx(1/np.sqrt(np.sum(shape**2/variance)),rel=2e-6)
    np.testing.assert_array_equal(cube.values,original_values)
    np.testing.assert_array_equal(cube.mask,original_mask)
    image=cube.cutout_slices(z=slice(0,1)).get_image(method='mean')
    assert image.values.dtype==image.err.dtype==np.dtype(dtype)
