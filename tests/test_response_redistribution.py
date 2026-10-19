"""Independent conservation, boundary and native-grid tests for Gaussian responses."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from astropy.io.misc import yaml
from astropy.modeling.models import Gaussian1D
from prism.modeling.operators import InstrumentResponse, SpectralResponse


def edges(x):
    return np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


@pytest.mark.parametrize('nonuniform', [False, True])
def test_weighted_columns_equal_retained_probability(nonuniform):
    x = np.linspace(-10., 10., 301)
    if nonuniform:
        x = np.r_[np.linspace(-10., 0., 220, endpoint=False), np.linspace(0., 10., 81)]
    sigma = .3 + .08*(x+10)
    matrix = InstrumentResponse._build_sparse_gaussian_matrix(x, sigma)
    bounds = edges(x)
    widths = np.diff(bounds)
    retained = ndtr((bounds[-1]-x)/sigma) - ndtr((bounds[0]-x)/sigma)
    np.testing.assert_allclose(widths @ matrix, widths*retained, atol=3e-15, rtol=2e-13)
    flux = np.exp(-.5*((x-.2)/.4)**2)
    np.testing.assert_allclose(widths @ (matrix @ flux), widths @ (flux*retained), atol=1e-14)


def test_nonuniform_matrix_against_gaussian_quadrature():
    x = np.r_[np.linspace(-5., 0., 35, endpoint=False), np.linspace(0., 5., 16)]
    sigma = .3 + .07*(x+5)
    matrix = InstrumentResponse._build_sparse_gaussian_matrix(x, sigma).toarray()
    bounds = edges(x)
    widths = np.diff(bounds)
    for i,j in [(0,0), (8,11), (34,35), (35,34), (50,48)]:
        probability = quad(lambda value: np.exp(-.5*((value-x[j])/sigma[j])**2)
                            /(np.sqrt(2*np.pi)*sigma[j]), bounds[i], bounds[i+1], epsabs=1e-14)[0]
        np.testing.assert_allclose(matrix[i,j], probability*widths[j]/widths[i], atol=3e-15, rtol=1e-11)


def test_edge_loss_is_not_renormalized():
    x = np.linspace(-5., 5., 201)
    response = InstrumentResponse.from_fixed_sigma(x, .8)
    bounds = edges(x)
    widths = np.diff(bounds)
    flux = np.zeros(x.size)
    flux[-1] = 1/widths[-1]
    retained = ndtr((bounds[-1]-x[-1])/.8) - ndtr((bounds[0]-x[-1])/.8)
    assert retained < .52
    np.testing.assert_allclose(widths @ response(flux), retained, atol=2e-15)


def test_gaussian_limit_converges_with_sampling():
    errors = []
    for step in [.1, .05, .025]:
        x = np.arange(-10., 10.+step/2, step)
        bounds = edges(x)
        widths = np.diff(bounds)
        source = np.diff(ndtr(bounds/.3))/widths
        expected = np.diff(ndtr(bounds/np.hypot(.3, .8)))/widths
        response = InstrumentResponse.from_fixed_sigma(x, .8)
        errors.append(np.max(np.abs(response(source)-expected)))
    assert errors[1] < errors[0]/3.5
    assert errors[2] < errors[1]/3.5


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('gaps', [False, True])
def test_window_keeps_all_available_source_contributions(gaps):
    native = np.linspace(-10., 10., 401)
    response = InstrumentResponse.from_fixed_sigma(native, .6)
    indices = np.arange(190,221)
    if gaps:
        indices = indices[(indices < 201) | (indices > 206)]
    requested = native[indices]
    source = Gaussian1D(2., -.9, .15)
    operator = SpectralResponse(instrument=response, wave=requested)
    assert operator.wavelength_grid.size > requested.size
    expected = response(source(native))[indices]
    np.testing.assert_allclose((source | operator)(requested), expected, atol=2e-15, rtol=1e-13)
    cropped = SpectralResponse(instrument=response, wave=requested, flexible=False)
    assert np.max(np.abs((source | cropped)(requested)-expected)) > .01


@pytest.mark.usefixtures('astropy_extensions')
def test_window_archive_recipe_does_not_grow_on_roundtrip(monkeypatch):
    native = np.linspace(4900., 5100., 501)
    response = InstrumentResponse.from_fixed_sigma(native, 1.)
    monkeypatch.setattr(InstrumentResponse, 'from_instrument', classmethod(lambda cls, name: response))
    requested = native[200:250]/1.1
    model = Gaussian1D(2., requested[0], 1.) | SpectralResponse(instrument='TEST', wave=requested, z=.1)
    original_grid = model.right.wavelength_grid.copy()
    original_flux = model(requested)
    for _ in range(3):
        model = yaml.load(yaml.dump(model))
        np.testing.assert_array_equal(model.right.wavelength_grid, original_grid)
        np.testing.assert_array_equal(model(requested), original_flux)


def test_crop_preserves_bin_widths_across_gaps_and_io(tmp_path):
    x = np.linspace(-5., 5., 101)
    response = InstrumentResponse.from_fixed_sigma(x, .3)
    selected = np.r_[20:35, 60:80]
    cropped = response.crop(x[selected])
    np.testing.assert_array_equal(cropped._bin_widths, np.diff(edges(x))[selected])
    np.testing.assert_array_equal(cropped.response_matrix.toarray(), response.response_matrix.toarray()[np.ix_(selected, selected)])
    target = tmp_path/'cropped.fits'
    cropped.save_fits(target)
    restored = InstrumentResponse.from_fits(target)
    np.testing.assert_array_equal(restored._bin_widths, cropped._bin_widths)
    np.testing.assert_array_equal(restored.response_matrix.toarray(), cropped.response_matrix.toarray())


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf])
def test_invalid_width_rejected(sigma):
    with pytest.raises(ValueError):
        InstrumentResponse.from_fixed_sigma(np.arange(5.), sigma)


def test_invalid_resolution_table_rejected():
    with pytest.raises(ValueError, match='Resolving powers'):
        InstrumentResponse.from_variable_gaussian_resolution(np.arange(1.,6.), [1.,5.], [1000., np.nan])


@pytest.mark.parametrize('backend', ['scipy', 'sherpa'])
def test_physical_response_fit_against_analytic_convolution(backend):
    from prism.modeling.fitting import ScipyTRF, SherpaLM, HAS_SHERPA
    if backend == 'sherpa' and not HAS_SHERPA:
        pytest.skip('Sherpa unavailable')
    native = np.linspace(-8., 8., 401)
    bounds = edges(native)
    widths = np.diff(bounds)
    response = InstrumentResponse.from_fixed_sigma(native, .8)
    indices = np.arange(90, 311, 2)
    indices = indices[(indices < 195) | (indices > 205)]
    observed = native[indices]
    # Exact continuous Gaussian convolution integrated over destination bins.
    total = 1.7*.5*np.sqrt(2*np.pi)
    truth = total*np.diff(ndtr((bounds-.2)/np.hypot(.5,.8)))/widths
    model = Gaussian1D(.7, -.4, 1.4, bounds={'amplitude': (.1, 5.), 'stddev': (.1, 3.)})
    model = model | SpectralResponse(instrument=response, wave=observed)
    fitter = ScipyTRF(calc_uncertainties=True) if backend == 'scipy' else SherpaLM(calc_uncertainties=True)
    fitted = fitter(model, observed, truth[indices], yerr=np.full(len(observed), .02))
    assert fitter.fit_info['success']
    np.testing.assert_allclose(fitted.left.parameters, [1.7, .2, .5], atol=2e-5)
    assert fitter.fit_info['param_cov'].shape == (3, 3)


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('shift', [-.02, .02])
def test_offset_sampling_keeps_bracketing_rows_even_for_identity(shift):
    native = np.linspace(-5., 5., 201)
    response = InstrumentResponse(native, np.eye(len(native)))
    requested = native[90:110] + shift
    source = Gaussian1D(2., .2, .3)
    operator = SpectralResponse(instrument=response, wave=requested)
    expected = np.interp(requested, native, source(native))
    np.testing.assert_allclose((source | operator)(requested), expected, atol=1e-14)
