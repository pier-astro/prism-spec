"""Vertical checks of native function components and mixed Astropy expressions."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import Model as AstropyModel
from astropy.modeling.models import Linear1D, Gaussian1D
from numpy.testing import assert_allclose

from prism.modeling import Component, Model
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.fitting.evaluation import _ModelEvaluation
from prism.modeling.io import load_model
from prism.modeling.operators.matop import LinearOperatorModel
from prism.modeling.lines.profiles import gaussian


def continuum(x, amplitude, slope):
    return amplitude + slope*x


def continuum_deriv(x, amplitude, slope):
    return np.array([np.ones_like(x), x])


def component(name='continuum', analytic=True):
    return Component(continuum, {'amplitude': 1., 'slope': .2},
                     derivative=continuum_deriv if analytic else None, name=name)


@pytest.mark.parametrize('analytic', [True, False])
@pytest.mark.parametrize('engine', [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_known_recovery(engine, analytic):
    x = np.linspace(-2, 2, 101)
    model = component(analytic=analytic)
    assert not isinstance(model.component, AstropyModel)
    model.amplitude.bounds = (0, 10)
    fitter = engine(covariance=True)
    fitted = fitter(model, x, 3+.7*x, yerr=np.full(x.size, .1), inplace=False)
    assert_allclose(fitted.parameters, [3, .7], atol=1e-6)
    assert_allclose(model.parameters, [1, .2])
    assert_allclose(fitted.predict(x), 3+.7*x, atol=1e-6)
    assert fitted.covariance.shape == (2, 2)


def test_mixed_links_copy_and_response(tmp_path):
    x = np.linspace(-3, 3, 101)
    native = component()
    line = Component(gaussian, {'amplitude': 2., 'center': 0., 'sigma': .5}, name='line')
    native.amplitude.link = line.amplitude*2
    source = native + line + Model.from_astropy(Linear1D(slope=0, intercept=0, fixed={'slope': True, 'intercept': True}))
    response = LinearOperatorModel(np.eye(x.size), x=x, xout=x)
    model = source | response
    copied = model.copy()
    copied['line'].amplitude = 3
    assert_allclose(native.amplitude.value, 4)
    assert_allclose(copied['continuum'].amplitude.value, 6)
    assert_allclose(model(x), continuum(x, 4, .2) + line(x))
    filename = tmp_path/'mixed.fits'
    model.save(filename)
    with pytest.raises(ValueError, match='Supply the custom Component'):
        load_model(filename)
    restored = load_model(filename, components={'continuum': component(), 'line': line})
    assert_allclose(restored(x), model(x))
    restored['line'].amplitude = 4
    assert_allclose(restored['continuum'].amplitude.value, 8)


def test_jacobian_and_astropy_baseline():
    x = np.linspace(-2, 2, 51)
    native = component()
    adapted = Model.from_astropy(Linear1D(slope=.2, intercept=1))
    assert_allclose(native(x), adapted(x))
    evaluation = _ModelEvaluation(native, x)
    assert evaluation.stateless
    assert_allclose(evaluation.fit_deriv(native.parameters), [np.ones_like(x), x])
    for i in range(2):
        p = native.parameters
        p[i] += 1e-6
        assert_allclose((evaluation(p)-native(x))/1e-6,
                        evaluation.fit_deriv(native.parameters)[i], atol=1e-8)


def test_native_link_validation():
    a, b = component('a'), component('b')
    a.amplitude.link = b.amplitude
    with pytest.raises(ValueError, match='Cyclic'):
        b.amplitude.link = a.amplitude
    with pytest.raises(ValueError, match='outside'):
        a(np.arange(3.))
    model = a+b
    assert_allclose(model(np.arange(3.)), 2+.4*np.arange(3.))
    a.amplitude.link = None
    a.amplitude = 5
    assert_allclose(a.amplitude.value, 5)


@pytest.mark.parametrize('parameters', [{'slope': 1, 'amplitude': 2}, {'amplitude': [1], 'slope': 2}, {'amplitude': np.nan, 'slope': 2}])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        Component(continuum, parameters)


def test_units_shapes_and_fixed_parameter():
    c = component()
    with pytest.raises(TypeError, match='numeric'):
        c(np.arange(3.)*u.AA)
    with pytest.raises(TypeError):
        c.amplitude = 1*u.Jy
    c.slope.fixed = True
    fitted = TRF()(c, np.arange(5.), 2+.2*np.arange(5.), inplace=False)
    assert_allclose(fitted.parameters, [2, .2])
    bad = Component(lambda x, a: a, {'a': 1})
    with pytest.raises(ValueError, match='one value'):
        bad(np.arange(3.))


def test_multifit_native():
    x = np.linspace(-1, 1, 41)
    data = np.array([2+.3*x, 3-.2*x])
    result = TRF().multifit(component(), x, data, axis=1, progress=False)
    assert_allclose(result.get_model(0)(x), data[0], atol=1e-7)
    assert_allclose(result.get_model(1)(x), data[1], atol=1e-7)


def test_parameter_mapping_preserves_links_during_fit_and_copy():
    first, second = component('first'), component('second')
    second.params['amplitude'].link = 2 * first.params['amplitude']
    second.params['slope'].value = 0
    second.params['slope'].fixed = True
    first.params['amplitude'].bounds = (0, 10)
    model = first + second
    params = model.params
    assert tuple(params) == model.param_names
    assert params['amplitude_0'] is first.amplitude
    assert model['second'].params['amplitude'] is second.amplitude
    first.amplitude = 2
    assert params['amplitude_0'].value == 2
    assert params['amplitude_1'].value == 4
    with pytest.raises(TypeError):
        params['amplitude_0'] = second.amplitude
    with pytest.raises(KeyError):
        params['missing']

    x = np.linspace(-2, 2, 81)
    fitted = TRF()(model, x, 12 + .7*x)
    assert_allclose(fitted.params['amplitude_0'].value, 4, atol=1e-7)
    assert_allclose(fitted.params['amplitude_1'].value, 8, atol=1e-7)
    assert_allclose(fitted.params['slope_0'].value, .7, atol=1e-7)
    assert fitted.params['amplitude_0'] is not params['amplitude_0']
    edited = fitted.model
    edited.params['amplitude_0'].value = 5
    assert edited.params['amplitude_1'].value == 10
    assert fitted.params['amplitude_1'].value == pytest.approx(8)
    assert params['amplitude_1'].value == 4


def test_parameter_mapping_for_lines_and_adapted_components():
    from prism.modeling.lines import GaussianLines

    lines = GaussianLines({'name': ['a', 'b'], 'position': np.array([5000, 5020])*u.AA,
                           'weight': [1, .5]}, name='lines')
    adapted = Model.from_astropy(Linear1D(slope=.2, intercept=1))
    for model in (component(), lines, adapted):
        assert tuple(model.params) == model.param_names
        for name, parameter in model.params.items():
            assert parameter is getattr(model, name)
        parameter = model.params[model.param_names[0]]
        parameter.value += 1
        assert model.parameters[0] == parameter.value


def test_link_to_astropy_parameter_is_preserved_on_copy():
    a = component()
    b = Model.from_astropy(Gaussian1D(2, 0, 1, name='line'))
    a.amplitude.link = b.amplitude
    copied = (a+b).copy()
    copied['line'].amplitude = 4
    assert_allclose(copied['continuum'].amplitude.value, 4)
    assert_allclose(a.amplitude.value, 2)


# Rest wavelengths in Angstrom; velocity parameters are numeric km/s.
positions = np.array([4861.3, 4958.9, 5006.8])
weights = np.array([1.0, 1.0/2.98, 1.0])

def lines(x, Hb, OIII, offset, fwhm):
    # First-order Doppler centers; width is defined at the rest wavelength.
    centers = positions * (1.0 + offset/299792.458)
    sigma = positions * fwhm/299792.458 / np.sqrt(8*np.log(2))
    amplitudes = np.array([Hb, OIII, OIII]) * weights
    total = np.zeros_like(x, dtype=float)
    for amplitude, center, width in zip(amplitudes, centers, sigma):
        total += gaussian(x, amplitude, center, width)
    return total



def test_weighted_group_known_recovery():
    x = np.linspace(4800, 5060, 1600)
    c = Component(lines, {'Hb': 2., 'OIII': 4., 'offset': 0., 'fwhm': 400.}, name='lines')
    c.fwhm.bounds = (50, 1500)
    truth = lines(x, 3, 6, -85, 280)
    fit = TRF()(c, x, truth, inplace=False)
    assert_allclose(fit.parameters, [3, 6, -85, 280], atol=1e-5)
    assert_allclose(lines(np.array([positions[1]]), 0, 2.98, 0, 20), [1.], atol=1e-10)
