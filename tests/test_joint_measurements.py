"""Intrinsic line errors use globally correlated drivers across observations."""
import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from scipy.sparse import eye
from numpy.testing import assert_allclose, assert_array_equal

from prism.modeling import Component, select_line, LinearOperatorModel
from prism.modeling.lines import GaussianLines
from prism.modeling.fitting import Dataset, FitResult, TRF, SherpaLM, HAS_SHERPA

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def level(x, amplitude):
    return np.full_like(x, amplitude)


def level_derivative(x, amplitude):
    return np.ones((1, x.size))


def setup(engine):
    table = {'name': ['line'], 'position': np.array([5000.])*u.AA}
    first = GaussianLines(table, amplitude=7., fwhm=150., name='line')
    second = GaussianLines(table, amplitude=1., fwhm=150., name='line')
    for model in (first, second):
        model.offset.fixed = model.fwhm.fixed = True
    continuum = Component(level, dict(amplitude=2.), derivative=level_derivative, name='continuum')
    second.amp_line.link = first.amp_line+continuum.amplitude
    blue_x = np.linspace(4993., 5007., 121)
    red_x = np.linspace(4991., 5009., 181)
    sigma = 5000.*150./c.to_value(u.km/u.s)/(2*np.sqrt(2*np.log(2)))
    blue_profile = np.exp(-.5*((blue_x-5000.)/sigma)**2)
    red_profile = np.exp(-.5*((red_x-5000.)/sigma)**2)
    blue = Dataset(blue_x, 10.*blue_profile, yerr=np.full(blue_x.size, 3.))
    red = Dataset(red_x, 5.+15.*red_profile, yerr=np.ones(red_x.size),
                  response=LinearOperatorModel(eye(red_x.size), x=red_x))
    result = engine(covariance=True).jointfit({'blue': (first, blue), 'red': (continuum+second, red)})
    # Independent weighted design in the two free drivers A and C.
    design = np.vstack([np.column_stack([blue_profile, np.zeros_like(blue_x)])/3.,
                        np.column_stack([red_profile, 1.+red_profile])])
    covariance = np.linalg.inv(design.T@design)
    return result, covariance, sigma, {'continuum': continuum}


@pytest.mark.parametrize('engine', ENGINES)
def test_linked_line_flux_variance_includes_cross_dataset_covariance(engine, tmp_path):
    result, covariance, sigma, definitions = setup(engine)
    assert_allclose(result.covariance, covariance, rtol=4e-5, atol=1e-8)
    assert abs(covariance[0, 1]) > .1*np.sqrt(covariance[0, 0]*covariance[1, 1])
    factor = sigma*np.sqrt(2*np.pi)
    expected_std = factor*np.sqrt(covariance[0, 0]+covariance[1, 1]+2*covariance[0, 1])
    independent_std = factor*np.sqrt(covariance[0, 0]+covariance[1, 1])
    assert abs(expected_std/independent_std-1.) > .1
    prediction = result['red'].predict().copy()
    covariance_snapshot = result.covariance.copy()
    selected = select_line(result['red'], 'line', components='line')
    measured = selected.measure()
    assert measured.flux.value == pytest.approx(15.*factor, rel=1e-7)
    sampled = selected.sample(method='covariance', n_samples=4000, num=1024, random_state=31)
    assert sampled.flux.std == pytest.approx(expected_std, rel=.035)
    assert_array_equal(result['red'].predict(), prediction)
    assert_array_equal(result.covariance, covariance_snapshot)
    # The loaded snapshot carries the same global parameter graph and draws.
    filename = tmp_path/'joint.fits'
    result.save(filename)
    restored = FitResult.load(filename, components=definitions)
    loaded = select_line(restored['red'], 'line', components='line')
    repeated = loaded.sample(method='covariance', n_samples=4000, num=1024, random_state=31)
    assert repeated.flux.std == pytest.approx(sampled.flux.std, rel=1e-12)
    assert_allclose(restored['red'].predict(), prediction)


def test_joint_result_requires_explicit_observation_and_missing_covariance_fails():
    result, _, _, _ = setup(TRF)
    with pytest.raises(TypeError, match='observation'):
        select_line(result, 'line')
    associations = result.associations
    no_covariance = TRF(covariance=False).jointfit(associations)
    selected = select_line(no_covariance['red'], 'line', components='line')
    with pytest.raises(ValueError, match='covariance'):
        selected.sample(method='covariance', n_samples=4)


def test_equivalent_width_honors_continuum_choice_with_global_covariance():
    result, covariance, sigma, _ = setup(TRF)
    selected = select_line(result['red'], 'line', components='line')
    factor = sigma*np.sqrt(2*np.pi)
    settings = dict(method='covariance', n_samples=4000, num=512, random_state=11)
    fixed = selected.sample_eqw(perturb_continuum=False, **settings)
    variable = selected.sample_eqw(perturb_continuum=True, **settings)
    # EW = K(A+C)/C. Holding C fixed gives a different propagation from varying
    # the same C in the line amplitude and continuum with their fitted covariance.
    fixed_gradient = np.array([factor/5., factor/5.])
    variable_gradient = np.array([factor/5., -factor*10./5.**2])
    fixed_std = np.sqrt(fixed_gradient@covariance@fixed_gradient)
    variable_std = np.sqrt(variable_gradient@covariance@variable_gradient)
    assert fixed.std == pytest.approx(fixed_std, rel=.04)
    assert variable.std == pytest.approx(variable_std, rel=.04)
    assert abs(fixed.std/variable.std-1.) > .1
