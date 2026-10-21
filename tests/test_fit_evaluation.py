"""Independent checks of source evaluation followed by a fixed response."""

from line_helpers import make_group
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from scipy.sparse import csr_matrix

from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.fitting.evaluation import _ModelEvaluation
from prism.modeling.operators.instrument import SpectralResponse


def setup_response():
    native = np.linspace(-8, 8, 161)
    matrix = np.exp(-.5 * ((native[:, None] - native) / .35)**2)
    matrix /= matrix.sum(axis=1)[:, None]
    response = SpectralResponse(response_matrix=csr_matrix(matrix), wave=native)
    return native, matrix, response


def test_prepared_response_jacobian_and_sampling():
    native, matrix, response = setup_response()
    observed = np.r_[-9., native[5:-5:3] + .012, 9.]
    model = Gaussian1D(2., .3, 1.2) | response
    original = model.parameters.copy()
    evaluation = _ModelEvaluation(model, observed)
    assert evaluation.stateless
    parameters = np.array([3., -.2, .7])
    intrinsic = parameters[0] * np.exp(-.5 * ((native-parameters[1])/parameters[2])**2)
    expected = np.interp(observed, native, matrix @ intrinsic, left=0, right=0)
    np.testing.assert_allclose(evaluation(parameters), expected, atol=1e-14)
    analytic = evaluation.fit_deriv(parameters)
    for index in range(3):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        finite = (evaluation(plus) - evaluation(minus)) / 2e-5
        np.testing.assert_allclose(analytic[index], finite, rtol=2e-7, atol=1e-9)
    np.testing.assert_array_equal(model.parameters, original)


@pytest.mark.parametrize('fitter_class', [TRF, pytest.param(
    SherpaLM, marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_response_known_input_recovery(fitter_class):
    native, matrix, response = setup_response()
    observed = native[7:-7:2] + .013
    truth = np.array([3., -.3, .8])
    flux = truth[0] * np.exp(-.5 * ((native-truth[1])/truth[2])**2)
    y = np.interp(observed, native, matrix @ flux)
    model = Gaussian1D(1.4, .4, 1.5, bounds={'amplitude': (.1, 8), 'stddev': (.2, 4)}) | response
    before = model.parameters.copy()
    fitter = fitter_class(covariance=True)
    result = fitter(model, observed, y, yerr=np.full(len(y), .03), inplace=False)
    np.testing.assert_allclose(result.parameters, truth, atol=2e-5)
    np.testing.assert_array_equal(model.parameters, before)
    assert fitter.fit_info['success']
    assert fitter.fit_info['param_cov'].shape == (3, 3)


def test_response_strict_grid_and_custom_operator_preserved():
    native, matrix, response = setup_response()
    response.flexible = False
    with pytest.raises(ValueError, match='flexible=False'):
        _ModelEvaluation(Gaussian1D() | response, native[::2])

    class CustomResponse(SpectralResponse):
        pass

    custom = CustomResponse(response_matrix=csr_matrix(matrix), wave=native)
    model = Gaussian1D() | custom
    evaluation = _ModelEvaluation(model, native)
    assert not evaluation.stateless
    np.testing.assert_array_equal(evaluation(model.parameters), model(native))


def test_response_velocity_units_are_not_discarded():
    import astropy.units as u

    native = np.linspace(4980., 5020., 161)
    response = SpectralResponse(response_matrix=csr_matrix(np.eye(len(native))), wave=native)
    group = make_group(
        names=['line'], pos=[5000.]*u.AA, amplitude=2.,
        offset=30.*u.km/u.s, fwhm=300.*u.km/u.s)
    model = group | response
    evaluation = _ModelEvaluation(model, native)
    assert evaluation.stateless
    parameters = model.parameters.copy()
    parameters[0] *= 1.3
    model.parameters = parameters
    np.testing.assert_allclose(evaluation(parameters), model(native), rtol=1e-14)


def test_response_with_frozen_and_linked_parameters():
    native, matrix, response = setup_response()
    observed = native[4:-4:3] + .01
    truth = 2*np.exp(-.5*((native+1)/.7)**2) + 4*np.exp(-.5*((native-1)/.7)**2)
    y = np.interp(observed, native, matrix @ truth)
    source = Gaussian1D(1., -1., 1.2) + Gaussian1D(2., 1., 1.2)
    source.mean_0.fixed = source.mean_1.fixed = True
    source.amplitude_1.tied = lambda model: 2*model.amplitude_0
    source.stddev_1.tied = lambda model: model.stddev_0
    # Equality inherits the positive driver domain; remove the redundant target
    # bound explicitly because an opaque Astropy callable cannot be compiled.
    source.stddev_1.bounds = (None, None)
    fitter = TRF(covariance=True)
    result = fitter(source | response, observed, y, yerr=np.full(len(y), .03))
    np.testing.assert_allclose(result.parameters, [2., -1., .7, 4., 1., .7], atol=1e-7)
    assert fitter.fit_info['param_cov'].shape == (2, 2)
