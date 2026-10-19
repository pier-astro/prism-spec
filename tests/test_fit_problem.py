"""Scientific residual/Jacobian contracts of the prepared numerical problem."""
import numpy as np
import pytest
from scipy.sparse import diags
from astropy.modeling.models import Gaussian1D

from prism.modeling import Model, LinearOperatorModel
from prism.modeling.models import GaussianLine
from prism.modeling.fitting.problem import _FitProblem


@pytest.mark.parametrize('native', [False, True])
def test_linked_residual_and_jacobian_with_response_and_selected_samples(native):
    x = np.linspace(4970., 5030., 401)
    selected = x[12:-12:3]
    a = GaussianLine(amplitude=1., position=5000., fwhm=200.)
    b = GaussianLine(amplitude=.4, position=5000., fwhm=300.)
    a.offset.fixed = True
    b.offset.link = a.offset
    b.amplitude.link = .4*a.amplitude
    b.fwhm.link = 1.5*a.fwhm
    matrix = diags([.25, .5, .25], [-1, 0, 1], shape=(len(x), len(x)), format='csr')
    source = Model.from_astropy(a) + b if native else a + b
    model = source | LinearOperatorModel(matrix, x=x)
    weights = np.linspace(.5, 1.5, len(selected))
    data = np.full(len(selected), .1)
    problem = _FitProblem(model, selected, data, weights)
    values = np.array([3., 350.])
    sigma = 350./299792.458*5000./(2*np.sqrt(2*np.log(2)))
    intrinsic = 3.*np.exp(-.5*((x-5000.)/sigma)**2)
    intrinsic += 1.2*np.exp(-.5*((x-5000.)/(1.5*sigma))**2)
    expected = (np.interp(selected, x, matrix@intrinsic)-data)*weights
    np.testing.assert_allclose(problem.residual(values), expected, rtol=2e-12, atol=1e-12)
    jacobian = problem.jacobian(values)
    for i in range(len(values)):
        step = 1e-4*max(1., abs(values[i]))
        high, low = values.copy(), values.copy()
        high[i] += step
        low[i] -= step
        derivative = (problem.residual(high)-problem.residual(low))/(2*step)
        np.testing.assert_allclose(jacobian[:, i], derivative, rtol=1e-6, atol=1e-8)
    finished = problem.finish(values)
    assert finished is model
    assert model.amplitude_0.value == 3.
    assert model.amplitude_1.value == pytest.approx(1.2)
    assert model.fwhm_1.value == 525.


def test_stateless_evaluation_commits_only_at_finish_and_caches_are_independent():
    x = np.linspace(-3., 3., 51)
    model = Model.from_astropy(Gaussian1D(1., 0., .7))
    first = _FitProblem(model, x, np.zeros_like(x))
    second = _FitProblem(model.copy(), x, np.zeros_like(x))
    assert first.evaluation.stateless
    initial = model.parameters.copy()
    values = np.array([2., .3, .9])
    expected = 2.*np.exp(-.5*((x-.3)/.9)**2)
    np.testing.assert_allclose(first.residual(values), expected)
    np.testing.assert_array_equal(model.parameters, initial)
    np.testing.assert_array_equal(second.cache, initial)
    first.finish(values)
    np.testing.assert_array_equal(model.parameters, values)
    np.testing.assert_array_equal(second.model.parameters, initial)
