"""Parameter ordering and ordered callable-link checks independent of Astropy internals."""

import numpy as np
import pytest
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Gaussian1D

from prism.modeling.fitting import ScipyTRF, SherpaLM, HAS_SHERPA, apply_tied
from prism.modeling.fitting.utils import _ParameterMap, _reduce_tied_analytic_jacobian


class VectorModel(Fittable1DModel):
    standard_broadcasting = False
    coefficients = Parameter(default=[1., 2.], bounds=(0., 8.))
    bias = Parameter(default=0.)

    @staticmethod
    def evaluate(x, coefficients, bias):
        coefficients = np.asarray(coefficients).ravel()
        return coefficients[0] + coefficients[1]*x + bias


def test_parameter_map_has_one_flattened_order():
    model = VectorModel(bias=3., fixed={'bias': True})
    parameters = _ParameterMap(model)
    np.testing.assert_array_equal(parameters.initial, [1., 2., 3.])
    np.testing.assert_array_equal(parameters.free, [0, 1])
    np.testing.assert_array_equal(parameters.values, [1., 2.])
    np.testing.assert_array_equal(parameters.bounds, [[0, 8], [0, 8]])
    assert parameters.slices == {'coefficients': slice(0, 2), 'bias': slice(2, 3)}
    model.coefficients.fixed = True
    model.bias.fixed = False
    parameters = _ParameterMap(model)
    np.testing.assert_array_equal(parameters.free, [2])


@pytest.mark.parametrize('fitter_class', [ScipyTRF, pytest.param(
    SherpaLM, marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_array_parameter_fit_and_covariance_order(fitter_class):
    x = np.linspace(-1, 1, 51)
    y = 3. + 4.*x + 2.
    model = VectorModel(bias=2., fixed={'bias': True})
    fitter = fitter_class(calc_uncertainties=True)
    result = fitter(model, x, y, yerr=np.full(x.size, .1), inplace=False)
    np.testing.assert_allclose(result.parameters, [3., 4., 2.], atol=1e-6)
    np.testing.assert_array_equal(model.parameters, [1., 2., 2.])
    assert fitter.fit_info['nfree'] == 2
    assert fitter.fit_info['dof'] == 49
    assert fitter.fit_info['param_cov'].shape == (2, 2)
    np.testing.assert_allclose(result.coefficients.std, np.sqrt(np.diag(fitter.fit_info['param_cov'])))


def chained_model():
    model = Gaussian1D(1.3, 0., 1.) + Gaussian1D(.1, 0., 1.) + Gaussian1D(.1, 0., 1.)
    for name in model.param_names:
        if not name.startswith('amplitude'):
            getattr(model, name).fixed = True
    model.amplitude_1.tied = lambda m: 2*m.amplitude_0
    model.amplitude_2.tied = lambda m: m.amplitude_1**2
    model.amplitude_0.bounds = (.1, 5.)
    return model


def test_chained_links_and_chain_rule_use_updated_values():
    model = chained_model()
    apply_tied(model)
    assert model.amplitude_1.value == 2.6
    assert model.amplitude_2.value == 2.6**2
    x = np.linspace(-4, 4, 81)
    original = model.parameters.copy()
    jacobian = _reduce_tied_analytic_jacobian(model, model.fit_deriv(x, *model.parameters))
    expected = (3 + 8*1.3)*np.exp(-.5*x**2)
    np.testing.assert_allclose(jacobian[0], expected, rtol=2e-7, atol=1e-8)
    np.testing.assert_array_equal(model.parameters, original)


@pytest.mark.parametrize('jac', ['auto', '2-point'])
def test_chained_link_fit_recovers_known_input(jac):
    x = np.linspace(-4, 4, 81)
    amplitude = 2.1
    y = (3*amplitude + 4*amplitude**2)*np.exp(-.5*x**2)
    fitter = ScipyTRF(jac=jac, calc_uncertainties=True)
    if jac == 'auto':
        with pytest.warns(UserWarning, match='tied parameters together'):
            result = fitter(chained_model(), x, y)
    else:
        result = fitter(chained_model(), x, y)
    np.testing.assert_allclose(result.parameters[[0, 3, 6]],
                               [amplitude, 2*amplitude, 4*amplitude**2], atol=1e-7)
    assert fitter.fit_info['param_cov'].shape == (1, 1)


def test_scalar_callable_link_broadcasts_to_array_parameter():
    model = VectorModel(bias=3.)
    model.coefficients.tied = lambda m: m.bias
    apply_tied(model)
    np.testing.assert_array_equal(model.parameters, [3., 3., 3.])
