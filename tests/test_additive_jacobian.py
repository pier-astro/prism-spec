"""Independent derivatives for prepared additive sources and explicit fallbacks."""
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Polynomial1D
from scipy.sparse import csr_matrix
from prism.modeling.fitting.evaluation import _ModelEvaluation
from prism.modeling.fitting import TRF
from prism.modeling.operators.instrument import SpectralResponse


@pytest.mark.parametrize('operation', ['sum', 'difference', 'product', 'quotient'])
def test_compound_derivatives_against_finite_differences(operation):
    x = np.linspace(-5, 5, 151)
    a = Gaussian1D(2., -.7, .8)
    b = Gaussian1D(1.5, .9, 1.2)
    if operation == 'sum':
        source = a + (b + Polynomial1D(2, c0=.3, c1=.01, c2=.001))
    elif operation == 'difference':
        source = a - (b - Polynomial1D(2, c0=.3, c1=.01, c2=.001))
    elif operation == 'product':
        source = a * (b + Polynomial1D(2, c0=.3, c1=.01, c2=.001))
    else:
        source = a / (b + Polynomial1D(2, c0=.3, c1=.01, c2=.001))
    # Nontrivial response, offset observed samples, and a masked gap.
    matrix = np.exp(-.5*((x[:, None]-x)/.25)**2)
    matrix /= matrix.sum(axis=1)[:, None]
    response = SpectralResponse(response_matrix=csr_matrix(matrix), wave=x)
    observed = np.r_[x[5:60:2], x[70:-5:2]] + .007
    model = source | response
    evaluation = _ModelEvaluation(model, observed)
    assert (evaluation.derivatives is not None) == (operation in ('sum', 'difference'))
    p = model.parameters.copy()
    jac = evaluation.fit_deriv(p)
    for i in range(len(p)):
        step = 1e-6
        plus, minus = p.copy(), p.copy()
        plus[i] += step
        minus[i] -= step
        numeric = (evaluation(plus)-evaluation(minus))/(2*step)
        np.testing.assert_allclose(jac[i], numeric, rtol=3e-6, atol=1e-8)
    np.testing.assert_array_equal(model.parameters, p)


def test_additive_jacobian_reads_new_parameter_values_after_preparation():
    x = np.linspace(-5, 5, 101)
    response = SpectralResponse(response_matrix=csr_matrix(np.eye(len(x))), wave=x)
    model = (Gaussian1D(1., -1., .8) + Gaussian1D(1., 1., .8)) | response
    evaluation = _ModelEvaluation(model, x)
    initial = model.parameters.copy()
    changed = np.array([3., -.3, .6, 2., .9, 1.1])
    before = evaluation.fit_deriv(initial)
    after = evaluation.fit_deriv(changed)
    assert not np.allclose(before, after)
    np.testing.assert_array_equal(evaluation.fit_deriv(initial), before)
    # Known-input recovery from a distant constructor, not a saved solution.
    truth = changed[0]*np.exp(-.5*((x-changed[1])/changed[2])**2)
    truth += changed[3]*np.exp(-.5*((x-changed[4])/changed[5])**2)
    fit = TRF()(model, x, truth, inplace=False)
    # The two Gaussian labels are interchangeable; compare ordered components.
    recovered = fit.parameters.reshape(2, 3)
    expected = changed.reshape(2, 3)
    np.testing.assert_allclose(recovered[np.argsort(recovered[:, 1])],
                               expected[np.argsort(expected[:, 1])], atol=1e-6)
    np.testing.assert_allclose(fit.predict(x), truth, atol=1e-8)
