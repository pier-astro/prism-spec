"""Optimizer controls must reach SciPy without changing subsequent fits."""
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from prism.modeling import Model
from prism.modeling.fitting import TRF
from prism.modeling.fitting.extension import get_max_evaluations, set_max_evaluations


def problem(native=False):
    x = np.linspace(-5, 5, 201)
    y = 3*np.exp(-.5*((x-.4)/.7)**2)
    model = Gaussian1D(1, 0, 1.2)
    return Model.from_astropy(model) if native else model, x, y


@pytest.mark.parametrize('alias', ['max_evaluations', 'max_nfev', 'maxiter', 'maxfev', 'maxfun'])
@pytest.mark.parametrize('constructor', [False, True])
def test_budget_reaches_optimizer(alias, constructor):
    model, x, y = problem()
    options = {alias: 1}
    fitter = TRF(**options) if constructor else TRF()
    fitter(model, x, y, **({} if constructor else options))
    assert fitter.fit_info['nfev'] == 1
    assert not fitter.fit_info['success']


def test_budget_precedence_and_no_call_leak():
    model, x, y = problem()
    previous = get_max_evaluations()
    try:
        fitter = TRF()
        set_max_evaluations(1)
        fitter(model, x, y)
        assert not fitter.fit_info['success']
        fitter = TRF(max_nfev=200)
        fitter(model, x, y, maxiter=1)
        assert not fitter.fit_info['success']
        fitted = fitter(model, x, y)
        assert fitter.fit_info['success']
        np.testing.assert_allclose(fitted.parameters, [3, .4, .7], atol=1e-7)
    finally:
        set_max_evaluations(previous)


@pytest.mark.parametrize('native', [False, True])
def test_copy_and_inplace(native):
    model, x, y = problem(native)
    initial = model.parameters.copy()
    fitter = TRF()
    fitted = fitter(model, x, y)
    assert fitted is not model
    np.testing.assert_array_equal(model.parameters, initial)
    fitted = fitter(model, x, y, inplace=True)
    assert fitted.model is not model
    np.testing.assert_allclose(model.parameters, [3, .4, .7], atol=1e-7)


@pytest.mark.parametrize('constructor', [False, True])
def test_conflicting_budgets(constructor):
    model, x, y = problem()
    with pytest.raises(ValueError, match='Conflicting'):
        if constructor:
            TRF(max_nfev=2, maxiter=3)
        else:
            TRF()(model, x, y, max_nfev=2, maxiter=3)
