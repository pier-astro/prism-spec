"""Check real optimizer configuration and conservative termination reporting."""
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from prism.modeling.fitting import SherpaLM, HAS_SHERPA
from prism.modeling.fitting.extension import set_max_evaluations, get_max_evaluations

pytestmark = pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable')


def problem():
    x = np.linspace(-5, 5, 400)
    y = 3 * np.exp(-.5 * ((x - .4) / .7)**2)
    model = Gaussian1D(1, 0, 1.2, bounds={
        'amplitude': (0, 10), 'mean': (-1, 1), 'stddev': (.1, 3)})
    return model, x, y


@pytest.mark.parametrize('alias', ['maxfev', 'max_nfev', 'max_evaluations', 'maxiter'])
def test_budget_reaches_native_optimizer(alias):
    model, x, y = problem()
    fitter = SherpaLM(maxfev=1000)
    fitter(model, x, y, yerr=np.full_like(x, .1), **{alias: 2})
    assert fitter.fit_info['nfev'] < 20
    assert not fitter.fit_info['success']
    assert 'budget' in fitter.fit_info['message']
    assert fitter.fit_info['param_cov'] is None


def test_global_budget_and_constructor_override():
    model, x, y = problem()
    previous = get_max_evaluations()
    try:
        set_max_evaluations(2)
        limited = SherpaLM()
        limited(model.copy(), x, y)
        assert not limited.fit_info['success']
        fitter = SherpaLM(maxfev=1000)
        result = fitter(model, x, y)
        assert fitter.fit_info['success']
        np.testing.assert_allclose(result.parameters, [3, .4, .7], atol=1e-7)
    finally:
        set_max_evaluations(previous)


def test_options_match_native_and_do_not_leak_between_calls():
    from sherpa.optmethods import LevMar
    model, x, y = problem()
    fitter = SherpaLM(epsfcn=1e-12)
    original = fitter.opt.config.copy()
    fitted = fitter(model, x, y, ftol=1e-10, inplace=False)
    def objective(p):
        residual = p[0] * np.exp(-.5 * ((x-p[1])/p[2])**2) - y
        return np.sum(residual**2), residual
    native = LevMar()
    native.config.update(epsfcn=1e-12, ftol=1e-10)
    expected = native.fit(objective, [1, 0, 1.2], [0, -1, .1], [10, 1, 3])
    np.testing.assert_allclose(fitted.parameters, expected[1], atol=1e-10, rtol=0)
    assert fitter.opt.config == original
    fitter(model, x, y, maxfev=2)
    assert not fitter.fit_info['success']
    assert fitter.opt.config == original


def test_invalid_options_and_conflicting_budgets():
    model, x, y = problem()
    with pytest.raises(TypeError, match='Unsupported Sherpa options'):
        SherpaLM()(model, x, y, typo=1)
    with pytest.raises(ValueError, match='Conflicting'):
        SherpaLM()(model, x, y, maxfev=2, max_nfev=3)
