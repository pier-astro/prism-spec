"""Refitting must replace uncertainty estimates rather than inherit them."""
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Linear1D
from prism.modeling import Model
from prism.modeling.fitting import (
    ScipyTRF, LMFitter, SeparableTRF, SherpaLM, HAS_SHERPA,
    TRFLSQFitter, LinearLSQFitter,
)
from prism.modeling.fitting.extension import _attach_covariance
from prism.modeling.fitting.multifit import MultiFitResult

ENGINES = [ScipyTRF, LMFitter, SeparableTRF, TRFLSQFitter,
           pytest.param(SherpaLM, marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))]


def seed(model):
    model._param_cov = np.eye(3)*99
    model._fit_indices = np.arange(3)
    for name in model.param_names:
        parameter = getattr(model, name)
        parameter.std = 9.
        parameter.median = 4.
        parameter.samples = np.array([1., 2.])
    return model


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('engine', ENGINES)
def test_refit_replaces_estimates_and_preserves_input(engine):
    x = np.linspace(-3, 3, 101)
    model = seed(Gaussian1D(1., 0., .8, fixed={'mean': True, 'stddev': True}))
    fitter = engine(calc_uncertainties=False)
    options = {'linear_params': ['amplitude']} if engine is SeparableTRF else {}
    fitted = fitter(model, x, 3*np.exp(-.5*(x/.8)**2), inplace=False, **options)
    assert fitted.amplitude.value == pytest.approx(3., rel=1e-6)
    current = fitter.fit_info.get('param_cov')
    if current is None:
        assert fitted.covariance is None
        assert not hasattr(fitted, '_fit_indices')
        assert fitted.amplitude.std is None
    else:
        np.testing.assert_array_equal(fitted.covariance, current)
        assert current.shape == (1, 1)
        np.testing.assert_array_equal(fitted._fit_indices, [0])
    assert fitted.mean.std is None
    assert fitted.stddev.std is None
    for name in fitted.param_names:
        for attribute in ('median', 'samples'):
            assert getattr(getattr(fitted, name), attribute, None) is None
    assert model.amplitude.std == 9.
    np.testing.assert_array_equal(model._param_cov, np.eye(3)*99)


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('native', [False, True])
def test_new_covariance_uses_current_free_order_and_analytic_variance(native):
    x = np.linspace(-3, 3, 101)
    profile = np.exp(-.5*(x/.8)**2)
    model = seed(Gaussian1D(1., 0., .8, fixed={'mean': True, 'stddev': True}))
    if native:
        model = Model.from_astropy(model)
        seed(model)
    fitted = ScipyTRF(calc_uncertainties=True)(model, x, 3*profile, yerr=np.full_like(x, .2), inplace=True)
    assert fitted is model
    variance = .2**2/np.dot(profile, profile)
    np.testing.assert_allclose(fitted.covariance, [[variance]], rtol=1e-12)
    np.testing.assert_array_equal(fitted._fit_indices, [0])
    assert fitted.amplitude.std == pytest.approx(np.sqrt(variance))
    assert fitted.mean.std is None
    assert fitted.stddev.std is None
    assert getattr(fitted.amplitude, 'samples', None) is None


def test_invalid_attachment_does_not_destroy_current_estimates():
    model = seed(Gaussian1D())
    with pytest.raises(ValueError, match='shape'):
        _attach_covariance(model, np.eye(2))
    assert model.amplitude.std == 9.


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('engine', [LMFitter, SeparableTRF])
def test_fixed_model_path_clears_prior_estimates(engine):
    model = seed(Gaussian1D(fixed={'amplitude': True, 'mean': True, 'stddev': True}))
    x = np.linspace(-3, 3, 21)
    fitted = engine()(model, x, np.zeros_like(x))
    assert fitted.covariance is None
    assert getattr(fitted.amplitude, 'std', None) is None


@pytest.mark.usefixtures('astropy_extensions')
def test_linear_adapter_covariance_has_order_and_replaces_prior_stds():
    x = np.linspace(-2, 2, 51)
    model = Linear1D(0., 0., fixed={'intercept': True})
    model.intercept.std = 10.
    fitted = LinearLSQFitter()(model, x, 2*x+.01*np.cos(x))
    assert fitted.covariance.shape == (1, 1)
    np.testing.assert_array_equal(fitted._fit_indices, [0])
    assert fitted.intercept.std is None


@pytest.mark.usefixtures('astropy_extensions')
def test_cube_reconstruction_does_not_inherit_template_covariance_or_samples():
    model = seed(Gaussian1D())
    result = MultiFitResult((1,), model.param_names, template_model=model, has_cov=False)
    for name, value in zip(model.param_names, [3., .1, .9]):
        result._params[name][0] = value
    fitted = result.get_model(0)
    assert fitted.covariance is None
    assert getattr(fitted.amplitude, 'samples', None) is None
    assert model.amplitude.std == 9.
