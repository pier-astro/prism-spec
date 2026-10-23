"""Single-fit snapshots, parameter mutation and observed-grid prediction."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import Parameter
from astropy.modeling.models import Linear1D
from scipy.sparse import csr_matrix, eye

from prism.modeling import Component, Model, LinearOperatorModel
from prism.modeling.fitting import Dataset, FitResult, TRF, SherpaLM, HAS_SHERPA
from prism.modeling.operators import SpectralResponse

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def line(x, intercept, slope):
    return intercept + slope*x


def derivative(x, intercept, slope):
    return np.array([np.ones_like(x), x])


def source():
    return Component(line, dict(intercept=1., slope=.1), derivative=derivative)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('native', [True, False])
@pytest.mark.parametrize('inplace', [False, True])
def test_values_only_mutation_and_independent_snapshot(engine, native, inplace):
    model = source() if native else Linear1D(.1, 1.)
    initial = model.parameters.copy()
    parameter = getattr(model, model.param_names[0])
    parameter.std = 99.
    model._param_cov = np.eye(2)*99.
    x = np.linspace(-2., 2., 51)
    y = 3.+.7*x+.02*np.cos(x)
    error = np.full(x.size, .1)
    fitter = engine(covariance=True)
    result = fitter(model, x, y, yerr=error, inplace=inplace)
    assert isinstance(result, FitResult)
    design = np.column_stack([np.ones_like(x), x]) if native else np.column_stack([x, np.ones_like(x)])
    expected = np.linalg.lstsq(design/error[:, None], y/error, rcond=None)[0]
    np.testing.assert_allclose(result.parameters, expected, atol=1e-7)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(design.T@design/.1**2), rtol=2e-6, atol=1e-14)
    assert getattr(model, model.param_names[0]) is parameter
    if inplace:
        np.testing.assert_allclose(model.parameters, expected, atol=1e-7)
        assert getattr(parameter, 'std', None) is None
        assert not hasattr(model, '_param_cov')
    else:
        np.testing.assert_array_equal(model.parameters, initial)
        assert parameter.std == 99.
    for name in result.param_names:
        assert result.params[name].std > 0.
        assert getattr(getattr(result.model, name), 'std', None) is None
    predicted = result.predict().copy()
    model.parameters = initial
    x[:] = 50.
    y[:] = -100.
    detached = result.model
    detached.parameters = initial
    result.dataset.y[:] = 20.
    np.testing.assert_array_equal(result.predict(), predicted)
    assert result.model is not detached
    with pytest.raises(TypeError):
        result(np.arange(4.))
    with pytest.raises(ValueError):
        result.covariance[0, 0] = 0.


class Gain(LinearOperatorModel):
    gain = Parameter(default=1.)
    def get_matrix(self, x, gain):
        return gain*self.matrix


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('inplace', [False, True])
def test_fitted_response_is_separate_and_updates_only_when_requested(engine, inplace):
    x = np.linspace(-2., 2., 61)
    model = source()
    model.slope.fixed = model.intercept.fixed = True
    response = Gain(matrix=eye(x.size), x=x)
    data = Dataset(x, 3.*model(x), response=response)
    result = engine(covariance=True)(model, data, inplace=inplace)
    np.testing.assert_allclose(result.predict(), data.y, rtol=1e-6)
    np.testing.assert_allclose(result.model(x), model(x), rtol=1e-6)
    assert result.dataset.response.gain.value == pytest.approx(3., rel=1e-6)
    assert response.gain.value == pytest.approx(3. if inplace else 1., rel=1e-6)
    assert getattr(response.gain, 'std', None) is None
    assert result.params['gain_1'].std > 0.
    assert getattr(result.dataset.response.gain, 'std', None) is None


@pytest.mark.parametrize('engine', ENGINES)
def test_predict_alternative_grid_folds_native_source_before_interpolating(engine):
    native = np.linspace(-2., 2., 81)
    # Deliberately non-diagonal: evaluating the source on plotting coordinates
    # before folding would give a different prediction or incompatible shapes.
    matrix = np.diag(np.full(native.size, .7)) + np.diag(np.full(native.size-1, .3), 1)
    response = SpectralResponse(response_matrix=csr_matrix(matrix), wave=native)
    original = native[2:-2:2]+.003
    design = np.column_stack([np.interp(original, native, matrix@np.ones_like(native)),
                              np.interp(original, native, matrix@native)])
    data = Dataset(original, design@np.array([3., .7]), response=response)
    result = engine()(source(), data)
    plot = np.r_[-3., np.linspace(-1.99, 1.99, 120), 3., np.nan]
    expected = np.interp(plot, native, matrix@(3.+.7*native), left=0., right=0.)
    expected[-1] = np.nan
    np.testing.assert_allclose(result.predict(plot), expected, atol=2e-6)
    np.testing.assert_allclose(data.predict(result.model, plot), expected, atol=2e-6)
    # Explicit pipes belong to the source and must not be stripped away.
    piped = engine()(source() | response, original, data.y)
    np.testing.assert_allclose(piped.model(plot[:-1]), expected[:-1], atol=2e-6)
    np.testing.assert_allclose(piped.predict(plot), expected, atol=2e-6)
    joint = engine().jointfit({'blue': (source(), data)})
    np.testing.assert_allclose(joint['blue'].predict(plot), expected, atol=2e-6)
    with pytest.raises(AttributeError, match='label'):
        joint.predict(plot)


def test_prediction_without_response_has_no_window_and_preserves_quantity():
    data = Dataset([0., 1.], [1., 1.], mask=[False, True])
    model = source()
    plot = np.array([-5., 0., 2., 10.])
    np.testing.assert_allclose(data.predict(model, plot), line(plot, 1., .1))
    physical = Dataset(np.arange(2)*u.AA, np.ones(2)*u.Jy)
    values = physical.predict(Linear1D(.1*u.Jy/u.AA, 1.*u.Jy), plot*u.AA)
    assert values.unit == u.Jy
    np.testing.assert_allclose(values.value, line(plot, 1., .1))
    with pytest.raises(ValueError, match='one-dimensional'):
        data.predict(model, np.ones((2, 2)))


def test_strict_response_still_rejects_alternative_grid():
    x = np.linspace(-2., 2., 41)
    response = SpectralResponse(response_matrix=eye(len(x)), wave=x, flexible=False)
    data = Dataset(x, 3.+.7*x, response=response)
    result = TRF()(source(), data)
    with pytest.raises(ValueError, match='grid'):
        result.predict(x+.001)


@pytest.mark.parametrize('engine', ENGINES)
def test_default_preserves_model_and_batch_rejects_inplace(engine):
    x = np.arange(5.)
    model = source()
    initial = model.parameters.copy()
    engine()(model, x, 3.+.7*x)
    np.testing.assert_array_equal(model.parameters, initial)
    with pytest.raises(TypeError, match='inplace'):
        engine().multifit(model, x, np.ones((2, 5)), inplace=False)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('inplace', [False, True])
def test_source_response_links_keep_complete_evaluation_context(engine, inplace):
    x = np.linspace(-1., 1., 51)
    model = source()
    model.intercept.fixed = True
    response = Gain(matrix=eye(x.size), x=x)
    model.slope.link = response.gain
    data = Dataset(x, 3.*(1.+3.*x), response=response)
    result = engine()(model, data, inplace=inplace)
    fitted_source = result.model
    np.testing.assert_allclose(fitted_source(x), 1.+3.*x, atol=1e-6)
    np.testing.assert_allclose(result.predict(), data.y, atol=1e-6)
    assert result.dataset.response.gain.value == pytest.approx(3., rel=1e-6)
    assert response.gain.value == pytest.approx(3. if inplace else 1., rel=1e-6)
    if inplace:
        np.testing.assert_allclose(model(x), 1.+3.*x, atol=1e-6)


def test_line_covariance_sampling_uses_result_without_mutating_snapshot():
    from prism.modeling.lines import GaussianLines
    from prism.modeling import select_line
    x = np.linspace(4990., 5010., 401)
    table = {'name': ['line'], 'position': np.array([5000.])*u.AA}
    model = GaussianLines(table, amplitude=1., fwhm=150., name='narrow', normalization='amplitude')
    model.offset.fixed = model.fwhm.fixed = True
    truth = model.copy()
    truth.amp_line.value = 3.
    result = TRF(covariance=True)(model, x, truth(x), yerr=np.full(x.size, .1))
    prediction = result.predict().copy()
    selection = select_line(result, 'line')
    summary = selection.sample(method='covariance', n_samples=64, random_state=41)
    assert np.isfinite(summary.flux.std) and summary.flux.std > 0.
    selection.source_model.parameters = model.parameters
    np.testing.assert_array_equal(result.predict(), prediction)
    assert getattr(result.model.amp_line, 'std', None) is None


def test_batch_and_x_perturbed_bootstrap_do_not_construct_single_snapshots(monkeypatch):
    from prism.modeling.fitting.uncertainty import bootstrap
    from prism.modeling.fitting import single
    def unexpected(*args, **kwargs):
        raise AssertionError('No public snapshot is needed inside a refitting loop.')
    monkeypatch.setattr(single, 'FitResult', unexpected)
    x = np.linspace(-1., 1., 31)
    model = source()
    y = 3.+.7*x
    result = TRF().multifit(model, x, np.array([y, y+.1]), progress=False)
    assert result.success.all()
    samples = bootstrap(model, TRF(), x=x, y=y, yerr=np.full(x.size, .1),
                        xerr=np.full(x.size, .01), perturb_x=True,
                        n_samples=8, seed=41, nproc=1, )
    assert all(np.all(np.isfinite(values)) for values in samples.samples.values())
