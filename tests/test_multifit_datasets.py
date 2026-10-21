"""Independent named observations retain grids, responses and local covariance."""
import numpy as np
import pytest
from astropy.modeling import Parameter
from scipy.sparse import eye
from numpy.testing import assert_allclose, assert_array_equal
from prism.modeling import Component, LinearOperatorModel, select_line
from prism.modeling.fitting import Dataset, FitResult, TRF, SherpaLM, HAS_SHERPA

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def trend(x, intercept, slope):
    return intercept+slope*x


def derivative(x, intercept, slope):
    return np.array([np.ones_like(x), x])


def source():
    return Component(trend, dict(intercept=.4, slope=.05), derivative=derivative, name='trend')


def observations():
    data = {}
    for label, size, gain, intercept, slope in [('blue', 41, 1.2, 3., .7), ('red', 57, .8, 2., -.2)]:
        x = np.linspace(-2., 3., size)
        mask = np.ones(size, dtype=bool)
        mask[7] = False
        weights = np.ones(size)/.1
        weights[4] = 0.
        y = gain*(intercept+slope*x)+.02*np.sin(3*x)
        y[4] = np.nan
        data[label] = Dataset(x, y, yerr=np.full(size, .5), weights=weights, mask=mask,
            response=LinearOperatorModel(gain*eye(size), x=x))
    return data


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('nproc', [1, 2])
def test_named_independent_fits_match_weighted_linear_reference(engine, nproc, tmp_path):
    model, data = source(), observations()
    fitter = engine(covariance=True)
    result = fitter.multifit(model, data, nproc=nproc, progress=False,
        initpars={'intercept': {'red': .8, 'blue': .6}},
        bounds={'intercept': ({'blue': 0., 'red': .1}, 6.)})
    assert isinstance(result, dict)
    assert tuple(result) == ('blue', 'red')
    assert fitter.fit_info['n_success'] == 2
    assert_array_equal(model.parameters, [.4, .05])
    for label, fitted in result.items():
        assert isinstance(fitted, FitResult)
        original = data[label]
        valid = original.mask & (original.weights != 0)
        gain = 1.2 if label == 'blue' else .8
        design = gain*np.column_stack([np.ones(original.x.size), original.x])
        weighted = design[valid]*original.weights[valid, None]
        expected = np.linalg.lstsq(weighted, original.y[valid]*original.weights[valid], rcond=None)[0]
        assert_allclose(fitted.parameters, expected, atol=1e-7)
        assert_allclose(fitted.covariance, np.linalg.inv(weighted.T@weighted), rtol=1e-6, atol=1e-14)
        assert_array_equal(fitted.dataset.mask, valid)
        assert fitted.fit_info['ndata'] == valid.sum()
        assert fitted.fit_info['nfree'] == 2
        assert fitted['data'].params['intercept'] is fitted.params['intercept_0']
        assert not fitted.parameters.flags.writeable
        assert not fitted.covariance.flags.writeable
        assert not fitted.fit_indices.flags.writeable
        with pytest.raises(TypeError):
            fitted.params['intercept_0'] = 1.
        assert fitted.model.intercept.bounds[0] == (0. if label == 'blue' else .1)
        assert fitted.predict().shape == original.x.shape
        assert fitted.predict(np.linspace(-1., 1., 15)).shape == (15,)
        path = tmp_path/(label+'.fits')
        fitted.save(path)
        restored = FitResult.load(path, components={'trend': model})
        assert_allclose(restored.predict(), fitted.predict(), rtol=1e-12)
        copied = fitted.dataset
        copied.y[:] = -99.
        assert_allclose(fitted.predict(), gain*trend(original.x, *expected), rtol=1e-7)
    assert not np.allclose(result['blue'].parameters, result['red'].parameters)


class Gain(LinearOperatorModel):
    gain = Parameter(default=.7, bounds=(.1, 3.))
    def get_matrix(self, x, gain):
        return gain*self.matrix


@pytest.mark.parametrize('nproc', [1, 2])
def test_source_response_links_and_reused_response_remain_local(nproc):
    x = np.linspace(-2., 2., 61)
    response = Gain(matrix=eye(x.size), x=x)
    model = source()
    model.slope.link = response.gain
    data = {label: Dataset(x, gain*(intercept+gain*x), yerr=np.full(x.size, .1), response=response)
            for label, gain, intercept in [('early', .8, 2.), ('late', 1.3, 3.)]}
    result = TRF(covariance=True).multifit(model, data, nproc=nproc, progress=False)
    for label, gain, intercept in [('early', .8, 2.), ('late', 1.3, 3.)]:
        fitted = result[label]
        assert fitted.fit_info['success']
        assert fitted.fit_info['nfree'] == 2
        assert fitted.dataset.response.gain.value == pytest.approx(gain, abs=1e-6)
        assert fitted.model.slope.value == pytest.approx(gain, abs=1e-6)
        assert fitted.model.intercept.value == pytest.approx(intercept, abs=1e-6)
        assert_allclose(fitted.predict(), data[label].y, atol=1e-6)
        source_copy, data_copy = fitted.associations['data']
        data_copy.response.gain.value = 1.5
        assert source_copy.slope.value == pytest.approx(1.5)
    assert response.gain.value == .7
    assert model.intercept.value == .4


@pytest.mark.parametrize('nproc', [1, 2])
@pytest.mark.parametrize('empty', [True, False])
def test_failed_fit_retains_diagnostics_and_retry_recipe(nproc, empty, tmp_path):
    data = observations()
    if empty:
        data['blue'].mask[:] = False
    else:
        data['blue'].y[:] = np.nan
    result = TRF(covariance=True).multifit(source(), data, nproc=nproc, progress=False)
    assert result['red'].fit_info['success']
    failed = result['blue']
    assert not failed.fit_info['success']
    assert failed.fit_info['message']
    assert np.isnan(failed.parameters).all()
    assert np.isnan(failed.params['intercept_0'].value)
    assert failed.covariance is None
    assert failed.dataset.x.size == data['blue'].x.size
    for action in (lambda: failed.model, failed.predict, lambda: failed.save(tmp_path/'failed.fits')):
        with pytest.raises(ValueError, match='No fitted model'):
            action()
    model, retry = failed.associations['data']
    assert_array_equal(model.parameters, [.4, .05])
    assert not (tmp_path/'failed.fits').exists()
    retry.mask = np.ones_like(retry.x, dtype=bool)
    retry.y = retry.predict(model)
    repeated = TRF()(model, retry)
    assert repeated.fit_info['success']


@pytest.mark.parametrize('nproc', [1, 2])
def test_empty_selection_strict_mode_propagates(nproc):
    data = observations()
    data['blue'].mask[:] = False
    with pytest.raises(ValueError, match='No valid spectral samples'):
        TRF().multifit(source(), data, skip_invalid=False, nproc=nproc, progress=False)


@pytest.mark.parametrize('options, error', [
    ({'initpars': {'missing': 1.}}, KeyError),
    ({'initpars': {'intercept': {'blue': 1.}}}, ValueError),
    ({'initpars': {'intercept': {'blue': 1., 'other': 2.}}}, ValueError),
    ({'initpars': {'intercept': [1., 2.]}}, TypeError),
    ({'initpars': {'intercept': {'blue': 8., 'red': 1.}},
      'bounds': {'intercept': (0., 5.)}}, ValueError),
    ({'bounds': {'intercept': ({'blue': 3., 'red': 0.}, 2.)}}, ValueError),
    ({'fixed': {'intercept': [True, False]}}, ValueError),
    ({'mask': np.ones(2, bool)}, TypeError),
    ({'axis': 0}, TypeError),
    ({'nproc': 0}, ValueError),
    ({'inplace': False}, TypeError),
    ({'statistic': 'poisson'}, NotImplementedError),
])
def test_configuration_errors_precede_every_fit(options, error, monkeypatch):
    fitter = TRF()
    def unexpected(*args, **kwargs):
        pytest.fail('An invalid configuration reached a worker')
    monkeypatch.setattr(fitter, '_fit_dataset_direct', unexpected)
    with pytest.raises(error):
        fitter.multifit(source(), observations(), progress=False, **options)


@pytest.mark.parametrize('data', [{}, {'': object()}, {3: object()}, {'blue': object()}])
def test_invalid_dataset_collections(data):
    with pytest.raises((ValueError, TypeError)):
        TRF().multifit(source(), data, progress=False)


def test_external_link_rejected_before_workers(monkeypatch):
    model, outside = source(), source()
    model.slope.link = outside.slope
    fitter = TRF()
    monkeypatch.setattr(fitter, '_fit_dataset_direct', lambda *args, **kwargs: pytest.fail('worker started'))
    with pytest.raises(ValueError, match='outside the assembled model'):
        fitter.multifit(model, observations(), progress=False)


@pytest.mark.parametrize('shape, axis', [((17, 2, 3), 0), ((2, 17, 3), 1), ((2, 3, 17), -1)])
def test_generic_axis_returns_original_layout(shape, axis):
    x = np.linspace(-1., 1., 17)
    model = source()
    model.slope.fixed = True
    intercept = np.arange(6.).reshape(2, 3)+1.
    y = np.moveaxis(intercept[..., None]+.05*x, -1, axis)
    result = TRF().multifit(model, x, y, axis=axis, progress=False)
    assert result.axis == axis % len(shape)
    assert result.evaluate().shape == shape
    assert_allclose(result.evaluate(), y, atol=1e-7)
    assert_allclose(result.intercept.value, intercept, atol=1e-7)


@pytest.mark.parametrize('field', ['weights', 'yerr', 'mask'])
def test_regular_input_shape_errors_precede_fits(field, monkeypatch):
    fitter = TRF()
    monkeypatch.setattr(fitter, '_fit_single_direct', lambda *args, **kwargs: pytest.fail('worker started'))
    with pytest.raises(ValueError, match='must have shape'):
        fitter.multifit(source(), np.arange(11.), np.ones((2, 11)),
            **{field: np.ones((11, 2))}, progress=False)


@pytest.mark.parametrize('shape', [(6,), (2, 3)])
def test_parameter_map_access_shares_storage_and_preserves_shape(shape, monkeypatch):
    x = np.linspace(-1., 1., 17)
    model = source()
    model.slope.fixed = True
    values = np.arange(6.).reshape(shape)+1.
    data = values[..., None]+.05*x
    result = TRF().multifit(model, x, data, bounds={'intercept': (0., 10.)}, progress=False)
    parameter = result.params['intercept']
    assert parameter.value.shape == shape
    assert parameter.std.shape == shape
    assert np.shares_memory(parameter.value, result._params['intercept'])
    assert np.shares_memory(parameter.std, result._errs['intercept'])
    assert_allclose(parameter.value, values, atol=1e-7)
    with pytest.raises(ValueError):
        parameter.value.flat[0] = 99.
    assert result._parameter_metadata['intercept']['bounds'] == (0., 10.)
    assert parameter.min.strides == (0,)*len(shape)
    # Reconstructing one target must not expand bounds over the whole collection.
    monkeypatch.setattr(result, 'get_bounds', lambda *args: pytest.fail('Full map allocated'))
    fitted = result.get_model(0)
    assert fitted.intercept.bounds == (0., 10.)
    assert fitted.intercept.value == pytest.approx(1.)


def test_mapped_bounds_and_declared_links_survive_local_reconstruction():
    x = np.linspace(-2., 2., 21)
    model = source()
    model.slope.link = .1*model.intercept
    amplitude = np.arange(6.).reshape(2, 3)+1.
    lower = np.zeros_like(amplitude)
    upper = amplitude+2.
    result = TRF().multifit(model, x, amplitude[..., None]*(1.+.1*x), progress=False,
        bounds={'intercept': (lower, upper)})
    lower[:] = 9.
    upper[:] = 10.
    assert_allclose(result.params['intercept'].min, 0.)
    assert_allclose(result.params['intercept'].max, amplitude+2.)
    for index in np.ndindex(amplitude.shape):
        fitted = result.get_model(index)
        assert_allclose(fitted(x), amplitude[index]*(1.+.1*x), atol=1e-7)
        fitted.intercept.value = 2.
        assert fitted.slope.value == pytest.approx(.2)
        assert fitted.intercept.bounds == (0., amplitude[index]+2.)
    assert_allclose(result.evaluate(), amplitude[..., None]*(1.+.1*x), atol=1e-7)


def test_regular_failed_targets_keep_nan_predictions():
    x = np.linspace(-1., 1., 17)
    model = source()
    mask = np.ones((2, 17), dtype=bool)
    mask[1] = False
    result = TRF().multifit(model, x, np.ones((2, 17)), mask=mask, progress=False)
    prediction = result.evaluate()
    assert_allclose(prediction[0], 1., atol=1e-7)
    assert np.isnan(prediction[1]).all()
    assert np.isnan(result.params['intercept'].value[1])
    with pytest.raises(ValueError, match='No finite fitted model'):
        result.get_model(1)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('nproc', [1, 2])
def test_named_astropy_sources_recover_the_same_local_least_squares(engine, nproc):
    from astropy.modeling.models import Linear1D
    model = Linear1D(.05, .4)
    data = observations()
    results = engine(covariance=True).multifit(model, data, nproc=nproc, progress=False)
    for label, fitted in results.items():
        observation = data[label]
        gain = 1.2 if label == 'blue' else .8
        valid = observation.mask & (observation.weights != 0)
        design = gain*np.column_stack([observation.x, np.ones_like(observation.x)])
        weighted = design[valid]*observation.weights[valid, None]
        expected = np.linalg.lstsq(weighted, observation.y[valid]*observation.weights[valid], rcond=None)[0]
        assert fitted.fit_info['success']
        assert_allclose(fitted.parameters, expected, atol=1e-7)
        assert_allclose(fitted.covariance, np.linalg.inv(weighted.T@weighted), rtol=1e-6, atol=1e-14)
    assert_array_equal(model.parameters, [.05, .4])
