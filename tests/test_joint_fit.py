"""Independent estimators and source/response separation for joint fits."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import diags, eye
from astropy.modeling import Parameter

from prism.modeling import Component, LinearOperatorModel
from prism.modeling.fitting import Dataset, FitResult, HAS_SHERPA, TRF, SherpaLM
from prism.modeling.fitting.joint import _JointProblem

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def linear(x, intercept, slope):
    return intercept + slope*x


def derivative(x, intercept, slope):
    return np.array([np.ones_like(x), x])


def source(name='continuum', analytic=True):
    return Component(linear, {'intercept': 1., 'slope': .1},
                     derivative=derivative if analytic else None, name=name)


def dataset(x, truth, gain, noise=.02):
    error = .1 + .01*(x-x.min())
    y = gain*linear(x, *truth) + noise*np.sin(2*x)
    valid = np.ones(x.shape, dtype=bool)
    valid[2:4] = False
    y[~valid] = np.nan
    response = LinearOperatorModel(gain*eye(len(x)), x=x, name='response')
    return Dataset(x, y, yerr=error, mask=valid, response=response)


def design(data):
    return data.response.matrix@np.column_stack([np.ones_like(data.x), data.x])


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('analytic', [True, False])
def test_shared_model_different_responses_matches_global_linear_estimator(engine, analytic):
    model = source(analytic=analytic)
    blue = dataset(np.linspace(-3, 1, 51), [3., .7], 1.)
    red = dataset(np.linspace(-1, 4, 73), [3., .7], 2.)
    matrices, observations = [], []
    for data in (blue, red):
        matrices.append(design(data)[data.mask]/data.yerr[data.mask, None])
        observations.append(data.y[data.mask]/data.yerr[data.mask])
    matrix = np.vstack(matrices)
    y = np.concatenate(observations)
    expected = np.linalg.lstsq(matrix, y, rcond=None)[0]
    covariance = np.linalg.inv(matrix.T@matrix)
    fitter = engine(covariance=True)
    result = fitter.jointfit({'blue': (model, blue), 'red': (model, red)})
    assert isinstance(result, FitResult)
    assert_allclose(result.parameters, expected, atol=2e-7)
    assert_allclose(result.covariance, covariance, rtol=3e-5, atol=1e-9)
    assert result.fit_info['nfree'] == 2
    assert result.fit_info['nparam'] == 2
    assert result.fit_info['ndata'] == len(y)
    assert result.fit_info['dof'] == len(y)-2
    assert result.fit_info['stat'] == pytest.approx(np.sum((matrix@expected-y)**2), rel=1e-7)
    assert result['blue'].params['slope'] is result['red'].params['slope']
    assert result['blue'].params['slope'].std == pytest.approx(np.sqrt(covariance[1, 1]), rel=3e-5)
    for label, data in [('blue', blue), ('red', red)]:
        assert_allclose(result[label].model(data.x), linear(data.x, *expected), atol=2e-7)
        assert_allclose(result[label].predict(), design(data)@expected, atol=2e-7)
        assert 'dof' not in result[label].fit_info
    assert result.fit_info['stat'] == pytest.approx(sum(result[k].fit_info['stat'] for k in result))
    assert_allclose(model.parameters, [1., .1])


@pytest.mark.parametrize('engine', ENGINES)
def test_distinct_models_with_cross_dataset_link_and_covariance(engine):
    first, second = source('early'), source('later')
    second.slope.link = first.slope
    early = dataset(np.linspace(-3, 1, 51), [3., .7], 1.)
    later = dataset(np.linspace(-1, 4, 73), [5., .7], 1.5)
    matrices, observations = [], []
    for data, epoch in [(early, 0), (later, 1)]:
        local = design(data)
        matrix = np.zeros((len(data.x), 3))
        matrix[:, 0 if epoch == 0 else 2] = local[:, 0]
        matrix[:, 1] = local[:, 1]
        matrices.append(matrix[data.mask]/data.yerr[data.mask, None])
        observations.append(data.y[data.mask]/data.yerr[data.mask])
    matrix, y = np.vstack(matrices), np.concatenate(observations)
    expected = np.linalg.lstsq(matrix, y, rcond=None)[0]
    result = engine(covariance=True).jointfit({'early': (first, early), 'later': (second, later)})
    assert result.fit_info['nfree'] == 3
    assert result.fit_info['nparam'] == 4
    assert_allclose(result.parameters, [expected[0], expected[1], expected[2], expected[1]], atol=2e-6)
    assert_allclose(result.covariance, np.linalg.inv(matrix.T@matrix), rtol=3e-5, atol=1e-9)
    assert abs(result.covariance[0, 2]) > 0
    assert_allclose(result['later'].model(later.x), linear(later.x, expected[2], expected[1]), atol=2e-6)
    assert_allclose(result['later'].predict(), design(later)@np.array([expected[2], expected[1]]), atol=2e-6)
    assert first.slope.value == .1 and second.slope.value == .1


def test_global_analytic_chain_rule_matches_residual_finite_differences():
    first, second = source(), source()
    second.slope.link = first.slope**2
    first.intercept.bounds = (0., 10.)
    problem = _JointProblem({
        'one': (first, dataset(np.linspace(-3, 1, 31), [3., .4], 1.)),
        'two': (second, dataset(np.linspace(-1, 4, 43), [5., .16], 2.))})
    values = np.array([2.5, .4, 4.5])
    jacobian = problem.jacobian(values)
    for index, value in enumerate(values):
        step = 1e-5*max(1., abs(value))
        high, low = values.copy(), values.copy()
        high[index] += step
        low[index] -= step
        expected = (problem.residual(high)-problem.residual(low))/(2*step)
        assert_allclose(jacobian[:, index], expected, rtol=1e-6, atol=1e-7)


def test_components_can_be_shared_between_distinct_source_expressions():
    common = Component(lambda x, amplitude: amplitude*x, {'amplitude': .1}, name='common')
    first = Component(lambda x, amplitude: np.full_like(x, amplitude), {'amplitude': 1.}, name='early')
    second = Component(lambda x, amplitude: np.full_like(x, amplitude), {'amplitude': 2.}, name='later')
    x = np.linspace(-2, 2, 31)
    early = Dataset(x, 3+.7*x)
    later = Dataset(x, 5+.7*x)
    result = TRF().jointfit({'early': (first+common, early), 'later': (second+common, later)})
    assert result.fit_info['nfree'] == 3
    assert_allclose(result['early'].predict(), early.y, atol=1e-7)
    assert_allclose(result['later'].predict(), later.y, atol=1e-7)
    assert result['early'].params['amplitude_1'] is result['later'].params['amplitude_1']


class Gain(LinearOperatorModel):
    gain = Parameter(default=.7, bounds=(.1, 3.))

    def get_matrix(self, x, gain):
        return self.matrix*float(np.asarray(gain).item())


@pytest.mark.parametrize('engine', ENGINES)
def test_response_parameter_is_fitted_and_retained_per_observation(engine):
    x = np.linspace(-1, 1, 31)
    model = source()
    model.slope.fixed = True
    first = Dataset(x, linear(x, 3., .1), yerr=np.full(x.shape, .1))
    response = Gain(matrix=eye(len(x)), x=x)
    second = Dataset(x, 1.5*linear(x, 3., .1), yerr=np.full(x.shape, .1), response=response)
    result = engine(covariance=True).jointfit({'one': (model, first), 'two': (model, second)})
    assert result.fit_info['nfree'] == 2
    assert result['two'].dataset.response.params['gain'].value == pytest.approx(1.5, rel=1e-6)
    assert_allclose(result['one'].predict(), first.y, atol=1e-6)
    assert_allclose(result['two'].predict(), second.y, atol=1e-6)
    jacobian = np.vstack([np.column_stack([np.ones(len(x)), np.zeros(len(x))]),
                          np.column_stack([np.full(len(x), 1.5), linear(x, 3., .1)])])/.1
    assert_allclose(result.covariance, np.linalg.inv(jacobian.T@jacobian), rtol=2e-4, atol=1e-8)
    assert result.covariance[0, 1] < 0
    assert response.gain.value == .7


@pytest.mark.parametrize('engine', ENGINES)
def test_unlinked_groups_match_separate_fits_without_implicit_name_sharing(engine):
    first, second = source(), source()
    early = dataset(np.linspace(-3, 1, 31), [3., .4], 1.)
    later = dataset(np.linspace(-1, 4, 43), [5., -.2], 2.)
    result = engine().jointfit({'early': (first, early), 'later': (second, later)})
    for label, model, data in [('early', first, early), ('later', second, later)]:
        independent = engine()(model, data, inplace=False)
        assert_allclose(result[label].predict(), independent.predict(data.x), atol=2e-6)
    assert result.fit_info['nfree'] == 4
    assert result['early'].params['slope'] is not result['later'].params['slope']


def test_fitted_snapshot_survives_edits_to_inputs_and_materialized_models():
    model = source()
    data = dataset(np.linspace(-2, 2, 31), [3., .4], 2.)
    result = TRF(covariance=True).jointfit({'obs': (model, data)})
    expected = result['obs'].predict().copy()
    covariance = result.covariance.copy()
    editable = result['obs'].model
    editable.intercept.value = 99.
    assert editable.covariance is None
    assert getattr(editable.intercept, 'std', None) is None
    model.intercept.value = 100.
    data.y[:] = 0.
    data.x[:] = 5.
    result['obs'].dataset.response.matrix.data[:] = 0.
    assert_allclose(result['obs'].predict(), expected)
    assert_allclose(result.covariance, covariance)
    with pytest.raises(ValueError):
        result.covariance[0, 0] = 0.
    with pytest.raises(FrozenInstanceError):
        result['obs'].params['intercept'].value = 0.
    with pytest.raises(TypeError):
        result.fit_info['stat'] = 0.
    with pytest.raises(AttributeError):
        result.params = {}


def test_opaque_ties_keep_their_local_model_context():
    model = source()
    model.slope.tied = lambda model: 2*model.intercept.value
    x = np.linspace(-1, 1, 31)
    data = Dataset(x, 3+6*x)
    result = TRF().jointfit({'obs': (model, data)})
    assert result.fit_info['nfree'] == 1
    assert_allclose(result['obs'].predict(), data.y, atol=1e-7)


@pytest.mark.parametrize('engine', ENGINES)
def test_line_profile_recovery_with_distinct_native_response_grids(engine):
    from astropy import units as u
    from prism.modeling.lines import GaussianLines
    table = {'name': ['L'], 'position': [5000.]*u.AA}
    truth = GaussianLines(table, amplitude=3., fwhm=280., offset=45.)
    start = GaussianLines(table, amplitude=1., fwhm=500., offset=0.,
                          bounds={'fwhm': (50., 1000.), 'offset': (-200., 200.)})
    associations = {}
    for label, size, coefficients in [('blue', 201, [.2, .6, .2]), ('red', 301, [.3, .4, .3])]:
        native = np.linspace(4975., 5025., size)
        matrix = diags(coefficients, [-1, 0, 1], shape=(size, size), format='csr')
        response = LinearOperatorModel(matrix, x=native)
        x = native[2:-2:2]
        expected = np.interp(x, native, matrix@truth(native))
        associations[label] = (start, Dataset(x, expected, yerr=np.full(x.shape, .02), response=response))
    result = engine(covariance=True).jointfit(associations)
    assert result.fit_info['nfree'] == 3
    for label, (_, data) in associations.items():
        assert_allclose(result[label].predict(), data.y, atol=2e-6)
        assert result[label].params['fwhm'].value == pytest.approx(280., rel=1e-5)
        assert result[label].params['offset'].value == pytest.approx(45., rel=1e-5)


def test_external_link_invalid_association_and_duplicate_data_fail_before_fitting():
    first, external = source(), source()
    first.slope.link = external.slope
    data = Dataset(np.arange(4.), np.ones(4))
    with pytest.raises(ValueError, match='outside'):
        TRF().jointfit({'obs': (first, data)})
    for invalid in ({}, {'obs': (source(), np.ones(4))}, {'obs': data}):
        with pytest.raises(TypeError):
            TRF().jointfit(invalid)
    with pytest.raises(ValueError, match='more than once'):
        TRF().jointfit({'one': (source(), data), 'two': (source(), data)})


def test_budget_termination_is_separate_from_finite_fit_statistic():
    data = dataset(np.linspace(-2, 2, 31), [3., .4], 2.)
    result = TRF().jointfit({'obs': (source(), data)}, max_nfev=1)
    assert not result.fit_info['success']
    assert result.fit_info['nfev'] == 1
    assert np.isfinite(result.fit_info['stat'])
