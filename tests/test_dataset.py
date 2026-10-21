"""Observed datasets preserve response, weighting and source-grid semantics."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import FittableModel, Parameter
from astropy.modeling.models import Const1D, Gaussian1D
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix, eye

from prism.data import Spectrum
from prism.modeling import Component, LinearOperatorModel, Model
from prism.modeling.fitting import Dataset, HAS_SHERPA, TRF, SherpaLM
from prism.modeling.fitting.problem import _FitProblem
from prism.modeling.fitting.data import _prepare_fit_data, _resolve_fit_inputs

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def continuum(x, intercept, slope):
    return intercept + slope*x


def continuum_derivative(x, intercept, slope):
    return np.array([np.ones_like(x), x])


def source():
    return Component(continuum, {'intercept': 1., 'slope': .1},
                     derivative=continuum_derivative, name='continuum')


def observation():
    native = np.linspace(-2., 2., 121)
    output = native[1:-1:2]
    rows = np.repeat(np.arange(len(output)), 3)
    cols = np.column_stack((np.arange(0, 119, 2), np.arange(1, 120, 2),
                            np.arange(2, 121, 2))).reshape(-1)
    matrix = csr_matrix((np.tile([.2, .6, .2], len(output)), (rows, cols)),
                        shape=(len(output), len(native)))
    response = LinearOperatorModel(matrix, x=native, xout=output)
    x = np.linspace(output[2], output[-3], 77)
    design = np.column_stack([np.interp(x, output, matrix@np.ones_like(native)),
                              np.interp(x, output, matrix@native)])
    y = design@np.array([3., .7]) + .025*np.sin(4*x)
    errors = .1 + .02*(x-x[0])
    valid = np.ones(x.shape, dtype=bool)
    valid[12:18] = False
    y[~valid] = np.nan
    return Dataset(x, y, yerr=errors, mask=valid, response=response), design


@pytest.mark.parametrize('engine', ENGINES)
def test_dataset_response_fit_matches_independent_weighted_estimator_and_pipe(engine):
    data, design = observation()
    start = source()
    valid = data.mask
    expected = np.linalg.lstsq(design[valid]/data.yerr[valid, None],
                               data.y[valid]/data.yerr[valid], rcond=None)[0]
    fitter = engine(covariance=True)
    fitted = fitter(start, data, inplace=False)
    info = fitter.fit_info.copy()
    reference = engine(covariance=True)(start | data.response,
        data.x, data.y, yerr=data.yerr, mask=valid, inplace=False)
    assert_allclose(fitted.parameters, expected, atol=2e-7)
    assert_allclose(fitted.parameters, reference.parameters, rtol=1e-12)
    assert_allclose(fitted.predict(data.x), design@expected, atol=2e-7)
    assert_allclose(fitted.covariance, reference.covariance, rtol=1e-12)
    assert info['ndata'] == np.count_nonzero(valid)
    assert info['dof'] == np.count_nonzero(valid)-2
    expected_stat = np.sum(((design@expected-data.y)[valid]/data.yerr[valid])**2)
    assert info['stat'] == pytest.approx(expected_stat, rel=1e-8)
    assert_array_equal(start.parameters, [1., .1])
    assert fitted.model.component is not start.component
    assert fitted.dataset.response is not data.response


def test_dataset_preparation_preserves_response_jacobian():
    data, design = observation()
    model, x, y, err, weights, mask = _resolve_fit_inputs(
        source(), data, None, None, None, None, None)
    x, y, err, weights, valid = _prepare_fit_data(x, y, err, weights, mask)
    problem = _FitProblem(model, x, y, 1/err, valid)
    values = np.array([2.5, .4])
    assert_allclose(problem.residual(values),
        ((design@values-data.y)/data.yerr)[valid], atol=1e-12)
    assert_allclose(problem.jacobian(values), design[valid]/data.yerr[valid, None],
                    atol=1e-12)


def test_prediction_includes_masked_samples_and_composes_declared_operators():
    data, design = observation()
    model = source()
    assert_allclose(data.predict(model), design@model.parameters, atol=1e-12)
    assert data.predict(model).shape == data.y.shape
    native = data.response.x
    extra = LinearOperatorModel(2*eye(len(native)), x=native)
    assert_allclose(data.predict(model | extra), 2*design@model.parameters, atol=1e-12)
    assert_allclose(model(native), continuum(native, 1., .1))


class Area(FittableModel):
    n_inputs = 2
    n_outputs = 1
    norm = Parameter(default=1.)

    @staticmethod
    def evaluate(x, flux, norm):
        return norm*flux/trapezoid(flux, x)


@pytest.mark.parametrize('engine', ENGINES)
def test_dataset_profile_transform_keeps_grid_and_copies_response_parameters(engine):
    x = np.linspace(-2., 2., 101)
    model = source()
    model.intercept.fixed = model.slope.fixed = True
    response = Model.from_astropy(Area())
    expected = 4.*model(x)/trapezoid(model(x), x)
    valid = np.ones(x.shape, dtype=bool)
    valid[20:60] = False
    y = expected.copy()
    y[~valid] = np.nan
    data = Dataset(x, y, mask=valid, response=response)
    fitter = engine()
    fitted = fitter(model, data, inplace=False)
    assert_allclose(fitted.predict(x), expected, rtol=1e-6)
    assert response.norm.value == 1.
    assert fitted.dataset.response.norm.value == pytest.approx(4., rel=1e-6)
    assert fitter.fit_info['ndata'] == np.count_nonzero(valid)


@pytest.mark.parametrize('engine', ENGINES)
def test_dataset_inplace_and_error_weight_selection(engine):
    model = Model.from_astropy(Const1D(0.))
    x = np.ma.array(np.arange(4.), mask=[False, False, True, False])
    y = np.array([1., 2., 999., np.nan])
    errors = np.ones(4)
    weights = np.array([1., 2., 50., 0.])
    data = Dataset(x, y, yerr=errors, weights=weights)
    fitter = engine()
    fitted = fitter(model, data, inplace=True)
    assert fitted.model is not model
    assert model.amplitude.value == pytest.approx(1.8, rel=1e-6)
    assert fitter.fit_info['ndata'] == 2
    assert_array_equal(data.y, y)
    assert_array_equal(data.weights, weights)
    assert_array_equal(np.ma.getmaskarray(data.x), [False, False, True, False])


def test_spectrum_adapter_references_current_arrays_metadata_and_units():
    spectrum = Spectrum(np.arange(20.)+5000., np.ones(20), yerr=np.ones(20),
                        xunit='AA', yunit='Jy', meta={'target': 'test'}, dtype=np.float64)
    spectrum.cutout(min=5005., max=5015., inplace=True)
    data = Dataset(spectrum)
    assert data.x is spectrum.x
    assert data.y is spectrum.y
    assert data.yerr is spectrum.yerr
    assert data.meta is spectrum.meta
    assert data.xunit == spectrum.xunit
    assert data.yunit == spectrum.yunit
    assert data.mask is None
    assert len(data.x) == 11 and len(spectrum.mask) == 20
    fitter = TRF()
    fitted = fitter(Model.from_astropy(Const1D(0.)), data)
    assert fitted.model.amplitude.value == pytest.approx(1.)
    assert fitter.fit_info['ndata'] == 11


@pytest.mark.parametrize('engine', ENGINES)
def test_nonfinite_filter_remains_explicit_fit_choice(engine):
    data = Dataset([0., 1., np.nan], [2., 2., np.nan])
    model = Model.from_astropy(Const1D(0.))
    with pytest.raises(ValueError, match='Non-finite'):
        engine()(model, data)
    fitter = engine()
    fitted = fitter(model, data, filter_non_finite=True, inplace=False)
    assert fitted.model.amplitude.value == pytest.approx(2.)
    assert fitter.fit_info['ndata'] == 2
    assert_allclose(data.predict(fitted.model), [2., 2., np.nan])


def test_quantity_coordinates_and_predictions_preserve_explicit_units():
    data = Dataset(np.array([0., 1., np.nan])*u.AA, np.array([2., 2., np.nan])*u.Jy)
    prediction = data.predict(Const1D(2*u.Jy))
    assert prediction.unit == u.Jy
    assert data.x.unit == u.AA
    assert_allclose(prediction.to_value(u.Jy), [2., 2., np.nan])
    with pytest.raises(TypeError, match='numeric'):
        TRF()(Model.from_astropy(Const1D(1.)), data, filter_non_finite=True)
    line = Gaussian1D(2*u.Jy, 5000*u.AA, 1*u.AA)
    wave = np.array([4999., 5000., 5001.])*u.AA
    prediction = Dataset(wave, np.ones(3)*u.Jy).predict(line)
    assert_allclose(prediction.to_value(u.Jy), 2*np.exp(-.5*np.array([-1., 0., 1.])**2))


@pytest.mark.parametrize('option,value', [
    ('yerr', [1.]), ('weights', [1.]), ('mask', [True]),
    ('yerr', [[1., 1.], [1., 1.]]),
])
def test_shapes_are_checked_before_fitting(option, value):
    with pytest.raises(ValueError, match='shape|Asymmetric'):
        Dataset([0., 1.], [1., 1.], **{option: value})


@pytest.mark.parametrize('options', [
    {'y': [1., 1.]}, {'z': [1., 1.]}, {'yerr': [1., 1.]},
    {'weights': [1., 1.]}, {'mask': [True, True]},
])
def test_conflicting_dataset_and_call_data_are_rejected(options):
    with pytest.raises(TypeError, match='Dataset'):
        TRF()(source(), Dataset([0., 1.], [1., 1.]), **options)


def test_invalid_response_or_missing_data_fail_clearly():
    with pytest.raises(TypeError, match='response'):
        Dataset([0., 1.], [1., 1.], response=lambda x: x)
    with pytest.raises(TypeError, match='Supply y'):
        Dataset([0., 1.])
    with pytest.raises(ValueError, match='one-dimensional'):
        Dataset(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match='No finite coordinates'):
        Dataset([np.nan], [1.]).predict(source())
