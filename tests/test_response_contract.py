"""Scientific and storage contracts for spectral response operators."""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from scipy.sparse import csr_matrix
from prism.modeling.operators import InstrumentResponse, SpectralResponse


@pytest.mark.parametrize('compress', [False, True])
def test_response_fits_roundtrip_is_lossless(tmp_path, compress):
    x = np.linspace(4990., 5010., 151)
    response = InstrumentResponse.from_fixed_sigma(x, .6)
    target = tmp_path/'response.fits'
    response.save_fits(target, compress=compress)
    restored = InstrumentResponse.from_fits(target)
    np.testing.assert_array_equal(restored.wavelength_grid, x)
    np.testing.assert_array_equal(restored.response_matrix.toarray(), response.response_matrix.toarray())


def test_sampling_cache_owns_observed_coordinates():
    native = np.linspace(-5., 5., 51)
    response = SpectralResponse(response_matrix=csr_matrix(np.eye(len(native))), wave=native)
    values = np.exp(-native**2)
    observed = np.array([-2.03, .26, 1.47])
    response._resample_from_native(values, observed)
    observed += .31
    actual = response._resample_from_native(values, observed)
    np.testing.assert_allclose(actual, np.interp(observed, native, values), atol=1e-15)


@pytest.mark.parametrize('grid', [[1., 1.], [2., 1.], [1., np.nan], [[1., 2.]], []])
def test_invalid_native_grid_is_rejected(grid):
    with pytest.raises(ValueError, match='Response wavelengths'):
        InstrumentResponse(grid, csr_matrix(np.eye(2)))


def test_response_owns_native_coordinates_and_validates_matrix():
    grid = np.arange(3.)
    response = InstrumentResponse(grid, csr_matrix(np.eye(3)))
    grid[0] = -100
    np.testing.assert_array_equal(response.wavelength_grid, np.arange(3.))
    with pytest.raises(ValueError, match='square'):
        SpectralResponse(response_matrix=np.ones((3, 2)), wave=np.arange(3.))
    with pytest.raises(ValueError, match='finite'):
        InstrumentResponse(np.arange(2.), np.array([[1., np.inf], [0., 1.]]))


@pytest.mark.usefixtures('astropy_extensions')
def test_off_grid_jacobian_matches_independent_finite_difference():
    native = np.linspace(-8., 8., 201)
    response = SpectralResponse(
        instrument=InstrumentResponse.from_fixed_sigma(native, .4), wave=native)
    source = Gaussian1D(2., -.3, .7) + Gaussian1D(.5, 1.2, 1.3)
    observed = np.r_[-9., native[4:-4:3] + .013, 9.]
    model = source | response
    jacobian = np.asarray(model.fit_deriv(observed, *model.parameters))
    for i in range(len(model.parameters)):
        plus, minus = model.copy(), model.copy()
        values = model.parameters.copy()
        values[i] += 1e-5
        plus.parameters = values
        values[i] -= 2e-5
        minus.parameters = values
        numerical = (plus(observed)-minus(observed))/2e-5
        np.testing.assert_allclose(jacobian[i], numerical, rtol=2e-7, atol=1e-9)


def test_instrument_response_is_a_unit_preserving_operator():
    import astropy.units as u
    matrix = np.array([[.8, .2], [.3, .7]])
    response = InstrumentResponse([1., 2.], matrix)
    values = np.array([[2., 4.], [3., 5.]])*u.Jy
    actual = response(values)
    assert actual.unit == u.Jy
    np.testing.assert_allclose(actual.value, matrix @ values.value)
    np.testing.assert_allclose(response(values[:, 0]).value, matrix @ values.value[:, 0])
    with pytest.raises(ValueError, match='input samples'):
        response(np.ones(3))


def test_spectral_response_has_one_matrix_and_one_grid():
    grid = np.arange(4.)
    response = SpectralResponse(response_matrix=np.eye(4), wave=grid)
    assert response.matrix is response.response_matrix
    assert response.x is response.wavelength_grid
    response._get_interp_cache(grid + .1)
    response.wavelength_grid = grid + 10.
    assert response._interp_cache is None
    np.testing.assert_array_equal(response.x, grid + 10.)
    response.response_matrix = 2*np.eye(4)
    np.testing.assert_array_equal(response.matrix.toarray(), 2*np.eye(4))


def test_plain_fixed_operator_uses_prepared_evaluation():
    from prism.modeling.operators import LinearOperatorModel
    from prism.modeling.fitting.evaluation import _ModelEvaluation
    x = np.linspace(-4., 4., 51)
    matrix = csr_matrix(np.diag(np.linspace(.5, 1., len(x))))
    model = Gaussian1D(2., .1, .8) | LinearOperatorModel(matrix, x=x)
    evaluation = _ModelEvaluation(model, x)
    assert evaluation.stateless
    parameters = np.array([3., -.2, 1.1])
    expected = matrix @ (3*np.exp(-.5*((x+.2)/1.1)**2))
    np.testing.assert_allclose(evaluation(parameters), expected, rtol=1e-14)


@pytest.mark.parametrize('flexible', [False, True])
def test_direct_response_yaml_roundtrip_preserves_operator(flexible):
    from astropy.io.misc import yaml
    native = np.linspace(-8., 8., 71)
    response = SpectralResponse(
        response_matrix=InstrumentResponse.from_fixed_sigma(native, .4).response_matrix,
        wave=native, flexible=flexible, name='custom_lsf')
    model = Gaussian1D(2., -.3, .7) | response
    restored = yaml.load(yaml.dump(model))
    assert restored.right.flexible is flexible
    assert restored.right.name == 'custom_lsf'
    np.testing.assert_array_equal(restored.right.matrix.toarray(), response.matrix.toarray())
    np.testing.assert_array_equal(restored(native), model(native))


def test_instrument_recipe_preserves_options(monkeypatch):
    from astropy.io.misc import yaml
    native = np.linspace(-8., 8., 71)
    response = InstrumentResponse.from_fixed_sigma(native, .4)
    monkeypatch.setattr(InstrumentResponse, 'from_instrument', classmethod(lambda cls, name: response))
    operator = SpectralResponse(instrument='TEST', wave=native, flexible=False, renormalize=False)
    model = Gaussian1D() | operator
    restored = yaml.load(yaml.dump(model))
    assert restored.right.flexible is False
    assert restored.right.renormalize is False
    np.testing.assert_array_equal(restored(native), model(native))


@pytest.mark.usefixtures('astropy_extensions')
def test_direct_response_model_fits_roundtrip(tmp_path):
    from prism.modeling.io import model_to_fits, load_model
    native = np.linspace(-6., 6., 51)
    response = SpectralResponse(
        response_matrix=InstrumentResponse.from_fixed_sigma(native, .4).response_matrix,
        wave=native, flexible=True)
    model = Gaussian1D(2., -.3, .7) | response
    target = tmp_path/'model.fits'
    model_to_fits(model, target)
    restored = load_model(target)
    np.testing.assert_array_equal(restored(native), model(native))
    observed = native[::3] + .02
    np.testing.assert_array_equal(restored(observed), model(observed))


@pytest.mark.usefixtures('astropy_extensions')
def test_parameter_dependent_operator_is_not_frozen():
    from astropy.modeling import Parameter
    from prism.modeling.operators import LinearOperatorModel
    from prism.modeling.fitting import ScipyTRF
    from prism.modeling.fitting.evaluation import _ModelEvaluation

    class Gain(LinearOperatorModel):
        gain = Parameter(default=.5)

        def get_matrix(self, x, *parameters):
            return self.matrix * parameters[0]

        def _prism_pipe_fit_deriv(self, left_deriv, left_inputs, left_params, right_params, left_model):
            transformed = super()._prism_pipe_fit_deriv(
                left_deriv, left_inputs, left_params, right_params, left_model)
            intrinsic = left_model.evaluate(left_inputs[0], *left_params)
            return np.vstack([transformed, self.matrix @ intrinsic])

    x = np.linspace(-4., 4., 81)
    source = Gaussian1D(2., .2, .8, fixed={'amplitude': True, 'mean': True, 'stddev': True})
    model = source | Gain(matrix=csr_matrix(np.eye(len(x))), x=x)
    assert not _ModelEvaluation(model, x).stateless
    y = 6*np.exp(-.5*((x-.2)/.8)**2)
    fitted = ScipyTRF()(model, x, y)
    np.testing.assert_allclose(fitted.right.gain.value, 3., atol=1e-9)


@pytest.mark.parametrize('recipe', [{'type': 'direct'}, {'type': 'unsupported'},
                                     {'type': 'instrument', 'instrument': 'prism-no-such-calibration'}])
def test_response_load_never_substitutes_unconvolved_source(recipe):
    from astropy.io.misc import yaml
    mapping = {'_source': Gaussian1D(2., 0., 1.), '_recipe': recipe, '_wave': [-1., 0., 1.]}
    serialized = '!prism.LinearOperatorPipe\n' + yaml.dump(mapping)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        yaml.load(serialized)


@pytest.mark.usefixtures('astropy_extensions')
def test_unsaveable_operator_fails_without_writing_partial_model(tmp_path):
    from prism.modeling.operators import LinearOperatorModel
    from scipy.sparse import eye
    x = np.linspace(-2, 2, 21)
    from astropy.modeling import Parameter
    class Dynamic(LinearOperatorModel):
        gain = Parameter(default=1.)
    model = Gaussian1D() | Dynamic(matrix=eye(len(x)), x=x)
    target = tmp_path/'unsupported.fits'
    with pytest.raises(ValueError, match='reconstruction rule'):
        model.save(target)
    assert not target.exists()
