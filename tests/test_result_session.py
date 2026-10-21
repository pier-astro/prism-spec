"""Portable fitted graphs retain sharing, links, observations and covariance."""
from pathlib import Path
import shutil

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.modeling.models import Linear1D, Planar2D
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import eye

from prism.modeling import Component, Model, LinearOperatorModel
from prism.modeling.fitting import Dataset, FitResult, TRF, SherpaLM, HAS_SHERPA
from prism.modeling.operators import InstrumentResponse, SpectralResponse

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def trend(x, intercept, slope):
    return intercept+slope*x


def derivative(x, intercept, slope):
    return np.array([np.ones_like(x), x])


def component(name='continuum'):
    return Component(trend, dict(intercept=1., slope=.1), derivative=derivative, name=name)


def data(x, intercept=3., slope=.7, gain=1.):
    weights = np.ones(x.size)/.1
    weights[4] = 0.
    values = gain*(intercept+slope*x)
    values[4] = np.nan
    valid = np.ones(x.size, dtype=bool)
    valid[7] = False
    return Dataset(x, values, yerr=np.full(x.size, .1), weights=weights,
                   mask=valid, response=LinearOperatorModel(gain*eye(len(x)), x=x))


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('shared', [True, False])
def test_save_load_reuse_complete_graph(engine, shared, tmp_path):
    first, second = component(), component()
    if shared:
        second = first
    else:
        second.slope.link = first.slope
    early = data(np.linspace(-2., 1., 41))
    later = data(np.linspace(-1., 3., 57), intercept=3. if shared else 5., gain=1.5)
    early.x = np.ma.array(early.x, mask=np.arange(early.x.size) == 1)
    early.meta = {'target': 'AGN', 'position': np.array([12., 24.]), 'filter': 'bluè'}
    early.xunit = u.AA
    early.yunit = u.Jy
    fitter = engine(covariance=True)
    result = fitter.jointfit({'early': (first, early), 'later': (second, later)})
    filename = tmp_path/'fit.fits'
    result.save(filename)
    with pytest.raises(FileExistsError):
        result.save(filename)
    result.save(filename, overwrite=True)
    restored = FitResult.load(filename, components={'continuum': first})
    assert tuple(restored) == ('early', 'later')
    assert restored.param_names == result.param_names
    assert_array_equal(restored.parameters, result.parameters)
    assert_array_equal(restored.fit_indices, result.fit_indices)
    assert_array_equal(restored.covariance, result.covariance)
    assert restored.fit_info['stat'] == result.fit_info['stat']
    assert restored.fit_info['success'] == result.fit_info['success']
    assert restored.fit_info['nfev'] == result.fit_info['nfev']
    for label in result:
        assert_allclose(restored[label].predict(), result[label].predict(), atol=1e-12)
    saved = restored['early'].dataset
    assert saved.xunit == u.AA and saved.yunit == u.Jy
    assert saved.meta['filter'] == 'bluè'
    assert_array_equal(saved.meta['position'], [12., 24.])
    assert_array_equal(saved.mask, result['early'].dataset.mask)
    assert_array_equal(np.ma.getmaskarray(saved.x), np.ma.getmaskarray(early.x))
    assert_array_equal(saved.weights, early.weights)
    inputs = restored.associations
    a, b = inputs['early'][0], inputs['later'][0]
    assert (a is b) is shared
    if not shared:
        a.slope.value = .8
        assert b.slope.value == pytest.approx(.8)
    assert_allclose(first.parameters, [1., .1])
    repeated = fitter.jointfit(restored.associations)
    assert_allclose(repeated.parameters, result.parameters, atol=2e-6)
    assert_allclose(repeated.covariance, result.covariance, rtol=3e-5, atol=1e-9)
    inputs['early'][1].y[:] = -500.
    assert_allclose(restored['early'].predict(), result['early'].predict(), atol=1e-12)
    # Relative response resources survive moving the complete session directory.
    moved = tmp_path/'moved'
    moved.mkdir()
    for path in tmp_path.glob('*.fits'):
        shutil.copy2(path, moved/path.name)
    relocated = FitResult.load(moved/'fit.fits', components={'continuum': first})
    assert_allclose(relocated['later'].predict(), result['later'].predict())


@pytest.mark.parametrize('explicit_pipe', [False, True])
def test_single_result_roundtrip_preserves_source_response_boundary(explicit_pipe, tmp_path):
    x = np.linspace(-1., 2., 41)
    source = Model.from_astropy(Linear1D(.1, 1.))
    response = LinearOperatorModel(2.*eye(len(x)), x=x)
    observed = Dataset(x, 2.*(3.+.7*x), response=None if explicit_pipe else response)
    if explicit_pipe:
        source = source | response
    result = TRF(covariance=True)(source, observed)
    filename = tmp_path/'single.fits'
    result.save(filename)
    restored = FitResult.load(filename)
    assert_array_equal(restored.parameters, result.parameters)
    assert_array_equal(restored.covariance, result.covariance)
    assert_allclose(restored.predict(), result.predict())
    assert_allclose(restored.model(x), result.model(x))
    assert (restored.dataset.response is None) is explicit_pipe
    model, dataset = restored.associations['data']
    repeated = TRF()(model, dataset)
    assert_allclose(repeated.predict(), result.predict(), atol=1e-7)


def test_shared_components_survive_independent_source_roots(tmp_path):
    a, b = component('a'), component('b')
    common = Model.from_astropy(Linear1D(.1, 0., fixed={'intercept': True}, name='common'))
    a.slope.fixed = b.slope.fixed = True
    x = np.linspace(-1., 2., 41)
    result = TRF().jointfit({'one': (a+common, Dataset(x, 3.+.8*x)),
                                 'two': (b+common, Dataset(x, 5.+.8*x))})
    filename = tmp_path/'shared.fits'
    result.save(filename)
    restored = FitResult.load(filename, components={'a': a, 'b': b})
    inputs = restored.associations
    assert inputs['one'][0].components['common'].component is inputs['two'][0].components['common'].component
    assert_allclose(restored.parameters, result.parameters)
    assert restored.fit_info['nfree'] == 3


def test_fitted_response_transform_and_cross_links_roundtrip(tmp_path):
    x = np.linspace(-1., 2., 41)
    source = component()
    source.slope.fixed = True
    response = Model.from_astropy(Planar2D(0., 1., 0., fixed={'slope_x': True, 'intercept': True}))
    # Direct parameter reference stays declarative, including Astropy drivers.
    source.intercept.link = response.slope_y
    result = TRF(covariance=True)(source, Dataset(x, 3.*(3.+.1*x), response=response))
    filename = tmp_path/'response.fits'
    result.save(filename)
    restored = FitResult.load(filename, components={'continuum': source})
    assert_allclose(restored.predict(), result.predict())
    model, dataset = restored.associations['data']
    dataset.response.slope_y.value = 4.
    assert_allclose(model(x), 4.+.1*x)
    assert_allclose(restored.predict(), result.predict())


def test_builtin_lines_and_no_covariance_roundtrip_without_definitions(tmp_path):
    from prism.modeling.lines import GaussianLines
    table = {'name': ['line'], 'position': np.array([5000.])*u.AA}
    model = GaussianLines(table, amplitude=1., fwhm=150., name='narrow')
    model.offset.fixed = model.fwhm.fixed = True
    x = np.linspace(4990., 5010., 101)
    response = SpectralResponse(instrument=InstrumentResponse.from_fixed_fwhm(x, .5), wave=x)
    result = TRF(covariance=False)(model, Dataset(x, Dataset(x, np.zeros(x.size), response=response).predict(model)*3., response=response))
    filename = tmp_path/'lines.fits'
    result.save(filename)
    restored = FitResult.load(filename)
    assert restored.covariance is None
    assert_allclose(restored.predict(), result.predict())
    assert_allclose(restored.model(x), result.model(x))


def test_callable_links_rejected_without_publishing_partial_session(tmp_path):
    x = np.linspace(-1., 1., 41)
    source = component()
    source.slope.tied = lambda model: model.intercept.value*.2
    result = TRF()(source, Dataset(x, 3.+.6*x))
    with pytest.raises(ValueError, match='Callable'):
        result.save(tmp_path/'invalid.fits')
    assert not list(tmp_path.iterdir())


def test_corrupted_session_and_missing_response_fail_explicitly(tmp_path):
    x = np.linspace(-1., 1., 41)
    result = TRF()(component(), data(x))
    filename = tmp_path/'fit.fits'
    result.save(filename)
    resource = next(tmp_path.glob('*operator*.fits'))
    resource.unlink()
    with pytest.raises(FileNotFoundError):
        FitResult.load(filename, components={'continuum': component()})
    result.save(filename, overwrite=True)
    with fits.open(filename, mode='update', checksum=False) as hdus:
        hdus['D0_Y'].data[0] += 1.
        hdus['D0_Y'].header['CHECKSUM'] = '0000000000000000'
    with pytest.raises(ValueError, match='checksum'):
        FitResult.load(filename, components={'continuum': component()})


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('joint', [False, True])
def test_actual_filter_selection_is_retained_for_reuse(engine, joint, tmp_path):
    x = np.linspace(-1., 1., 41)
    y = 3.+.7*x
    y[3] = np.nan
    x[5] = np.nan
    original = Dataset(x, y)
    fitter = engine()
    if joint:
        result = fitter.jointfit({'obs': (component(), original)}, filter_non_finite=True)
        view = result['obs']
    else:
        result = fitter(component(), original, filter_non_finite=True)
        view = result
    expected = np.isfinite(x) & np.isfinite(y)
    assert_array_equal(view.dataset.mask, expected)
    assert original.mask is None
    filename = tmp_path/'filtered.fits'
    result.save(filename)
    restored = FitResult.load(filename, components={'continuum': component()})
    if joint:
        repeated = fitter.jointfit(restored.associations)
    else:
        model, observed = restored.associations['data']
        repeated = fitter(model, observed)
    assert repeated.fit_info['ndata'] == np.count_nonzero(expected)
    assert_allclose(repeated.parameters, result.parameters, atol=1e-6)


def test_equal_response_content_preserves_independent_names_and_shared_identity(tmp_path):
    x = np.linspace(-1., 1., 41)
    model = component()
    first = LinearOperatorModel(eye(len(x)), x=x, name='first')
    second = LinearOperatorModel(eye(len(x)), x=x, name='second')
    result = TRF().jointfit({'one': (model, Dataset(x, 3.+.7*x, response=first)),
        'two': (model, Dataset(x, 3.+.7*x, response=second)),
        'three': (model, Dataset(x, 3.+.7*x, response=first))})
    filename = tmp_path/'same.fits'
    result.save(filename)
    assert len(list(tmp_path.glob('*operator*.fits'))) == 1
    restored = FitResult.load(filename, components={'continuum': model})
    inputs = restored.associations
    assert inputs['one'][1].response.name == 'first'
    assert inputs['two'][1].response.name == 'second'
    assert inputs['one'][1].response is inputs['three'][1].response
    assert inputs['one'][1].response is not inputs['two'][1].response


def test_incompatible_custom_definitions_with_same_name_are_rejected(tmp_path):
    first = component()
    def other(x, intercept, slope):
        return intercept+slope*x*x
    second = Component(other, dict(intercept=1., slope=.1), name='continuum')
    x = np.linspace(-1., 1., 41)
    result = TRF().jointfit({'one': (first, Dataset(x, 3.+.7*x)),
                                 'two': (second, Dataset(x, 3.+.7*x*x))})
    with pytest.raises(ValueError, match='distinct component names'):
        result.save(tmp_path/'invalid.fits')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('loader', ['load_params', 'load_model'])
def test_model_only_loaders_identify_fitted_sessions(loader, tmp_path):
    from prism.modeling import io
    x = np.linspace(-1., 1., 41)
    result = TRF()(component(), data(x))
    filename = tmp_path/'fit.fits'
    result.save(filename)
    with pytest.raises(ValueError, match='FitResult.load'):
        getattr(io, loader)(filename)


def test_missing_definition_and_unsupported_version_fail_clearly(tmp_path):
    x = np.linspace(-1., 1., 41)
    result = TRF()(component(), data(x))
    filename = tmp_path/'fit.fits'
    result.save(filename)
    with pytest.raises(ValueError, match='custom Component'):
        FitResult.load(filename)
    with fits.open(filename, mode='update') as hdus:
        hdus[0].header['VERSION'] = 99
    with pytest.raises(ValueError, match='version'):
        FitResult.load(filename, components={'continuum': component()})


def test_valid_checksums_do_not_hide_wrong_covariance_order(tmp_path):
    x = np.linspace(-1., 1., 41)
    result = TRF(covariance=True)(component(), data(x))
    filename = tmp_path/'fit.fits'
    result.save(filename)
    with fits.open(filename, mode='update') as hdus:
        document = str(hdus['MODEL_DEF'].data['YAML'][0])
        assert 'fit_indices: [0, 1]' in document
        hdus['MODEL_DEF'].data['YAML'][0] = document.replace('fit_indices: [0, 1]', 'fit_indices: [1, 0]')
        for hdu in hdus:
            hdu.add_checksum()
    with pytest.raises(ValueError, match='covariance order'):
        FitResult.load(filename, components={'continuum': component()})


def test_parameterized_operator_without_reconstruction_rule_is_not_dropped(tmp_path):
    from astropy.modeling import Parameter
    class Gain(LinearOperatorModel):
        gain = Parameter(default=1.)
        def get_matrix(self, x, gain):
            return gain*self.matrix
    x = np.linspace(-1., 1., 41)
    model = component()
    model.intercept.fixed = model.slope.fixed = True
    response = Gain(matrix=eye(x.size), x=x)
    result = TRF()(model, Dataset(x, 3.*model(x), response=response))
    with pytest.raises(ValueError, match='reconstruction rule'):
        result.save(tmp_path/'invalid.fits')
    assert not list(tmp_path.iterdir())
