"""Generic fixed operators: independent projection, persistence and failures."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.modeling import Parameter
from astropy.modeling.models import Gaussian1D
from scipy.sparse import csr_matrix
from prism.modeling.operators import LinearOperatorModel, SpectralResponse, InstrumentResponse
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.io import load_model


def rectangular():
    x = np.linspace(-3, 3, 63)
    xout = np.arange(21.)
    matrix = np.zeros((21, 63))
    for row in range(21):
        matrix[row, 3*row:3*row+3] = 1/3
    return x, xout, matrix


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('backend', [TRF, pytest.param(SherpaLM,
    marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_rectangular_projection_derivatives_and_known_input_fit(backend):
    x, xout, matrix = rectangular()
    operator = LinearOperatorModel(matrix, x=x, xout=xout)
    model = Gaussian1D(1., -.4, 1.2, bounds={'stddev': (.2, 2.)}) | operator
    truth = matrix @ (2.7*np.exp(-.5*((x-.2)/.6)**2))
    np.testing.assert_allclose(model(xout), matrix @ np.exp(-.5*((x+.4)/1.2)**2))
    jac = model.fit_deriv(xout, *model.parameters)
    for i in range(3):
        plus, minus = model.parameters.copy(), model.parameters.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        np.testing.assert_allclose(jac[i], (model.evaluate(xout,*plus)-model.evaluate(xout,*minus))/2e-6,
                                   rtol=1e-6, atol=1e-8)
    fitted = backend()(model, xout, truth, inplace=False)
    np.testing.assert_allclose(fitted.parameters, [2.7, .2, .6], atol=1e-5)


@pytest.mark.usefixtures('astropy_extensions')
def test_rectangular_grids_and_values_keep_units_in_storage(tmp_path):
    x, xout, matrix = rectangular()
    op = LinearOperatorModel(csr_matrix(matrix), x=x*u.AA, xout=xout*u.s)
    target = tmp_path/'operator.fits'
    op.save(target)
    loaded = LinearOperatorModel.from_fits(target)
    assert loaded.x.unit == u.AA and loaded.xout.unit == u.s
    np.testing.assert_array_equal(loaded.matrix.toarray(), matrix)
    values = np.column_stack([np.exp(-x*x), x*x])*u.Jy
    np.testing.assert_allclose(loaded.evaluate(values).value, matrix @ values.value)
    assert loaded.evaluate(values).unit == u.Jy
    model = Gaussian1D(2.*u.Jy, .2*u.AA, .6*u.AA) | loaded
    np.testing.assert_allclose(model(xout*u.s).value, matrix @ (2*np.exp(-.5*((x-.2)/.6)**2)))


@pytest.mark.usefixtures('astropy_extensions')
@pytest.mark.parametrize('spectral', [False, True])
def test_session_resource_is_relative_portable_and_shared(tmp_path, spectral):
    directory = tmp_path/'bundle';directory.mkdir()
    x = np.linspace(-3, 3, 61)
    op = SpectralResponse(response_matrix=np.eye(len(x)), wave=x) if spectral else LinearOperatorModel(np.eye(len(x)), x=x)
    model = Gaussian1D(2., .2, .6) | op
    target = directory/'fit.fits'
    model.save(target, operator_file='response.fits')
    model.save(directory/'second.fits', operator_file='response.fits')
    assert len(list(directory.glob('response*.fits'))) == 1
    with fits.open(target) as hdus:
        definition = hdus['MODEL_DEF'].data['YAML'][0]
        assert 'response.fits' in definition
        assert str(directory) not in definition
        assert '_matrix' not in definition
    moved = tmp_path/'moved';shutil.move(str(directory), moved)
    restored = load_model(moved/'fit.fits')
    np.testing.assert_array_equal(restored(x), model(x))
    assert type(restored.right) is type(op)


@pytest.mark.usefixtures('astropy_extensions')
def test_default_resource_name_and_rectangular_session(tmp_path):
    x,xout,matrix = rectangular()
    model = Gaussian1D(2.,.1,.8) | LinearOperatorModel(matrix,x=x,xout=xout)
    path = tmp_path/'session.fits';model.save(path)
    files = list(tmp_path.glob('session_operator*.fits'))
    assert len(files) == 1
    np.testing.assert_array_equal(load_model(path)(xout), model(xout))
    files[0].unlink()
    with pytest.raises(FileNotFoundError):load_model(path)


@pytest.mark.usefixtures('astropy_extensions')
def test_modified_resource_is_rejected(tmp_path):
    x = np.linspace(-2, 2, 21)
    model = Gaussian1D() | LinearOperatorModel(np.eye(len(x)), x=x)
    path = tmp_path/'session.fits';model.save(path, operator_file='op.fits')
    LinearOperatorModel(2*np.eye(len(x)), x=x).save(tmp_path/'op.fits', overwrite=True)
    with pytest.raises(ValueError,match='differs from the saved session'):load_model(path)
    with pytest.raises(FileExistsError,match='different content'):
        model.save(path, operator_file='op.fits',overwrite=True)


@pytest.mark.usefixtures('astropy_extensions')
def test_failed_session_write_keeps_old_session_and_resources(tmp_path, monkeypatch):
    x = np.linspace(-2, 2, 21)
    op = LinearOperatorModel(np.eye(len(x)),x=x)
    path = tmp_path/'session.fits';(Gaussian1D()|op).save(path)
    original = path.read_bytes();resources = set(tmp_path.glob('session_operator*.fits'))
    native = fits.HDUList.writeto
    def fail_session(self, *args, **kwargs):
        if self[0].header.get('FORMAT') != 'PRISMLIN':raise OSError('forced session write failure')
        return native(self,*args,**kwargs)
    monkeypatch.setattr(fits.HDUList,'writeto',fail_session)
    changed = Gaussian1D() | LinearOperatorModel(2*np.eye(len(x)),x=x)
    with pytest.raises(OSError,match='forced'):changed.save(path,overwrite=True)
    assert path.read_bytes() == original
    assert set(tmp_path.glob('session_operator*.fits')) == resources


@pytest.mark.usefixtures('astropy_extensions')
def test_dynamic_operator_is_not_saved_as_frozen_snapshot(tmp_path):
    class Dynamic(LinearOperatorModel):
        gain = Parameter(default=1.)
    model = Gaussian1D() | Dynamic(matrix=np.eye(3),x=np.arange(3.))
    with pytest.raises(ValueError,match='reconstruction rule'):model.save(tmp_path/'dynamic.fits')
    assert not list(tmp_path.iterdir())


@pytest.mark.usefixtures('astropy_extensions')
def test_multiple_resources_are_unique_and_rebuilt(tmp_path):
    x = np.linspace(-3,3,31)
    a = Gaussian1D(2.,-.5,.6) | LinearOperatorModel(np.eye(len(x)),x=x)
    b = Gaussian1D(1.,.8,.9) | LinearOperatorModel(2*np.eye(len(x)),x=x)
    model = a+b
    path = tmp_path/'sum.fits';model.save(path)
    assert len(list(tmp_path.glob('sum_operator*.fits'))) == 2
    np.testing.assert_array_equal(load_model(path)(x),model(x))
    with pytest.raises(ValueError,match='exactly one'):model.save(tmp_path/'bad.fits',operator_file='one.fits')


@pytest.mark.usefixtures('astropy_extensions')
def test_output_subset_and_jacobian_columns_use_common_sampling():
    x, xout, matrix = rectangular()
    operator = LinearOperatorModel(matrix, x=x, xout=xout)
    source = Gaussian1D(2., .1, .8)
    model = source | operator
    selected = xout[2:-2:2]+.07
    native = matrix @ source(x)
    np.testing.assert_allclose(model(selected), np.interp(selected, xout, native))
    from prism.modeling.fitting.evaluation import _ModelEvaluation
    prepared = _ModelEvaluation(model, selected)
    assert prepared.stateless
    np.testing.assert_array_equal(prepared(model.parameters), model(selected))
    np.testing.assert_allclose(prepared.fit_deriv(model.parameters),model.fit_deriv(selected,*model.parameters))


@pytest.mark.usefixtures('astropy_extensions')
def test_resource_reuse_preserves_session_specific_operator_name(tmp_path):
    x = np.linspace(-2,2,21)
    for name in ['alpha','beta']:
        model = Gaussian1D() | LinearOperatorModel(np.eye(len(x)),x=x,name=name)
        path = tmp_path/(name+'.fits');model.save(path,operator_file='shared.fits')
        assert load_model(path).right.name == name
    assert len(list(tmp_path.glob('shared*.fits'))) == 1


def test_cube_and_individual_spectrum_store_the_same_operator(tmp_path):
    x = np.linspace(-3,3,61)
    initial = Gaussian1D(.8,0.,.7,fixed={'mean': True,'stddev': True})
    model = initial | LinearOperatorModel(np.eye(len(x)),x=x)
    data = np.array([2.,3.])[:,None]*np.exp(-.5*(x/.7)**2)
    result = TRF(covariance=True).multifit(model,x,data,progress=False)
    result.save(tmp_path/'cube.fits',operator_file='response.fits')
    np.testing.assert_array_equal(load_model(tmp_path/'cube.fits').right.matrix.toarray(),np.eye(len(x)))
    path = tmp_path/'spectrum.fits';result[1].save(path,operator_file='response.fits')
    loaded = load_model(path)
    np.testing.assert_allclose(loaded.parameters,result[1].model.parameters)
    assert loaded.left.amplitude.value == pytest.approx(3.)
    np.testing.assert_allclose(loaded._param_cov,result[1].model._param_cov,equal_nan=True)
    np.testing.assert_array_equal(loaded._fit_indices,result[1].model._fit_indices)


def test_corrupt_operator_checksum_is_rejected(tmp_path):
    path = tmp_path/'operator.fits'
    LinearOperatorModel(np.eye(3)).save(path)
    with fits.open(path) as hdus:
        position = hdus['DATA'].fileinfo()['datLoc']
    with path.open('r+b') as stream:
        stream.seek(position);first=stream.read(1);stream.seek(position);stream.write(bytes([first[0]^1]))
    with pytest.raises(ValueError,match='checksum'):LinearOperatorModel.from_fits(path)


@pytest.mark.usefixtures('astropy_extensions')
def test_archive_recipe_is_verified_and_can_be_snapshotted(tmp_path, monkeypatch):
    x = np.linspace(-3.,3.,61)
    calibration = {'response': InstrumentResponse.from_fixed_sigma(x,.2)}
    monkeypatch.setattr(InstrumentResponse,'from_instrument',classmethod(lambda cls,name: calibration['response']))
    op = SpectralResponse(instrument='test-calibration',wave=x)
    model = Gaussian1D(2.,.1,.7)|op
    recipe_file=tmp_path/'recipe.fits';model.save(recipe_file)
    assert not list(tmp_path.glob('recipe_operator*.fits'))
    np.testing.assert_array_equal(load_model(recipe_file)(x),model(x))
    snapshot=tmp_path/'snapshot.fits';model.save(snapshot,operator_file='calibration.fits')
    calibration['response'] = InstrumentResponse.from_fixed_sigma(x,.4)
    with pytest.raises(ValueError,match='calibration or response configuration differs'):load_model(recipe_file)
    np.testing.assert_array_equal(load_model(snapshot)(x),model(x))


@pytest.mark.usefixtures('astropy_extensions')
def test_gzip_session_and_operator_roundtrip(tmp_path):
    x=np.linspace(-2,2,21)
    op=LinearOperatorModel(np.eye(len(x)),x=x)
    op.save(tmp_path/'operator.fits.gz')
    assert (tmp_path/'operator.fits.gz').read_bytes()[:2] == b'\x1f\x8b'
    np.testing.assert_array_equal(LinearOperatorModel.from_fits(tmp_path/'operator.fits.gz').matrix.toarray(),op.matrix.toarray())
    model=Gaussian1D()|op
    model.save(tmp_path/'fit.fits.gz')
    assert (tmp_path/'fit.fits.gz').read_bytes()[:2] == b'\x1f\x8b'
    np.testing.assert_array_equal(load_model(tmp_path/'fit.fits.gz')(x),model(x))
