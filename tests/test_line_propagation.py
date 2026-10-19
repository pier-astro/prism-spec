"""Independent propagation checks for covariance, links and equivalent width."""
import numpy as np
import pytest
from astropy.modeling.models import Const1D
from prism.modeling.models.lines import GaussianLine, profiles
from prism.modeling.line_analysis import select_line


def line_model():
    line = GaussianLine(position=5000., amplitude=4., fwhm=200., name='line')
    line.offset.fixed = True
    line.fwhm.fixed = True
    return line


def test_covariance_without_parameter_std():
    line = line_model()
    line._param_cov = np.array([[.04]])
    selected = select_line(line, 'line')
    result, samples = selected.sample(n_samples=64, method='covariance', random_state=71, return_samples=True)
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    draws = np.random.default_rng(71).multivariate_normal([4.], [[.04]], size=64)[:, 0]
    np.testing.assert_allclose(samples['flux'], draws*sigma*np.sqrt(2*np.pi), rtol=1e-12)
    assert result.flux.std > 0.
    np.testing.assert_array_equal(line.parameters, line_model().parameters)


@pytest.mark.parametrize('method', ['std', 'covariance'])
def test_selected_tied_amplitude_follows_external_driver(method):
    model = line_model() + Const1D(2., name='driver')
    model.amplitude_0.tied = lambda m: 2.*m.amplitude_1
    model.amplitude_1.std = .2
    model._param_cov = np.array([[.04]])
    initial = model.parameters.copy()
    selected = select_line(model, 'line')
    result, samples = selected.sample(n_samples=64, method=method, distribution='gaussian',
                                     random_state=71, return_samples=True)
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    if method == 'covariance':
        draws = np.random.default_rng(71).multivariate_normal([2.], [[.04]], size=64)[:, 0]
    else:
        draws = np.random.default_rng(71).normal(2., .2, size=64)
    np.testing.assert_allclose(samples['flux'], 2.*draws*sigma*np.sqrt(2*np.pi), rtol=1e-12)
    np.testing.assert_array_equal(model.parameters, initial)
    assert result.flux.std > 0.


def test_fixed_parameters_with_stale_errors_are_not_drawn():
    line = line_model()
    line.position.std = 100.
    with pytest.raises(ValueError, match='uncertainty information'):
        select_line(line, 'line').sample(method='std', n_samples=8)


@pytest.mark.parametrize('cov', [np.eye(2), [[-1.]], [[np.nan]]])
def test_explicit_invalid_covariance_rejected(cov):
    line = line_model()
    line._param_cov = np.asarray(cov)
    with pytest.raises(ValueError, match='covariance'):
        select_line(line, 'line').sample(method='covariance', n_samples=8)


def test_eqw_joint_covariance_and_reproducible_generator():
    model = line_model() + Const1D(3., name='continuum')
    covariance = np.array([[.01,.015],[.015,.04]])
    model._param_cov = covariance
    selected = select_line(model, 'line')
    kwargs = dict(method='covariance', perturb_continuum=True, n_samples=64)
    result = selected.sample_eqw(random_state=np.random.default_rng(71), **kwargs)
    again = selected.sample_eqw(random_state=np.random.default_rng(71), **kwargs)
    draws = np.random.default_rng(71).multivariate_normal([4.,3.], covariance, size=64)
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    expected = draws[:,0]*sigma*np.sqrt(2*np.pi)/draws[:,1]
    np.testing.assert_allclose(result.std, np.std(expected,ddof=1), rtol=1e-12)
    assert result.std == again.std
    assert result.value == again.value
    assert selected.eqw(continuum=3.).value == result.value


@pytest.mark.parametrize('method', ['measure','sample'])
def test_eqw_descending_quantity_grid(method):
    from astropy import units as u
    line = line_model()
    line.amplitude.std = .1
    selected = select_line(line, 'line')
    x = np.linspace(4980.,5020.,1001)*u.AA
    function = selected.eqw if method == 'measure' else selected.sample_eqw
    kwargs = {} if method == 'measure' else dict(method='std', random_state=71,n_samples=8)
    a = function(continuum=3., x=x, **kwargs)
    b = function(continuum=3., x=x[::-1], **kwargs)
    assert a.value == b.value
    assert a.std == b.std


def test_expanded_cube_covariance_uses_explicit_free_indices():
    line = line_model()
    line._param_cov = np.full((len(line.param_names),)*2, np.nan)
    line._param_cov[0,0] = .04
    line._fit_indices = np.array([0])
    result = select_line(line, 'line').sample(method='covariance', n_samples=16, random_state=71)
    assert result.flux.std > 0.


def test_sampled_eqw_keeps_explicit_member_of_tied_template():
    from prism.modeling.models.lines import GaussianLines
    group = GaussianLines.from_arrays(names=['pair','pair'], pos=[5000.,5010.],
                                      weights=[1.,.5], amplitude=4., fwhm=200., position_unit='AA')
    group.offset.fixed = True
    group.fwhm.fixed = True
    group.amp_pair.std = .2
    selected = select_line(group, 'pair', position=5010.)
    result = selected.sample_eqw(continuum=3., method='std', distribution='gaussian',
                                 n_samples=64, random_state=71)
    sigma = 5010.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    draws = np.random.default_rng(71).normal(4.,.2,64)
    expected = .5*draws*sigma*np.sqrt(2*np.pi)/3.
    np.testing.assert_allclose(result.value, .5*4.*sigma*np.sqrt(2*np.pi)/3.,rtol=1e-12)
    np.testing.assert_allclose(result.std, np.std(expected,ddof=1),rtol=1e-12)


def test_covariance_with_stale_free_indices_is_rejected():
    line = line_model()
    line._param_cov = np.array([[.04]])
    line._fit_indices = [1]
    with pytest.raises(ValueError, match='indices'):
        select_line(line,'line').sample(method='covariance', n_samples=8)


@pytest.mark.parametrize('n_samples', [0,-1,1.5])
def test_sampling_requires_positive_integer_count(n_samples):
    line = line_model()
    line.amplitude.std = .2
    with pytest.raises(ValueError,match='n_samples'):
        select_line(line,'line').sample(n_samples=n_samples)


def test_explicit_response_selection_retains_driver_covariance_and_intrinsic_flux():
    from scipy.sparse import eye
    from prism.modeling.operators import SpectralResponse
    x = np.linspace(4980., 5020., 201)
    source = line_model() + Const1D(2., name='driver')
    source.amplitude_0.tied = lambda m: 2*m.amplitude_1
    model = source | SpectralResponse(response_matrix=.5*eye(len(x), format='csr'), wave=x)
    model._param_cov = np.array([[.04]])
    original = model.parameters.copy()
    with pytest.raises(ValueError, match='components='):
        select_line(model, 'line')
    selected = select_line(model, 'line', components='line')
    assert selected.source_model is model
    result, samples = selected.sample(n_samples=64, method='covariance', random_state=71, return_samples=True)
    draws = np.random.default_rng(71).multivariate_normal([2.], [[.04]], size=64)[:, 0]
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    np.testing.assert_allclose(samples['flux'], 2*draws*sigma*np.sqrt(2*np.pi), rtol=1e-12)
    np.testing.assert_allclose(selected(5000.), 4.)
    np.testing.assert_array_equal(model.parameters, original)
    assert result.flux.std > 0
    with pytest.raises(ValueError, match='explicit intrinsic continuum'):
        selected.eqw()
    np.testing.assert_allclose(selected.eqw(continuum=2.).value, 2*sigma*np.sqrt(2*np.pi), rtol=1e-12)


def test_sampling_uses_ordered_nonlinear_link_chain_from_shared_mapping():
    model = Const1D(2., name='driver') + Const1D(4., name='middle') + line_model()
    model.amplitude_1.tied = lambda m: m.amplitude_0**2
    model.amplitude_2.tied = lambda m: 3*m.amplitude_1
    model.amplitude_2 = 12.
    model._param_cov = np.array([[.01]])
    selected = select_line(model, 'line')
    _, samples = selected.sample(n_samples=64, method='covariance', random_state=71, return_samples=True)
    draws = np.random.default_rng(71).multivariate_normal([2.], [[.01]], size=64)[:, 0]
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    np.testing.assert_allclose(samples['flux'], 3*draws**2*sigma*np.sqrt(2*np.pi), rtol=1e-12)


@pytest.mark.usefixtures('astropy_extensions')
def test_serialized_callable_link_requires_explicit_reconstruction(tmp_path):
    from prism.modeling.io import load_model
    model = line_model() + Const1D(2., name='driver')
    model.amplitude_0.tied = lambda m: 2*m.amplitude_1
    path = tmp_path/'linked.fits'
    model.save(path)
    with pytest.warns(UserWarning, match='Python links are not serialized'):
        loaded = load_model(path)
    assert not loaded.amplitude_0.tied
    loaded.amplitude_0.tied = lambda m: 2*m.amplitude_1
    loaded.amplitude_1 = 3.
    from prism.modeling.fitting import apply_tied
    apply_tied(loaded)
    assert loaded.amplitude_0.value == 6.
