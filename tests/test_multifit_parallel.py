"""Serial/real-spawn equivalence for masked, linked spectra and bin maps."""
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from prism.modeling.fitting import ScipyTRF, SherpaLM, HAS_SHERPA
from prism.modeling.operators.instrument import SpectralResponse
from scipy.sparse import eye


@pytest.mark.parametrize('backend', [ScipyTRF, pytest.param(
    SherpaLM, marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_parallel_response_links_masks_covariance_and_overrides(backend):
    x = np.linspace(-4, 4, 81)
    source = Gaussian1D(.6, 0, .6) + Gaussian1D(.2, 1.5, .9)
    for name in ('mean_0', 'stddev_0', 'mean_1', 'stddev_1'):
        getattr(source, name).fixed = True
    source.amplitude_1.tied = lambda m: .4*m.amplitude_0
    model = source | SpectralResponse(response_matrix=eye(len(x), format='csr'), wave=x)
    original = model.parameters.copy()
    amplitude = np.linspace(1, 2, 6).reshape(2, 3)
    profile = np.exp(-.5*(x/.6)**2) + .4*np.exp(-.5*((x-1.5)/.9)**2)
    data = profile[:, None, None]*amplitude
    mask = np.ones_like(data, dtype=bool)
    mask[12:20, 0, 1] = False
    mask[:, 1, 1] = False
    data[:, 1, 0] = np.nan  # A local failure must not poison the next task.
    options = dict(yerr=np.full_like(data, .1), mask=mask, spectral_axis=0,
                   initpars={'amplitude_0': np.full((2, 3), .5)},
                   bounds={'amplitude_0': (np.zeros((2, 3)), 4.)}, progress=False)
    serial = backend(calc_uncertainties=True).multifit(model, x, data, nproc=1, **options)
    parallel = backend(calc_uncertainties=True).multifit(model, x, data, nproc=2, **options)
    np.testing.assert_array_equal(model.parameters, original)
    expected = np.array([[True, True, True], [False, False, True]])
    np.testing.assert_array_equal(parallel._success.reshape(2, 3), expected)
    np.testing.assert_array_equal(parallel._success, serial._success)
    for name in model.param_names:
        np.testing.assert_allclose(parallel._params[name], serial._params[name], atol=1e-11, rtol=1e-11, equal_nan=True)
        np.testing.assert_allclose(parallel._errs[name], serial._errs[name], atol=1e-11, rtol=1e-11, equal_nan=True)
    np.testing.assert_allclose(parallel.covariance, serial.covariance, atol=1e-11, rtol=1e-11, equal_nan=True)
    np.testing.assert_allclose(parallel._params['amplitude_0'][expected.ravel()], amplitude[expected], atol=1e-6)
    np.testing.assert_allclose(parallel.nfev, serial.nfev, equal_nan=True)
    np.testing.assert_allclose(parallel.get_stat('cost'), serial.get_stat('cost'), equal_nan=True)
    assert parallel.fit_info['n_success'] == 4


def test_parallel_binmap_fits_representatives_and_preserves_exclusions():
    x = np.linspace(-3, 3, 61)
    model = Gaussian1D(.5, 0, .8, fixed={'mean': True, 'stddev': True})
    amplitudes = np.array([[1., 9., 2.], [9., 3., 9.]])
    data = amplitudes[..., None]*np.exp(-.5*(x/.8)**2)
    bins = np.array([[2, 2, 7], [7, 12, -1]])
    serial = ScipyTRF().multifit(model, x, data, binmap=bins, progress=False)
    parallel = ScipyTRF().multifit(model, x, data, binmap=bins, nproc=2, progress=False)
    expected = np.array([[1., 1., 2.], [2., 3., np.nan]])
    np.testing.assert_allclose(parallel.amplitude.value, expected, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(parallel.amplitude.value, serial.amplitude.value, equal_nan=True)


def test_parallel_strict_mask_failure_propagates():
    x = np.linspace(-3, 3, 31)
    data = np.ones((2, len(x)))
    mask = np.ones_like(data, dtype=bool)
    mask[1] = False
    with pytest.raises(ValueError, match='No valid spectral samples'):
        ScipyTRF().multifit(Gaussian1D(), x, data, mask=mask,
                           skip_invalid=False, nproc=2, progress=False)


@pytest.mark.parametrize('nproc', [1, 2])
def test_shuffled_bin_labels_keep_first_spatial_member_and_its_overrides(nproc):
    x = np.linspace(-3, 3, 51)
    rng = np.random.default_rng(20261018)
    bins = rng.choice([-1, 0, 19, 1000000000], size=(7, 8))
    amplitude = np.linspace(.8, 3., bins.size).reshape(bins.shape)
    data = amplitude[:, None, :] * np.exp(-.5*(x[None, :, None]/.7)**2)
    expected = np.full(bins.shape, np.nan)
    mask = np.zeros_like(data, dtype=bool)
    for label in (0, 19, 1000000000):
        first = tuple(np.argwhere(bins == label)[0])
        expected[bins == label] = amplitude[first]
        mask[first[0], :, first[1]] = True
    result = ScipyTRF(calc_uncertainties=True).multifit(
        Gaussian1D(.4, 0, .7, fixed={'mean': True, 'stddev': True}),
        x, data, mask=mask, binmap=bins, spectral_axis=1, nproc=nproc,
        initpars={'amplitude': np.full(bins.shape, .3)},
        bounds={'amplitude': (0., amplitude + .5)}, progress=False)
    np.testing.assert_allclose(result.amplitude.value, expected, atol=1e-7, equal_nan=True)
    np.testing.assert_array_equal(result._success.reshape(bins.shape), bins >= 0)


def test_binmap_with_no_included_targets_is_rejected():
    x = np.linspace(-3, 3, 21)
    with pytest.raises(ValueError, match='non-negative bin labels'):
        ScipyTRF().multifit(Gaussian1D(), x, np.ones((2, len(x))),
                           binmap=np.array([-1, -3]), nproc=2, progress=False)
