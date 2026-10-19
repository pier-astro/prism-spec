"""Independent finite-window Gaussian moments and grid contracts."""
import numpy as np
import pytest
from astropy import units as u
from prism.modeling.models.lines import GaussianLine, profiles
from prism.modeling.line_analysis import select_line


def test_gaussian_moments_and_descending_quantity_grid():
    line = GaussianLine(position=5000., amplitude=2., fwhm=200., name="line")
    selected = select_line(line, "line")
    sigma = 5000. * 200. / profiles.C_KMS / profiles.SIGMA2FWHM
    x = np.linspace(5000. - 10*sigma, 5000. + 10*sigma, 10001)
    ascending = selected.measure(x=x)
    descending = selected.measure(x=x[::-1]*u.AA)
    np.testing.assert_allclose(ascending.flux.value, 2.*sigma*np.sqrt(2*np.pi), rtol=1e-12)
    np.testing.assert_allclose(ascending.moment1.value, 5000., atol=1e-10)
    np.testing.assert_allclose(ascending.sigma.value, sigma, rtol=1e-10)
    np.testing.assert_allclose(ascending.skewness.value, 0., atol=1e-10)
    np.testing.assert_allclose(ascending.kurtosis.value, 0., atol=1e-10)
    np.testing.assert_allclose(ascending.fwhm.value, sigma*profiles.SIGMA2FWHM, rtol=1e-6)
    for name in ascending.metric_units:
        assert getattr(ascending, name).value == getattr(descending, name).value
    windowed = selected.measure(window=(499.*u.nm, 501.*u.nm), num=10001)
    np.testing.assert_allclose(windowed.flux.value, ascending.flux.value, rtol=1e-10)


@pytest.mark.parametrize('x', [[1.,2.], [1.,2.,2.], [1.,3.,2.], [1.,np.nan,3.], [1.,2.,np.inf]])
@pytest.mark.parametrize('sample', [False, True])
def test_measurement_and_sampling_reject_invalid_grid_before_sampling(x, sample):
    selected = select_line(GaussianLine(name="line"), "line")
    operation = selected.sample if sample else selected.measure
    with pytest.raises(ValueError, match='spectral grid|monotonic'):
        operation(x=x)


def test_sampling_uses_same_grid_orientation_and_leaves_model_unchanged():
    line = GaussianLine(position=5000., amplitude=2., fwhm=200., name="line")
    line.amplitude.std = .1
    selected = select_line(line, "line")
    x = np.linspace(4980., 5020., 1001)
    initial = line.parameters.copy()
    first, samples = selected.sample(x=x, n_samples=12, method='std',
                                     random_state=21, return_samples=True)
    second, reverse_samples = selected.sample(x=x[::-1], n_samples=12, method='std',
                                              random_state=21, return_samples=True)
    np.testing.assert_array_equal(line.parameters, initial)
    np.testing.assert_array_equal(samples.to_numpy(), reverse_samples.to_numpy())
    assert first.flux.value == second.flux.value
    assert first.flux.std == second.flux.std
