import warnings
import numpy as np
import pytest
from astropy.modeling.models import Const1D, Linear1D
from prism.modeling.fitting import TRF
from prism.modeling.fitting.uncertainty.resample import bootstrap, prepare_noise_and_weights, ResampleError

@pytest.mark.parametrize("statistic", ["gauss", "poisson"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_resampling_recovers_independent_weighted_mean(statistic, weights_only):
    x = np.arange(3.)
    y = np.array([4., 7., 12.])
    errors = np.array([0.4, 1.2, 2.5])
    rng = np.random.default_rng(81)
    draws = rng.normal(y, errors, size=(16, 3)) if statistic == "gauss" else rng.poisson(y, size=(16, 3))
    expected = (draws / errors**2).sum(axis=1) / (1 / errors**2).sum()
    kwargs = {"weights": 1 / errors} if weights_only else {"yerr": errors}
    with warnings.catch_warnings(record=True) as caught:
        samples = bootstrap(Const1D(-30.), TRF(), x=x, y=y, n_samples=16,
                            seed=81, statistic=statistic, **kwargs)
    np.testing.assert_allclose(samples.samples["amplitude"], expected, rtol=1e-7, atol=1e-7)
    assert not any("barely" in str(item.message) or "reoptimiz" in str(item.message) for item in caught)

def test_excluded_values_and_errors_do_not_enter_noise():
    raw, effective, weights = prepare_noise_and_weights([2., np.nan], [0.3, np.nan], [2., 0.], "gauss")
    np.testing.assert_equal(raw, [0.3, 0.])
    np.testing.assert_equal(weights, [2., 0.])
    samples = bootstrap(Const1D(-10), TRF(), x=[0., 1.], y=[2., np.nan],
                        yerr=[0.3, np.nan], weights=[2., 0.], n_samples=8, seed=7, )
    expected = np.random.default_rng(7).normal([2., 0.], [0.3, 0.], size=(8, 2))[:, 0]
    np.testing.assert_allclose(samples.samples["amplitude"], expected, atol=1e-7)

@pytest.mark.parametrize("errors, weights", [([-1., 1.], None), ([1.], None), ([1., 1.], [-1., 1.]), ([1., 1.], [0., 0.])])
def test_invalid_noise_contract(errors, weights):
    with pytest.raises(ResampleError):
        prepare_noise_and_weights([1., 2.], errors, weights, "gauss")

def test_poisson_rejects_negative_counts():
    with pytest.raises(ResampleError):
        prepare_noise_and_weights([-1., 2.], None, None, "poisson")

def test_numerical_fit_rejects_unsupported_dimension_and_statistic():
    fitter = TRF()
    with pytest.raises(NotImplementedError):
        fitter(Const1D(), [0., 1.], [1., 2.], z=[1., 2.])
    with pytest.raises(NotImplementedError):
        fitter(Const1D(), [0., 1.], [1., 2.], statistic="poisson")

def test_robust_stat_is_squared_residual_not_robust_cost():
    x = np.arange(10.)
    y = 2 * x + 1
    y[-1] += 40
    fitter = TRF()
    fitted = fitter(Linear1D(), x, y, loss="soft_l1")
    expected = np.sum((fitted.predict(x) - y)**2)
    assert fitter.fit_info["stat"] == pytest.approx(expected)
    assert fitter.fit_info["stat"] != pytest.approx(2 * fitter.fit_info["cost"])


def test_x_perturbation_does_not_count_failed_fits():
    class FailedFitter:
        fit_info = {"success": False, "message": "Evaluation budget exhausted"}

        def __call__(self, model, *args, **kwargs):
            return model

    with pytest.warns(RuntimeWarning, match="Only 0.0%"):
        samples = bootstrap(Const1D(), FailedFitter(), x=[0., 1.], y=[1., 2.],
                            yerr=[0.2, 0.3], xerr=[0.1, 0.1], perturb_x=True,
                            n_samples=3, seed=1, )
    assert np.all(np.isnan(samples.samples["amplitude"]))
