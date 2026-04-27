import os
import tempfile
import warnings

import numpy as np
import pytest
from astropy.modeling import fitting as ast_fitting, models

import prism.modeling.fitting as prism_fitting
from prism.modeling import select_line
from prism.modeling.fitting import TRFLSQFitter
from prism.modeling.fitting.scipyfit import ScipyTRF
from prism.modeling.fitting.uncertainty import covar_attach
from prism.modeling.io import load_model
from prism.modeling.models.lines import GaussianLine


def _build_gaussian_dataset(seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, 400)
    truth = models.Gaussian1D(amplitude=5.0, mean=0.35, stddev=0.65)
    y = truth(x) + 0.03 * rng.normal(size=x.size)
    init = models.Gaussian1D(amplitude=0.2, mean=-3.0, stddev=4.0)
    return x, y, init


def _build_covariance_dataset(seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(4800.0, 5200.0, 200)
    model = GaussianLine(amplitude=10.0, position=5007.0, offset=50.0, fwhm=300.0)
    yerr = np.full_like(x, 0.3)
    y = model(x) + rng.normal(0.0, 0.3, size=x.size)
    return x, y, yerr, model


def _warning_messages(records):
    return [str(w.message).lower() for w in records]


def test_native_astropy_fitters_remain_unmodified():
    fitter = ast_fitting.TRFLSQFitter()
    assert not hasattr(fitter, 'multifit')
    assert not hasattr(ast_fitting, 'set_max_evaluations')
    assert not hasattr(ast_fitting, 'get_max_evaluations')


def test_prism_trf_wrapper_supports_single_fit_and_multifit():
    rng = np.random.default_rng(3)
    fitter = prism_fitting.TRFLSQFitter()
    model = models.Gaussian1D(amplitude=10.0, mean=5.0, stddev=1.0)

    x = np.linspace(0.0, 10.0, 100)
    y = model(x) + rng.normal(0.0, 0.1, size=x.size)
    yerr = np.full_like(x, 0.1)
    fitted = fitter(model, x, y, yerr=yerr)
    assert hasattr(fitted, 'amplitude')

    y_cube = np.stack([y, 1.01 * y])
    result = fitter.multifit(model, x, y_cube, progress=False)
    assert result.shape == (2,)
    assert np.all(result.success)


def test_prism_trf_max_evaluations_controls():
    x, y, init = _build_gaussian_dataset()

    with warnings.catch_warnings(record=True) as rec_low:
        warnings.simplefilter('always')
        low = prism_fitting.TRFLSQFitter()
        low(init, x, y, max_evaluations=1)

    low_warnings = _warning_messages(rec_low)
    assert any('maximum number of function evaluations is exceeded' in m for m in low_warnings)
    assert low.fit_info['nfev'] == 1

    with warnings.catch_warnings(record=True) as rec_high:
        warnings.simplefilter('always')
        high = prism_fitting.TRFLSQFitter()
        high.max_evaluations = 4000
        high(init, x, y)

    high_warnings = _warning_messages(rec_high)
    assert not any('maximum number of function evaluations is exceeded' in m for m in high_warnings)
    assert bool(high.fit_info.get('success', False))

    previous = prism_fitting.get_max_evaluations()
    try:
        prism_fitting.set_max_evaluations(1)
        with warnings.catch_warnings(record=True) as rec_global:
            warnings.simplefilter('always')
            global_limited = prism_fitting.TRFLSQFitter()
            global_limited(init, x, y)

        global_warnings = _warning_messages(rec_global)
        assert any('maximum number of function evaluations is exceeded' in m for m in global_warnings)
        assert global_limited.fit_info['nfev'] == 1
    finally:
        prism_fitting.set_max_evaluations(previous)


def test_covariance_attach_and_roundtrip_from_saved_model():
    x, y, yerr, model = _build_covariance_dataset()
    fitter = ScipyTRF(calc_uncertainties=True)
    fitted = fitter(model.copy(), x, y, yerr=yerr)
    covar_attach(fitted, fitter)

    assert fitter.fit_info['param_cov'] is not None
    assert hasattr(fitted, '_param_cov')
    assert hasattr(fitted.amplitude, 'std') and fitted.amplitude.std > 0
    assert np.isfinite(fitted.flux.std) and fitted.flux.std > 0

    path = os.path.join(tempfile.gettempdir(), 'prism_public_cov_roundtrip.fits')
    try:
        fitted.save(path, overwrite=True)
        loaded = load_model(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert loaded.covariance is not None
    assert np.allclose(fitted.covariance, loaded.covariance)
    assert np.isfinite(loaded.amplitude.std)

    selected = select_line(loaded, selector='GaussianLine')
    sampled = selected.sample(n_samples=64, method='covariance', random_state=7)
    assert np.isfinite(sampled.flux.std)


def test_prism_astropy_wrapper_attaches_covariance_without_native_patch():
    x, y, yerr, model = _build_covariance_dataset(seed=12)
    fitter = TRFLSQFitter(calc_uncertainties=True)
    fitted = fitter(model.copy(), x, y, yerr=yerr)

    assert fitter.fit_info['param_cov'] is not None
    assert hasattr(fitted, '_param_cov')
    assert np.isfinite(fitted.flux.std) and fitted.flux.std > 0


@pytest.mark.skipif(not prism_fitting.HAS_SHERPA, reason='Sherpa is not installed')
def test_sherpa_wrapper_honors_inplace_and_replaces_stale_out_of_bounds_state():
    x = np.linspace(-5.0, 5.0, 400)
    truth = models.Gaussian1D(amplitude=3.0, mean=0.4, stddev=0.7)
    y = truth(x)
    yerr = np.full_like(x, 0.1)

    model = models.Gaussian1D(
        amplitude=1.0,
        mean=0.0,
        stddev=1.2,
        bounds={
            'amplitude': (0.0, 10.0),
            'mean': (-1.0, 1.0),
            'stddev': (0.1, 3.0),
        },
    )

    # Simulate notebook state after loading persisted parameters that predate current bounds.
    model.parameters = np.array([2.5, 2.5, 0.8])
    assert model.mean.value == 2.5
    assert model.mean.bounds == (-1.0, 1.0)

    fitter = prism_fitting.SherpaLM()
    fitted = fitter(model, x, y, yerr=yerr)

    assert fitted is model
    assert model.mean.bounds[0] <= model.mean.value <= model.mean.bounds[1]
    assert np.isclose(model.mean.value, truth.mean.value, atol=1e-2)

    untouched = models.Gaussian1D(
        amplitude=1.0,
        mean=0.0,
        stddev=1.2,
        bounds={
            'amplitude': (0.0, 10.0),
            'mean': (-1.0, 1.0),
            'stddev': (0.1, 3.0),
        },
    )
    untouched.parameters = np.array([2.5, 2.5, 0.8])

    separate = fitter(untouched, x, y, yerr=yerr, inplace=False)
    assert separate is not untouched
    assert untouched.mean.value == 2.5
    assert separate.mean.bounds[0] <= separate.mean.value <= separate.mean.bounds[1]