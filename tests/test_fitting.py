from prism import experimental
from prism.modeling.interop import astropy as astropy_fitting
from line_helpers import make_line
import os
import tempfile
import warnings

import numpy as np
import pytest
from astropy.modeling import fitting as ast_fitting, models

from prism.data import Cube
import prism.modeling.fitting as prism_fitting
from prism.modeling import select_line
from prism.modeling.interop.astropy import TRFLSQFitter
from prism.modeling.fitting.scipyfit import TRF
from prism.modeling.fitting.uncertainty import covar_attach
from prism.modeling.io import load_model
from prism.modeling.models import BSpline, FixedTemplate


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
    model = make_line(amplitude=10.0, position=5007.0, offset=50.0, fwhm=300.0)
    yerr = np.full_like(x, 0.3)
    y = model(x) + rng.normal(0.0, 0.3, size=x.size)
    return x, y, yerr, model


def _build_separable_empirical_dataset(seed=21):
    rng = np.random.default_rng(seed)
    x = np.linspace(4860.0, 5150.0, 180)
    spline = BSpline(
        np.concatenate(([x[0]] * 3, np.arange(x[0] + 40.0, x[-1], 40.0), [x[-1]] * 3)),
        degree=3,
        name='spl',
    )
    aspline = BSpline(
        np.concatenate(([x[0]] * 3, np.linspace(x[0], x[-1], 3), [x[-1]] * 3)),
        degree=3,
        name='aspl',
    )
    template = FixedTemplate(x, 2.0 + 0.2 * np.sin((x - x.mean()) / 18.0), name='tmpl')
    truth = spline * template + aspline
    truth.parameters = rng.normal(scale=0.5, size=truth.parameters.size)
    yerr = np.full_like(x, 0.05)
    y = truth(x) + rng.normal(0.0, yerr[0], size=x.size)
    init = truth.copy()
    init.parameters = np.ones(init.parameters.size)
    return x, y, yerr, truth, init


def _warning_messages(records):
    return [str(w.message).lower() for w in records]


def test_native_astropy_fitters_remain_unmodified():
    fitter = ast_fitting.TRFLSQFitter()
    assert not hasattr(fitter, 'multifit')
    assert not hasattr(ast_fitting, 'set_max_evaluations')
    assert not hasattr(ast_fitting, 'get_max_evaluations')


def test_prism_trf_wrapper_supports_single_fit_and_multifit():
    rng = np.random.default_rng(3)
    fitter = astropy_fitting.TRFLSQFitter()
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


def test_multifit_uses_cube_validity_mask_and_skips_fully_masked_spectra():
    fitter = astropy_fitting.TRFLSQFitter()
    model = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.0)
    x = np.linspace(-2.0, 2.0, 81)
    y = model(x)
    values = np.stack([y, y], axis=1)[:, :, None]
    err = np.full_like(values, 0.05)
    cube = Cube(values=values, z=x, err=err)
    cube.mask[:, 1, 0] = False

    result = fitter.multifit(model.copy(), x, cube, progress=False, nproc=1)

    assert result.shape == (2, 1)
    assert result.success[0, 0]
    assert not result.success[1, 0]
    assert np.isfinite(result.amplitude.value[0, 0])
    assert np.isnan(result.amplitude.value[1, 0])
    assert 'skipped' in result.messages[1, 0]


def test_multifit_skip_invalid_false_raises_for_fully_masked_spectrum():
    fitter = astropy_fitting.TRFLSQFitter()
    model = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.0)
    x = np.linspace(-2.0, 2.0, 81)
    y = model(x)
    values = np.stack([y, y])
    mask = np.ones_like(values, dtype=bool)
    mask[1, :] = False

    with pytest.raises(ValueError, match='No valid spectral samples remain after masking'):
        fitter.multifit(model.copy(), x, values, mask=mask, skip_invalid=False, progress=False, nproc=1)


def test_multifit_ignores_non_finite_initpar_overrides():
    fitter = astropy_fitting.TRFLSQFitter()
    model = models.Linear1D(slope=0.5, intercept=1.0)
    x = np.linspace(-2.0, 2.0, 81)
    y = model(x)
    values = np.stack([y, y])
    yerr = np.full_like(values, 0.05)
    initpars = {
        'slope': np.array([np.nan, 0.4]),
        'intercept': np.array([np.nan, 0.9]),
    }

    result = fitter.multifit(
        model.copy(), x, values, yerr=yerr, initpars=initpars, progress=False, nproc=1
    )

    assert np.all(result.success)
    assert np.isfinite(result.slope.value).all()


def test_prism_trf_max_evaluations_controls():
    x, y, init = _build_gaussian_dataset()

    with warnings.catch_warnings(record=True) as rec_low:
        warnings.simplefilter('always')
        low = astropy_fitting.TRFLSQFitter()
        low(init, x, y, max_evaluations=1)

    low_warnings = _warning_messages(rec_low)
    assert any('maximum number of function evaluations is exceeded' in m for m in low_warnings)
    assert low.fit_info['nfev'] == 1

    with warnings.catch_warnings(record=True) as rec_high:
        warnings.simplefilter('always')
        high = astropy_fitting.TRFLSQFitter()
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
            global_limited = astropy_fitting.TRFLSQFitter()
            global_limited(init, x, y)

        global_warnings = _warning_messages(rec_global)
        assert any('maximum number of function evaluations is exceeded' in m for m in global_warnings)
        assert global_limited.fit_info['nfev'] == 1
    finally:
        prism_fitting.set_max_evaluations(previous)


@pytest.mark.usefixtures('astropy_extensions')
def test_covariance_attach_and_roundtrip_from_saved_model():
    x, y, yerr, model = _build_covariance_dataset()
    fitter = TRF(covariance=True)
    result = fitter(model.copy(), x, y, yerr=yerr)
    fitted = result.model
    covar_attach(fitted, result)

    assert fitter.fit_info['param_cov'] is not None
    assert hasattr(fitted, '_param_cov')
    assert hasattr(fitted.amplitude, 'std') and fitted.amplitude.std > 0
    assert np.isfinite(select_line(fitted, 'line').sample(n_samples=64, method='covariance', random_state=7).flux.std) and select_line(fitted, 'line').sample(n_samples=64, method='covariance', random_state=7).flux.std > 0

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
    x, y, model = _build_gaussian_dataset(seed=12)
    fitter = TRFLSQFitter(calc_uncertainties=True)
    fitted = fitter(model.copy(), x, y, yerr=np.full_like(x,.03))
    assert fitter.fit_info['param_cov'] is not None
    assert hasattr(fitted, '_param_cov')
    assert np.isfinite(fitted.amplitude.std)


def test_separable_trf_matches_exact_linear_solution_for_empirical_compound():
    x, y, yerr, _, init = _build_separable_empirical_dataset()

    separable = experimental.SeparableTRF()
    fitted_sep = separable(init.copy(), x, y, yerr=yerr, inplace=False)

    plain = astropy_fitting.TRFLSQFitter()
    fitted_plain = plain(init.copy(), x, y, yerr=yerr, inplace=False)

    # Use the linear estimator, not iterative stopping, as the coefficient reference.
    basis_model = init.copy()
    basis_model.parameters = np.zeros(init.parameters.size)
    offset = basis_model(x)
    columns = []
    for index in range(init.parameters.size):
        coefficients = np.zeros(init.parameters.size)
        coefficients[index] = 1.
        basis_model.parameters = coefficients
        columns.append(basis_model(x) - offset)
    design = np.column_stack(columns) / yerr[:, None]
    expected = np.linalg.lstsq(design, (y - offset) / yerr, rcond=None)[0]
    np.testing.assert_allclose(fitted_sep.parameters, expected, atol=1e-8, rtol=1e-8)
    expected_stat = np.sum((design @ expected - (y - offset) / yerr)**2)
    for fitted in (fitted_sep, fitted_plain):
        assert np.sum(((fitted(x) - y) / yerr)**2) == pytest.approx(expected_stat, rel=1e-8)
    assert separable.fit_info['n_linear'] == len(init.param_names)
    assert separable.fit_info['n_nonlinear'] == 0


def test_inspect_separable_reports_linear_and_nonlinear_blocks():
    model = make_line(amplitude=1.0, position=5007.0, offset=0.0, fwhm=200.0)

    info = experimental.inspect_separable(model)

    assert info['supported']
    assert info['linear_params'] == ('amplitude',)
    assert info['nonlinear_params'] == ('offset', 'fwhm')
    assert info['fixed_params'] == ('position', 'redshift')
    assert info['tied_params'] == ()


def test_separable_trf_solves_gaussian_amplitude_inside_outer_nonlinear_fit():
    rng = np.random.default_rng(11)
    x = np.linspace(4990.0, 5025.0, 240)
    truth = make_line(amplitude=12.0, position=5007.0, offset=35.0, fwhm=240.0)
    yerr = np.full_like(x, 0.12)
    y = truth(x) + rng.normal(0.0, yerr[0], size=x.size)

    init = make_line(amplitude=1.0, position=5007.0, offset=-120.0, fwhm=500.0)
    fitter = experimental.SeparableTRF()
    fitted = fitter(init, x, y, yerr=yerr, inplace=False)

    assert fitter.fit_info['n_linear'] == 1
    assert fitter.fit_info['n_nonlinear'] == 2
    assert fitter.fit_info['outer_jacobian'] == 'analytic'
    assert np.isclose(fitted.amplitude.value, truth.amplitude.value, rtol=0.05)
    assert np.isclose(fitted.offset.value, truth.offset.value, atol=10.0)
    assert np.isclose(fitted.fwhm.value, truth.fwhm.value, atol=40.0)


@pytest.mark.usefixtures('astropy_extensions')
def test_separable_trf_exposes_covariance_for_fully_linear_case():
    x, y, yerr, _, init = _build_separable_empirical_dataset(seed=27)

    fitter = experimental.SeparableTRF(covariance=False)
    fitted = fitter(init.copy(), x, y, yerr=yerr, inplace=False)

    assert fitter.fit_info['param_cov'] is not None
    assert hasattr(fitted, '_param_cov')
    assert fitted.covariance is not None
    assert np.all(np.isfinite(np.diag(fitted.covariance)))
    assert all(np.isfinite(getattr(fitted, name).std) for name in fitted.param_names)


def test_separable_trf_respects_linear_parameter_bounds():
    x = np.linspace(4990.0, 5025.0, 200)
    truth = make_line(amplitude=4.0, position=5007.0, offset=0.0, fwhm=200.0)
    yerr = np.full_like(x, 0.05)
    y = truth(x)

    init = make_line(amplitude=1.0, position=5007.0, offset=0.0, fwhm=200.0)
    init.amplitude.bounds = (0.0, 1.5)
    fitter = experimental.SeparableTRF()
    fitted = fitter(init, x, y, yerr=yerr, inplace=False)

    assert fitter.fit_info['outer_jacobian'] == '2-point'
    assert np.isclose(fitted.amplitude.value, 1.5, atol=1e-8)


def test_separable_trf_supports_tied_linear_amplitudes():
    x = np.linspace(4990.0, 5025.0, 220)
    left = make_line(amplitude=3.0, position=5000.0, offset=0.0, fwhm=160.0, name='left')
    right = make_line(amplitude=1.5, position=5010.0, offset=0.0, fwhm=160.0, name='right')
    truth = left + right
    truth.amplitude_1.tied = lambda m: 0.5 * m.amplitude_0

    yerr = np.full_like(x, 0.03)
    y = truth(x)

    init_left = make_line(amplitude=1.0, position=5000.0, offset=0.0, fwhm=160.0, name='left')
    init_right = make_line(amplitude=0.2, position=5010.0, offset=0.0, fwhm=160.0, name='right')
    model = init_left + init_right
    model.offset_0.fixed = True
    model.offset_1.fixed = True
    model.fwhm_0.fixed = True
    model.fwhm_1.fixed = True
    model.amplitude_1.tied = lambda m: 0.5 * m.amplitude_0

    fitter = experimental.SeparableTRF()
    fitted = fitter(model, x, y, yerr=yerr, inplace=False)

    assert fitter.fit_info['n_linear'] == 1
    assert np.isclose(fitted.amplitude_0.value, 3.0, atol=1e-6)
    assert np.isclose(fitted.amplitude_1.value, 1.5, atol=1e-6)


def test_separable_trf_uses_projected_analytic_jacobian_with_smooth_ties():
    x = np.linspace(4990.0, 5025.0, 220)
    left = make_line(amplitude=4.0, position=5000.0, offset=0.0, fwhm=170.0, name='left')
    right = make_line(amplitude=2.0, position=5010.0, offset=0.0, fwhm=170.0, name='right')
    truth = left + right
    truth.amplitude_1.tied = lambda m: 0.5 * m.amplitude_0
    truth.fwhm_1.tied = lambda m: m.fwhm_0

    yerr = np.full_like(x, 0.04)
    y = truth(x)

    init_left = make_line(amplitude=0.8, position=5000.0, offset=0.0, fwhm=320.0, name='left')
    init_right = make_line(amplitude=0.1, position=5010.0, offset=0.0, fwhm=120.0, name='right')
    model = init_left + init_right
    model.offset_0.fixed = True
    model.offset_1.fixed = True
    model.amplitude_1.tied = lambda m: 0.5 * m.amplitude_0
    model.fwhm_1.tied = lambda m: m.fwhm_0

    fitter = experimental.SeparableTRF()
    fitted = fitter(model, x, y, yerr=yerr, inplace=False)

    assert fitter.fit_info['outer_jacobian'] == 'analytic'
    assert np.isclose(fitted.amplitude_0.value, 4.0, atol=1e-4)
    assert np.isclose(fitted.amplitude_1.value, 2.0, atol=1e-4)
    assert np.isclose(fitted.fwhm_0.value, 170.0, atol=1e-2)
    assert np.isclose(fitted.fwhm_1.value, 170.0, atol=1e-2)


def test_separable_trf_rejects_nonlinear_ties_inside_linear_block():
    x = np.linspace(4990.0, 5025.0, 220)
    left = make_line(amplitude=2.0, position=5000.0, offset=0.0, fwhm=160.0, name='left')
    right = make_line(amplitude=4.0, position=5010.0, offset=0.0, fwhm=160.0, name='right')
    truth = left + right
    y = truth(x)

    init_left = make_line(amplitude=1.0, position=5000.0, offset=0.0, fwhm=160.0, name='left')
    init_right = make_line(amplitude=0.5, position=5010.0, offset=0.0, fwhm=160.0, name='right')
    model = init_left + init_right
    model.offset_0.fixed = True
    model.offset_1.fixed = True
    model.fwhm_0.fixed = True
    model.fwhm_1.fixed = True
    model.amplitude_1.tied = lambda m: m.amplitude_0 ** 2

    fitter = experimental.SeparableTRF()
    with pytest.raises(ValueError, match='do not preserve linear superposition'):
        fitter(model, x, y, inplace=False)


def test_separable_trf_raises_when_no_linear_block_is_available():
    x, y, init = _build_gaussian_dataset()
    fitter = experimental.SeparableTRF()
    with pytest.raises(ValueError, match='no free linear parameters'):
        fitter(init, x, y, inplace=False)


@pytest.mark.usefixtures('astropy_extensions')
def test_linear_lsq_wrapper_exposes_covariance_and_std_by_default():
    x = np.linspace(-2.0, 2.0, 50)
    y = 3.5 * x - 1.2

    fitter = astropy_fitting.LinearLSQFitter(calc_uncertainties=False)
    fitted = fitter(models.Linear1D(), x, y)

    assert not fitter.calc_uncertainties
    assert fitter.fit_info['param_cov'] is not None
    assert fitted.covariance is not None
    assert np.isfinite(fitted.slope.std)
    assert np.isfinite(fitted.intercept.std)


@pytest.mark.skipif(not prism_fitting.HAS_SHERPA, reason='Sherpa is not installed')
def test_sherpa_wrapper_requires_feasible_starts_and_honors_inplace():
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
    with pytest.raises(ValueError, match='Initial values'):
        fitter(model, x, y, yerr=yerr, inplace=True)
    assert model.mean.value == 2.5
    model.mean.value = 0.
    fitted = fitter(model, x, y, yerr=yerr, inplace=True)

    assert fitted.model is not model
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

    with pytest.raises(ValueError, match='Initial values'):
        fitter(untouched, x, y, yerr=yerr, inplace=False)
    assert untouched.mean.value == 2.5
    untouched.mean.value = 0.
    separate = fitter(untouched, x, y, yerr=yerr, inplace=False)
    assert separate is not untouched
    assert untouched.mean.value == 0.
    assert separate.model.mean.bounds[0] <= separate.model.mean.value <= separate.model.mean.bounds[1]
