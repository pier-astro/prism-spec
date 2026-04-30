import os
import tempfile
import inspect

import numpy as np
import astropy.units as u
from astropy.modeling import models

import prism
from prism.modeling import (
    LineResult,
    line_analysis,
    measure_line,
    sample_line_measurements,
    select_line,
)
from prism.modeling.fitting import ScipyTRF, TRFLSQFitter
from prism.modeling.fitting.uncertainty import covar_attach
from prism.modeling.io import load_model
from prism.modeling.line_analysis import _draw_selection_samples
from prism.modeling.models import GaussianLine, GaussianLines, get_components


def _build_selection_model():
    hb_narrow = GaussianLines.from_arrays(
        names=['Hb4861'],
        pos=[4861.333],
        name='hb_narrow',
        amplitude=2.0,
        offset=0.0,
        fwhm=300.0,
        position_unit=u.AA,
    )
    hb_broad = GaussianLines.from_arrays(
        names=['Hb4861'],
        pos=[4861.333],
        name='hb_broad',
        amplitude=1.5,
        offset=150.0,
        fwhm=1800.0,
        position_unit=u.AA,
    )
    oiii = GaussianLines.from_arrays(
        names=['[OIII]5007'],
        pos=[5006.803],
        name='oiii',
        amplitude=4.0,
        offset=0.0,
        fwhm=250.0,
        position_unit=u.AA,
    )
    hb_narrow.amp_hb4861.std = 0.10
    hb_broad.amp_hb4861.std = 0.08
    return hb_narrow + hb_broad + oiii


def _build_fitted_model():
    wave = np.linspace(4820.0, 4930.0, 600)
    hb_narrow = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_narrow',
        position_unit=u.AA,
        amplitude=1.0, offset=0.0, fwhm=300.0)
    hb_broad = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_broad',
        position_unit=u.AA,
        amplitude=0.7, offset=60.0, fwhm=1600.0)
    true_model = hb_narrow + hb_broad

    rng = np.random.default_rng(0)
    flux = true_model(wave) + rng.normal(0.0, 0.03, wave.size)
    yerr = np.full_like(wave, 0.03)

    init_narrow = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_narrow',
        amplitude=0.8, offset=10.0, fwhm=420.0,
        position_unit=u.AA,
        bounds={'amplitude': (0, 5), 'offset': (-200, 200), 'fwhm': (80, 800)})
    init_broad = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_broad',
        amplitude=0.5, offset=30.0, fwhm=1200.0,
        position_unit=u.AA,
        bounds={'amplitude': (0, 5), 'offset': (-400, 400), 'fwhm': (700, 4000)})

    fitter = ScipyTRF(calc_uncertainties=True)
    fitted = fitter(init_narrow + init_broad, x=wave, y=flux, weights=1.0 / yerr)
    covar_attach(fitted, fitter)
    return fitted, wave


def _build_multifit_cube(wave):
    continuum = models.Const1D(amplitude=0.12, name='continuum')
    hb_narrow = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_narrow',
        position_unit=u.AA,
        amplitude=1.2, offset=0.0, fwhm=300.0)
    hb_broad = GaussianLines.from_arrays(
        names=['Hb4861'], pos=[4861.333], name='hb_broad',
        position_unit=u.AA,
        amplitude=0.8, offset=85.0, fwhm=1700.0)
    true_model = continuum + hb_narrow + hb_broad

    cube = np.empty((2, 2, wave.size), dtype=float)
    yerr = np.full_like(cube, 0.025)
    rng = np.random.default_rng(11)
    for j in range(2):
        for i in range(2):
            model = true_model.copy()
            model['hb_narrow'].amp_hb4861.value = 1.2 + 0.10 * j + 0.05 * i
            model['hb_broad'].amp_hb4861.value = 0.8 + 0.08 * j + 0.04 * i
            model['continuum'].amplitude.value = 0.12 + 0.01 * j - 0.005 * i
            cube[j, i] = model(wave) + rng.normal(0.0, yerr[j, i], wave.size)

    init_model = (
        models.Const1D(amplitude=0.10, name='continuum')
        + GaussianLines.from_arrays(
            names=['Hb4861'], pos=[4861.333], name='hb_narrow',
            position_unit=u.AA,
            amplitude=1.0, offset=15.0, fwhm=420.0)
        + GaussianLines.from_arrays(
            names=['Hb4861'], pos=[4861.333], name='hb_broad',
            position_unit=u.AA,
            amplitude=0.6, offset=40.0, fwhm=1400.0)
    )
    return cube, yerr, init_model


def test_line_selection_and_measurement_workflow():
    model = _build_selection_model()
    wave = np.linspace(4800.0, 4925.0, 5000)

    selected = select_line(model, 'Hb4861')
    alias = select_line(model, 'Hb4861')
    expected = get_components(model, additive=True)['hb_narrow'](wave) + get_components(model, additive=True)['hb_broad'](wave)
    np.testing.assert_allclose(selected(wave), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(alias(wave), expected, rtol=1e-8, atol=1e-10)

    explicit = select_line(model, 'Hb4861', components=['hb_narrow', 'hb_broad'])
    np.testing.assert_allclose(explicit(wave), expected, rtol=1e-8, atol=1e-10)

    measurements = selected.measure(x=wave)
    alias_measurements = measure_line(selected, x=wave)
    assert isinstance(alias_measurements, LineResult)
    assert measurements.flux.value > 0.0
    assert 4860.0 < measurements.peak_position.value < 4864.0
    assert np.isfinite(measurements.fw10m.value)
    assert prism.modeling.select_line is select_line
    assert line_analysis.select_line is select_line

    component = select_line(model['hb_broad'], 'Hb4861')
    component_measurements = measure_line(component, x=wave)
    assert component_measurements.flux.value > 0.0

    summary, samples = sample_line_measurements(selected, x=wave, n_samples=64, random_state=123, return_samples=True)
    assert np.isfinite(summary.flux.std)
    assert samples.shape[0] == 64


def test_selected_line_position_and_component_breakdown():
    fitted, wave = _build_fitted_model()
    selected = select_line(fitted, 'Hb4861')

    assert abs(selected.position - 4861.333) < 1.0

    component_fluxes = selected.evaluate_components(x=wave)
    assert {'hb_narrow', 'hb_broad'} <= set(component_fluxes)
    np.testing.assert_allclose(sum(component_fluxes.values()), selected.evaluate(wave), rtol=1e-10)


def test_line_measurement_public_docs_and_quantity_axis():
    model = _build_selection_model()
    selected = select_line(model, 'Hb4861')
    wave = np.linspace(4805.0, 4915.0, 2048) * u.AA

    result = measure_line(selected, x=wave)

    assert result.axis_unit == u.AA
    assert result.peak_position.unit == u.AA
    assert result.fwhm.unit == u.AA

    for obj in (measure_line, sample_line_measurements, LineResult):
        doc = inspect.getdoc(obj)
        assert doc is not None
        assert 'Parameters' in doc
        assert 'Returns' in doc


def test_line_sampling_methods_and_bounds():
    fitted, wave = _build_fitted_model()
    selected = select_line(fitted, 'Hb4861')

    covariance_sample = selected.sample(n_samples=32, x=wave, method='covariance', random_state=10)
    gaussian_std_sample = selected.sample(n_samples=32, x=wave, method='std', distribution='gaussian', random_state=10)
    uniform_std_sample = selected.sample(n_samples=32, x=wave, method='std', distribution='uniform', random_state=10)

    assert np.isfinite(covariance_sample.flux.std)
    assert np.isfinite(gaussian_std_sample.flux.std)
    assert np.isfinite(uniform_std_sample.flux.std)

    specs, draws = _draw_selection_samples(selected, n_samples=256, method='std', distribution='gaussian', random_state=0)
    for column, spec in enumerate(specs):
        lower, upper = spec.bounds
        if lower is not None:
            assert np.all(draws[:, column] >= lower)
        if upper is not None:
            assert np.all(draws[:, column] <= upper)


def test_saved_model_retains_uncertainty_for_line_sampling():
    fitted, wave = _build_fitted_model()

    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as handle:
        path = handle.name
    try:
        fitted.save(path, overwrite=True)
        loaded = load_model(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    selected = select_line(loaded, 'Hb4861')
    sampled = selected.sample(n_samples=64, x=wave, random_state=1)
    assert np.isfinite(sampled.flux.std)


def test_line_analysis_multifit_collection_workflow():
    wave = np.linspace(4820.0, 4905.0, 700)
    cube, yerr, init_model = _build_multifit_cube(wave)

    fitter = ScipyTRF(calc_uncertainties=True)
    result = fitter.multifit(init_model, wave, cube, weights=1.0 / yerr, nproc=1, progress=False)

    selected = select_line(result, 'Hb4861')
    evaluated = selected(wave)
    measurements = selected.measure(x=wave)
    direct = measure_line(result, 'Hb4861', x=wave, index=(1, 1))
    sampled, draws = sample_line_measurements(result, 'Hb4861', x=wave, index=(0, 1), n_samples=32, random_state=5, return_samples=True)

    assert evaluated.shape == cube.shape
    assert measurements.flux.shape == result.shape
    assert np.all(np.isfinite(measurements.flux))
    assert np.isfinite(direct.flux.value)
    assert np.isfinite(sampled.flux.std)
    assert draws.shape[0] == 32