import numpy as np
from astropy.stats import sigma_clip

from prism.modeling.fitting import FittingWithOutlierRemoval, ScipyTRF
from prism.modeling.models import BSpline, FixedTemplate, get_components


def _bspline_kwargs(knots, values):
    degree = 3
    ncoeffs = len(knots) - degree - 1
    return {f'c{i}': value for i, value in enumerate(values[:ncoeffs])}


def _build_empirical_model(wave, template_flux, spline_values, additive_values):
    degree = 3
    knots = np.concatenate((
        [wave[0]] * degree,
        np.linspace(wave[0], wave[-1], 4),
        [wave[-1]] * degree,
    ))
    spline = BSpline(knots, degree=degree, name='spline', **_bspline_kwargs(knots, spline_values))
    additive = BSpline(knots, degree=degree, name='aspline', **_bspline_kwargs(knots, additive_values))
    template = FixedTemplate(wave, template_flux, name='template')
    return spline * template + additive


def test_empirical_subtraction_multifit_workflow():
    wave = np.linspace(5000.0, 5100.0, 96)
    template_flux = 1.0 + 0.12 * np.sin((wave - wave[0]) / 6.0)

    truth = _build_empirical_model(
        wave,
        template_flux,
        spline_values=[0.95, 1.00, 1.05, 1.02, 0.98, 1.01],
        additive_values=[0.04, 0.03, 0.02, 0.02, 0.03, 0.04],
    )
    init_model = _build_empirical_model(
        wave,
        template_flux,
        spline_values=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        additive_values=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    )

    clean_cube = np.empty((2, 2, wave.size), dtype=float)
    cube = np.empty_like(clean_cube)
    yerr = np.full_like(clean_cube, 0.03)
    rng = np.random.default_rng(17)

    for j in range(2):
        for i in range(2):
            model = truth.copy()
            comps = get_components(model, additive=False)
            comps['spline'].c0.value += 0.02 * j
            comps['spline'].c1.value += 0.01 * i
            comps['aspline'].c2.value += 0.01 * (j - i)
            clean_cube[j, i] = model(wave)
            cube[j, i] = clean_cube[j, i] + rng.normal(0.0, yerr[j, i], wave.size)

    cube[0, 1, 20] += 0.8
    cube[1, 0, 70] += 0.7

    fitter = FittingWithOutlierRemoval(
        ScipyTRF(calc_uncertainties=True),
        outlier_func=sigma_clip,
        niter=2,
        sigma_lower=3.0,
        sigma_upper=2.0,
    )
    result = fitter.multifit(init_model, wave, cube, yerr=yerr, nproc=1, progress=False)

    fitted_cube = result(wave)
    spline_cube = result.comps['spline'](wave)

    assert result.shape == (2, 2)
    assert fitted_cube.shape == cube.shape
    assert spline_cube.shape == cube.shape

    raw_mae = np.mean(np.abs(cube - clean_cube))
    fit_mae = np.mean(np.abs(fitted_cube - clean_cube))
    assert fit_mae < raw_mae