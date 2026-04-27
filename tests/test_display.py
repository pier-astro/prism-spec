import warnings

import numpy as np
from astropy.modeling import models
from astropy.modeling.powerlaws import PowerLaw1D

import prism.modeling.display as model_display
from prism.modeling.fitting import TRFLSQFitter
from prism.modeling.fitting.utils import tie
from prism.modeling.models import GaussianLines
from prism.modeling.models.components import get_components
from prism.modeling.models.display import (
    format_fit_html,
    format_fit_text,
    format_model_html,
    format_model_text,
    get_model_expression,
)
from prism.modeling.fitting import MultiFitResult


def _build_multifit_cube():
    x = np.linspace(-5.0, 5.0, 121)
    base = (
        models.Gaussian1D(amplitude=2.0, mean=-1.0, stddev=0.8, name='g1')
        + models.Gaussian1D(amplitude=1.2, mean=1.4, stddev=0.9, name='g2')
        + models.Const1D(amplitude=0.15, name='c0')
    )
    cube = np.empty((2, 3, x.size), dtype=float)
    for j in range(2):
        for i in range(3):
            model = base.copy()
            model['g1'].amplitude.value = 2.0 + 0.05 * j + 0.02 * i
            model['g2'].amplitude.value = 1.2 + 0.03 * j + 0.01 * i
            model['c0'].amplitude.value = 0.15 + 0.01 * j - 0.005 * i
            cube[j, i] = model(x)
    return x, cube, base


def test_model_display_workflow():
    single = models.Gaussian1D(1.234, 5000.0, 2.5, name='gauss')
    assert hasattr(single, '_repr_html_')

    try:
        model_display.disable_model_display()
        assert not hasattr(single, '_repr_html_')
        model_display.enable_model_display()
        assert hasattr(single, '_repr_html_')

        single = models.Gaussian1D(1.234, 5000.0, 2.5, name='gauss')
        single.mean.fixed = True
        single.amplitude.std = 0.01
        single.amplitude.lolim = 1.20
        single.amplitude.uplim = 1.26

        assert 'Expression:' in format_model_text(single)
        assert '<table' in format_model_html(single).lower()
        assert '± 0.01' in format_fit_text(single)
        assert '#0f1115' in format_fit_html(single)

        compound = models.Linear1D(0.1, 2.0, name='cont') + models.Gaussian1D(3.0, 5007.0, 1.2, name='oiii')
        compound.mean_1.fixed = True
        compound.intercept_0.tied = tie('oiii', lambda m: m.amplitude)
        compound.amplitude_1.std = 0.2
        compound.amplitude_1.lolim = 2.7
        compound.amplitude_1.uplim = 3.3

        fit_text = compound.show(mode='text')
        fit_html = compound.show(mode='html')
        assert 'cont + oiii' in str(compound)
        assert '= amplitude_1' in fit_text
        assert 'colspan="2"' in fit_html
        assert get_model_expression(models.Const1D(1.0, name='a') + models.Const1D(2.0, name='b')) == 'a + b'

        tied_lines = GaussianLines.from_arrays(
            names=['Hb4861'], pos=[4861.333], name='nlr', amplitude=2.0, offset=0.0, fwhm=300.0,
        ) + GaussianLines.from_arrays(
            names=['OIII5007'], pos=[5006.803], name='hhe_nlr', amplitude=1.0, offset=0.0, fwhm=300.0,
        )
        tied_lines.offset_1.tied = tie('nlr', lambda m: m.offset)
        assert '= offset_0' in format_fit_text(tied_lines)

        prism_model = PowerLaw1D(name='pl') + GaussianLines.from_arrays(
            names=['Hb4861'], pos=[4861.333], name='hb', amplitude=2.0, offset=0.0, fwhm=300.0,
        )
        assert 'pl + hb' in format_model_text(prism_model)

        plain = models.Gaussian1D(1.0, 0.0, 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plain.show(mode='text')
        assert any('No parameter std or interval limits were found' in str(w.message) for w in caught)
    finally:
        model_display.enable_model_display()


def test_model_components_accessor():
    model = (
        models.Gaussian1D(1.0, -1.0, 0.5, name='g1')
        + models.Gaussian1D(2.0, 1.0, 0.7, name='g2')
        + models.Const1D(0.1, name='c0')
    )
    comps = get_components(model, additive=True)
    assert len(comps) == 3
    assert comps[0] is comps['g1']
    assert comps[1] is comps['g2']
    assert comps[2] is comps['c0']


def test_multifit_component_accessors():
    x, cube, model = _build_multifit_cube()
    fitter = TRFLSQFitter()
    result_auto = fitter.multifit(model.copy(), x, cube, nproc=1, progress=False)
    assert isinstance(result_auto, MultiFitResult)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fitter.multifit(model.copy(), x, cube, inplace=True, nproc=1, progress=False)

    assert any('inplace=True' in str(w.message) and 'multifit mode' in str(w.message) for w in caught)
    assert set(result.comps.names) == {'g1', 'g2', 'c0'}

    comp_cube = result.comps['g1'](x)
    np.testing.assert_allclose(result.evaluate_component('g1', x=x), comp_cube, rtol=1e-6, atol=1e-8)
    assert result.evaluate_addcomponent(result.addcomps.names[0], x=x).shape == cube.shape