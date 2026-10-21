"""Declarative links: scientific constraints, ownership and portable sessions."""
from line_helpers import make_line, make_group
import copy
import pickle
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import Parameter, models
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.fitting.utils import _ParameterMap, apply_tied
from prism.modeling.interop.astropy import tie
from prism.modeling.io import load_model


def pair():
    a = make_line(amplitude=2, position=5000, fwhm=240, name='a')
    b = make_line(amplitude=1, position=5020, fwhm=400, name='b')
    b.fwhm.link = a.fwhm
    b.offset.link = a.offset
    b.amplitude.link = a.amplitude / 2.98
    return a, b


def test_live_values_units_and_unlink():
    a, b = pair()
    a.amplitude.value = 5
    assert b.amplitude.value == 5 / 2.98
    a.fwhm.value = 360
    assert b.fwhm.value == 360
    b.fwhm.link = None
    a.fwhm.value = 500
    assert b.fwhm.value == 360
    assert not b.fwhm.tied


def test_direct_evaluation_and_copy_independence():
    a, b = pair(); model = a + b
    x = np.linspace(4990, 5030, 151)
    model.amplitude_0.value = 8
    expected_b = make_line(amplitude=8/2.98, position=5020, fwhm=240)
    np.testing.assert_allclose(model(x), a(x) + expected_b(x))
    for duplicate in [model.copy(), copy.deepcopy(model), pickle.loads(pickle.dumps(model))]:
        duplicate.amplitude_0.value = 7
        duplicate['a'].name = 'renamed'
        assert duplicate.amplitude_1.value == 7/2.98
        assert model.amplitude_1.value == 8/2.98
        _ParameterMap(duplicate)


def test_reverse_order_chain_and_cycle_rejection():
    a, b, c = make_line(), make_line(), make_line()
    a.amplitude.link = b.amplitude * 2
    b.amplitude.link = c.amplitude + 1
    model = a + b + c
    model.amplitude_2.value = 3
    apply_tied(model)
    assert model.amplitude_0.value == 8
    names = [p.name for _, _, p in _ParameterMap(model).tied]
    assert len(names) == 2
    with pytest.raises(ValueError, match='Cyclic'):
        c.amplitude.link = a.amplitude
    assert not c.amplitude.tied
    with pytest.raises(ValueError, match='Cyclic'):
        c.amplitude.link = c.amplitude + 1


@pytest.mark.usefixtures('astropy_extensions')
def test_external_dependencies_rejected_before_fit_and_save(tmp_path):
    a, b = pair()
    with pytest.raises(ValueError, match='outside'):
        _ParameterMap(b)
    with pytest.raises(ValueError, match='outside'):
        b.save(tmp_path/'bad.fits')
    assert not (tmp_path/'bad.fits').exists()
    # Copying only the dependent component also copies its external dependency;
    # assembling it with the original driver cannot silently redirect that link.
    with pytest.raises(ValueError, match='outside'):
        _ParameterMap(a + b.copy())


@pytest.mark.parametrize('value', [np.nan, np.inf, [1, 2], 'a+1', lambda x: x])
def test_invalid_expression_assignment_is_atomic(value):
    a, b = pair(); previous = b.amplitude.link
    with pytest.raises((ValueError, TypeError)):
        b.amplitude.link = value
    assert b.amplitude.link is previous




@pytest.mark.parametrize('engine', [TRF, pytest.param(SherpaLM,
    marks=pytest.mark.skipif(not HAS_SHERPA, reason='Sherpa unavailable'))])
def test_known_input_fit_with_bounds_links_and_covariance(engine):
    a, b = pair(); truth = a + b
    truth.amplitude_0.value = 5; truth.fwhm_0.value = 320; truth.offset_0.value = 40
    x = np.linspace(4980, 5040, 240)
    y = truth(x)
    start = truth.copy()
    start.amplitude_0.value = 2; start.fwhm_0.value = 230; start.offset_0.value = 0
    start.amplitude_0.bounds = (0.01, 10)
    start.fwhm_0.bounds = (100, 700)
    start.offset_0.bounds = (-150, 150)
    fitter = engine(covariance=True)
    result = fitter(start, x, y, yerr=np.full(x.size, .02), inplace=False)
    np.testing.assert_allclose(result.predict(x), y, atol=1e-6)
    np.testing.assert_allclose([result.model.amplitude_0.value, result.model.fwhm_0.value, result.model.offset_0.value], [5, 320, 40], rtol=1e-5)
    assert result.model.amplitude_1.value == result.model.amplitude_0.value/2.98
    assert fitter.fit_info['param_cov'].shape == (3, 3)
    assert start.amplitude_0.value == 2


@pytest.mark.usefixtures('astropy_extensions')
def test_arithmetic_roundtrip_with_response(tmp_path):
    from prism.modeling.operators import LinearOperatorModel
    a, b = pair()
    b.amplitude.link = -(a.amplitude + 3) ** 2 / 4 + 100
    x = np.linspace(4980, 5040, 100)
    model = (a + b) | LinearOperatorModel(np.eye(len(x)), x=x)
    file = tmp_path/'session.fits'; model.save(file)
    loaded = load_model(file)
    loaded.amplitude_0.value = 7; loaded.offset_0.value = 2
    assert loaded.amplitude_1.value == 75
    assert loaded.offset_1.value == 2
    model.amplitude_0.value = 7; model.offset_0.value = 2
    np.testing.assert_array_equal(loaded(x), model(x))
    _ParameterMap(loaded)
    # Native Astropy parameter behavior has not been changed globally.
    assert type(models.Gaussian1D().mean) is Parameter
    assert not hasattr(models.Gaussian1D().mean, 'link')


@pytest.mark.usefixtures('astropy_extensions')
def test_constant_and_quantity_roundtrip(tmp_path):
    a = make_line(); a.fwhm.link = a.fwhm_G if hasattr(a, 'fwhm_G') else 250.
    a.amplitude.link = 3
    a.save(tmp_path/'constant.fits'); loaded = load_model(tmp_path/'constant.fits')
    assert loaded.fwhm.value == 250
    assert loaded.amplitude.value == 3
    assert loaded.fwhm.link is not None


def test_native_astropy_callable_adapter_preserved():
    a = models.Gaussian1D(name='a'); b = models.Gaussian1D(name='b')
    model = a + b
    model.stddev_1.tied = tie('a', lambda component: 2*component.stddev)
    model.stddev_0.value = 4
    apply_tied(model)
    assert model.stddev_1.value == 8


def test_group_parameters_support_links():
    table = dict(name=['a', 'b'], position=[5000, 5020], weight=[1, 1])
    # Use the same public group constructor as the existing scientific examples.
    import pandas as pd
    a = make_group(table=pd.DataFrame(table), position_unit=u.AA)
    b = a.copy()
    b.fwhm.link = 1.5 * a.fwhm
    model = a + b
    assert model.fwhm_1.value == 1.5*model.fwhm_0.value
    _ParameterMap(model)


def test_numpy_scalar_coefficients_remain_live():
    a, b = pair()
    b.amplitude.link = np.float64(2)*a.amplitude + np.int64(1)
    a.amplitude.value = 5
    assert b.amplitude.value == 11


def test_linked_cube_serial_parallel_and_saved_view(tmp_path):
    a, b = pair()
    for component in [a, b]:
        component.fwhm.fixed = True
        component.offset.fixed = True
    model = a+b
    x = np.linspace(4980, 5040, 120)
    profile = model(x)/model.amplitude_0.value
    data = profile[:, None]*np.array([2., 3., 4.])
    options = dict(yerr=np.full_like(data, .05), axis=0, progress=False)
    serial = TRF(covariance=True).multifit(model,x,data,nproc=1,**options)
    parallel = TRF(covariance=True).multifit(model,x,data,nproc=2,**options)
    for name in model.param_names:
        np.testing.assert_allclose(parallel._params[name],serial._params[name])
    np.testing.assert_allclose(parallel._params['amplitude_0'],[2,3,4],atol=1e-7)
    np.testing.assert_allclose(parallel._params['amplitude_1'],np.array([2,3,4])/2.98,atol=1e-7)
    parallel.save(tmp_path/'cube.fits')
    loaded=load_model(tmp_path/'cube.fits')
    loaded.amplitude_0.value=6
    assert loaded.amplitude_1.value == 6/2.98


def test_malformed_restored_links_are_rejected():
    from prism.modeling.parameters import _restore_links
    for payload in [
        {'amplitude': {'parameter':'missing'}},
        {'amplitude': {'parameter':'fwhm'}},
        {'amplitude': {'constant':2}},
        {'amplitude': {'operation':'exec','operands':[]}},
        {'amplitude': {'operation':'mul','operands':[{'constant':1}]}},
        {'amplitude': {'operation':'ref','operands':[{'parameter':'amplitude'}]}},
    ]:
        with pytest.raises(ValueError):
            _restore_links(make_line(),payload)




@pytest.mark.usefixtures('astropy_extensions')
def test_repeated_component_instances_do_not_silently_rewire_links(tmp_path):
    a,b=pair()
    with pytest.raises(ValueError, match='Repeated component'):
        _ = a+b+a


def test_quantity_constants_on_right_and_numeric_parameter_representation():
    a,b=pair()
    b.offset.link=a.offset + 10.
    b.amplitude.link=a.amplitude * (2*u.dimensionless_unscaled)
    a.offset.value=40; a.amplitude.value=5
    assert b.offset.value==50
    assert b.amplitude.value==10
    assert np.isfinite(a.offset.value)
    assert b.offset.name == 'offset'
    assert 'Model' in repr(b)
