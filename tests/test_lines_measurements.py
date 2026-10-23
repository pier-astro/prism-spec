"""Explicit component tables and EW covariance, checked against ratio calculus."""
import copy
import types
import numpy as np
import pytest
from astropy import units as u
from prism.modeling import Component, select_line
from prism.modeling.lines import GaussianLines, LineSet
from prism.modeling.lines.measurements import lines
from prism.modeling.fitting import Dataset, FitResult, TRF
from prism.modeling.fitting.utils import _free_parameter_indices
from prism.modeling.fitting.uncertainty.resample import BootstrapResult


def source():
    model=GaussianLines(dict(name=['A','A','B'],position=[4959.,5007.,5100.]*u.AA,
                             weight=[1/3,1.,.4]),flux=8.,fwhm=200.,name='narrow')
    model.offset.fixed=model.fwhm.fixed=True
    model.flux_b.fixed=True
    return model


def result(link=False):
    model=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),flux=8.,fwhm=200.,name='narrow')
    model.offset.fixed=model.fwhm.fixed=True
    continuum=Component(lambda x,c: np.full_like(x,c),{'c':2.},name='continuum')
    if link:model.flux_a.link=4*continuum.c
    graph=model+continuum
    graph._param_cov=np.array([[.09]]) if link else np.array([[.64,.12],[.12,.04]])
    graph._fit_indices=_free_parameter_indices(graph)
    x=np.linspace(4950.,5050.,301)
    return FitResult({'data':(graph,Dataset(x,graph(x)))},graph,{'success':True},
                     {'data':{'stat':0.}},_single=True)


def test_single_lineset_catalog_has_every_transition_and_no_model_flux_method():
    model=source();prepared=lines(model);table=prepared.flux()
    assert list(table['name'])==['A','A','B']
    np.testing.assert_allclose(table['flux'],[8/3,8,3.2])
    assert table['position'].unit==u.AA
    assert np.isnan(table['std']).all()
    assert not hasattr(model,'flux') and not hasattr(model,'eqw')
    model.flux_a=12.
    np.testing.assert_allclose(prepared.flux()['flux'],[4,12,3.2])
    ew=prepared.eqw(continuum=2.)
    assert ew['eqw'].unit==u.AA
    assert ew['eqw'][0].value==pytest.approx(2.,rel=1e-8)


def test_explicit_single_component_contract_and_module_namespace():
    from prism.modeling import lines as module
    assert isinstance(module,types.ModuleType)
    model=source();continuum=Component(lambda x,c:np.full_like(x,c),{'c':2.},name='continuum')
    compound=model+continuum
    with pytest.raises(ValueError,match='single LineSet'):lines(compound)
    with pytest.raises(TypeError,match='one component'):lines(compound,component=['narrow'])
    with pytest.raises(TypeError,match='LineSet'):lines(compound,component='continuum')
    assert len(lines(compound,component='narrow').flux())==3
    with pytest.raises(ValueError,match='explicit continuum'):lines(model).eqw(None)
    with pytest.raises(TypeError):select_line(compound)


def test_ew_covariance_matches_ratio_with_continuum_cross_term():
    fitted=result();before=fitted.parameters.copy()
    selected=select_line(fitted,'A',components='narrow')
    measured=selected.eqw(continuum='continuum',window=(4950.,5050.),num=2048)
    gradient=np.array([.5,-2.])
    expected=np.sqrt(gradient@fitted.covariance@gradient)
    assert measured.value==pytest.approx(4.,rel=1e-12)
    assert measured.std==pytest.approx(expected,rel=2e-7)
    assert measured.unit==u.AA
    held=selected.eqw(continuum=2.,window=(4950.,5050.))
    assert held.std==pytest.approx(.4,rel=2e-7)
    table=lines(fitted,component='narrow').eqw('continuum',window=(4950.,5050.))
    assert table['std'][0].value==pytest.approx(expected,rel=2e-7)
    assert np.isnan(table['lolim']).all()
    np.testing.assert_array_equal(fitted.parameters,before)
    assert lines(fitted,component='narrow').flux()['std'][0]==pytest.approx(.8)


def test_named_continuum_link_can_cancel_the_entire_ew_uncertainty():
    fitted=result(link=True)
    selected=select_line(fitted,'A',components='narrow')
    measured=selected.eqw('continuum',window=(4950.,5050.))
    assert measured.value==pytest.approx(4.,rel=1e-12)
    assert measured.std<1e-8
    assert selected.eqw(2.,window=(4950.,5050.)).std==pytest.approx(.6,rel=2e-6)


def test_bootstrap_ew_uses_correlated_refits_instead_of_independent_errors():
    fitted=result();graph=fitted.model
    free=[graph.param_names[i] for i in _free_parameter_indices(graph)]
    values=np.array([[7.,1.8],[9.,2.3],[8.5,1.9],[np.nan,np.nan]])
    bootstrap=BootstrapResult(graph,dict(zip(free,values.T)),{'success':True})
    table=lines(bootstrap,component='narrow').eqw('continuum',window=(4950.,5050.))
    expected=values[:3,0]/values[:3,1]
    assert table['eqw'][0].value==pytest.approx(4.)
    assert table['std'][0].value==pytest.approx(np.std(expected,ddof=1))
    np.testing.assert_allclose([table['lolim'][0].value,table['uplim'][0].value],np.percentile(expected,[16,84]))


def test_incompatible_continuum_units_restore_model_on_failure():
    fitted=result();selected=select_line(fitted,'A',components='narrow')
    before=selected._parameter_model.parameters.copy()
    with pytest.raises(ValueError,match='units'):selected.eqw(continuum=lambda x: np.ones_like(x)*u.Jy)
    np.testing.assert_array_equal(selected._parameter_model.parameters,before)


def test_single_component_fit_does_not_require_identifier():
    model=source();x=np.linspace(4900.,5150.,501)
    fit=TRF()(model,x,model(x),yerr=np.full(x.size,.1))
    assert len(lines(fit).flux())==3


def test_preparation_copies_fit_context_only_once_for_all_measurements():
    from unittest.mock import patch
    from prism.modeling.lines.measurements.catalog import _line_context
    fitted=result()
    with patch('prism.modeling.lines.measurements.catalog._line_context',wraps=_line_context) as prepare:
        view=lines(fitted,component='narrow')
        view.flux();view.eqw('continuum')
        assert prepare.call_count==1


def test_invalid_covariance_is_rejected_consistently():
    fitted=result();selected=select_line(fitted,'A',components='narrow')
    selected._parameter_model._param_cov=np.array([[1.,2.],[0.,1.]])
    with pytest.raises(ValueError,match='symmetric'):selected.eqw('continuum')
    with pytest.raises(ValueError,match='symmetric'):selected.flux()


def test_variable_continuum_and_kinematic_derivatives_match_independent_integral():
    from scipy.integrate import quad
    model=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),flux=8.,fwhm=200.,name='narrow')
    model.offset.fixed=True
    cont=Component(lambda x,c,slope:c+slope*(x-5000.),{'c':2.,'slope':.015},name='continuum')
    graph=model+cont;free=_free_parameter_indices(graph)
    covariance=np.diag([25.,.16,.01,1e-6])
    graph._param_cov=covariance;graph._fit_indices=free
    x=np.linspace(4970.,5030.,501)
    fitted=FitResult({'data':(graph,Dataset(x,graph(x)))},graph,{'success':True},{'data':{'stat':0.}},_single=True)
    selected=select_line(fitted,'A',components='narrow')
    metric=selected.eqw('continuum',window=(4970.,5030.),num=8192)
    def reference(p):
        width,flux,c,slope=p
        sigma=5000.*width/299792.458/np.sqrt(8*np.log(2))
        return quad(lambda t:flux*np.exp(-.5*(t/sigma)**2)/(sigma*np.sqrt(2*np.pi))/(c+slope*t),-30.,30.,epsabs=1e-11)[0]
    values=np.array([200.,8.,2.,.015]);gradient=[]
    for i,v in enumerate(values):
        h=abs(v)*1e-4;up=values.copy();down=values.copy();up[i]+=h;down[i]-=h
        gradient.append((reference(up)-reference(down))/(2*h))
    gradient=np.array(gradient)
    assert metric.value==pytest.approx(reference(values),rel=1e-10)
    assert metric.std==pytest.approx(np.sqrt(gradient@covariance@gradient),rel=2e-5)


def test_single_sampler_reports_optional_ew_from_the_same_parameter_draws():
    fitted=result();selected=select_line(fitted,'A',components='narrow')
    summary,draws=selected.sample(continuum='continuum',method='covariance',
                                  n_samples=256,random_state=18,return_samples=True,
                                  window=(4950.,5050.),num=1024)
    expected=np.random.default_rng(18).multivariate_normal([8.,2.],fitted.covariance,size=256)
    np.testing.assert_allclose(draws['flux'],expected[:,0],rtol=1e-10)
    np.testing.assert_allclose(draws['eqw'],expected[:,0]/expected[:,1],rtol=1e-10)
    assert summary.eqw.std==pytest.approx(np.std(draws['eqw'],ddof=1))
    assert 'eqw' in summary.to_frame().index
    plain=selected.sample(method='covariance',n_samples=8,random_state=18)
    assert not hasattr(plain,'eqw')
    assert 'eqw' not in plain.metric_units
    assert summary.metric_units['eqw'] == u.AA
    assert not hasattr(selected,'sample_eqw')


def test_single_sampler_reuses_bootstrap_draws_for_profile_and_ew_together():
    fitted=result();graph=fitted.model
    names=[graph.param_names[i] for i in _free_parameter_indices(graph)]
    values=np.array([[7.,1.8],[9.,2.3],[8.5,1.9],[np.nan,np.nan]])
    bootstrap=BootstrapResult(graph,dict(zip(names,values.T)),{'success':True})
    selected=select_line(bootstrap,'A',components='narrow')
    summary,draws=selected.sample(continuum='continuum',return_samples=True,window=(4950.,5050.))
    assert len(draws)==3
    np.testing.assert_allclose(draws['flux'],values[:3,0],rtol=1e-10)
    np.testing.assert_allclose(draws['eqw'],values[:3,0]/values[:3,1],rtol=1e-10)
    assert summary.eqw.std==pytest.approx(selected.eqw('continuum',window=(4950.,5050.)).std)
    view=lines(bootstrap,component='narrow')
    first=view.flux();view.eqw('continuum');second=view.flux()
    np.testing.assert_array_equal(first['flux'],second['flux'])
    np.testing.assert_array_equal(first['std'],second['std'])


def test_ew_covariance_handles_zero_slope_in_unscaled_physical_flux_units():
    line=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),flux=8e-15,fwhm=200.,name='narrow')
    line.offset.fixed=line.fwhm.fixed=True
    cont=Component(lambda x,c,slope:c+slope*(x-4990.),{'c':2e-15,'slope':0.},name='continuum')
    graph=line+cont;graph._param_cov=np.diag([.64e-30,.04e-30,1e-36])
    graph._fit_indices=_free_parameter_indices(graph)
    x=np.linspace(4950.,5050.,301)
    fitted=FitResult({'data':(graph,Dataset(x,graph(x)))},graph,{'success':True},{'data':{'stat':0.}},_single=True)
    gradient=np.array([1/2e-15,-8e-15/(2e-15)**2,-8e-15*10/(2e-15)**2])
    measured=select_line(fitted,'A',components='narrow').eqw('continuum',window=(4950.,5050.))
    assert measured.std==pytest.approx(np.sqrt(gradient@fitted.covariance@gradient),rel=2e-6)
