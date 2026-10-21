"""Native profiles reuse line measurements and parent-covariance propagation."""
from line_helpers import make_group
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose
from prism.modeling import Component,LineSet,select_line,measure_line
from prism.modeling.lines import GaussianProfile
from prism.modeling.lines.profiles import gaussian
from prism.modeling.lines.measurements.sampling import _selection_parameter_specs
from prism.modeling.fitting import TRF
from prism.modeling.operators.matop import LinearOperatorModel
from prism.modeling.io import load_model


def table():
    return dict(name=['Hb','OIII','OIII'],position=[4861.3,4958.9,5006.8]*u.AA,weight=[1.,1/2.98,1.])


def native(name='narrow',fwhm=300.):
    profile=GaussianProfile(instfwhm=[[4800,80],[5100,120]])
    return LineSet(profile,table(),shared=dict(offset=-35.,fwhm=fwhm,redshift=0.),
                         parameters={'Hb':{'amplitude':2.},'OIII':{'amplitude':3.}},
                         position='position',derivative=profile.derivative,name=name)


def test_selected_member_matches_analytic_profile_and_named_component_identity():
    g=native()
    x=np.linspace(4920,5050,4096)
    new=select_line(g,'OIII')
    center=5006.8*np.exp(-35/299792.458)
    instrumental=80.+(center-4800)*40./300.
    sigma=np.hypot(300.,instrumental)*center/299792.458/np.sqrt(8*np.log(2))
    assert_allclose(new(x),3.*np.exp(-.5*((x-center)/sigma)**2),rtol=1e-12)
    assert_allclose(new.infer_window(),[center-6*sigma*np.sqrt(8*np.log(2)),center+6*sigma*np.sqrt(8*np.log(2))])
    assert_allclose(new.measure(x=x).flux.value,3*sigma*np.sqrt(2*np.pi),rtol=1e-10)
    chosen=select_line(g,'OIII',components=g,position=4958.9*u.AA)
    expected_peak=3/2.98
    assert abs(chosen(np.array([4958.9*np.exp(-35/299792.458)]))[0]-expected_peak)<1e-12


def test_native_single_profile_and_automatic_grid():
    profile=GaussianProfile()
    line=Component(profile,dict(amplitude=2.,position=5000.,offset=0.,fwhm=300.,redshift=0.),name='line')
    selection=select_line(line,'line')
    measured=selection.measure()
    sigma=5000*300/299792.458/np.sqrt(8*np.log(2))
    assert_allclose(measured.flux.value,2*np.sqrt(2*np.pi)*sigma,rtol=1e-6)
    assert_allclose(measured.fwhm.value,5000*300/299792.458,rtol=2e-5)


def test_raw_gaussian_group_and_independent_component_list():
    g=LineSet(gaussian,table(),shared={'sigma':2.},parameters={'Hb':{'amplitude':2.},'OIII':{'amplitude':3.}},name='raw')
    b=native('broad',900.)
    x=np.linspace(4800,4900,2048)
    selection=select_line([g,b],'Hb')
    assert_allclose(selection(x),select_line(g,'Hb')(x)+select_line(b,'Hb')(x))
    assert selection.measure().flux.value>0


def exponential(x,flux,center,width):
    return flux*np.exp(-np.abs(x-center)/width)/(2*width)


def test_custom_flux_scale_requires_explicit_grid_and_relevant_parameters():
    g=LineSet(exponential,dict(name=['A','A','B'],position=[10,20,40]*u.AA,weight=[.25,.75,1]),
                    shared={},parameters={'A':{'flux':8.,'width':.5},'B':{'flux':3.,'width':1.}},scale='flux',name='custom')
    with pytest.raises(ValueError,match='unique matching member'):
        select_line(g,'A')
    selection=select_line(g,'A',position=20.)
    with pytest.raises(ValueError,match='explicit measurement'):
        selection.measure()
    x=np.linspace(0,40,20000)
    assert_allclose(selection.measure(x=x).flux.value,6.,rtol=1e-6)
    assert_allclose(selection.eqw(continuum=2.,x=x).value,3.,rtol=1e-6)
    assert set(spec.param_name for spec in _selection_parameter_specs(selection))=={'flux_a','width_a'}


def test_parent_links_response_and_roundtrip(tmp_path):
    a,b=native(),native('broad',900.)
    b.amp_hb.link=a.amp_hb/2
    x=np.linspace(4800,5100,1001)
    model=(a+b)|LinearOperatorModel(np.eye(x.size),x=x,xout=x)
    selection=select_line(model,'Hb',components=[a,b])
    a.amp_hb=4.
    assert_allclose(selection(x),select_line(a,'Hb')(x)+select_line(native('reference',900.),'Hb')(x))
    with pytest.raises(ValueError,match='explicit intrinsic continuum'):
        selection.eqw(x=x)
    assert selection.eqw(continuum=2.,x=x).value>0
    path=tmp_path/'parent.fits';model.save(path)
    restored=load_model(path)
    assert_allclose(select_line(restored,'Hb',components=['narrow','broad'])(x),selection(x))


def test_covariance_sampling_retains_correlated_native_components():
    a,b=native(),native('broad',900.)
    for component in [a,b]:
        for name in component.param_names:getattr(component,name).fixed=name!='amp_hb'
    model=a+b
    model._param_cov=np.array([[.01,.012],[.012,.04]])
    model._fit_indices=np.array([model.param_names.index('amp_hb_0'),model.param_names.index('amp_hb_1')])
    selection=select_line(model,'Hb')
    summary=selection.sample(n_samples=1536,method='covariance',random_state=41,num=1200)
    scales=[]
    for component in [a,b]:
        p=component.copy();p.amp_hb=1.
        scales.append(select_line(p,'Hb').measure(num=4096).flux.value)
    expected=np.sqrt(np.array(scales)@model._param_cov@np.array(scales))
    assert_allclose(summary.flux.std,expected,rtol=.08)


def test_native_cube_selection_and_fits_view():
    g=native();g.offset.fixed=g.fwhm.fixed=g.redshift.fixed=True
    x=np.linspace(4800,5100,1001)
    data=np.array([g(x),2*g(x)])
    result=TRF().multifit(g,x,data,axis=1,progress=False)
    measured=measure_line(result,'Hb')
    assert_allclose(measured.flux[1],2*measured.flux[0],rtol=1e-7)


def test_native_cube_missing_fit_remains_nan_in_measurements():
    g=native();g.offset.fixed=g.fwhm.fixed=g.redshift.fixed=True
    x=np.linspace(4800,5100,1001)
    data=np.array([np.full(x.size,np.nan),g(x)])
    result=TRF().multifit(g,x,data,axis=1,progress=False,filter_non_finite=True)
    selected=select_line(result,'Hb')
    measured=selected.measure()
    assert np.isnan(measured.flux[0]) and np.isfinite(measured.flux[1])
    assert selected.axis_unit==u.AA
    assert np.isnan(selected(x)[0]).all()
    assert np.isfinite(selected(x)[1]).all()
    with pytest.raises(ValueError,match='No finite fitted model'):
        selected.get_profile(0)


def test_complete_native_multiplet_uses_joint_profile_and_covariance():
    g=LineSet(gaussian,dict(name=['pair','pair'],position=[10.,20.]*u.AA,weight=[.25,.75]),
                    shared={'sigma':.4},parameters={'pair':{'amplitude':4.}},name='pair')
    g.sigma.fixed=True
    g._param_cov=np.array([[.04]])
    selected=select_line(g,'pair',position=[10.,20.,20.]*u.AA)
    assert len(selected.entries)==2
    x=np.linspace(0.,30.,4096)
    assert selected.component_names == ['pair']
    assert_allclose(selected.evaluate_components(x=x)['pair'],g(x))
    expected=4*.4*np.sqrt(2*np.pi)
    assert_allclose(selected.measure().flux.value,expected,rtol=1e-6)
    result,samples=selected.sample(n_samples=64,method='covariance',random_state=71,return_samples=True)
    draws=np.random.default_rng(71).multivariate_normal([4.],[[.04]],size=64)[:,0]
    assert_allclose(samples['flux'],draws*.4*np.sqrt(2*np.pi),rtol=1e-6)
    with pytest.raises(ValueError,match='finite rest positions'):
        select_line(g,'pair',position=[])
