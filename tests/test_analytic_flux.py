from prism.modeling.lines.measurements import lines
"""Total integrals, physical gradients and their result-aware propagation."""
import copy
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from scipy.integrate import quad
from prism.modeling import LineSet, Component, select_line
from prism.modeling.lines import GaussianLines, LorentzianLines, VoigtLines
from prism.modeling.lines.profiles import GaussianProfile, LorentzianProfile, VoigtProfile
from prism.modeling.fitting import TRF, Dataset, FitResult
from prism.modeling.fitting.uncertainty import BootstrapResult, bootstrap
from prism.modeling.io import load_params, load_model


@pytest.mark.parametrize('cls,widths', [(GaussianProfile,[300.]),(LorentzianProfile,[300.]),(VoigtProfile,[250.,170.])])
@pytest.mark.parametrize('unit,position',[(u.AA,5000.),(u.eV,6.4),(u.Hz,1.e14)])
@pytest.mark.parametrize('instrumental',[0.,80.,'variable'])
def test_integrals_and_analytic_gradient_against_independent_numerics(cls,widths,unit,position,instrumental):
    inst = np.array([[position*.5,50.],[position*2,150.]]) if instrumental=='variable' else instrumental
    profile=cls(instfwhm=inst,position_unit=unit)
    args=np.array([2.3,position,-120.,*widths,.04])
    center,width=profile._center_width(*args)
    value=quad(lambda t: float(profile(center+width*t,*args))*width,-np.inf,np.inf,epsabs=1e-12)[0]
    np.testing.assert_allclose(profile._integral(*args),value,rtol=1e-8,atol=abs(value)*1e-9)
    gradient=profile._integral_gradient(*args)
    for i in range(len(args)):
        step=max(abs(args[i])*1e-5,1e-6)
        high,low=args.copy(),args.copy();high[i]+=step;low[i]-=step
        expected=(profile._integral(*high)-profile._integral(*low))/(2*step)
        np.testing.assert_allclose(gradient[i],expected,rtol=3e-5,atol=max(abs(value),1e-20)*1e-8)


def source(name='lines'):
    return GaussianLines({'name':['A','A'],'position':[5000.,5010.]*u.AA,'weight':[1.,.3]},
                         amplitude=2.,offset=0.,fwhm=250.,name=name, normalization='amplitude')


def test_ready_set_tables_typed_views_and_roundtrip(tmp_path):
    table=QTable({'name':['A'],'position':[5000.]*u.AA,'weight':[1.]})
    table.write(tmp_path/'one.ecsv');table['position']=[5010.]*u.AA;table['weight']=[.3];table.write(tmp_path/'two.ecsv')
    model=GaussianLines([tmp_path/'one.ecsv',tmp_path/'two.ecsv'],window=(4990.,5020.),name='lines', normalization='amplitude')
    assert isinstance(model,LineSet) and len(lines(model).flux())==2
    with pytest.raises(ValueError):GaussianLines(table,window=(5020.,4990.), normalization='amplitude')
    expression=model+Component(lambda x,a: np.full_like(x,a),{'a':1.},name='constant')
    assert isinstance(expression.components['lines'],GaussianLines)
    filename=tmp_path/'model.fits';model.save(filename)
    restored=load_model(filename)
    assert isinstance(restored,GaussianLines)
    np.testing.assert_allclose(lines(restored).flux()['flux'],lines(model).flux()['flux'])


def test_shared_scale_remains_generic_and_has_no_flux_convention():
    def profile(x,flux,center,sigma):return flux*np.exp(-.5*((x-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    model=LineSet(profile,{'name':['A','B'],'position':[0.,3.]*u.AA,'weight':[1.,2.]},
                  shared={'flux':2.,'sigma':.3},parameters={'A':{},'B':{}},scale='flux')
    assert not callable(model.flux)  # It is the user's shared parameter.
    x=np.linspace(-2.,5.,3001)
    integral=getattr(np,'trapezoid',None) or np.trapz
    assert integral(model(x),x)==pytest.approx(6.,rel=1e-8)
    with pytest.raises(NotImplementedError):select_line(model,'A').flux()


def test_covariance_with_links_matches_full_free_parameter_differences():
    first,second=source('first'),source('second')
    second.amp_a.link=first.amp_a*1.5
    second.fwhm.link=first.fwhm
    model=first+second
    x=np.linspace(4970.,5040.,401);y=model(x)+.01*np.sin(x)
    result=TRF()(model,x,y,yerr=np.full(x.size,.1))
    selected=select_line(result,'A',components=['first','second'],position=5000.)
    metric=selected.flux();graph=selected._parameter_model
    values=graph.parameters.copy();free=graph._fit_indices;gradient=[]
    for index in free:
        h=max(abs(values[index])*1e-5,1e-6);up=values.copy();down=values.copy();up[index]+=h;down[index]-=h
        graph.parameters=up;upper=selected.flux().value
        graph.parameters=down;lower=selected.flux().value
        gradient.append((upper-lower)/(2*h))
    graph.parameters=values
    expected=np.sqrt(np.array(gradient)@result.covariance@gradient)
    assert metric.std==pytest.approx(expected,rel=2e-5)
    assert lines(source()).flux()['flux'][0]==pytest.approx(select_line(source(),'A',position=5000.).flux().value)
    assert select_line(result.model,'A',components='first',position=5000.).flux().std is None


def test_joint_external_link_uses_global_covariance():
    first,second=source('blue'),source('red');second.amp_a.link=first.amp_a
    x=np.linspace(4970.,5040.,301)
    result=TRF().jointfit({'blue':(first,Dataset(x,first(x),yerr=np.full(x.size,.1))),
                           'red':(second,Dataset(x,first(x),yerr=np.full(x.size,.2)))})
    assert select_line(result['red'],'A',components='red',position=5000.).flux().std>0


def test_bootstrap_owns_estimates_and_flux_intervals_without_model_mutation():
    model=source();before=model.parameters.copy();x=np.linspace(4970.,5040.,201)
    result=bootstrap(model,TRF(covariance=False),x=x,y=model(x),yerr=np.full(x.size,.1),n_samples=16,seed=5)
    assert isinstance(result,BootstrapResult)
    np.testing.assert_array_equal(model.parameters,before)
    assert not hasattr(model.amp_a,'std')
    assert result.params['amp_a'].std>0
    selected=select_line(result,'A',position=5000.);flux=selected.flux()
    working=model.copy();draws=[]
    for values in zip(*result.samples.values()):
        for name,value in zip(result.samples,values):working.params[name].value=value
        draws.append(lines(working).flux()['flux'][0])
    assert flux.std==pytest.approx(np.std(draws,ddof=1))
    np.testing.assert_allclose([flux.lolim,flux.uplim],np.percentile(draws,[16,84]))
    assert 'BootstrapResult' in result.show(mode='text')
    with pytest.raises(ValueError):result.samples['amp_a'][0]=0.


def test_light_parameter_loading_does_not_rebuild_custom_models(tmp_path):
    model=Component(lambda x,slope: slope*x,{'slope':1.},name='custom')
    x=np.linspace(-1.,1.,31);result=TRF()(model,x,3*x)
    filename=tmp_path/'result.fits';result.save(filename)
    assert load_params(filename)=={'slope':pytest.approx(3.)}
    assert 'FitResult' in result.show(mode='text')
    assert not hasattr(result.model.slope,'std')
