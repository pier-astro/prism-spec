"""Intervals are summaries of existing draws, explicitly attached to fit snapshots."""
import copy
import numpy as np
import pytest
from astropy.modeling.models import Const1D, Linear1D
from prism.modeling import Model, Component
from prism.modeling.fitting import TRF, FitResult
from prism.modeling.fitting.uncertainty.resample import BootstrapResult, bootstrap
from prism.modeling.fitting.uncertainty.limits import _sample_interval
from prism.modeling.metrics import Metric


def fitted():
    x=np.arange(10.)
    return TRF()(Model.from_astropy(Const1D(2.,bounds={'amplitude':(0.,10.)})),x,
                 2.+.02*np.sin(x),yerr=np.full(10,.1))


def draws(result):
    name=result.param_names[0]
    return BootstrapResult(result.model,{name:np.linspace(1.,3.,101)},dict(n_samples=101,n_success=101,seed=7))


def test_recompute_levels_attach_replace_clear_and_centered_errors():
    result=fitted();b=draws(result);before=result.parameters.copy();cov=result.covariance.copy()
    a=b.limits(confidence=68);z=b.limits(confidence=90)
    np.testing.assert_allclose([a['lolim'][0],a['uplim'][0]],[1.32,2.68])
    np.testing.assert_allclose([z['lolim'][0],z['uplim'][0]],[1.1,2.9])
    name=result.param_names[0];std=result.params[name].std
    result.set_limits(a)
    assert result.limits_info['confidence']==68
    np.testing.assert_allclose(result.params[name].errors,[before[0]-1.32,2.68-before[0]])
    result.set_limits(z)
    assert result.params[name].lolim==pytest.approx(1.1)
    assert result.limits_info['confidence']==90
    assert result.params[name].std==std
    np.testing.assert_array_equal(result.parameters,before)
    np.testing.assert_array_equal(result.covariance,cov)
    assert result['data'].params[name] is result.params[name]
    a['lolim'][0]=5.;a['uplim'][0]=6.;result.set_limits(a)
    assert np.isnan(result.params[name].errors).all()
    result.clear_limits()
    assert result.limits_info is None and result.params[name].lolim is None
    assert np.isnan(result.params[name].errors).all()


def test_bad_intervals_do_not_partially_replace_existing_estimates():
    result=fitted();table=draws(result).limits();result.set_limits(table)
    old=result.params
    for change in ('state','names','units','order','confidence'):
        bad=copy.deepcopy(table)
        if change=='state':bad.meta['state']['parameters'][0]['value']=100
        if change=='names':bad['name']=['missing']
        if change=='units':bad['unit']=['km']
        if change=='order':bad['lolim'][0]=9.;bad['uplim'][0]=1.
        if change=='confidence':bad.meta['confidence']=101
        with pytest.raises(ValueError):result.set_limits(bad)
        assert result.params is old


def test_boundary_accumulation_not_just_exact_hits_and_custom_threshold():
    bound=(0.,None)
    regular=np.linspace(.1,2.,1000)
    lower,upper,sides=_sample_interval(regular,bound,68,.05)
    assert not sides and np.isfinite(lower)
    # No sample equals zero, but a fifth of the refits pile close to it.
    pile=np.r_[np.linspace(1e-8,1e-6,200),np.linspace(.1,2.,800)]
    lower,upper,sides=_sample_interval(pile,bound,68,.05)
    assert np.isnan(lower) and sides==['lower']
    assert upper==pytest.approx(np.percentile(pile,84))
    assert np.isfinite(_sample_interval(pile,bound,68,0)[0])
    assert np.isfinite(_sample_interval(pile,(None,None),68,.05)[0])
    assert np.isnan(_sample_interval(np.zeros(20),bound,68,.05)[0])
    assert np.isfinite(_sample_interval(np.ones(20),bound,68,.05)[0])
    assert np.isnan(_sample_interval(2-pile,(None,2.),68,.05)[1])
    # Fraction threshold follows the excluded tail: 10% pile affects 90%, not 68%.
    small=np.r_[np.full(100,1e-8),np.linspace(.1,2.,900)]
    assert np.isfinite(_sample_interval(small,bound,68,.05)[0])
    assert np.isnan(_sample_interval(small,bound,90,.05)[0])


def test_samples_save_reload_associated_model_and_limits_session(tmp_path):
    result=fitted();b=draws(result);result.set_limits(b.limits(confidence=90))
    path=tmp_path/'fit.fits';result.save(path)
    restored=FitResult.load(path)
    name=result.param_names[0]
    assert restored.params[name].lolim==pytest.approx(result.params[name].lolim)
    assert restored.limits_info==result.limits_info
    samples=tmp_path/'samples.npz';b.save(samples)
    with pytest.raises(FileExistsError):b.save(samples)
    loaded=BootstrapResult.load(samples,restored)
    np.testing.assert_array_equal(loaded.samples[name],b.samples[name])
    assert not loaded.samples[name].flags.writeable
    assert loaded.limits(confidence=95).meta['confidence']==95
    other=TRF()(Model.from_astropy(Const1D(5.,bounds={'amplitude':(0.,10.)})),np.arange(10.),np.full(10,5.))
    with pytest.raises(ValueError,match='do not match'):BootstrapResult.load(samples,other)
    result.clear_limits();result.save(path,overwrite=True)
    assert FitResult.load(path).limits_info is None


def test_samples_invalid_missing_and_linked_parameters():
    model=Component(lambda x,slope,intercept:slope*x+intercept,{'slope':2.,'intercept':1.})
    model.intercept.link=3*model.slope
    b=BootstrapResult(model,{'slope':np.array([1.,2.,3.,np.nan])})
    table=b.limits();assert table.meta['n_success']==3
    np.testing.assert_allclose(table['lolim'][1],3*np.percentile([1.,2.,3.],16))
    no=BootstrapResult(model,{'slope':np.array([np.nan])}).limits()
    assert np.isnan(no['lolim']).all()
    for kw in [dict(confidence=0),dict(confidence=100),dict(boundary_tolerance=-1)]:
        with pytest.raises(ValueError):b.limits(**kw)
    with pytest.raises(ValueError):BootstrapResult(model,{'wrong':np.ones(2)})
    assert np.isnan(Metric(3.,lolim=0.,uplim=2.).errors).all()


def test_boundary_heuristic_scales_with_physical_units_and_missing_side_errors():
    pile=np.r_[np.linspace(1e-8,1e-6,200),np.linspace(.1,2.,800)]
    for scale in (1e-15,1.,1e9):
        lo,hi,sides=_sample_interval(pile*scale,(0.,None),68,.05)
        assert np.isnan(lo) and sides==['lower']
        assert hi/scale==pytest.approx(np.percentile(pile,84))
    assert np.isnan(Metric(1.,lolim=np.nan,uplim=2.).errors).all()
    assert np.isnan(Metric(1.).errors).all()
    np.testing.assert_allclose(Metric(1.,lolim=0.,uplim=3.).errors,(1.,2.))


def test_persistence_keeps_unavailable_endpoint_and_metadata(tmp_path):
    result=fitted();name=result.param_names[0]
    b=BootstrapResult(result.model,{name:np.r_[np.full(40,1e-9),np.linspace(.1,4.,160)]})
    table=b.limits();assert np.isnan(table['lolim'][0])
    result.set_limits(table);result.save(tmp_path/'fit.fits')
    restored=FitResult.load(tmp_path/'fit.fits')
    assert np.isnan(restored.params[name].lolim)
    assert restored.limits_info['boundary'][name]==['lower']
    assert np.isnan(restored.params[name].errors).all()


def test_array_parameter_summaries_preserve_shapes():
    model=Linear1D(slope=[1.,2.],intercept=[3.,4.],n_models=2)
    samples={f'{name}[{i}]':np.arange(5.)+i for name in model.param_names for i in range(2)}
    result=BootstrapResult(model,samples)
    table=result.limits(confidence=90)
    np.testing.assert_allclose(table['lolim'][0],np.percentile(np.column_stack([samples['slope[0]'],samples['slope[1]']]),5,axis=0))
    assert result.params['slope'].value.shape==(2,)
    assert table.meta['n_success']==5
