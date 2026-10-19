"""Independent checks for fitting sample selection and residual weights."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import FittableModel, Parameter
from astropy.modeling.models import Const1D, Gaussian1D
from scipy.integrate import trapezoid
from prism.modeling import Model
from prism.modeling.fitting import (ScipyTRF, SherpaLM, HAS_SHERPA,
    TRFLSQFitter, LMFitter, SeparableTRF, FittingWithOutlierRemoval)
from prism.modeling.fitting.data import _prepare_fit_data

ENGINES=[ScipyTRF,pytest.param(SherpaLM,marks=pytest.mark.skipif(not HAS_SHERPA,reason='Sherpa unavailable'))]


@pytest.mark.parametrize('engine',ENGINES)
def test_weighted_mean_uses_errors_through_outlier_wrapper(engine):
    x=np.arange(3.)
    y=np.array([1.,2.,9.]);error=np.array([1.,2.,10.])
    expected=np.sum(y/error**2)/np.sum(1/error**2)
    fitter=FittingWithOutlierRemoval(engine(),niter=0)
    result=fitter(Model.from_astropy(Const1D(0.)),x,y,yerr=error)
    assert result.amplitude.value==pytest.approx(expected,rel=1e-6)


@pytest.mark.parametrize('engine',ENGINES)
@pytest.mark.parametrize('masked',[True,False])
def test_errors_cannot_reintroduce_rejected_samples(engine,masked):
    x=np.arange(4.)
    y=np.array([2.,2.,2.,30.]);error=np.array([1.,2.,3.,1.])
    reject=np.array([False,False,False,True])
    def clip(residual):
        return np.ma.array(residual,mask=reject) if masked else reject
    fitter=FittingWithOutlierRemoval(engine(),outlier_func=clip,niter=1)
    result=fitter(Model.from_astropy(Const1D(1.)),x,y,yerr=error)
    assert fitter.outlier_mask[-1]
    assert result.amplitude.value==pytest.approx(2.,rel=1e-6)
    assert fitter.fit_info['ndata']==3


@pytest.mark.parametrize('engine',ENGINES)
def test_zero_weights_exclude_nan_and_override_errors(engine):
    fitter=engine()
    result=fitter(Model.from_astropy(Const1D(0.)),np.arange(3.),np.array([1.,2.,np.nan]),
                  weights=np.array([1.,2.,0.]),yerr=np.ones(3))
    assert result.amplitude.value==pytest.approx(1.8,rel=1e-6)
    assert fitter.fit_info['ndata']==2
    assert fitter.fit_info['dof']==1


@pytest.mark.parametrize('engine',ENGINES)
def test_masked_arrays_and_explicit_filter_share_selection(engine):
    x=np.arange(5.);x[3]=np.nan
    y=np.ma.array([1.,2.,100.,100.,np.nan],mask=[False,False,True,False,False])
    fitter=engine()
    result=fitter(Model.from_astropy(Const1D(0.)),x,y,filter_non_finite=True)
    assert result.amplitude.value==pytest.approx(1.5,rel=1e-6)
    assert fitter.fit_info['ndata']==2


@pytest.mark.parametrize('field,value,message',[
    ('yerr',[1.,0.,1.],'strictly positive'),
    ('yerr',[1.,-1.,1.],'strictly positive'),
    ('weights',[1.,-1.,1.],'nonnegative'),
    ('yerr',[1.,np.nan,1.],'Non-finite'),
    ('x',[0.,np.nan,2.],'Non-finite'),
    ('yerr',[[1.,1.,1.],[1.,1.,1.]],'Asymmetric'),
    ('weights',[1.,1.],'same shape'),
])
def test_invalid_included_data_fail_before_minimization(field,value,message):
    inputs=dict(x=np.arange(3.),y=np.ones(3));inputs[field]=np.asarray(value)
    with pytest.raises(ValueError,match=message):_prepare_fit_data(**inputs)


def test_explicit_mask_excludes_invalid_values_and_keeps_units():
    x=np.arange(3.)*u.AA;y=np.array([1.,2.,np.nan])*u.Jy
    a,b,_,_,valid=_prepare_fit_data(x,y,mask=[True,True,False])
    assert a.unit==u.AA and b.unit==u.Jy
    np.testing.assert_array_equal(valid,[True,True,False])


class Area(FittableModel):
    n_inputs=2;n_outputs=1
    norm=Parameter(default=1.)
    @staticmethod
    def evaluate(x,flux,norm):return norm*flux/trapezoid(flux,x)


@pytest.mark.parametrize('engine',ENGINES)
def test_profile_transform_keeps_full_grid_when_samples_are_masked(engine):
    x=np.linspace(-3.,3.,151)
    source=Gaussian1D(1.,.1,.8,fixed={'amplitude':True,'mean':True,'stddev':True})
    model=Model.from_astropy(source)|Model.from_astropy(Area())
    profile=np.exp(-.5*((x-.1)/.8)**2)
    truth=4.*profile/trapezoid(profile,x)
    mask=(x<-.5)|(x>1.4)
    y=truth.copy();y[~mask]=np.nan
    fitter=engine()
    result=fitter(model,x,y,mask=mask)
    assert result.norm_1.value==pytest.approx(4.,rel=1e-6)
    assert trapezoid(result(x),x)==pytest.approx(4.,rel=1e-6)
    assert fitter.fit_info['ndata']==np.sum(mask)


@pytest.mark.parametrize('engine',ENGINES)
@pytest.mark.parametrize('nproc',[1,2])
def test_cube_mask_keeps_grid_for_profile_wide_transform(engine,nproc):
    x=np.linspace(-3.,3.,101)
    source=Gaussian1D(1.,.1,.8,fixed={'amplitude':True,'mean':True,'stddev':True})
    model=Model.from_astropy(source)|Model.from_astropy(Area())
    p=np.exp(-.5*((x-.1)/.8)**2);p/=trapezoid(p,x)
    y=np.array([3.,5.])[:,None]*p
    mask=np.ones_like(y,dtype=bool);mask[:,35:65]=False
    y[~mask]=np.nan
    fitted=engine().multifit(model,x,y,mask=mask,nproc=nproc,progress=False)
    np.testing.assert_allclose(fitted._params['norm_1'],[3.,5.],rtol=1e-6)


@pytest.mark.parametrize('engine',[TRFLSQFitter,LMFitter,SeparableTRF])
def test_legacy_adapters_share_one_dimensional_mask_contract(engine):
    x=np.arange(4.);y=np.array([2.,2.,np.nan,2.])
    fitted=engine()(Const1D(0.),x,y,mask=np.array([True,True,False,True]))
    assert fitted.amplitude.value==pytest.approx(2.,rel=1e-6)
