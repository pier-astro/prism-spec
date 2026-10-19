"""LM prepared evaluation and consistent least-squares diagnostics."""
import numpy as np
import pytest
from astropy.modeling import FittableModel, Parameter
from astropy.modeling.models import Gaussian1D, Const1D
from scipy.integrate import trapezoid
from prism.modeling import Model
from prism.modeling.fitting import LMFitter, SeparableTRF, ScipyTRF, SherpaLM, HAS_SHERPA


@pytest.mark.parametrize('bounds',[(None,None),(.1,None),(None,8.),(.1,8.)])
def test_lm_bounds_and_covariance_match_weighted_linear_solution(bounds):
    x=np.linspace(-3.,3.,101)
    profile=np.exp(-.5*(x/.8)**2)
    error=np.linspace(.1,.2,len(x))
    y=3.*profile+.03*np.sin(4*x)
    model=Model.from_astropy(Gaussian1D(1.,0.,.8,fixed={'mean':True,'stddev':True},bounds={'amplitude':bounds}))
    initial=model.parameters.copy()
    fitter=LMFitter(calc_uncertainties=True)
    result=fitter(model,x,y,yerr=error,inplace=False)
    expected=np.sum(profile*y/error**2)/np.sum(profile**2/error**2)
    residual=(expected*profile-y)/error
    variance=np.dot(residual,residual)/(len(x)-1)/np.sum(profile**2/error**2)
    assert fitter.fit_info['success']
    assert result.amplitude.value==pytest.approx(expected,rel=1e-6)
    assert result.covariance[0,0]==pytest.approx(variance,rel=2e-5)
    np.testing.assert_array_equal(model.parameters,initial)


class Area(FittableModel):
    n_inputs=2;n_outputs=1
    norm=Parameter(default=1.,bounds=(.1,10.))
    @staticmethod
    def evaluate(x,flux,norm):return norm*flux/trapezoid(flux,x)


def test_lm_mask_preserves_whole_profile_normalization():
    x=np.linspace(-3.,3.,151)
    source=Gaussian1D(1.,.1,.8,fixed={'amplitude':True,'mean':True,'stddev':True})
    model=Model.from_astropy(source)|Model.from_astropy(Area())
    profile=np.exp(-.5*((x-.1)/.8)**2);profile/=trapezoid(profile,x)
    valid=(x<-.5)|(x>1.4)
    y=4.*profile;y[~valid]=np.nan
    fitter=LMFitter()
    result=fitter(model,x,y,mask=valid)
    assert result.norm_1.value==pytest.approx(4.,rel=1e-6)
    assert fitter.fit_info['ndata']==sum(valid)


@pytest.mark.parametrize('engine',[LMFitter,SeparableTRF])
def test_fixed_model_reports_actual_weighted_residual_without_optimization(engine):
    x=np.arange(4.)
    model=Model.from_astropy(Const1D(1.,fixed={'amplitude':True}))
    error=np.array([1.,2.,1.,1.]);y=np.array([2.,3.,np.nan,2.])
    valid=np.array([True,True,False,True])
    fitter=engine();result=fitter(model,x,y,yerr=error,mask=valid)
    assert fitter.fit_info['nfev']==0
    assert fitter.fit_info['success']
    assert fitter.fit_info['stat']==pytest.approx(3.)
    assert fitter.fit_info['cost']==pytest.approx(1.5)
    assert fitter.fit_info['ndata']==3 and fitter.fit_info['dof']==3
    assert result.amplitude.value==1.


def test_lm_incomplete_fit_retains_finite_residual_diagnostics():
    x=np.linspace(-3.,3.,101)
    truth=Gaussian1D(3.,.3,.6)(x)
    fitter=LMFitter(maxfev=1)
    result=fitter(Model.from_astropy(Gaussian1D(.5,1.,1.4)),x,truth)
    assert not fitter.fit_info['success']
    expected=np.sum((result(x)-truth)**2)
    assert np.isfinite(fitter.fit_info['stat'])
    assert fitter.fit_info['stat']==pytest.approx(expected)
    assert fitter.fit_info['cost']==pytest.approx(.5*expected)
    assert fitter.fit_info['param_cov'] is None


@pytest.mark.parametrize('engine',[ScipyTRF,LMFitter,SeparableTRF,
    pytest.param(SherpaLM,marks=pytest.mark.skipif(not HAS_SHERPA,reason='Sherpa unavailable'))])
def test_least_squares_cost_and_stat_share_convention(engine):
    x=np.arange(5.);y=np.array([1.,2.,4.,2.,1.]);error=np.full(5,.3)
    fitter=engine();result=fitter(Const1D(.5),x,y,yerr=error)
    expected=np.sum(((result(x)-y)/error)**2)
    assert fitter.fit_info['success']
    assert fitter.fit_info['stat']==pytest.approx(expected)
    assert fitter.fit_info['cost']==pytest.approx(.5*expected)
