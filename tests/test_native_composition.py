"""Native expression semantics, operators, fitting and portable sessions."""
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import FittableModel, Parameter, CompoundModel
from astropy.modeling.models import Gaussian1D, Const1D, Planar2D
from scipy.sparse import csr_matrix, eye
from scipy.integrate import trapezoid
from prism.modeling import Model, LinearOperatorModel
from prism.modeling.models import GaussianLine, get_components, format_model_text
from prism.modeling.fitting import ScipyTRF, SherpaLM, SeparableTRF, FittingWithOutlierRemoval, HAS_SHERPA
from prism.modeling.fitting.evaluation import _ModelEvaluation
from prism.modeling.io import load_model
from prism.modeling.line_analysis import select_line


_ENGINES = [ScipyTRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


class Gain(LinearOperatorModel):
    gain = Parameter(default=.4, bounds=(.01, 5.))
    def get_matrix(self, x, *parameters):
        return self.matrix*parameters[0]


class Normalize(FittableModel):
    """Grid-preserving, nonlinear profile normalization, like a flux transform."""
    n_inputs=2
    n_outputs=1
    norm=Parameter(default=2., bounds=(.01,10.))
    @staticmethod
    def evaluate(x, flux, norm):
        return norm*flux/trapezoid(flux,x)


def response(x):
    return LinearOperatorModel(eye(len(x),format='csr'),x=x,name='response')


def lines():
    a=GaussianLine(amplitude=2.,position=5000.,fwhm=250.,name='narrow')
    b=GaussianLine(amplitude=.7,position=5000.,fwhm=500.,name='broad')
    b.offset.link=a.offset
    b.amplitude.link=.4*a.amplitude
    return a,b


@pytest.mark.parametrize('op', ['+','-','*','/'])
def test_arithmetic_matches_independent_formula_and_central_jacobian(op):
    x=np.linspace(-3,3,101)
    a=Gaussian1D(2,-.7,.6,name='a');b=Gaussian1D(.3,.8,.9,name='b')
    native=Model.from_astropy(a)
    expression={'+':native+b,'-':native-b,'*':native*b,'/':native/(b+Const1D(1.))}[op]
    pa=2*np.exp(-.5*((x+.7)/.6)**2)
    pb=.3*np.exp(-.5*((x-.8)/.9)**2)
    expected={'+':pa+pb,'-':pa-pb,'*':pa*pb,'/':pa/(pb+1.)}[op]
    np.testing.assert_allclose(expression(x),expected,rtol=1e-14,atol=1e-14)
    p=expression.parameters.copy();evaluation=_ModelEvaluation(expression,x)
    jac=evaluation.fit_deriv(p)
    for i,value in enumerate(p):
        step=1e-5*max(1,abs(value));high=p.copy();low=p.copy();high[i]+=step;low[i]-=step
        np.testing.assert_allclose(jac[i],(evaluation(high)-evaluation(low))/(2*step),rtol=1e-5,atol=1e-8)


def test_negative_base_integer_power_evaluates_without_fake_log_derivative():
    x=np.linspace(-1,1,11)
    model=Model.from_astropy(Const1D(-2.))**2
    np.testing.assert_array_equal(model(x),np.full(len(x),4.))
    assert model.fit_deriv is None


def test_model_preserves_reference_identity_complete_copy_and_constraints():
    a,b=lines();model=Model.from_astropy(a)+b
    assert model['narrow'] is a
    a.amplitude.value=4
    assert model.amplitude_1.value==1.6
    model['narrow'].name='renamed'
    duplicate=model.copy();duplicate.amplitude_0.value=3
    assert duplicate.amplitude_1.value==pytest.approx(1.2)
    assert model.amplitude_1.value==1.6
    assert duplicate['renamed'] is not model['renamed']
    with pytest.raises(TypeError):model.fixed['amplitude_0']=True
    model.amplitude_0.fixed=True
    assert model.fixed['amplitude_0']


def test_rectangular_operator_nested_grid_and_no_astropy_compound_patch():
    x=np.linspace(-4,4,81);out=x[::4]
    matrix=np.zeros((len(out),len(x)))
    for i in range(len(out)):matrix[i,4*i]=1.
    op=LinearOperatorModel(csr_matrix(matrix),x=x,xout=out)
    a=Gaussian1D(2,.2,.7);b=Gaussian1D(.3,1.5,.4)
    model=((Model.from_astropy(a)+b)|response(x))|op
    expected=2*np.exp(-.5*((out-.2)/.7)**2)+.3*np.exp(-.5*((out-1.5)/.4)**2)
    assert not getattr(CompoundModel, '_prism_linear_operator_patch', False)
    np.testing.assert_allclose(model(out),expected)
    evaluation=_ModelEvaluation(model,out)
    np.testing.assert_allclose(evaluation(model.parameters),expected)
    assert evaluation.fit_deriv(model.parameters).shape==(6,len(out))


def test_operator_order_is_not_commuted_with_a_multiplicative_factor():
    x=np.linspace(-4,4,81)
    matrix=np.eye(len(x))*.5+np.eye(len(x),k=2)*.5
    op=LinearOperatorModel(csr_matrix(matrix),x=x)
    a=Model.from_astropy(Gaussian1D(2.,0.,.3))
    t=1.-Model.from_astropy(Gaussian1D(.7,.4,1.))
    before=(t*a)|op;after=t*(a|op)
    np.testing.assert_allclose(before(x),matrix@(t(x)*a(x)))
    np.testing.assert_allclose(after(x),t(x)*(matrix@a(x)))
    assert np.max(abs(before(x)-after(x)))>.01


@pytest.mark.parametrize('engine',_ENGINES)
def test_cold_fit_linked_source_and_response_with_covariance(engine,tmp_path):
    x=np.linspace(4970,5030,200);a,b=lines()
    a.offset.fixed=True;b.fwhm.fixed=True
    truth=(Model.from_astropy(a)+b)|response(x)
    truth.amplitude_0.value=4.;truth.fwhm_0.value=320.
    y=truth(x)
    start=truth.copy();start.amplitude_0.value=1.5;start.fwhm_0.value=200.
    start.amplitude_0.bounds=(.01,10);start.fwhm_0.bounds=(100,700)
    fitter=engine(calc_uncertainties=True)
    result=fitter(start,x,y,yerr=np.full(len(x),.02),inplace=False)
    assert isinstance(result,Model)
    np.testing.assert_allclose(result(x),y,atol=1e-6)
    assert result.amplitude_0.value==pytest.approx(4.,rel=1e-5)
    assert start.amplitude_0.value==1.5
    assert result.covariance.shape==(2,2)
    result.save(tmp_path/'fit.fits')
    loaded=load_model(tmp_path/'fit.fits')
    assert isinstance(loaded,Model)
    np.testing.assert_array_equal(loaded(x),result(x))
    np.testing.assert_array_equal(loaded.covariance,result.covariance)
    loaded.amplitude_0.value=6
    assert loaded.amplitude_1.value==pytest.approx(2.4)


@pytest.mark.parametrize('engine',_ENGINES)
def test_parameterized_linear_operator_is_not_frozen_and_includes_its_sensitivity(engine):
    x=np.linspace(-3,3,100)
    source=Gaussian1D(2.,.1,.7,fixed={'amplitude':True,'mean':True,'stddev':True})
    model=Model.from_astropy(source)|Gain(matrix=eye(len(x)),x=x)
    y=3.2*2*np.exp(-.5*((x-.1)/.7)**2)
    assert model.fit_deriv is None
    result=engine()(model,x,y,yerr=np.full(len(x),.02),inplace=False)
    assert result.gain_1.value==pytest.approx(3.2,rel=1e-6)
    np.testing.assert_allclose(result(x),y,atol=1e-6)


@pytest.mark.parametrize('engine',_ENGINES)
def test_nonlinear_flux_transform_and_instrument_response(engine):
    x=np.linspace(-5,5,201)
    source=Gaussian1D(1.,.2,.7,bounds={'stddev':(.2,2.)},fixed={'amplitude':True,'mean':True})
    model=(Model.from_astropy(source)|Model.from_astropy(Normalize(norm=4.)))|response(x)
    y=model(x)
    assert trapezoid(y,x)==pytest.approx(4.)
    start=model.copy();start.stddev_0.value=1.1;start.norm_1.value=2.
    result=engine()(start,x,y,yerr=np.full(len(x),.02),inplace=False)
    np.testing.assert_allclose(result(x),y,atol=1e-6)
    assert result.stddev_0.value==pytest.approx(.7,rel=1e-5)
    assert result.norm_1.value==pytest.approx(4.,rel=1e-5)
    with pytest.raises(ValueError,match='reconstruction rule'):
        result.save('/private/tmp/unsupported-custom-transform-prism.fits',overwrite=True)


def test_builtin_general_transform_roundtrips_without_assuming_linearity(tmp_path):
    x=np.linspace(-3,3,50)
    source=Model.from_astropy(Gaussian1D(2,.3,.7))
    model=source|Model.from_astropy(Planar2D(slope_x=.2,slope_y=3,intercept=1))
    np.testing.assert_allclose(model(x),.2*x+3*source(x)+1)
    assert model.fit_deriv is None
    assert len(get_components(model,additive=True).names)==1
    model.save(tmp_path/'general.fits')
    np.testing.assert_array_equal(load_model(tmp_path/'general.fits')(x),model(x))


def test_unit_aware_ordinary_evaluation_and_native_operator_coordinates():
    x=np.linspace(4990,5010,101)*u.AA
    source=GaussianLine(amplitude=2*u.Jy,position=5000*u.AA,fwhm=250.)
    model=Model.from_astropy(source)|LinearOperatorModel(eye(len(x)),x=x.to(u.nm))
    np.testing.assert_allclose(model(x).to_value(u.Jy),source(x).to_value(u.Jy))
    # Numeric kernels convert the declared native coordinate grid once.
    expected=source(x).to_value(u.Jy)
    np.testing.assert_allclose(_ModelEvaluation(model,x.to_value(u.nm))(model.parameters),expected)
    with pytest.raises(TypeError,match='numeric data'):
        ScipyTRF()(model,x,expected,yerr=np.ones(len(x)))


def test_selected_intrinsic_lines_preserve_parent_covariance_and_saved_links(tmp_path):
    x=np.linspace(4970,5030,1201);a,b=lines()
    a.offset.fixed=True;a.fwhm.fixed=True;b.fwhm.fixed=True
    model=(Model.from_astropy(a)+b)|response(x)
    model._param_cov=np.array([[.04]])
    model._fit_indices=np.array([0])
    selected=select_line(model,'Hb',components=['narrow','broad'])
    assert selected.source_model is model
    np.testing.assert_allclose(selected(x),a(x)+b(x))
    _,samples=selected.sample(method='covariance',n_samples=16,random_state=12,return_samples=True)
    assert np.all(np.isfinite(samples['flux']))
    assert np.std(samples['flux'])>0
    assert 'narrow' in format_model_text(model)


def test_separable_fit_and_outlier_rejection_use_the_native_expression():
    x=np.linspace(-3,3,101)
    source=Gaussian1D(.7,.1,.6,fixed={'mean':True,'stddev':True})
    model=Model.from_astropy(source)
    y=3.*np.exp(-.5*((x-.1)/.6)**2)
    fitted=SeparableTRF(linear_params=['amplitude'])(model,x,y,inplace=False)
    np.testing.assert_allclose(fitted(x),y,atol=1e-7)
    corrupted=y.copy();corrupted[30]+=20
    fitter=FittingWithOutlierRemoval(ScipyTRF(),niter=3,sigma=3)
    result=fitter(model,x,corrupted,yerr=np.full(len(x),.05),inplace=False)
    assert fitter.outlier_mask[30]
    assert result.amplitude.value==pytest.approx(3.,rel=1e-5)


def test_reused_fixed_operator_on_distinct_branches_and_duplicate_parameterized_leaf():
    x=np.linspace(-3,3,51);op=response(x)
    a=Model.from_astropy(Gaussian1D(1,-1,.4));b=Model.from_astropy(Gaussian1D(2,1,.5))
    model=(a|op)+(b|op)
    np.testing.assert_allclose(model(x),a(x)+b(x))
    with pytest.raises(ValueError,match='Repeated component'):
        _=a+a


def test_invalid_transform_placement_and_unsupported_astropy_composition_fail_early():
    transform=Model.from_astropy(Planar2D())
    with pytest.raises(TypeError,match='two-input'):
        _=transform+1
    with pytest.raises(TypeError,match='linear operator'):
        _=Model.from_astropy(Gaussian1D())|Model.from_astropy(Gaussian1D())
    with pytest.raises(ValueError,match='Unsupported Astropy'):
        Model.from_astropy(Gaussian1D() & Gaussian1D())


def test_additive_operator_terms_keep_links_to_the_full_parent():
    from prism.modeling.models import get_components
    x=np.linspace(4970,5030,101)
    a,b=lines()
    model=(Model.from_astropy(a)+b)|response(x)
    terms=get_components(model,additive=True)
    model.amplitude_0.value=5.
    np.testing.assert_allclose(sum(term(x) for term in terms),model(x))


@pytest.mark.parametrize('engine',_ENGINES)
def test_native_cube_serial_and_spawn_agree(engine):
    x=np.linspace(4970,5030,81)
    a,b=lines()
    for line in (a,b):
        line.offset.fixed=True
        line.fwhm.fixed=True
    model=(Model.from_astropy(a)+b)|response(x)
    amplitude=np.array([[1.,2.],[3.,4.]])
    model.amplitude_0.value=1.
    y=model(x)[:,None,None]*amplitude
    mask=np.ones_like(y,dtype=bool)
    mask[20:30,0,1]=False
    mask[:,1,1]=False
    options=dict(yerr=np.full_like(y,.1),mask=mask,spectral_axis=0,progress=False)
    serial=engine(calc_uncertainties=True).multifit(model,x,y,nproc=1,**options)
    parallel=engine(calc_uncertainties=True).multifit(model,x,y,nproc=2,**options)
    np.testing.assert_allclose(serial._params['amplitude_0'].reshape(2,2),[[1.,2.],[3.,np.nan]],rtol=1e-5,equal_nan=True)
    np.testing.assert_allclose(serial.covariance,parallel.covariance,rtol=1e-10,atol=1e-10,equal_nan=True)
    for name in model.param_names:
        np.testing.assert_allclose(serial._params[name],parallel._params[name],rtol=1e-10,atol=1e-10,equal_nan=True)


def test_scipy_covariance_is_symmetric_for_correlated_profile_parameters():
    x=np.linspace(-4.,4.,151)
    source=Model.from_astropy(Gaussian1D(1.,0.,.8))
    y=2.*np.exp(-.5*((x-.3)/.9)**2)
    fitted=ScipyTRF(calc_uncertainties=True)(source,x,y,yerr=np.full(len(x),.1),inplace=False)
    covariance=fitted.covariance
    np.testing.assert_array_equal(covariance,covariance.T)
    assert np.all(np.isfinite(covariance))
    assert np.min(np.linalg.eigvalsh(covariance))>0.
    # Compare with an independently written Gaussian derivative and Fisher matrix.
    profile=np.exp(-.5*((x-.3)/.9)**2)
    jac=np.array([profile,2.*profile*(x-.3)/.9**2,2.*profile*(x-.3)**2/.9**3]).T/.1
    np.testing.assert_allclose(covariance,np.linalg.inv(jac.T@jac),rtol=1e-7,atol=1e-12)
