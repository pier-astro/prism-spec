from prism.modeling.lines.measurements import lines
"""Independent integrals, velocity derivatives and graph contracts for flux scales."""
import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad
from prism.modeling import select_line
from prism.modeling.lines import GaussianLines, LorentzianLines, VoigtLines, LineSet
from prism.modeling.fitting import TRF
from prism.modeling.io import load_model

CASES=[(GaussianLines,dict(fwhm=230.)),(LorentzianLines,dict(fwhm=230.)),
       (VoigtLines,dict(fwhm_G=180.,fwhm_L=210.))]

@pytest.mark.parametrize('factory,widths',CASES)
@pytest.mark.parametrize('unit,position',[(u.AA,5000.),(u.nm,500.),(u.eV,2.5)])
def test_unit_area_shift_and_instrument_independence(factory,widths,unit,position):
    model=factory(dict(name=['A'],position=[position]*unit),flux=7.3,
                  redshift=.11,offset=78.,instfwhm=130.,**widths)
    profile=model.component._profile
    args=[7.3,position,78.,*widths.values(),.11]
    center,width=profile._center_width(*args)
    # Integrate the actual evaluated density, not the analytic flux helper.
    measured=quad(lambda t: float(profile(np.array([center+width*t]),*args)[0])*width,
                  -np.inf,np.inf,epsabs=1e-8)[0]
    assert measured==pytest.approx(7.3,rel=2e-8)
    assert lines(model).flux()['flux'][0]==pytest.approx(7.3)
    assert select_line(model,'A').flux().value==pytest.approx(7.3)

@pytest.mark.parametrize('factory,widths',CASES)
@pytest.mark.parametrize('unit,position',[(u.AA,5000.),(u.eV,2.5)])
def test_analytic_derivatives_include_width_and_chromatic_normalization(factory,widths,unit,position):
    curve=np.array([[position*.9,90.],[position*1.3,230.]])
    model=factory(dict(name=['A'],position=[position]*unit),flux=-3.,offset=60.,redshift=.07,
                  instfwhm=curve,**widths)
    profile=model.component._profile
    args=np.array([-3.,position,60.,*widths.values(),.07])
    center,width=profile._center_width(*args)
    x=np.linspace(center-3*width,center+3*width,301)
    analytic=np.asarray(profile.derivative(x,*args))
    for index,value in enumerate(args):
        h=max(abs(value)*1e-5,1e-6) if index!=1 else abs(value)*1e-7
        up=args.copy();down=args.copy();up[index]+=h;down[index]-=h
        numeric=(profile(x,*up)-profile(x,*down))/(2*h)
        np.testing.assert_allclose(analytic[index],numeric,rtol=3e-4,
                                   atol=max(1e-10,np.max(np.abs(numeric))*2e-6))

@pytest.mark.parametrize('factory,widths',CASES)
def test_doublet_flux_ratio_and_roundtrip_with_links(factory,widths,tmp_path):
    table=dict(name=['OIII','OIII','Hb'],position=[4959.,5007.,4861.]*u.AA,weight=[1/2.98,1.,1.])
    model=factory(table,flux=5.,bounds={'flux':(0.,100.)},name='lines',**widths)
    assert model.normalization=='flux' and model.flux_oiii.bounds==(0.,100.)
    assert 'amp_oiii' not in model.param_names
    model.flux_hb.link=2*model.flux_oiii
    fluxes=lines(model).flux()['flux']
    assert fluxes[1]/fluxes[0]==pytest.approx(2.98)
    for name in model.param_names:
        if not name.startswith('flux_'):getattr(model,name).fixed=True
    truth=model.copy();truth.flux_oiii=12.
    x=np.linspace(4800.,5050.,2001)
    fitted=TRF()(model,x,truth(x),yerr=np.full(len(x),.01))
    assert fitted.fit_info['success']
    assert fitted.model.flux_oiii.value==pytest.approx(12.,rel=1e-7)
    assert fitted.model.flux_hb.value==pytest.approx(24.,rel=1e-7)
    metric=select_line(fitted,'OIII',position=4959.).flux()
    assert metric.value==pytest.approx(12./2.98)
    assert metric.std>0
    path=tmp_path/'lines.fits';fitted.model.save(path);loaded=load_model(path)
    assert loaded.normalization=='flux'
    np.testing.assert_allclose(loaded(x),fitted.predict(x),rtol=1e-13)
    np.testing.assert_allclose(loaded.fit_deriv(x,*loaded.parameters),fitted.model.fit_deriv(x,*fitted.parameters))


def test_amplitude_mode_still_has_its_explicit_peak_semantics():
    table=dict(name=['OIII','OIII'],position=[4959.,5007.]*u.AA,weight=[1/3,1.])
    model=GaussianLines(table,normalization='amplitude',amplitude=2.,fwhm=200.)
    assert model.normalization=='amplitude'
    assert lines(model).flux()['flux'][1]/lines(model).flux()['flux'][0]==pytest.approx(3*5007/4959)
    with pytest.raises(ValueError,match='consistently'):GaussianLines(table,amplitude=2.)
    with pytest.raises(ValueError,match='consistently'):GaussianLines(table,normalization='amplitude',flux=2.)
    with pytest.raises(ValueError,match='normalization'):GaussianLines(table,normalization='other')
    with pytest.raises(KeyError):GaussianLines(table,bounds={'amplitude':(0,10)})


def test_zero_width_cannot_represent_finite_integrated_flux():
    model=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),fwhm=0.)
    with pytest.raises(ValueError,match='positive total width'):model(np.array([5000.]))


def test_flux_does_not_change_generic_lineset_contract():
    def profile(x,flux,center,sigma):return flux*np.exp(-.5*((x-center)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    model=LineSet(profile,dict(name=['A'],position=[5000.]*u.AA),shared={'sigma':2.},
                  parameters={'A':{'flux':3.}},scale='flux')
    assert model.flux_a.value==3.
    assert 'normalization' not in model.param_names


@pytest.mark.parametrize('factory,widths',CASES)
def test_response_preserves_intrinsic_flux_and_can_be_saved(factory,widths,tmp_path):
    from prism.modeling.operators import InstrumentResponse, SpectralResponse
    x=np.linspace(4900.,5100.,2001)
    source=factory(dict(name=['A'],position=[5000.]*u.AA),flux=9.,name='source',**widths)
    response=SpectralResponse(instrument=InstrumentResponse.from_fixed_sigma(x,.9),wave=x)
    expression=source|response
    np.testing.assert_allclose(expression(x),response(source(x)),rtol=1e-13)
    widths_bin=np.full(len(x),x[1]-x[0])
    # The operator conserves the sampled finite-domain integral apart from edge loss.
    assert widths_bin@expression(x)<=9.*1.000001
    assert select_line(expression,'A',components='source').flux().value==pytest.approx(9.)
    path=tmp_path/'response.fits';expression.save(path);loaded=load_model(path)
    np.testing.assert_allclose(loaded(x),expression(x),rtol=1e-13)


@pytest.mark.parametrize('nproc',[1,2])
def test_flux_parameters_survive_parallel_multifit(nproc):
    x=np.linspace(4960.,5040.,301)
    model=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),flux=1.,fwhm=250.,
                        name='line',bounds={'flux':(0.,100.)})
    model.offset.fixed=model.fwhm.fixed=True
    truth=model.copy();truth.flux_a=8.
    data=np.array([truth(x),2*truth(x),3*truth(x)])
    results=TRF().multifit(model,x,data,yerr=np.full_like(data,.1),nproc=nproc,progress=False)
    np.testing.assert_allclose(results.params['flux_a'].value,[8.,16.,24.],rtol=1e-7)
    assert results.get_model(0).normalization=='flux'
