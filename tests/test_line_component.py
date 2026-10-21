from line_helpers import reference_profile
"""Native line scopes, weighted chain rule, kernels and saved reconstruction."""
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from prism.modeling import LineSet
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.io import load_model
from prism.modeling.lines import profiles


def table():
    return {'name': ['Hb', 'OIII', 'OIII'],
            'position': [4861.3, 4958.9, 5006.8]*u.AA,
            'weight': [1., 1/2.98, 1.]}


def gaussian_deriv(x, amplitude, center, sigma):
    return profiles.gaussian_deriv(x, amplitude, center, sigma)[1:]


def group(analytic=True):
    return LineSet(profiles.gaussian, table(), shared={'sigma': 2.},
                         parameters={'Hb': {'amplitude': 2.}, 'OIII': {'amplitude': 3.}},
                         derivative=gaussian_deriv if analytic else None, name='narrow')


def test_template_scopes_and_analytic_chain_rule():
    g = group()
    assert g.param_names == ('sigma', 'amp_hb', 'amp_oiii')
    x = np.linspace(4800, 5060, 1600)
    expected = sum(profiles.gaussian(x, a*w, p, 2.) for a,w,p in
                   zip([2,3,3],table()['weight'],table()['position'].value))
    assert_allclose(g(x), expected)
    jac = g.fit_deriv(x, *g.parameters)
    for i in range(len(g.parameters)):
        upper, lower = g.parameters, g.parameters
        upper[i] += 1e-5
        lower[i] -= 1e-5
        assert_allclose((g.evaluate(x,*upper)-g.evaluate(x,*lower))/2e-5,
                        jac[i], rtol=1e-7, atol=1e-9)
    # A changed shared width affects all three transitions through one parameter.
    g.sigma = 3.
    assert len(g.parameters) == 3


@pytest.mark.parametrize('analytic',[True,False])
@pytest.mark.parametrize('engine',[TRF,pytest.param(SherpaLM,marks=pytest.mark.skipif(not HAS_SHERPA,reason='Sherpa unavailable'))])
def test_known_input_recovery(engine,analytic):
    x=np.linspace(4800,5060,1600)
    truth=group();truth.parameters=[1.4,4.,7.]
    g=group(analytic);g.sigma.bounds=(.3,8)
    fitted=engine()(g,x,truth(x),inplace=False)
    assert_allclose(fitted.parameters,truth.parameters,atol=1e-6)
    assert_allclose(g.parameters,[2,2,3])


@pytest.mark.parametrize('kind,widths',[
    ('gaussian',{'fwhm':280.}),
    ('lorentzian',{'fwhm':280.}),
    ('voigt',{'fwhm_G':180.,'fwhm_L':240.})])
@pytest.mark.parametrize('domain',['wavelength','linear'])
def test_builtin_kernels_match_independent_reference(kind,widths,domain):
    t=table()
    if domain=='linear':t['position']=[1.,1.2,1.4]*u.keV
    shared=dict(offset=-85.,**widths,redshift=.08)
    profile=partial(getattr(profiles,'_'+kind+'_line'),instfwhm=150.,domain_family=domain)
    derivative=partial(getattr(profiles,'_'+kind+'_line_deriv'),instfwhm=150.,domain_family=domain)
    g=LineSet(profile,t,shared=shared,
                    parameters={'Hb':{'amplitude':2.},'OIII':{'amplitude':3.}},
                    position='position',derivative=derivative,name='lines')
    lo,hi=(4800,5600) if domain=='wavelength' else (.8,1.6)
    x=np.linspace(lo,hi,2500)
    physical_widths=([280.,0.] if kind=='gaussian' else [0.,280.] if kind=='lorentzian' else [180.,240.])
    expected=sum(reference_profile(x,a*w,p,-85.,physical_widths,.08,150.,t['position'].unit)
                 for a,w,p in zip([2.,3.,3.],t['weight'],t['position'].value))
    assert_allclose(g(x),expected,rtol=1e-12,atol=1e-12)
    jac=g.fit_deriv(x,*g.parameters)
    for i in range(len(g.parameters)):
        upper,lower=g.parameters,g.parameters
        delta=max(1e-6,abs(upper[i])*1e-6)
        upper[i]+=delta;lower[i]-=delta
        assert_allclose((g.evaluate(x,*upper)-g.evaluate(x,*lower))/(2*delta),
                        jac[i],rtol=5e-5,atol=1e-7)


def test_state_restores_saved_table_scopes_and_links(tmp_path):
    g=group();g.amp_oiii.link=g.amp_hb*2
    g.sigma.fixed=True
    path=tmp_path/'lines.fits';g.save(path)
    with pytest.raises(ValueError,match='Supply the custom LineSet'):
        load_model(path)
    # The definition supplies code; saved membership and weights supply the setup.
    definition=group();definition.component._setup['lines']['weight']=[1,1,1]
    restored=load_model(path,components={'narrow':definition})
    x=np.linspace(4800,5060,500)
    assert_allclose(restored(x),g(x))
    restored.amp_hb=5
    assert_allclose(restored.amp_oiii.value,10)
    assert restored.sigma.fixed
    assert_allclose(g.amp_hb.value,2)


def test_omitted_weights_preserve_membership():
    t=table();del t['weight']
    g=LineSet(profiles.gaussian,t,shared={'sigma':2},
                    parameters={'Hb':{'amplitude':2},'OIII':{'amplitude':3}})
    assert len(g.param_names)==3


@pytest.mark.parametrize('shared,local',[
    ({'sigma':2,'amplitude':1},{'Hb':{},'OIII':{}}),
    ({'sigma':2},{'Hb':{'amplitude':1}}),
    ({'sigma':2},{'Hb':{'amplitude':1,'sigma':3},'OIII':{'amplitude':2}})])
def test_invalid_scopes(shared,local):
    with pytest.raises(ValueError):
        LineSet(profiles.gaussian,table(),shared=shared,parameters=local)


def exponential(x, flux, center, width):
    return flux*np.exp(-np.abs(x-center)/width)/(2*width)


def test_custom_flux_scale_and_independent_shape():
    t={'name':['A','A','B'],'position':[10,20,30]*u.AA,'weight':[.25,.75,1]}
    g=LineSet(exponential,t,shared={},
                    parameters={'A':{'flux':8.,'width':.5},'B':{'flux':3.,'width':1.}},
                    scale='flux',name='wings')
    x=np.linspace(0,50,50000)
    assert_allclose(np.trapezoid(g(x),x),11,rtol=1e-6)
    assert g.param_names==('flux_a','width_a','flux_b','width_b')
