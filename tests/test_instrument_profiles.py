from line_helpers import reference_profile
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose
from prism.modeling import Component,LineSet
from prism.modeling.lines import GaussianProfile, LorentzianProfile, VoigtProfile
from prism.modeling.io import load_model


@pytest.mark.parametrize('profile_class,widths',[(GaussianProfile,[280.]),(LorentzianProfile,[280.]),(VoigtProfile,[180.,240.])])
@pytest.mark.parametrize('unit,position,window',[(u.AA,5000.,[4800.,5600.]),(u.keV,1.2,[.9,1.4])])
def test_variable_instrument_profile_jacobian_and_roundtrip(profile_class,widths,unit,position,window,tmp_path):
    curve=np.array([[window[0],100.],[window[1],250.]])
    profile=profile_class(instfwhm=curve,position_unit=unit)
    params=dict(amplitude=2.,position=position,offset=-85.)
    params.update(dict(fwhm=widths[0]) if len(widths)==1 else dict(fwhm_G=widths[0],fwhm_L=widths[1]))
    params['redshift']=.02
    c=Component(profile,params,derivative=profile.derivative,name='line')
    x=np.linspace(*window,2000)
    center=position*((1+.02)*np.exp(-85/299792.458) if unit.is_equivalent(u.m) else 1/((1+.02)*np.exp(-85/299792.458)))
    instrumental=np.interp(center,curve[:,0],curve[:,1])
    physical_widths=([widths[0],0.] if profile_class is GaussianProfile else
                     [0.,widths[0]] if profile_class is LorentzianProfile else widths)
    expected=reference_profile(x,2.,position,-85.,physical_widths,.02,instrumental,unit)
    assert_allclose(c(x),expected,rtol=1e-12,atol=1e-12)
    jac=c.fit_deriv(x,*c.parameters)
    for i in range(len(c.parameters)):
        upper,lower=c.parameters,c.parameters
        step=max(1e-7,abs(upper[i])*1e-7)
        upper[i]+=step;lower[i]-=step
        assert_allclose((c.evaluate(x,*upper)-c.evaluate(x,*lower))/(2*step),jac[i],rtol=3e-4,atol=2e-7)
    path=tmp_path/'line.fits';c.save(path)
    restored=load_model(path)
    assert_allclose(restored(x),c(x),rtol=1e-13,atol=1e-13)
    assert_allclose(restored.fit_deriv(x,*restored.parameters),jac)


def test_gaussian_quadrature_at_observed_center():
    profile=GaussianProfile(instfwhm=[[4800,100],[5200,200]])
    position,offset,fwhm,redshift=5000.,-85.,280.,.01
    center=position*(1+redshift)*np.exp(offset/299792.458)
    instrumental=100+(center-4800)*.25
    effective=np.hypot(fwhm,instrumental)*center/299792.458
    x=np.array([center,center-effective/2,center+effective/2])
    assert_allclose(profile(x,2.,position,offset,fwhm,redshift),np.array([2.,1.,1.])/(1+redshift),rtol=1e-12)


def test_native_group_restores_instrument_curve_without_definition(tmp_path):
    profile=GaussianProfile(instfwhm=[[4800,100],[5200,200]])
    group=LineSet(profile,dict(name=['A','A'],position=[4959,5007]*u.AA,weight=[1/3,1]),
                        shared=dict(offset=0.,fwhm=300.,redshift=0.),parameters={'A':{'amplitude':3.}},
                        position='position',derivative=profile.derivative,name='narrow')
    x=np.linspace(4900,5050,500)
    path=tmp_path/'group.fits';group.save(path)
    restored=load_model(path)
    assert_allclose(restored(x),group(x))
    assert_allclose(restored.fit_deriv(x,*restored.parameters),group.fit_deriv(x,*group.parameters))


def test_reject_inconsistent_coordinate_units():
    with pytest.raises(ValueError,match='same numeric spectral unit'):
        LineSet(GaussianProfile(),dict(name=['A'],position=[500.]*u.nm),
                      shared=dict(offset=0.,fwhm=300.,redshift=0.),parameters={'A':{'amplitude':3.}},position='position')
