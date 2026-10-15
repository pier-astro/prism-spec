"""Known-flux recovery, conditional uncertainties and extraction semantics."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.models import Gaussian2D, Const2D
from astropy.table import Table
from scipy.special import erf

from prism.data import Cube
from prism.modeling.psf import ChromaticPSF
from prism.modeling.extraction import aperture, optimal


def _source(background=0):
    wave=np.linspace(5000,7000,7)
    y,x=np.indices((11,11))
    center=(5.2,4.8)
    values=[]; profiles=[]
    flux=100+20*np.sin(np.linspace(0,np.pi,7))
    for w,f in zip(wave,flux):
        sx=1.3+.15*(w-6000)/1000
        px=.5*(erf((x+.5-center[1])/(np.sqrt(2)*sx))-erf((x-.5-center[1])/(np.sqrt(2)*sx)))
        py=.5*(erf((y+.5-center[0])/(np.sqrt(2)*1.5))-erf((y-.5-center[0])/(np.sqrt(2)*1.5)))
        profiles.append(px*py);values.append(f*px*py+background)
    model=Gaussian2D(x_mean=12,y_mean=13,x_stddev=1.3,y_stddev=1.5)
    for n in ['x_mean','y_mean','y_stddev','theta']:model.fixed[n]=True
    psf=ChromaticPSF(model,{'x_stddev':{'coefficients':[1.3,.15]}},[5000,7000],unit=u.AA,factor=15)
    cube=Cube(values,z=wave,zunit=u.AA,unit='erg / (s cm2 Angstrom)',err=np.full((7,11,11),.2),dtype=float)
    mask=(x-5)**2+(y-5)**2<=3**2
    return cube,psf,mask,center,flux,np.asarray(profiles)


def test_aperture_flux_and_uncertainty_with_fractional_pixels(tmp_path):
    cube,psf,mask,center,flux,profiles=_source()
    weights=mask.astype(float);weights[5,5]=.5
    spectrum,table=aperture(cube,psf,weights,center=center)
    np.testing.assert_allclose(spectrum.y,flux,rtol=4e-4)
    np.testing.assert_allclose(table['fraction'],np.sum(weights*profiles,axis=(1,2)),rtol=4e-4)
    np.testing.assert_allclose(spectrum.yerr,np.sqrt(np.sum((weights*cube.err)**2,axis=(1,2)))/table['fraction'])
    assert spectrum.dtype==np.dtype(float) and spectrum.yunit==cube.unit
    assert np.all(table['success'])
    table.write(tmp_path/'spectrum.ecsv')
    loaded=Table.read(tmp_path/'spectrum.ecsv')
    assert loaded['flux'].unit==cube.unit
    np.testing.assert_array_equal(loaded['fraction'],table['fraction'])


def test_optimal_recovers_point_source_and_constant_background(tmp_path):
    cube,psf,mask,center,flux,profiles=_source(background=3)
    spectrum,table=optimal(cube,psf,mask,center=center,background=Const2D())
    np.testing.assert_allclose(spectrum.y,flux,rtol=8e-4)
    np.testing.assert_allclose(table['background'],3,atol=8e-4)
    assert np.all(table['success']) and np.all(table['dof']==mask.sum()-2)
    assert np.all(np.asarray(table['covariance'])[:,0,1]<0)
    np.testing.assert_allclose(table['redchi2'],table['chi2']/table['dof'])
    assert np.all(spectrum.yerr>0)
    table.write(tmp_path/'optimal.ecsv')
    loaded=Table.read(tmp_path/'optimal.ecsv')
    np.testing.assert_allclose(loaded['covariance'],table['covariance'])
    assert loaded.meta['covnames']==['flux','background']
    # Composite aperture extraction intentionally retains and rescales host light.
    composite,diag=aperture(cube,psf,mask,center=center)
    np.testing.assert_allclose(composite.y,flux+3*mask.sum()/diag['fraction'],rtol=8e-4)


def test_invalid_pixels_do_not_silently_change_aperture():
    cube,psf,mask,center,flux,_=_source()
    cube.mask[2,5,5]=False
    cube.err[3,5,5]=0
    with pytest.warns(UserWarning,match='2 of 7'):
        spectrum,table=aperture(cube,psf,mask,center=center)
    assert not table['success'][2] and not table['success'][3]
    assert np.isnan(spectrum.y[[2,3]]).all()
    fitted,diagnostics=optimal(cube,psf,mask,center=center)
    assert np.all(diagnostics['success'])
    assert diagnostics['npix'][2]==mask.sum()-1
    np.testing.assert_allclose(fitted.y,flux,rtol=8e-4)


def test_single_pixel_equivalence_and_degeneracy():
    cube,psf,_,center,_,_=_source()
    mask=np.zeros(cube.shape[1:],dtype=bool);mask[5,5]=True
    summed,_=aperture(cube,psf,mask,center=center)
    fitted,table=optimal(cube,psf,mask,center=center)
    np.testing.assert_allclose(summed.y,fitted.y)
    np.testing.assert_allclose(summed.yerr,fitted.yerr)
    assert np.isnan(table['redchi2']).all() and np.all(table['dof']==0)
    with pytest.warns(UserWarning,match='7 of 7'):
        fitted,table=optimal(cube,psf,mask,center=center,background=Const2D())
    assert not np.any(table['success']) and np.isnan(fitted.y).all()


def test_known_background_validation_and_domain():
    cube,psf,mask,center,flux,_=_source(background=3)
    background=Const2D(3,fixed={'amplitude':True})
    fitted,table=optimal(cube,psf,mask,center=center,background=background)
    np.testing.assert_allclose(fitted.y,flux,rtol=8e-4)
    np.testing.assert_array_equal(table['background_err'],0)
    with pytest.raises(ValueError,match='binary'):
        optimal(cube,psf,mask*.5,center=center)
    with pytest.raises(ValueError,match='bounds'):
        optimal(cube,psf,mask,center=center,background=Const2D(bounds={'amplitude':(0,10)}))
    with pytest.raises(ValueError,match='mask'):
        aperture(cube,psf,np.full(cube.shape[1:],np.nan),center=center)
    with pytest.raises(ValueError,match='center'):
        aperture(cube,psf,mask,center=(np.nan,0))
    cube.z[0]=4900
    with pytest.warns(UserWarning,match='1 of 7'):
        _,table=aperture(cube,psf,mask,center=center)
    assert not table['success'][0] and 'outside' in table['message'][0]


def test_conditional_covariance_matches_noise_realizations():
    # Independent Gaussian pixel integrals with known F and flat background.
    rng=np.random.default_rng(193)
    y,x=np.indices((5,5));sx=1.1
    p=.5*(erf((x+.5-2.1)/(np.sqrt(2)*sx))-erf((x-.5-2.1)/(np.sqrt(2)*sx)))
    p*=.5*(erf((y+.5-1.9)/(np.sqrt(2)*sx))-erf((y-.5-1.9)/(np.sqrt(2)*sx)))
    model=Gaussian2D(x_stddev=sx,y_stddev=sx)
    for n in model.param_names:
        if n!='amplitude':model.fixed[n]=True
    psf=ChromaticPSF(model,{},[5000,7000],unit=u.AA,factor=15)
    values=40*p+3+rng.normal(0,.5,size=(300,5,5))
    cube=Cube(values,z=np.linspace(5000,7000,300),zunit=u.AA,err=np.full_like(values,.5),dtype=float)
    spectrum,table=optimal(cube,psf,np.ones((5,5),bool),center=(1.9,2.1),background=Const2D())
    assert abs(np.mean(spectrum.y)-40)<.25*np.mean(spectrum.yerr)
    assert np.std(spectrum.y,ddof=1)==pytest.approx(np.mean(spectrum.yerr),rel=.15)
    assert np.mean((spectrum.y-40)**2/spectrum.yerr**2)==pytest.approx(1,abs=.2)


def test_public_import_and_wavelength_unit_conversion():
    from prism.modeling.psf import aperture as public_aperture, optimal as public_optimal
    assert public_aperture is aperture and public_optimal is optimal
    cube,psf,mask,center,flux,_=_source()
    cube.z = cube.z / 10
    cube.zunit = u.nm
    spectrum,table=aperture(cube,psf,mask,center=center)
    assert spectrum.xunit==u.nm and table['wave'].unit==u.nm
    np.testing.assert_allclose(spectrum.y,flux,rtol=4e-4)
