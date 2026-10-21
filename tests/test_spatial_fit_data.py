"""Spatial adapters share valid-pixel/error semantics with numerical fitting."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy.modeling.models import Gaussian2D
from prism.modeling.interop.astropy import TRFLSQFitter


@pytest.mark.parametrize('use_errors',[True,False])
def test_spatial_weighted_amplitude_matches_analytic_estimator(use_errors):
    y,x=np.mgrid[-3:3:25j,-3:3:27j]
    shape=np.exp(-.5*(x/.7)**2-.5*(y/1.2)**2)
    z=3*shape+.03*np.sin(x*3+y)
    errors=.1+.03*(x+3)
    valid=np.ones(z.shape,dtype=bool);valid[4:7,5:8]=False
    z[~valid]=np.nan
    expected=np.sum(shape[valid]*z[valid]/errors[valid]**2)/np.sum(shape[valid]**2/errors[valid]**2)
    model=Gaussian2D(1,0,0,.7,1.2,theta=0)
    for name in model.param_names:model.fixed[name]=name!='amplitude'
    fitter=TRFLSQFitter()
    options={'yerr':errors} if use_errors else {'weights':1/errors}
    fitted=fitter(model,x,y,z=z,mask=valid,inplace=False,**options)
    assert_allclose(fitted.amplitude.value,expected,atol=1e-9)
    assert model.amplitude.value==1
    assert fitter.fit_info['ndata']==np.count_nonzero(valid)
    assert fitted(x,y).shape==z.shape


def test_spatial_filter_checks_both_coordinates_and_image_values():
    y,x=np.mgrid[-3:3:15j,-3:3:17j]
    z=4*np.exp(-.5*x*x-.5*y*y)
    model=Gaussian2D(1,0,0,1,1,theta=0)
    for name in model.param_names:model.fixed[name]=name!='amplitude'
    x[0,0]=np.nan;y[1,1]=np.nan;z[2,2]=np.nan
    with pytest.raises(ValueError,match='Non-finite'):
        TRFLSQFitter()(model,x,y,z=z,inplace=False)
    fitter=TRFLSQFitter()
    fit=fitter(model,x,y,z=z,yerr=np.ones(z.shape),filter_non_finite=True,inplace=False)
    assert_allclose(fit.amplitude.value,4,atol=1e-9)
    assert fitter.fit_info['ndata']==z.size-3


def test_spatial_error_shape_and_sign_are_explicit():
    y,x=np.mgrid[:3,:4];z=np.ones(x.shape)
    with pytest.raises(ValueError,match='same shape'):
        TRFLSQFitter()(Gaussian2D(),x,y,z=z,yerr=np.ones((2,3,4)))
    with pytest.raises(ValueError,match='strictly positive'):
        TRFLSQFitter()(Gaussian2D(),x,y,z=z,yerr=-np.ones(z.shape))


def test_spatial_quantity_coordinates_and_error_units():
    from astropy import units as u
    from astropy.modeling.models import Const2D
    y,x=np.mgrid[:3,:4]
    fitter=TRFLSQFitter()
    fitted=fitter(Const2D(1*u.Jy),x*u.arcsec,y*u.arcsec,
                  z=np.full(x.shape,3.)*u.Jy,yerr=np.full(x.shape,100.)*u.mJy,inplace=False)
    assert_allclose(fitted.amplitude.quantity.to_value(u.Jy),3.,atol=1e-9)
