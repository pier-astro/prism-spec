"""Scientific contracts exercised by the introductory notebooks."""
from pathlib import Path
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import BSpline
from prism.data import Spectrum
from prism.modeling import Component
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
from prism.modeling.lines import GaussianLines
from prism.modeling.display import format_model_html
from prism.psf import fittrends


@pytest.mark.parametrize('asymmetric', [False, True])
def test_spectrum_snapshot_keeps_active_values_errors_units_and_metadata(tmp_path, asymmetric):
    x = np.linspace(4800., 5100., 21)
    error = np.array([np.full(21,.1),np.full(21,.2)]) if asymmetric else np.full(21,.1)
    spectrum = Spectrum(x, 1.+x/10000., yerr=np.full(21,.1), xerr=error,
        xunit=u.AA, yunit=1e-15*u.erg/(u.s*u.cm**2*u.AA),
        ra=13.*u.deg, dec=-25.*u.deg, redshift=.0622,
        meta={'source': Path('input.txt'), 'stage':'corrected'}, name='prepared',dtype=np.float64)
    spectrum = spectrum.cutout(min=4850.,max=5050.)
    path = tmp_path/'spectrum.fits'
    spectrum.save(path)
    restored = Spectrum.load(path)
    for name in ('x','y','xerr','yerr'):
        np.testing.assert_array_equal(getattr(restored,name),getattr(spectrum,name))
    assert restored.xunit==spectrum.xunit and restored.yunit==spectrum.yunit
    assert restored.name==spectrum.name and restored.redshift==spectrum.redshift
    assert restored.ra==13. and restored.dec==-25.
    assert restored.dtype==np.dtype(np.float64)
    assert restored.meta=={'source':'input.txt','stage':'corrected'}
    with pytest.raises(OSError):spectrum.save(path)


@pytest.mark.parametrize('engine', [TRF,pytest.param(SherpaLM,marks=pytest.mark.skipif(not HAS_SHERPA,reason='Sherpa unavailable'))])
def test_covariance_default_and_explicit_opt_out(engine):
    x=np.linspace(-1.,1.,61)
    model=Component(lambda x,slope: slope*x,{'slope':1.},name='slope')
    y=3.*x+np.random.default_rng(4).normal(0,.01,x.size)
    with_cov=engine()(model,x,y,yerr=np.full(x.size,.01))
    without_cov=engine(covariance=False)(model,x,y,yerr=np.full(x.size,.01))
    assert with_cov.covariance.shape==(1,1) and with_cov.params['slope'].std>0
    assert without_cov.covariance is None and without_cov.params['slope'].std is None
    np.testing.assert_allclose(with_cov.parameters,without_cov.parameters,rtol=1e-12,atol=1e-12)


def test_builtin_display_exposes_units_without_changing_native_arithmetic():
    model=GaussianLines({'name':['A'],'position':[5000.]*u.AA},name='lines')
    before=model.parameters.copy()
    html=format_model_html(model)
    assert 'km / s' in html
    np.testing.assert_array_equal(model.parameters,before)
    assert model.fwhm.unit is None


def test_cubic_psf_trend_recovers_edge_curvature_from_interior_centers():
    from prism.modeling.models import MoffatPSF
    from prism.psf.psf import _state
    model=MoffatPSF(fwhm=2.,fixed={'beta':True,'x_0':True,'y_0':True})
    centers=np.linspace(-.95,.95,20)
    grid=np.linspace(-1.,1.,201)
    edges=np.linspace(-1.,1.,21)
    channels=np.array([(grid>=lo)&(grid<=hi) for lo,hi in zip(edges[:-1],edges[1:])])
    bands=Table({'wave':6000+1000*centers,'success':np.ones(20,dtype=bool),
                 'fwhm':2+.2*centers+.3*centers**2,'fwhm_err':np.full(20,.02),
                 'channels':channels})
    bands.meta={'profile':_state(model),'waveunit':'Angstrom','wavelengths':(6000+1000*grid).tolist()}
    psf=fittrends(bands,{'fwhm':5},display=False)
    np.testing.assert_allclose(psf.fwhm(6000+1000*grid).value[:,0],2+.2*grid+.3*grid**2,atol=1e-12)
    law=psf.trends['fwhm']
    spline=BSpline(law['knots'],law['coefficients'],law['degree'])
    np.testing.assert_allclose(spline.derivative(2)([-1.,1.]),[.6,.6],atol=1e-12)
