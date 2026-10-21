"""Model-independent measurements, independently normalized Gaussian mixture."""
import numpy as np
import pytest
from astropy import units as u
from prism.modeling.lines.measurements import profile_metrics


def test_sampled_gaussian_mixture_moments_and_units():
    x = np.linspace(-30.,40.,14001)
    # Unit-area Gaussian components, with integrated weights 2:1.
    y = 2.*np.exp(-.5*((x+2.)/1.5)**2)/(1.5*np.sqrt(2*np.pi))
    y += np.exp(-.5*((x-4.)/3.)**2)/(3.*np.sqrt(2*np.pi))
    mean = (2.*(-2.)+4.)/3.
    variance = (2.*(1.5**2+(-2.-mean)**2)+(3.**2+(4.-mean)**2))/3.
    result = profile_metrics(x*u.AA, y*u.Jy)
    reverse = profile_metrics(x[::-1]*u.AA, y[::-1]*u.Jy)
    np.testing.assert_allclose(result.flux.value,3.,rtol=1e-12)
    np.testing.assert_allclose(result.moment1.value,mean,atol=1e-12)
    np.testing.assert_allclose(result.moment2.value,variance,rtol=1e-12)
    assert result.flux.unit == u.Jy*u.AA
    assert result.fwhm.unit == u.AA
    assert result.moment2.unit == u.AA**2
    for name in result.metric_units:
        assert getattr(result,name).value == getattr(reverse,name).value


@pytest.mark.parametrize('x,y', [([1,2],[1,2]),([1,2,3],[1,2]),([1,2,3],[1,np.nan,2]),([1,2,2],[1,2,3])])
def test_sampled_profile_rejects_invalid_input(x,y):
    with pytest.raises(ValueError):
        profile_metrics(x,y)
