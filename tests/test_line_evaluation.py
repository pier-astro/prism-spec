import numpy as np
import pytest
import astropy.units as u
from astropy.constants import c
from scipy.special import voigt_profile

from prism.modeling.models.lines import GaussianLine, GaussianLines, LorentzianLines, VoigtLines


@pytest.mark.parametrize('kind', [GaussianLines, LorentzianLines, VoigtLines])
@pytest.mark.parametrize('unit', [u.AA, u.eV])
def test_group_profiles_match_independent_reference(kind, unit):
    group = kind.from_arrays(
        names=['doublet', 'doublet'], pos=[5000, 5010] * unit,
        weights=[1.0, 0.33], amplitude=2.0, offset=35.0,
        redshift=0.025, instfwhm=80.0,
        **({'fwhm_G': 180.0, 'fwhm_L': 110.0} if kind is VoigtLines else {'fwhm': 180.0}),
    )
    centers = np.array([5000.0, 5010.0]) * 1.025 * np.exp(35.0 / c.to_value('km/s'))
    if unit == u.eV:
        centers = np.array([5000.0, 5010.0]) / (1.025 * np.exp(35.0 / c.to_value('km/s')))
    x = np.linspace(centers.min() - 20.0, centers.max() + 20.0, 1201)
    expected = np.zeros_like(x)
    for center, weight in zip(centers, [1.0, 0.33]):
        sigma = center / c.to_value('km/s') / np.sqrt(8.0 * np.log(2.0))
        sigma *= np.hypot(180.0, 80.0) if kind is not LorentzianLines else 80.0
        gamma = 0.0
        if kind is not GaussianLines:
            fwhm = 110.0 if kind is VoigtLines else 180.0
            gamma = center / c.to_value('km/s') * fwhm / 2.0
        shape = voigt_profile(x - center, sigma, gamma)
        shape /= voigt_profile(0.0, sigma, gamma)
        expected += 2.0 * weight / 1.025 * shape
    np.testing.assert_allclose(group(x), expected, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(group(x[500]), expected[500], rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(group(x * unit), expected, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize('kind', [GaussianLines, LorentzianLines, VoigtLines])
@pytest.mark.parametrize('unit', [u.AA, u.eV])
@pytest.mark.parametrize('amplitude', [0.0, 2.0])
def test_group_derivatives_match_central_differences(kind, unit, amplitude):
    group = kind.from_arrays(
        names=['doublet', 'doublet'], pos=[5000, 5010] * unit,
        weights=[1.0, 0.33], amplitude=amplitude, offset=35.0,
        redshift=0.025, instfwhm=80.0,
        **({'fwhm_G': 180.0, 'fwhm_L': 110.0} if kind is VoigtLines else {'fwhm': 180.0}),
    )
    center = 5000.0 * 1.025 if unit == u.AA else 5000.0 / 1.025
    x = np.linspace(center - 30.0, center + 30.0, 801)
    parameters = group.parameters.copy()
    jacobian = np.asarray(group.fit_deriv(x, *parameters))
    for index, value in enumerate(parameters):
        step = 1e-6 * max(abs(value), 0.1)
        upper = group.copy()
        lower = group.copy()
        values = parameters.copy()
        values[index] += step
        upper.parameters = values
        values = parameters.copy()
        values[index] -= step
        lower.parameters = values
        expected = (upper(x) - lower(x)) / (2.0 * step)
        tolerance = 1e-6
        if kind is LorentzianLines:
            # Forward differences have first-order error near zero crossings.
            tolerance = 1e-4
        np.testing.assert_allclose(
            jacobian[index], expected, rtol=2e-3,
            atol=tolerance * max(1.0, np.max(np.abs(expected))),
            err_msg=group.param_names[index],
        )


def test_domain_family_follows_replaced_reference_unit_after_copy():
    group = GaussianLines.from_arrays(names=['line'], pos=[5000] * u.AA)
    assert group.domain_family == 'wavelength'
    copied = group.copy()
    copied._position_unit = u.eV
    assert copied.domain_family == 'linear'
    assert group.domain_family == 'wavelength'
    copied._position_unit = u.dimensionless_unscaled
    with pytest.raises(ValueError, match='no spectral unit'):
        _ = copied.domain_family
    copied._position_unit = u.AA
    assert copied.domain_family == 'wavelength'
    line = GaussianLine(position=5000 * u.AA)
    assert line.domain_family == 'wavelength'
    line._position_unit = u.eV
    assert line.domain_family == 'linear'
