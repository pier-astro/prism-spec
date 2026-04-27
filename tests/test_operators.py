import warnings

import astropy.units as u
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter as NativeTRFLSQFitter
from astropy.modeling.powerlaws import ExponentialCutoffPowerLaw1D, PowerLaw1D

from prism.modeling.fitting import ScipyTRF, TRFLSQFitter, tie
from prism.modeling.models import BSpline, FixedTemplate
from prism.modeling.models.lines import GaussianLine
from prism.modeling.operators.instrument import InstrumentResponse, SpectralResponse


def _build_tied_double_gaussian(offset=-120.0, fwhm=700.0):
    return (
        GaussianLine(
            name='g1', amplitude=1.5, position=5000.0, offset=offset, fwhm=fwhm,
            bounds={'amplitude': (0, 10), 'offset': (-300, 300), 'fwhm': (50, 900)},
        )
        + GaussianLine(
            name='g2', amplitude=1.0, position=5007.0, offset=offset, fwhm=fwhm,
            bounds={'amplitude': (0, 10), 'offset': (-300, 300), 'fwhm': (50, 900)},
        )
    )


def _tie_double_gaussian(model):
    model['g2'].offset.tied = tie('g1', lambda m: m.offset)
    model['g2'].fwhm.tied = tie('g1', lambda m: m.fwhm)
    return model


def _assert_tied_warning(records):
    messages = [
        str(item.message)
        for item in records
        if issubclass(item.category, UserWarning)
        and 'tied parameters together with analytic derivatives' in str(item.message)
    ]
    assert len(messages) == 1


def test_spectral_response_pipe_supports_quantity_aware_prism_trf():
    x = np.linspace(4990.0, 5020.0, 128) * u.AA
    response = InstrumentResponse.from_fixed_fwhm(x.value, fwhm=2.0)
    rsp = SpectralResponse(instrument=response, wave=x.value, flexible=False)

    truth = GaussianLine(amplitude=3.0 * u.Jy, position=5007.0 * u.AA, fwhm=350.0 * u.km / u.s) | rsp
    y = truth(x)
    model = GaussianLine(amplitude=2.5 * u.Jy, position=5007.0 * u.AA, fwhm=400.0 * u.km / u.s) | rsp

    fitted = TRFLSQFitter()(model, x, y)
    assert fitted.left.amplitude.unit == u.Jy
    assert fitted.left.position.unit == u.AA
    assert fitted.left.fwhm.unit == (u.km / u.s)


def test_fixed_template_and_bspline_support_quantity_fitting():
    x = np.linspace(1.0, 10.0, 128) * u.AA
    template = FixedTemplate(x.value, 1.0 + 0.1 * np.sin(x.value))
    truth = template * PowerLaw1D(amplitude=2.0 * u.Jy, x_0=5.0 * u.AA, alpha=0.0)
    fitted_template = TRFLSQFitter()(template * PowerLaw1D(amplitude=1.7 * u.Jy, x_0=5.0 * u.AA, alpha=0.0), x, truth(x))
    assert fitted_template.right.amplitude.unit == u.Jy
    assert fitted_template.right.x_0.unit == u.AA

    x2 = np.linspace(0.0, 10.0, 128) * u.AA
    truth_spline = BSpline(np.linspace(0.0, 10.0, 8), degree=3, c0=1.0 * u.Jy, c1=1.2 * u.Jy, c2=0.8 * u.Jy, c3=1.1 * u.Jy)
    fitted_spline = TRFLSQFitter()(BSpline(np.linspace(0.0, 10.0, 8), degree=3, c0=0.9 * u.Jy, c1=1.0 * u.Jy, c2=0.9 * u.Jy, c3=1.0 * u.Jy), x2, truth_spline(x2))
    assert fitted_spline.c0.unit == u.Jy


def test_cutoff_powerlaw_supports_quantity_evaluation_and_fitting():
    x = np.linspace(1000.0, 9000.0, 256) * u.AA
    truth = ExponentialCutoffPowerLaw1D(
        amplitude=2.0 * u.Jy,
        x_0=3000.0 * u.AA,
        alpha=1.2,
        x_cutoff=5000.0 * u.AA,
        bounds={'x_0': (1.0 * u.AA, None), 'x_cutoff': (1.0 * u.AA, None), 'alpha': (0.0, None)},
    )

    fitted = TRFLSQFitter()(
        ExponentialCutoffPowerLaw1D(
            amplitude=1.7 * u.Jy,
            x_0=3000.0 * u.AA,
            alpha=1.0,
            x_cutoff=4500.0 * u.AA,
            bounds={'x_0': (1.0 * u.AA, None), 'x_cutoff': (1.0 * u.AA, None), 'alpha': (0.0, None)},
        ),
        x,
        truth(x),
    )

    assert fitted.amplitude.unit == u.Jy
    assert fitted.x_0.unit == u.AA
    assert fitted.x_cutoff.unit == u.AA


def test_native_astropy_tied_jacobian_patch_remains_active():
    x = np.linspace(4990.0, 5020.0, 128)
    truth = _tie_double_gaussian(_build_tied_double_gaussian(offset=-90.0, fwhm=540.0))
    y = truth(x)
    model = _tie_double_gaussian(_build_tied_double_gaussian())

    fitter = NativeTRFLSQFitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted_analytic = fitter(model.copy(), x, y, estimate_jacobian=False)

    _assert_tied_warning(caught)
    fitted_numeric = fitter(model.copy(), x, y, estimate_jacobian=True)
    assert np.allclose(fitted_analytic(x), fitted_numeric(x), rtol=1e-6, atol=1e-6)


def test_scipytrf_tied_pipe_warns_and_converges():
    x = np.linspace(4990.0, 5020.0, 128)
    rsp = SpectralResponse(
        instrument=InstrumentResponse.from_fixed_fwhm(x, fwhm=2.0),
        wave=x,
        flexible=False,
    )
    truth = _tie_double_gaussian(_build_tied_double_gaussian(offset=-75.0, fwhm=510.0)) | rsp
    y = truth(x)
    model = _tie_double_gaussian(_build_tied_double_gaussian()) | rsp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = ScipyTRF()(model, x, y)

    _assert_tied_warning(caught)
    assert np.allclose(fitted(x), y, rtol=1e-5, atol=1e-5)