"""Prism-owned adapters must not change native Astropy fitter classes."""
from prism.modeling.interop import astropy as astropy_fitting
import subprocess
import sys
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Gaussian2D
from prism.modeling.interop.astropy import TRFLSQFitter, DogBoxLSQFitter, LevMarLSQFitter


def test_import_leaves_native_astropy_fitter_methods_unchanged():
    code = '''
from astropy.modeling import fitting
classes = [fitting.TRFLSQFitter, fitting.DogBoxLSQFitter,
           fitting.LevMarLSQFitter, fitting.LinearLSQFitter,
           fitting.SimplexLSQFitter, fitting.SLSQPLSQFitter]
before = [(cls.__call__, getattr(cls, '_wrap_deriv', None)) for cls in classes]
import prism.modeling.fitting
import prism.modeling.interop.astropy
assert before == [(cls.__call__, getattr(cls, '_wrap_deriv', None)) for cls in classes]
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('backend', [TRFLSQFitter, DogBoxLSQFitter, LevMarLSQFitter])
def test_adapter_linked_amplitudes_recover_known_input(backend):
    x = np.linspace(-4, 4, 101)
    model = Gaussian1D(1., -.7, .6) + Gaussian1D(2., .8, .9)
    for name in ('mean_0', 'mean_1', 'stddev_0', 'stddev_1'):
        getattr(model, name).fixed = True
    model.amplitude_1.tied = lambda m: 2*m.amplitude_0
    truth = 3*np.exp(-.5*((x+.7)/.6)**2) + 6*np.exp(-.5*((x-.8)/.9)**2)
    fitted = backend()(model, x, truth, inplace=False)
    np.testing.assert_allclose(fitted.parameters[[0, 3]], [3., 6.], atol=1e-7)
    assert model.amplitude_0.value == 1.


def test_adapter_two_dimensional_linked_width_matches_numeric_jacobian():
    y, x = np.mgrid[-3:3:25j, -3:3:27j]
    truth = 3*np.exp(-.5*((x-.1)/.7)**2-.5*((y+.2)/1.4)**2)
    model = Gaussian2D(1., .1, -.2, 1.1, 2.2, theta=0.)
    for name in ('x_mean', 'y_mean', 'theta'):
        getattr(model, name).fixed = True
    model.y_stddev.tied = lambda m: 2*m.x_stddev
    analytic = TRFLSQFitter()(model, x, y, z=truth, inplace=False)
    numeric = TRFLSQFitter()(model, x, y, z=truth, estimate_jacobian=True, inplace=False)
    np.testing.assert_allclose(analytic.parameters, numeric.parameters, atol=1e-7)
    np.testing.assert_allclose(analytic.parameters[[0, 3, 4]], [3., .7, 1.4], atol=1e-7)


def test_slsqp_adapter_keeps_equality_constraint_capability():
    from astropy.modeling.models import Linear1D
    from prism.modeling.interop.astropy import SLSQPLSQFitter
    x = np.linspace(-1, 1, 51)
    y = 2*x + 3
    model = Linear1D(0., 1., eqcons=[lambda p, *args: p[0]+p[1]-1.])
    result = SLSQPLSQFitter()(model, x, y, inplace=False)
    expected_slope = np.sum((x-1)*(y-1))/np.sum((x-1)**2)
    np.testing.assert_allclose(result.parameters, [expected_slope, 1-expected_slope], atol=1e-5)
