"""The default fitter works without loading the optional optimizer backend."""
import subprocess
import sys
import warnings

import numpy as np
import pytest
from prism.modeling import Component
from prism.modeling.fitting import TRF


@pytest.mark.parametrize('missing', [False, True])
def test_default_fit_does_not_import_sherpa(missing):
    code = '''
import importlib.util
import sys
missing = MISSING
if missing:
    original = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *args, **kwargs: (
        None if name == 'sherpa' else original(name, *args, **kwargs))
from prism.modeling import Component
from prism.modeling.fitting import TRF, SherpaLM, HAS_SHERPA
import numpy as np
assert not any(name == 'sherpa' or name.startswith('sherpa.') for name in sys.modules)
x = np.linspace(-1, 1, 31)
model = Component(lambda x, slope: slope*x, {'slope': 1.})
result = TRF()(model, x, 3*x)
np.testing.assert_allclose(result.model.parameters, [3.], atol=1e-7)
assert not any(name == 'sherpa' or name.startswith('sherpa.') for name in sys.modules)
if missing:
    assert not HAS_SHERPA
    try:
        SherpaLM()
    except ImportError as exc:
        assert 'Install' in str(exc)
    else:
        raise AssertionError('Missing optional backend must fail when requested')
'''.replace('MISSING', repr(missing))
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_analytic_linked_fit_is_quiet_and_recovers_input():
    calls = []
    def profile(x, amplitude):
        return amplitude*np.exp(-.5*x*x)
    def derivative(x, amplitude):
        calls.append(amplitude)
        return [np.exp(-.5*x*x)]
    first = Component(profile, {'amplitude': 1.}, derivative=derivative, name='first')
    second = Component(profile, {'amplitude': 2.}, derivative=derivative, name='second')
    second.amplitude.link = 2*first.amplitude
    x = np.linspace(-4, 4, 101)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = TRF(covariance=True)(first+second, x, 9*np.exp(-.5*x*x))
    np.testing.assert_allclose(result.model.parameters, [3., 6.], atol=1e-7)
    assert len(calls) > 2  # Both profiles are differentiated beyond the initial probe.
    assert result.covariance.shape == (1, 1)
    assert not any('tied parameters together' in str(item.message) for item in caught)
