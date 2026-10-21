"""Native response fitting and persistence work without Astropy class patches."""
import os
import subprocess
import sys
from pathlib import Path


def test_native_fit_save_and_response_without_global_extensions():
    script = r'''
import tempfile
from pathlib import Path
import numpy as np
from astropy.modeling import Model as AstropyModel, CompoundModel
from astropy.modeling.models import Gaussian1D
names = {AstropyModel: ('save', 'to_fits', 'covariance'),
         CompoundModel: ('_evaluate', 'evaluate', 'fit_deriv')}
original = {(cls, name): getattr(cls, name, None) for cls in names for name in names[cls]}
from prism.modeling import Model, LinearOperatorModel
from prism.modeling.fitting import TRF
from prism.modeling.io import load_model, model_to_fits
from prism.modeling.interop.astropy import install_model_io_methods, install_pipe_patch
x = np.linspace(-3., 3., 81)
source = Gaussian1D(1., 0., .8, fixed={'mean': True, 'stddev': True})
response = LinearOperatorModel(np.eye(len(x))*.5, x=x)
model = Model.from_astropy(source) | response
expected = 1.5*np.exp(-.5*(x/.8)**2)
fitted = TRF(covariance=True)(model, x, expected)
assert np.allclose(fitted.predict(x), expected)
assert fitted.covariance.shape == (1, 1)
with tempfile.TemporaryDirectory() as directory:
    filename = Path(directory)/'fit.fits'
    fitted.model.save(filename)
    restored = load_model(filename)
    assert np.allclose(restored(x), expected)
    model_to_fits(source, Path(directory)/'astropy.fits')
assert all(getattr(cls, name, None) is value for (cls, name), value in original.items())
assert not getattr(CompoundModel, '_prism_linear_operator_patch', False)
assert not getattr(AstropyModel, '_prism_io_installed', False)
install_model_io_methods()
install_pipe_patch()
install_model_io_methods()
install_pipe_patch()
legacy = source | response
assert np.allclose(legacy(x), .5*source(x))
with tempfile.TemporaryDirectory() as directory:
    legacy.save(Path(directory)/'legacy.fits')
'''
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                   env=environment, check=True, capture_output=True, text=True)
