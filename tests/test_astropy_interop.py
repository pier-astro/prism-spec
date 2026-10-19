"""Opt-in Astropy extensions must not be required for native Prism display."""
import os
import subprocess
import sys
from pathlib import Path


def test_display_boundary_in_fresh_process():
    script = r'''
import sys
from astropy.modeling import Model as AstropyModel, CompoundModel
from astropy.modeling.models import Gaussian1D
classes = (AstropyModel, CompoundModel)
names = ('__repr__', '__str__', '_repr_html_', 'show', 'expression')
original = {(cls, name): getattr(cls, name, None) for cls in classes for name in names}
from prism.modeling import Model
import prism.modeling.display
assert 'prism.modeling.interop.astropy' not in sys.modules
from prism.modeling.interop.astropy import enable_model_display, disable_model_display, tie
assert all(getattr(cls, name, None) is value for (cls, name), value in original.items())
native = Model.from_astropy(Gaussian1D(name='core'))
assert '<table' in native._repr_html_()
assert native.expression == 'core'
for _ in range(2):
    enable_model_display()
    assert '<table' in Gaussian1D()._repr_html_()
    disable_model_display()
    assert all(getattr(cls, name, None) is value for (cls, name), value in original.items())
assert '<table' in native._repr_html_()
assert native.expression == 'core'
'''
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                   env=environment, check=True, capture_output=True, text=True)
