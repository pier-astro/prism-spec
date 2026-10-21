"""Stable imports keep development solvers and legacy adapters separate."""
import subprocess
import sys


def test_supported_import_does_not_load_experimental_solvers():
    code = """
import sys
from prism.modeling import fitting
assert 'prism.experimental' not in sys.modules
for name in ('LMFitter', 'SeparableTRF', 'TRFLSQFitter', 'SLSQPLSQFitter'):
    assert not hasattr(fitting, name), name
from prism.psf import ChromaticPSF, aperture, optimal
from prism.modeling.interop.astropy import TRFLSQFitter, SLSQPLSQFitter
from prism.experimental import LMFitter, SeparableTRF
from prism.modeling.lines import LineSet, GaussianProfile
from prism.modeling.lines.measurements import select_line, profile_metrics
assert 'prism.modeling.line_analysis' not in sys.modules
from prism.modeling import models
from importlib.util import find_spec
assert find_spec('prism.modeling.models.lines') is None
for name in ('GaussianLine', 'GaussianLines', 'LorentzianLine', 'VoigtLine'):
    assert not hasattr(models, name), name
from prism.modeling import Model
assert isinstance(models.nlr(), Model)
assert isinstance(models.blr(), Model)
from prism.modeling.fitting.multifit import MultiFitResult
from prism.modeling.fitting.multifit.results import MultiFitResult as Implementation
assert MultiFitResult is Implementation
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
