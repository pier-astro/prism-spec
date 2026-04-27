from pathlib import Path
import inspect

from prism.data import Cube, Image, RegionMeasurement, Spectrum
from prism.modeling.fitting import FittingWithOutlierRemoval, LMFitter, ScipyDogBox, ScipyTRF
from prism.modeling.line_analysis import (
    LineResult,
    MultiLineMeasurements,
    measure_line,
    sample_line_measurements,
)
from prism.modeling.models import (
    BSpline,
    FixedTemplate,
    GaussianLine,
    GaussianLines,
    LorentzianLine,
    VoigtLine,
    balmer,
    blr,
    fe,
    fe_uv,
    nlr,
)


def test_core_public_api_docstrings_are_substantive():
    objects = [
        Spectrum,
        Cube,
        Image,
        RegionMeasurement,
        nlr,
        blr,
        fe,
        fe_uv,
        balmer,
        GaussianLine,
        GaussianLines,
        LorentzianLine,
        VoigtLine,
        FixedTemplate,
        BSpline,
        ScipyTRF,
        ScipyDogBox,
        LMFitter,
        FittingWithOutlierRemoval,
        measure_line,
        sample_line_measurements,
        LineResult,
        MultiLineMeasurements,
    ]

    for obj in objects:
        doc = inspect.getdoc(obj)
        assert doc is not None
        assert len(doc) >= 80


def test_readme_uses_current_public_names():
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    text = readme.read_text()

    for stale_name in ('AstroTRF', 'zcorrect', 'wbounds', 'ResponseModel', 'ConvolvedModel'):
        assert stale_name not in text

    for current_name in ('ScipyTRF', 'TRFLSQFitter', 'select_line', 'measure_line', 'InstrumentResponse'):
        assert current_name in text