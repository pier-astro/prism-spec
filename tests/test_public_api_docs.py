from prism.modeling.lines import LineSet, GaussianProfile
from pathlib import Path
import inspect

from prism.data import Cube, Image, RegionMeasurement, Spectrum
from prism.modeling.fitting import FittingWithOutlierRemoval, ScipyDogBox, TRF
from prism.experimental import LMFitter
from prism.modeling.lines.measurements import (
    LineResult,
    MultiLineMeasurements,
    measure_line,
    sample_line_measurements,
)
from prism.modeling.models import (
    BSpline,
    FixedTemplate,
    balmer,
    blr,
    fe,
    fe_uv,
    nlr,
)


def test_core_public_api_docstrings_are_substantive():
    objects = [
        LineSet,
        GaussianProfile,
        Spectrum,
        Cube,
        Image,
        RegionMeasurement,
        nlr,
        blr,
        fe,
        fe_uv,
        balmer,
                        FixedTemplate,
        BSpline,
        TRF,
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

    for current_name in ('TRF', 'TRFLSQFitter', 'select_line', 'measure_line', 'InstrumentResponse'):
        assert current_name in text