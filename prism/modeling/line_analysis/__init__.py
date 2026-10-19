"""Select profiles, measure sampled arrays and optionally propagate errors.

``profile_metrics(x, y)`` measures continuum-subtracted sampled profiles without
model selection. ``select_line`` adapts fitted components to the same numerical
core; its ``measure``, ``sample`` and EW methods remain small convenience tools.
Scientific inference and complete analysis workflows belong in user notebooks.
"""
from .selection import (
    SelectedLineProfile,
    SelectedLineCollection,
    select_line,
)
from .measurement import (
    measure_line,
    profile_metrics,
)
from .results import LineResult, MultiLineMeasurements
from .sampling import (
    sample_line_measurements, _selection_parameter_specs, _draw_selection_samples,
)
from ..metrics import Metric
__all__ = [
    'LineResult',
    'Metric',
    'MultiLineMeasurements',
    'SelectedLineProfile',
    'SelectedLineCollection',
    'measure_line',
    'profile_metrics',
    'sample_line_measurements',
    'select_line',
]
