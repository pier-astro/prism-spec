"""
prism.modeling.line_analysis
============================

Selects, measures, and extracts metrics (like fluxes, FWHMs, centroids) from composite 
emission-line models built with `prism.modeling.lines`. 

The primary interface is `select_line`, which extracts a sub-profile from a model 
based on a physical tag (e.g., 'Hb4861') automatically collapsing `LineGroup`s. 
The returned profile acts as an independent 1D model that can be evaluated, measured 
(`measure()`), or sampled (`sample()`) with uncertainty propagation via Monte Carlo.

Core Components:
- `select_line`: Isolates a particular line by tag. Returns `SelectedLineProfile` (or `SelectedLineCollection` for multi-fit models).
- `SelectedLineProfile`: Extracted line profile providing `evaluate()`, `measure()`, `eqw()`, and `sample()` methods.
- `measure_line` / `sample_line_measurements`: Core functions driving the physical measurements, retrieving kinematic and flux quantities (e.g., flux, fwhm, asymmetry, moment1, moment2).
"""
from .selection import (
    SelectedLineProfile,
    SelectedLineCollection,
    select_line,
)
from .measurement import (
    LineResult,
    MultiLineMeasurements,
    measure_line,
    sample_line_measurements,
    _selection_parameter_specs,
    _draw_selection_samples,
)
from ..models.lines import Metric
__all__ = [
    'LineResult',
    'Metric',
    'MultiLineMeasurements',
    'SelectedLineProfile',
    'SelectedLineCollection',
    'measure_line',
    'sample_line_measurements',
    'select_line',
]
