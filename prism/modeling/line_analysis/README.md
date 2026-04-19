# prism.modeling.line_analysis

Post-fit line selection, measurement, and uncertainty propagation for Prism models.

This module operates on already constructed spectral models and extracts physically meaningful line-level measurements from them without forcing the user to manually rebuild submodels.

## Concept

`line_analysis` answers a simple question:

Given a composite spectral model, what is the profile and what are the measured properties of one physical line or one tied line component?

It provides:

- selection of one physical line from a compound model
- direct evaluation of the extracted profile
- numerical measurements of flux, width, centroids, and moments
- equivalent width estimation
- Monte Carlo propagation of attached parameter uncertainties

The API is designed as an analysis layer on top of the models in `prism.modeling.models.lines`.

## Structure

- `selection.py`
  Resolves a requested physical line into one or more source-model components and exposes `SelectedLineProfile` and `SelectedLineCollection`.
- `measurement.py`
  Computes summary measurements and Monte Carlo uncertainty propagation.
- `__init__.py`
  Public entry points.

## Physical Overview

The extracted line profile is treated as a spectral-density profile on the same spectral axis as the source model.

This means:

- if the source model is in wavelength, measurements are in wavelength units
- if the source model is in frequency, measurements are in frequency units
- if the source model output carries quantities, the extracted profile and its measurements preserve those units

The principal measured quantities are:

- `flux`
  Integrated line flux over the selected spectral axis.
- `peak_value`
  Peak spectral density of the selected line profile.
- `peak_position`
  Spectral coordinate of the peak.
- `fwhm`, `hw_blue`, `hw_red`
  Width measures on the selected axis.
- `moment1`, `moment2`, `sigma`
  First and second moments and the corresponding dispersion.
- `skewness`, `kurtosis`, `bisector_span`
  Shape diagnostics.

Equivalent width is computed relative to a supplied or inferred continuum in the same spectral domain as the selected profile.

## Main Objects

### `select_line(...)`

Resolves a selector, such as `Hb4861` or `oiii`, into a line-analysis object.

Returns:

- `SelectedLineProfile` for a single model
- `SelectedLineCollection` for multi-spaxel or multi-fit containers

### `SelectedLineProfile`

Represents one extracted line profile.

Important attributes and properties:

- `source_model`
- `selector`
- `entries`
- `component_names`
- `domain`
- `axis_unit`
- `output_unit`
- `flux_unit`
- `eqw_unit`
- `position`

Important methods:

- `evaluate(x)` or `__call__(x)`
- `evaluate_components(...)`
- `infer_window(...)`
- `measure(...)`
- `eqw(...)`
- `sample(...)`
- `sample_eqw(...)`

### `SelectedLineCollection`

Represents the same logical selection across a multi-fit result.

Important properties:

- `shape`
- `axis_unit`
- `output_unit`
- `eqw_unit`

Important methods:

- `get_profile(index)`
- `evaluate(...)`
- `measure(...)`
- `eqw(...)`

### `LineResult`

Container for measured quantities of one extracted line.

Each measurement is stored as a `Metric`, with:

- `value`
- `std`
- `lolim`
- `uplim`
- `unit`
- `quantity`

`LineResult.to_frame()` returns a compact table including the unit column.

## Complete Examples

### 1. Select and Evaluate a Line from a Composite Model

```python
import numpy as np
from prism.modeling.models import GaussianLines
from prism.modeling.line_analysis import select_line

hb_narrow = GaussianLines.from_arrays(
    names=['Hb4861'],
    pos=[4861.333],
    name='hb_narrow',
    amplitude=2.0,
    offset=0.0,
    fwhm=300.0,
)
hb_broad = GaussianLines.from_arrays(
    names=['Hb4861'],
    pos=[4861.333],
    name='hb_broad',
    amplitude=1.5,
    offset=150.0,
    fwhm=1800.0,
)

model = hb_narrow + hb_broad
hb = select_line(model, 'Hb4861')

x = np.linspace(4800.0, 4925.0, 4000)
y = hb(x)

print(hb.component_names)
print(hb.position)
```

### 2. Measure Flux, Width, and Moments

```python
result = hb.measure(x=x)

print(result.flux)
print(result.peak_position)
print(result.fwhm)
print(result.moment1)
print(result.to_frame())
```

### 3. Equivalent Width Against an Explicit Continuum

```python
ew = hb.eqw(continuum=100.0)
print(ew)
```

If the selected line carries quantity units, the EW unit follows the spectral axis.

### 4. Quantity-Aware Selection in Frequency Space

```python
import astropy.units as u
from prism.modeling.models import GaussianLine
from prism.modeling.models.lines import profiles
from prism.modeling.line_analysis import select_line

line = GaussianLine(
    amplitude=3.0 * u.Jy,
    position=5007.0 * u.AA,
    fwhm=500.0 * u.km / u.s,
    domain='frequency',
    name='oiii',
)

sel = select_line(line, 'oiii')
nu0 = (profiles.C_AA_S / 5007.0) * u.Hz

print(sel(nu0))
print(sel.eqw(1.5 * u.Jy).quantity)
print(sel.measure(num=2048).flux.quantity)
```

### 5. Monte Carlo Sampling from Attached Parameter Uncertainties

```python
hb_narrow.amp_hb4861.std = 0.10
hb_broad.amp_hb4861.std = 0.08

summary, samples = hb.sample(
    n_samples=128,
    random_state=123,
    return_samples=True,
)

print(summary.flux)
print(samples.head())
```

### 6. Evaluate Individual Contributing Components

```python
parts = hb.evaluate_components(x=x)
for name, values in parts.items():
    print(name, values.shape)
```

This is useful when one selected physical line is built from multiple components, such as narrow and broad systems.

## Notes

- `line_analysis` works on intrinsic model components, not on linear-operator pipe wrappers.
- The selected profile stays in the same spectral domain as the source model.
- Unit propagation now follows Astropy units consistently through profile evaluation, measurement results, and EW.
- This file is a compact addendum intended to complement fuller documentation later.