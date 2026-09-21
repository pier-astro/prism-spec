# prism.modeling.models.lines

Emission-line models for Prism, implemented as Astropy `Fittable1DModel` subclasses.

This module provides two closely related layers:

- analytic single-line models: `GaussianLine`, `LorentzianLine`, `VoigtLine`
- analytic line-group models: `GaussianLines`, `LorentzianLines`, `VoigtLines`

The goal is to keep the public API simple while preserving physically meaningful parameterisation and native Astropy behaviour.

## Concept

These models are built for spectral lines described by:

- a rest-frame transition position
- a velocity offset relative to that rest position
- a velocity width
- an optional redshift
- an optional instrumental broadening term

The line shape is evaluated internally in wavelength space in Angstrom, and can then be exposed in:

- wavelength
- frequency
- energy

through a thin spectral-domain transform. The internal physical reference remains the rest wavelength of the transition.

## Structure

- `profiles.py`
  Pure profile functions, flux formulae, Jacobians, and domain conversion helpers.
- `base.py`
  Shared line-model base class plus flux and EW handling.
- `tables.py`
  Line-table normalisation, spectral-domain conversion, local-resource setup, and wavelength trimming.
- `single.py`
  Single-line Astropy model subclasses.
- `groups.py`
  Dynamic multi-line models built from line templates.
- `__init__.py`
  Public entry points.

## Physical Overview

### Rest Positions

All line templates are stored internally as rest wavelengths in Angstrom.

For the built-in resources under `resources/lines`, this means:

- positions are stored explicitly in Angstrom inside ECSV files
- the line table metadata records `medium='air'`

For user-provided inputs, Prism normalises everything into the same internal wavelength representation.

### Velocity Parameters

Offsets and widths are parameterised in velocity units, typically `km / s`.

This keeps the parameterisation stable across different lines and observational setups:

- `offset` shifts the observed centre relative to the rest transition
- `fwhm`, `fwhm_G`, and `fwhm_L` describe the intrinsic line width in velocity space
- `instfwhm` adds instrumental broadening in the same velocity convention

### Spectral Domains

The model `domain` controls the x-axis on which the model is evaluated:

- `domain='wavelength'` returns spectral density per Angstrom
- `domain='frequency'` returns spectral density per Hz
- `domain='energy'` returns spectral density per eV

If the amplitude is a quantity, the returned profile, flux, and EW follow that domain consistently.

### Line Lists and Media

Line-group models are built from line tables with columns:

- `name`
- `position`
- optional `weight`

Internally Prism stores them as `QTable` with:

- `position` in Angstrom
- `weight` as a numeric scaling factor
- `meta['medium']` set to `air` or `vacuum`

Medium resolution follows a strict precedence rule:

- an explicit `medium=...` passed to the line model constructor wins
- otherwise, if an input ECSV line table declares `meta['medium']`, that file header wins
- otherwise Prism falls back to the current session default medium
- the session default medium is `air` unless changed with `set_medium(...)`

If a file declares one medium and the line model is constructed with another, Prism converts the tabulated line positions accordingly before building the model.

If the input line positions are supplied in frequency or energy units, Prism converts them to wavelength immediately and assigns `medium='vacuum'`, because those spectral conversions are vacuum definitions.

For arrays or other inputs that do not carry medium metadata, Prism assumes the values are already expressed in the requested medium. If no medium is requested, the session default applies.

## Units and Astropy Conventions

Prism follows Astropy-style parameter unit conventions:

- parameters with declared units accept either plain floats or `Quantity`
- plain floats are interpreted in the declared parameter unit
- amplitudes are unitless by default unless the user passes a `Quantity`
- unit strings such as `'km / s'` or `'Jy'` are normalised with `astropy.units.Unit`

That means all of the following are valid:

```python
from prism.modeling.models import GaussianLine

line = GaussianLine(amplitude=5.0, position=5007.0, offset=0.0, fwhm=500.0)
line_q = GaussianLine(amplitude=3.0, position=5007.0, fwhm=500.0)
```

and, with explicit quantities:

```python
import astropy.units as u
from prism.modeling.models import GaussianLine

line = GaussianLine(
    amplitude=3.0 * u.Jy,
    position=5007.0 * u.AA,
    offset=0.0 * u.km / u.s,
    fwhm=500.0 * u.km / u.s,
)
```

## Main Attributes and Properties

Single-line models expose Astropy parameters such as:

- `amplitude`
- `position`
- `offset`
- `fwhm` or `fwhm_G` and `fwhm_L`
- `redshift`

Shared convenience properties include:

- `domain`
- `medium`
- `flux`
- `ew`

`flux` and `ew` return `Metric` objects with:

- `value`
- `std`
- `lolim`
- `uplim`
- `unit`
- `quantity`

Line-group models additionally expose:

- `lines`
  The normalised `QTable` of transitions used by the model.
- `_templates`
  The template names used to build one amplitude parameter per tied group.
- `_param_names_list`
  The generated amplitude parameter names.

Group `flux` and `eqw(...)` return `DataFrame` objects with columns:

- `value`
- `std`
- `lolim`
- `uplim`
- `unit`

## Complete Examples

### 1. Single Gaussian Line

```python
import numpy as np
from prism.modeling.models import GaussianLine

line = GaussianLine(
    amplitude=5.0,
    position=5007.0,
    offset=0.0,
    fwhm=500.0,
    name='oiii',
)

x = np.linspace(4980.0, 5035.0, 2000)
y = line(x)

print(line.flux)
print(line.eqw(100.0))
```

### 2. Quantity Amplitude in Frequency Space

```python
import astropy.units as u
from prism.modeling.models import GaussianLine
from prism.modeling.models.lines import profiles

line = GaussianLine(
    amplitude=3.0 * u.Jy,
    position=5007.0 * u.AA,
    fwhm=500.0 * u.km / u.s,
    domain='frequency',
)

nu0 = (profiles.C_AA_S / 5007.0) * u.Hz

print(line(nu0))
print(line.flux.quantity)
print(line.eqw(1.5 * u.Jy).quantity)
```

### 3. Line Group from Built-in ECSV Files

```python
from prism.modeling import models

hhe_nlr = models.GaussianLines.from_csv(
    name='hhe_nlr',
  csv_files=['hydrogen.ecsv', 'helium.ecsv'],
    amplitude=1,
    offset=0,
    fwhm=500,
    instfwhm=120,
    bounds={
        'amplitude': (0, 100),
        'offset': (-300, 300),
        'fwhm': (0, 800),
    },
)

print(hhe_nlr.lines)
print(hhe_nlr.flux)
```

This remains valid. In this example:

- `amplitude=1` stays unitless
- `offset=0` is interpreted as `0 km / s`
- `fwhm=500` is interpreted as `500 km / s`

### 4. Custom ECSV with Energy Positions

If an ECSV stores transition positions in energy or frequency units, the unit lives in the file itself.

Example ECSV payload:

```yaml
# %ECSV 1.0
# ---
# datatype:
# - {name: name, datatype: string}
# - {name: position, datatype: float64, unit: eV}
# - {name: weight, datatype: float64}
# meta: !!omap
# - {medium: vacuum}
# schema: astropy-2.0
name position weight
Ha 1.8891391061 1.0
NII 1.8833990031 1.0
```

Usage:

```python
from prism.modeling.models import GaussianLines
from prism.modeling.models.lines import set_medium

set_medium('vacuum')

custom = GaussianLines.from_csv(
  csv_files=['energy_lines.ecsv'],
    amplitude=5.0,
    fwhm=300.0,
)

print(custom.lines['position'])
print(custom.lines.meta['medium'])
```

The table is converted internally to Angstrom and marked as vacuum.

### 5. Plain CSV with Explicit Position Unit

If a user keeps a plain CSV, Prism requires an explicit `position_unit`.

```text
name,position,weight
Ha,1.8891391061,1.0
NII,1.8833990031,1.0
```

```python
custom = GaussianLines.from_csv(
  csv_files=['energy_lines.csv'],
  amplitude=5.0,
  fwhm=300.0,
  position_unit='eV',
)
```

### 6. Header Medium vs Explicit Model Medium

If a file declares `medium: air` but the model is constructed with `medium='vacuum'`, Prism converts the stored wavelengths before creating the line model.

```python
air_lines = GaussianLines.from_csv(csv_files=['hydrogen.ecsv'])
vac_lines = GaussianLines.from_csv(csv_files=['hydrogen.ecsv'], medium='vacuum')

print(air_lines.lines.meta['medium'])
print(vac_lines.lines.meta['medium'])
print(air_lines.lines['position'][0], vac_lines.lines['position'][0])
```

### 7. Array Input with Quantities

```python
import astropy.units as u
from prism.modeling.models import GaussianLines

group = GaussianLines.from_arrays(
    names=['Ha', 'NII'],
    pos=[1.8891391061, 1.8833990031] * u.eV,
    amplitude=5.0,
    fwhm=300.0,
)

print(group.lines)
```

In this path the spectral unit is inferred automatically from the quantity array.

## Notes

- The current internal reference for transition lists is always wavelength in Angstrom.
- The built-in resource files remain air-wavelength line lists.
- User inputs are not limited to Angstrom as long as the spectral unit is explicitly declared or inferable from a `Quantity`.
- This file is intended as a compact addendum. The fuller design rationale and extended usage notes should move into the main documentation later.
