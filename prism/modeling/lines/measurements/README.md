# Line analysis

Small tools for extracting a fitted line and measuring its sampled profile.
They do not prescribe a scientific analysis pipeline.

## From a fitted result

```python
line = select_line(result['red'], 'OIII', components='lines')
measurement = line.sample(method='covariance', n_samples=1000, random_state=7)
measurement.flux
```

Choose an observation explicitly for a joint result; a single FitResult can be
passed directly. Profiles are intrinsic to the selected source. Sampling retains
the complete fitted parameter graph and the global covariance, including drivers
in another observation and cross-observation correlations. The selected source,
shared parameter graph and links are copied together; sampling never changes the
result snapshot. Linked target errors are derived from these draws, rather than
assigned independent marginal errors. A loaded FitResult supports the same path.

This is Gaussian propagation of the fitted covariance with the existing bound
clipping policy, not joint posterior sampling. A separately supplied continuum
is held fixed; use the documented continuum treatment below when its uncertainty
must be included. Separate calls do not constitute a common multi-observable
sample archive.

## From a model or from arrays

```python
from prism.modeling.lines.measurements import select_line, profile_metrics

line = select_line(model, 'Hb4861', components=['narrow', 'broad'])
y = line(x)
parts = line.evaluate_components(x=x)

metrics = line.measure(x=x)
# The same numerical measurements on any continuum-subtracted profile:
metrics = profile_metrics(x, y)
```

`profile_metrics(x, y)` accepts matching finite one-dimensional arrays with at
least three samples. Coordinates must be strictly monotonic; descending inputs
are oriented internally. Quantity inputs retain their coordinate and density
units. Plain arrays do not imply physical units. Coordinate transformations and
corresponding spectral-density transformations remain the caller's choice.

`LineResult` exposes `Metric` objects such as `flux`, `peak_value`,
`peak_position`, `fwhm`, `sigma`, `moment1`, `moment2`, `skewness` and `kurtosis`.
Each metric carries `value`, `std`, `lolim`, `uplim`, `unit` and `quantity`.
`to_frame()` returns a table. Deterministic measurements do not estimate errors.

`select_line` resolves the requested components and physical transition once.
Parameter edits remain live; rebuild the selection after replacing components or
changing the compiled line-table layout. Prefer an explicit choice:

```python
from prism.modeling.models import get_components

parts = get_components(model, additive=False)
line = select_line(model, 'Hb4861', components=[parts['narrow'], parts['broad']])
```

Names, indices, a single component, or a list of component objects belonging to
the parent are supported. Aliases of the same leaf count once. Object references
are matched by identity; an unrelated object with the same name is rejected.
Keeping the parent preserves joint covariance and links to other components.
For independent components, a shorter standalone form is available:

```python
line = select_line([narrow, broad], 'Hb4861')
```

This form does not recover a former parent's covariance; components with links
must be selected from their original parent. For a whole template complex or
band integral, evaluate the component directly and use `profile_metrics`.

Without `components=`, discovery only matches tags in a single group or a simple
sum of components. It does not distribute algebraic expressions: non-additive
models require explicit intrinsic leaves. The old `additive` argument is retained
for existing calls but no longer requests algebraic decomposition. Response
pipes require `components=` explicitly, keeping the fitted parent for joint
covariance and links while evaluating the chosen intrinsic leaves. For example:

```python
line = select_line(fitted_model, 'Hb4861', components=['blr', 'hhe_nlr'])
```

Equivalent widths of such selections require an explicit intrinsic continuum;
Prism does not subtract an intrinsic line from a response-folded parent.

For multi-transition templates, `position=` chooses a rest position (a scalar
uses native coordinates; a Quantity is converted). Otherwise the unique member
with weight=1 is selected, or the sole member. Ambiguity raises. Selected
components must use the same native coordinate unit; equivalent amplitude units
are converted before summation, and unitless/quantity amplitudes cannot be mixed.

For cube selections, specify components by names or indices so the same choice
can be applied to each reconstructed model. Explicit component objects from one
spectrum are not reused across different models.

## Keep physical choices visible

```python
summary, samples = line.sample(
    x=x, n_samples=1000, method='std', distribution='gaussian',
    random_state=7, return_samples=True,
)
```

Sampling is optional. The same numerical measurement functions process every
realization. The source model is left unchanged. Fixed parameters stay fixed;
derived parameters follow their links through the parent model. Arbitrary
callable links require retaining all parent free parameters as possible drivers.
Only parameters with usable uncertainty information are independently perturbed;
others stay nominal. This is conditional error propagation, not a posterior fit.

- `method='covariance'`: joint Gaussian draws from the parent covariance in free
  parameter order. Both compressed fit covariance and cube-expanded covariance
  with explicit fit indices are supported. Invalid matrices raise.
- `method='std'`: independent draws from attached standard deviations. Gaussian
  draws are explicit; `distribution='auto'` retains the existing variance-matched
  uniform rule, value ± sqrt(3)*std.
- `method='limits'`: uniform draws between attached finite lower/upper limits.
- `method='auto'`: use valid covariance first, otherwise attached limits/std.
  Unusable covariance emits a warning before falling back.

Draws are clipped to the free-parameter intervals, including bounds propagated
from supported affine linked targets, for compatibility. Clipping creates mass
at boundaries and is not sampling a truncated posterior. Reported intervals
summarize propagated draws and retain the existing boundary-limit convention.
A seed gives reproducibility within this implementation; corrected parameter
selection/link handling can change samples relative to earlier versions.

Equivalent width uses the same selected profile:

```python
ew = line.eqw(continuum="host", x=x)
summary, samples = line.sample(
    continuum="host", x=x, method="covariance", random_state=7,
    return_samples=True,
)
ew_samples = summary.eqw
```

A named continuum is resolved in the fitted graph, preserving its uncertainty
and correlations with the line. A scalar or external callable is treated as
exact. `.eqw()` propagates available covariance using a local delta approximation;
a bootstrap input instead reuses successful refits. Omitting the continuum in
`.eqw()` retains the legacy local-median estimate of source minus selected line,
not a continuum fit. The tabular `lines(...).eqw()` requires an explicit continuum.
Positive sampled continua are required for numerical integration; signed nonzero
scalar continua remain supported.

`.sample()` draws correlated **parameters**, without generating data or refitting.
Pass `continuum=` to include EW in the same samples as flux and shape metrics;
otherwise the returned result has no `eqw` attribute. This replaces the separate
`sample_eqw` method. A bootstrap input reuses its actual successful refits;
`n_samples`, distribution and seed apply only to newly generated parameter draws.
A model without uncertainty information cannot supply meaningful samples.

## One component, all transitions

```python
from prism.modeling.lines.measurements import lines
component = lines(result, component="narrow")
fluxes = component.flux()
equivalent_widths = component.eqw(continuum="host")
```

Exactly one component is accepted. Omit `component` only for a standalone
`LineSet` or its single-component fit. Every physical transition gets a row;
weights are applied once and rows are never implicitly summed. `select_line`
remains the route to one identified transition across several components.
Tables contain values, units and available uncertainty summaries; missing
uncertainties are NaN. A bare model gives nominal values; use the fit or bootstrap
result to retain its statistical context. Methods compute on request: retain the
returned table for reuse, rather than relying on a hidden cache.

## Numerical conventions

Flux and moments describe the supplied finite window, not an infinite analytic
profile. Heavy-tailed moments may depend strongly on that window. FWHM and other
fractional widths use the nearest crossings about the highest positive peak,
not the outer extent of all peaks in a blend. Negative-only profiles have signed
flux/moments and undefined crossing widths. Widths remain in the coordinate unit;
converting them to velocity requires an explicit reference and convention.

`select_line` on a multi-fit result returns `SelectedLineCollection`.
`collection.measure(x=x)` produces spatial maps with the same numerical core;
`collection[index].sample(...)` propagates uncertainties for one spectrum.
There is no separate cube-specific implementation of the physics.

## Implementation map

- `numerical.py`: flux, crossings, moments and EW on numeric arrays; NumPy only.
- `measurement.py`: array/Quantity entry point and deterministic selection adapters.
- `sampling.py`: parameter draws, parent-model updates, shared propagation and summaries.
- `selection.py`: component/member resolution and thin user-facing methods.
- `results.py`: scalar/spatial results, units and table presentation.

The existing root imports remain supported. Direct imports of the sampler from
`measurement.py` should use `prism.modeling.lines.measurements` or `sampling.py`.
Luminosities, mass calibrations, extinction assumptions, outflow geometry and
energetics belong in notebooks or downstream applications.

## Loading fitted models

Callable Python links are analysis code and are not reconstructed by model YAML.
Newly saved parameter states record whether a link existed and loading warns
which constraints need restoring. Reapply those `.tied` functions before
refitting or propagating covariance. The saved covariance is usable only when
the restored free-parameter order matches the fit. The Blended Line Analysis
example shows this explicitly after regenerating its input with Model Single
Spectrum. No callable pickle is loaded automatically.


## Native components

The same `select_line`, `measure_line`, equivalent-width and covariance-sampling
tools now accept native LineSet groups and single Component objects built
from GaussianProfile, LorentzianProfile or VoigtProfile. Native groups also
support custom profile functions. Keep the full parent model when parameters
are linked or correlated; `components=` may identify a component by name, index
or the original native Component object.

Selection chooses one transition within a template. A sole member or a unique
weight-one member is the default; otherwise supply its rest `position=` explicitly.
The selected scale is weighted exactly as during fitting, whether the profile
uses peak amplitude, integrated flux or another explicit convention. Native
sampling includes the selected template's independent shape parameters and its
shared parameters, and retains the parent's covariance and parameter links.

Built-in profiles can prepare a finite measurement window using their effective
width, including configured instrumental broadening. This is only a grid estimate,
not an analytic measurement of the selected line. Custom profiles require an
explicit x grid or window; zero-width native profiles do too. If continuum is
inferred for custom equivalent widths, also supply a window, or pass an explicit
continuum. Numerical fluxes, moments and wings always refer to the chosen finite
integration domain.

Native functions use numeric output units chosen by the caller. The line table
records the coordinate unit; an output physical unit is not invented for a numeric
amplitude or flux parameter. Source-profile selection preserves all broadening
configured inside that component, including instfwhm. It removes an external
response operator from the measured leaf; it does not automatically subtract
instrumental widths. Response-pipe selections still require explicit source
components and an explicit continuum for equivalent widths.

Missing cube fits retain NaNs in evaluation and measurement maps. Unit metadata
is obtained from the template, so an invalid first spectrum does not prevent
measurement of the valid spectra. Requesting an individual missing profile raises
a clear error rather than constructing a model with invalid scalar parameters.

## Complete multiplets

Pass an explicit sequence to `position` to select several members of one template:

```python
doublet = select_line(best, "OIII", components="narrow",
                      position=[4958.9, 5006.8]*u.AA)
result = doublet.measure()
uncertainty = doublet.sample(n_samples=1000, random_state=41)
```

Each weight is applied once. The summed profile shares the parent parameter graph
and covariance; uncertainties are not combined as independent member errors.
Repeated requested positions are counted once. Omit `position` to retain the
single-member rule (unique unit-weight anchor or sole member).

### Analytic total integrals

`select_line(result, tag, components=...).flux()` returns a `Metric` for the total
intrinsic integral of built-in Gaussian, Lorentzian and Voigt profiles. It uses
analytic profile derivatives and the full fit covariance, including links.
`std=None` means no covariance is available. An unsupported custom integral
raises `NotImplementedError`; use `.measure(x=...)` or `.measure(window=...)`.
With a `BootstrapResult`, `.flux()` propagates actual successful refits and
reports their standard deviation and central interval. Use `lines(model).flux()`
for nominal per-transition values, or `lines(result, component=...).flux()` to
retain the statistical context. Line components do not expose measurement methods.
