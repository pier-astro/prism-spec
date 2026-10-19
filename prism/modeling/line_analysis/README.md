# Line analysis

Small tools for extracting a fitted line and measuring its sampled profile.
They do not prescribe a scientific analysis pipeline.

## From a model or from arrays

```python
from prism.modeling.line_analysis import select_line, profile_metrics

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

Draws are clipped to parameter bounds for compatibility. Clipping creates mass
at boundaries and is not sampling a truncated posterior. Reported intervals
summarize propagated draws and retain the existing boundary-limit convention.
A seed gives reproducibility within this implementation; corrected parameter
selection/link handling can change samples relative to earlier versions.

Equivalent width is another optional operation:

```python
ew = line.eqw(continuum=continuum, x=x)
ew = line.sample_eqw(
    continuum=continuum, x=x, method='covariance', random_state=7,
)
```

A scalar or callable continuum is treated as exact. Omitting it retains the
legacy local-median estimate of source model minus selected line; this is a
convenience policy, not a continuum fit. With an inferred continuum,
`perturb_continuum=True` includes non-line free parameters in the same joint
propagation. Links can also couple line and continuum drivers. EW uses the same
selected transition, grid validation and parameter-update path as other measures.
Positive sampled continua are required for numerical integration; legacy signed
nonzero scalar continua remain supported.

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
`measurement.py` should use `prism.modeling.line_analysis` or `sampling.py`.
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
