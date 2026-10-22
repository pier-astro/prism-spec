# Numerical fitting

SciPy and Sherpa are the supported numerical fitting backends. Both use the same
prepared parameter mapping, model evaluation, residuals and Jacobian.

```python
from prism.modeling.fitting import TRF, SherpaLM
from prism.modeling.interop.astropy import TRFLSQFitter
from prism.experimental import LMFitter, SeparableTRF
```

The Astropy adapter retains Astropy models, Quantity handling and spatial 2D
support. It is deliberately outside the numerical spectral fitter namespace.
LM and Separable remain development solvers, imported explicitly; they are not
loaded or exported by the supported fitter package.

Batch fitting is organized in `multifit/`: `configuration.py` prepares shared and
mapped parameter controls, `execution.py` runs independent fits and collects
results, and `results.py` owns result arrays and spectrum/component views.
Each worker keeps private state. Regular arrays retain compact parameter maps;
named Datasets return a dictionary of independent FitResult snapshots. Both paths
reuse snapshot startup and streamed completion, without sharing fitted states.

# prism.modeling.fitting

The `prism.modeling.fitting` module provides optimization, uncertainty estimation and batch execution for native Prism spectral expressions and supported Astropy models.

Native `prism.modeling.Model` expressions use the same SciPy, Sherpa and separable
engines, masks, covariance ordering, outlier rejection and cube execution path.
Adapt an existing expression with `Model.from_astropy(model)` before fitting.
These native roots require numeric `x`, `y`, `yerr` and weights in consistent
native units; convert Quantities explicitly. Ordinary model evaluation remains
unit-aware. Astropy fitter wrappers require Astropy model roots. Fitting does not add methods
to Astropy classes: use `fitter.covariance` and `io.model_to_fits` for those roots,
or explicitly activate the optional class extensions in `interop.astropy`.
Native numerical fits return `FitResult`; use its `.covariance`, and obtain a
source model with `.model` for evaluation, editing or model-only persistence.

Fixed linear operators transform analytic source derivatives in one matrix
operation. Parameterized operators and general `(x, flux)` transforms use a
numerical Jacobian of the whole expression, so their effects on both the source
and their own parameters are included.

## Prepared numerical problem

SciPy, Sherpa and LMFitter share a fit-local `_FitProblem`: canonical parameter mapping,
reusable parameter buffer, prepared model/operator evaluation, residuals, analytic
residual derivatives and final parameter updates. Residuals are explicitly
`(prediction - data) * weights`. The Jacobian follows the same free-parameter
order, resolves links and uses the same residual weights.

Adapters retain model copying, uncertainty-to-weight conventions, optimizer
options, convergence diagnostics and covariance estimation. In particular,
Sherpa's current conversion from statistical errors is preserved exactly.
Input preparation is shared as described below. SeparableTRF retains its
separate variable-projection calculation.

SciPy budget controls use call options before constructor options, then the global
`set_max_evaluations` default. Supported aliases are `max_evaluations`, `max_nfev`,
`maxiter`, `maxfev` and `maxfun`; conflicting aliases raise. Call overrides do not
change the next fit. Single fits copy the input model by default; `inplace=True` updates its
parameter values and still returns a FitResult. Joint and batch fits do not
support inplace updates.

## One-dimensional input contract

Single-spectrum fitting and cube workers use the same preparation of `x`, `y`,
`yerr`, residual `weights` and validity masks. Cube workers delegate selection
to the spectrum fitter rather than slicing and validating the spectrum twice.

- `x` and `y` are matching one-dimensional arrays; errors and weights have the
  same shape. Errors are symmetric standard deviations, not inverse variances.
- `mask=True` means included, as for Prism cubes. NumPy masked-array masks mean
  excluded; their masks are combined with the explicit validity mask.
- Included errors must be finite and strictly positive; weights must be finite
  and nonnegative. Zero weights exclude samples. Explicit residual weights take
  precedence over `yerr`; chi-square weights derived from errors are `1/yerr`.
- Non-finite included coordinates/data/errors/weights raise by default.
  `filter_non_finite=True` explicitly excludes them. Values outside the selection
  are ignored. Empty selections raise, or become skipped cube targets according
  to `skip_invalid`.
- `ndata` and degrees of freedom count selected samples, including after clipping.
  The preparation does not modify user arrays or infer unit conversions.
  Result Datasets store the effective fit mask, including filtering and zero
  weights, so reusing a result preserves the fitted sample set.

For the prepared SciPy/Sherpa/LM path, models are evaluated on all finite input
coordinates, then residuals/Jacobian rows are selected. Excluding observations
therefore does not redefine a profile-wide normalization or convolution. A
response with a declared native grid still evaluates its source on that grid.
Other adapters retain their selected-coordinate evaluation; full-grid support
for general profile transforms is currently specific to the prepared path.

Native `Model` fits continue to require numerical data in coherent native units.
Quantity arrays retain units through preparation for supported Astropy adapters;
this does not add automatic unit conversion to native numerical kernels.
Spatial Astropy fitting retains its existing separate two-dimensional path.

Outlier rejection prepares one effective residual weighting from the errors or
explicit weights, then permanently zeros rejected samples. Both SciPy and Sherpa
respect these exclusions. The clipping function receives masked raw residuals;
it can return a masked array or a boolean rejection mask (True means rejected).
Clipping is not silently changed to clipping error-normalized residuals. Inner
fit exceptions propagate, rather than returning an unexplained `None` result.

## Algorithm and diagnostic scope

For ordinary squared loss, the numerical fitters report `stat = sum(residual**2)`
and `cost = stat/2`. Sherpa's cost now follows this convention. LMFitter and
SeparableTRF evaluate the actual weighted residual when no parameter is free,
with zero optimizer calls. An incomplete LM fit retains its finite final
statistic while reporting `success=False`; convergence and fit quality are
separate information.

LMFitter now uses the shared prepared model/response and preserves the full
finite coordinate grid with masks. Its existing log/logistic bound transforms
remain experimental: near a bound they can suppress numerical derivatives,
and extreme internal coordinates can overflow. Sharing the evaluation engine
alone does not establish numerical equivalence to XSPEC's bounded LM method.
The alternative physical-step and separable-basis experiments live outside the
package until their scientific and performance benefits are established.

## Features
- **MultiFitMixin**: Automatically handles broadcasting 1-D models over 2-D/3-D spatial maps via the `.multifit()` method, utilizing multiprocessing and masking.
- **Bootstrap Uncertainties**: Robust parameter uncertainties using empirical bootstrap resampling (`.bootstrap_uncertainties()`).
- **Sherpa Fitters**: Wrappers around CIAO's `sherpa` optimizers allowing chi-square optimization.
  - `SherpaLM`: Levenberg-Marquardt (fast, local gradient-based).
  - `SherpaSimplex`: Nelder-Mead (robust, gradient-free).
  - `SherpaMonCar`: Monte Carlo global optimization.

## Usage
Simply instantiate a fitter, pass it a parameterized model and the observed data, and receive a FitResult with the optimized model and diagnostics.

```python
from prism.modeling.fitting.sherpafit import SherpaLM
from astropy.modeling.models import Gaussian1D

fitter = SherpaLM()
model = Gaussian1D(amplitude=1, mean=5000, stddev=10)

result = fitter(model, x, y, yerr=yerr)
best_fit = result.model
```

## Observations and response

`Dataset` associates one 1D observation with its coordinates, errors or residual
weights, validity mask and optional response. Models remain independent of the
observation and can also be evaluated intrinsically.

```python
from prism.modeling.fitting import Dataset, TRF

data = Dataset(spectrum, response=response)
initial_prediction = data.predict(source)
result = TRF()(source, data)
prediction = result.predict()
```

The result separates the fitted source, observation and observable prediction.
An operator explicitly piped into the source stays in that source expression.

For arrays, use `Dataset(x, y, yerr=errors, mask=selection, response=response)`.
Data controls belong to the Dataset: repeating `yerr`, `weights` or `mask` on
the fitter call raises instead of silently replacing its definition. With array
inputs, the direct `fitter(model, x, y, ...)` route remains available. The two
routes use exactly the same validation and numerical evaluation.

`Dataset(spectrum)` references the Spectrum's current working arrays and
metadata. Construct it after cutout or rebinning; replacing a working array in
Spectrum later does not update existing Dataset references. Its original-grid
mask is already reflected in the selected arrays. An additional Dataset mask
therefore refers to the current working grid. Coordinate/value units are
retained, with no implicit numeric conversion at the fitting boundary.

`predict(source)` returns predictions on all finite observation coordinates,
including masked samples; non-finite coordinates receive NaN. Masks select
residuals and Jacobian rows during fitting. They do not trim the finite source
grid before profile-wide transforms or response evaluation. Responses with native
grids retain their native-grid folding and observation-grid sampling.

Explicit `source | response` composition remains supported. Every declared
operator is applied in order: a Dataset response follows operators already in
the supplied source expression. Use profile `instfwhm` or a response that includes
the same instrumental broadening, taking care not to count that effect twice.

## Single-fit results and mutation

```python
result = fitter(source, data)
source_fit = result.model       # Detached source, with explicit pipes retained
intrinsic = source_fit(x_plot)
observed = result.predict()    # Original observation grid
smooth = result.predict(x_plot)
result.params                  # Read-only fitted estimates, including std
result.covariance              # Free-parameter order selected by fit_indices
result.fit_info                # Termination and objective diagnostics
```

Single fits accept arrays or a Dataset and always return FitResult. Both SciPy
and Sherpa default to `inplace=False`. With `inplace=True`, values update in the
supplied source and any fitted Dataset response, preserving parameter identities
and links. Errors, covariance and diagnostics belong to the result; prior model
uncertainty attributes are cleared on the updated objects. Copy-based fitting
leaves the original objects untouched. Results snapshot models, response and data,
so later edits cannot change the recorded prediction or covariance.

A Dataset response stays separate from `result.model`; an operator explicitly
piped into the input model remains part of that model. `params` and `param_names`
cover the complete fitted expression, including response parameters; array-valued
Astropy estimates retain their shapes. `parameters` is the flattened vector and
`fit_indices` identifies its free entries. A result is not callable or a model
proxy: use `.model` or `.predict()` explicitly.

`Dataset.predict(model, x=None)` and `result.predict(x=None)` use the original
observation grid when x is omitted. Masks do not cut predictions. A spectral
response evaluates on its native source grid, applies its matrix, then linearly
interpolates the folded values to x, with zero outside the response output range.
This is point sampling, not integration or flux-conserving rebinning onto new
bins. Strict responses reject a different grid. Without a response, the model is
evaluated directly without window padding. Non-finite coordinates receive NaN.

Bootstrap and regular-array multifit use the same optimization path without
constructing a full result/data snapshot for every realization. Named Dataset
collections retain one complete FitResult per observation. `multifit` rejects `inplace`:
each target always receives an independent model. Outlier removal returns a
FitResult whose Dataset mask includes the final rejected samples.

For line measurements with covariance, pass a single result directly:
`select_line(result, 'Hb', components='narrow').sample(method='covariance')`.
The measurement uses a detached complete fitted graph; it does not attach errors
to `result.model`. Explicit `covar_attach(model, result)` remains available for
model-only exports or utilities that intentionally require attached estimates.
The TRF rename remains tracked in TODO.md.

## Fit observations independently

For a cube or other regular array, use the coordinate sample dimension:

```python
results = fitter.multifit(source, x, y, yerr=errors, axis=0,
                         initpars={'amplitude': initial_map})
amplitude = results.params['amplitude'].value  # nonsample input shape
prediction = results.evaluate()  # same sample-axis position as y
```

Only the sample dimension is special: remaining dimensions index independent
fits and define parameter-map shapes. Thus `(n_spectra, n_samples)` yields
1D parameter arrays and `(n_samples, ny, nx)` yields `(ny, nx)` maps. Parameter
value/std access returns read-only views over stored arrays, without stacking all
parameters or reconstructing models. `axis` is inferred when exactly one array
dimension matches `len(x)`; ambiguous shapes require it explicitly. Errors,
weights and masks must match the complete data shape. There is no implicit
broadcasting between axes. Parameter starts and each side of their bounds may be
scalar or mapped over the nonsample shape. `binmap` retains first-member fitting
and replication, without constructing a combined likelihood for a bin.

For different grids, masks or responses, supply named Datasets and omit y:

```python
results = fitter.multifit(source, {'blue': blue, 'red': red}, nproc=2,
    initpars={'amplitude': {'blue': 1., 'red': 2.}},
    bounds={'amplitude': (0., {'blue': 10., 'red': 20.})})
results['blue'].predict()
results['blue'].params
results['blue'].save('blue-fit.fits')
```

The result is a normal dictionary in input-label order. Each value follows the
single-fit contract, including separate source/response, prediction on another
grid, covariance and persistence. Dataset sampling is neither interpolated nor
reordered to form a rectangle. Source controls are scalars or explicit mappings
containing every Dataset label. The two bound sides can be mapped independently;
response parameters are configured on each Dataset response. Fixed/tied overrides
apply to each source copy, without sharing optimized values. Configure a
response link explicitly: ordinary arithmetic on an Astropy Parameter may return
a numeric value rather than a Prism link expression.

Every local graph is independent, even if the same source or response object was
supplied repeatedly. Links within a source and its response remain intact; a
reference outside that local graph is a configuration error. Use jointfit for
cross-observation links. Configuration checks precede workers; per-target sample
validity and numerical optimization remain local. Included nonfinite samples
require `filter_non_finite=True`, as in a single fit.

A local exception produces missing estimates and diagnostic fit_info, while other
targets continue. Empty selections raise when `skip_invalid=False`. Named failed
results retain Dataset and associations for retry, but model, predict and save
raise because no fitted model exists. A completed optimization that reports
`success=False` still retains its last finite model, as for a single fit; inspect
termination separately from solution quality. The fitter's batch fit_info reports
completion counts. Covariances belong to individual fits; no cross-target
covariance is constructed. The regular-array path retains compact numeric result
maps, avoiding full Dataset snapshots per pixel or bootstrap realization. Only
parameter access for a selected target reconstructs its model. Scalar bounds are
stored once; mapped bounds retain array storage rather than per-target tuples.
Full covariance storage (enabled by default; omit with `TRF(covariance=False)`) scales as the target count
times the square of the parameter count; this cost is separate from map access.

## Fit observations jointly

`jointfit` always accepts a named mapping of `(model, Dataset)` pairs. Reuse
the same model for observations that estimate exactly the same parameters.
Distinct model objects have their own parameter nodes; connect selected nodes
with `.link` when required. Components can also be shared between distinct
source expressions. The complete graph is copied together before fitting.

```python
result = fitter.jointfit({
    'blue': (source, blue),
    'red': (source, red),
})

intrinsic = result['blue'].model(x)
blue_prediction = result['blue'].predict(x_blue)
red_prediction = result['red'].predict(x_red)
```

Each Dataset retains its own grid, selection and response, including fitted
response parameters. There is no common observed model when the responses
differ. The model accessor returns the source expression supplied by the user;
operators explicitly composed into that expression remain present. Only the
Dataset response is separate.

`FitResult` stores a snapshot: changing input models/data or the detached copies
returned by `.model` and `.dataset` cannot alter its predictions or covariance.
Per-observation `.params` gives read-only source estimates in local parameter
names. Shared nodes refer to the same estimate. Global `.params` contains unique
nodes, including response parameters, named by their first observation and local
parameter name. Global `fit_indices` selects the covariance order from
`param_names`.

`result.covariance` and `result.fit_info` describe the complete optimization.
Observation views expose their statistic contributions and sample counts through
`.fit_info`. No local degrees of freedom or reduced statistic is assigned to a
dataset that participates in shared parameter estimation.

Residuals from statistically independent observations are concatenated without
interpolating the data. The global Jacobian scatters local model derivatives
into unique parameter columns and applies the link chain rule. A numerical
Jacobian remains available for profiles/transforms without analytic derivatives.
Bounds on affine linked targets are intersected exactly at their free roots,
including cross-observation and source–response links. Unsupported bounded links
raise before optimization; see the linked-bound contract below.

Source copies returned by the result are suitable for evaluation and editing.
Their standalone uncertainty attributes are cleared: marginal errors cannot
replace the global covariance for derived measurements. Fitted estimates and
their available standard deviations are in `result[label].params`; covariance
propagation for one observation uses the complete fitted parameter graph:
`select_line(result[label], tag, components=...).sample(method="covariance")`.
This includes cross-observation drivers and their covariance terms.


## Save and reuse a fitted problem

```python
from prism.modeling.fitting import FitResult

result.save('results/joint.fits')
restored = FitResult.load('results/joint.fits', components={'continuum': continuum})
observations = restored.associations
observations['red'][0].components['continuum'].slope.fixed = True
updated = fitter.jointfit(observations)
```

`associations` copies the complete mapping once. Shared model/component instances,
response nodes and declarative cross-observation links stay shared within that
copy; editing it never changes the saved snapshot. Do not assemble a refit by
separately reading each `.model` and `.dataset`, which returns independent copies.
For a single fit, unpack `model, data = restored.associations['data']` and use the
ordinary `fitter(model, data)` call.

The FITS file stores the source/response association graph, original coordinate
and value arrays, errors/weights, masked-array exclusions, effective selection,
units and metadata, fitted constraints/links, covariance and its explicit free
order, statistic contributions and portable termination diagnostics. Optimizer
runtime objects such as SciPy's OptimizeResult are not persisted. Fixed response
matrices use the existing fingerprinted sidecar files; move them with the main
FITS file. Archive recipes are verified against the available calibration.

Saved sessions currently require scalar parameters and operators with an existing
reconstruction rule. Custom functions/derivatives are supplied explicitly through
`components`; Python code is never stored or executed from the session. Identical
named definitions can have independent fitted instances, while different custom
functions require different names. Opaque callable links are rejected at saving:
use declarative expressions when a fitted problem must be reopened with its links
and covariance intact. In-memory fitting still supports Astropy array parameters
and analysis callables.

---

## What Lives Here

- Prism-native fitters based on SciPy and Sherpa
- batch fitting through `multifit()`
- outlier-removal wrappers
- covariance and resampling uncertainty helpers
- compatibility code that lets Prism-enhanced models and Astropy-style fitting stay interoperable

## Parameter and covariance order

SciPy, Sherpa and Prism LM prepare free values, bounds and parameter links from
public parameter names, sizes and constraints. Covariances use the flattened
free-parameter order. FitResult stores this order in `fit_indices` and the
estimates in `params`; numerical internals and explicit covariance attachment
use `_fit_indices` and `_param_cov`. The same order is used by batch results and
resampling. Astropy adapters and experimental solvers retain model returns.

A returned fit replaces earlier Prism uncertainty estimates: covariance, its
parameter order, parameter standard deviations and bootstrap summaries/samples.
If the new fit has no covariance, none is inherited from its starting model.
Fixed and linked parameters do not retain standard deviations from an earlier
free fit. Recompute covariance propagation or bootstrap summaries for the new
solution. Copy-based fitting leaves the original model and its estimates intact;
reconstructing a cube result likewise replaces template uncertainties.

## Declarative parameter links

Prism line models accept arithmetic links before or after model composition:

```python
component_b.fwhm.link = component_a.fwhm
component_b.amplitude.link = component_a.amplitude / 2.98
model = component_a + component_b
model.amplitude_1.link = 2.0 * model.amplitude_0
```

Supported expressions contain parameter references, finite scalar constants,
`+`, `-`, `*`, `/`, powers and unary minus. Values update when their drivers
change, including ordinary model evaluation. Use `.value` or `.quantity` for
immediate numerical arithmetic rather than building an expression. Set
`parameter.link = None` to detach the link and retain its current value.

Unit-bearing references obey dimensional arithmetic. For example, an additive
velocity offset must be a Quantity, such as `component_a.offset + 10*u.km/u.s`.
Keep the parameter expression on the left of Quantity constants: Astropy can
otherwise evaluate a leading Quantity eagerly and produce a fixed constant.
Ordinary scalar coefficients, including NumPy scalars, can be on either side.
A bare constant assigned directly as a link uses the target's native numeric
unit. Linked values are not independently optimized or clipped: their bounds
restrict the driver through the preparation described below.

Dependencies are ordered before fitting. Cycles and references outside the
assembled model are rejected. A copy of the complete model preserves internal
links and is independent of the original. Copying an isolated dependent component
also copies its external references: it does not redirect them by component name.
Assemble first and copy the complete model, or explicitly reassign the link.
Renaming components does not affect reference identity. Repeating the same
component instance in a linked expression is rejected because its parameter
aliases would be ambiguous after reconstruction; use independent components.

`model.save()` stores a versioned expression tree with parameter references in
the saved model's order. `load_model()` reconstructs these links automatically,
including models with an instrumental response. It neither evaluates code strings
nor pickles Python functions. Standalone YAML export rejects these links because
it lacks the full session graph. Existing development files containing only
callable markers cannot be upgraded by guessing their mathematical meaning.

The new `.link` property belongs to Prism line parameters; native Astropy
parameters retain their existing behavior. For arbitrary Python functions and
native Astropy parameters, retain `.tied` and the `tie()` helper imported from
`prism.modeling.interop.astropy`. Callables are
not automatically persisted and their hidden dependencies cannot be sorted.
Callable-only chains retain parameter order; declared links can explicitly depend
on callable targets. Restore analysis callables before fitting or propagating
covariance after loading. Analytic model Jacobians still use numerically estimated
link sensitivities; a numeric residual Jacobian remains
available for comparison. Prism's Astropy adapter preserves physical constants
and parameter-unit conventions during temporary unit stripping.

## Astropy interoperability and migration

Astropy models and Prism models remain accepted through the existing interfaces.
The native SciPy and Sherpa adapters share prepared evaluation for arithmetic
sources followed by a fixed `SpectralResponse`: source evaluation on the native
grid, matrix application, then sampling on the data grid. Other model/operator
cases retain their existing evaluation path.

For a prepared source made only of sums and differences, parameter slices are
computed once and component derivatives fill the final source Jacobian directly.
Products, quotients and mixed compositions retain Astropy's chain rules.
Profile formulas, parameter ordering, link propagation and response sampling are
unchanged; the same response is still applied to all derivative columns.

Astropy fitter wrappers remain available during migration. TRF and dogbox overlap
with the native SciPy adapters, but removing their wrappers also requires checking
unit handling, multidimensional models, diagnostics and reference workflows.
The linear solver and SLSQP provide distinct capabilities (linear least squares
and equality/inequality constraints); they should not be removed merely because
their implementations use SciPy. Algorithm choice and Astropy compatibility are
separate concerns.

Linked-parameter Jacobians in the Astropy adapters are handled by a Prism-owned
override. Importing Prism does not modify native Astropy fitter methods. The old
patch-installation functions have been removed. The CompoundModel operator patch
still supports response composition and has a separate migration boundary.

## Sherpa options and termination

Optimizer options such as `ftol`, `xtol` and `epsfcn` can be set on the
constructor or the fit call. Call-time settings override constructor settings
for that call only. Evaluation-budget aliases use the shared Prism controls;
unknown optimizer options raise an error. The adapter passes settings through
Sherpa's optimizer configuration.

Sherpa's LevMar can automatically run Nelder–Mead when a solution lies near a
bound. Its success flag does not establish a good minimum. In Sherpa 4.18.0,
MINPACK budget exhaustion can also be reported as success. Prism conservatively
marks a fit incomplete and omits its covariance when the reported evaluation
count reaches an explicitly configured budget; this may also flag convergence
on the final allowed evaluation. With no configured budget, Sherpa's native
termination reporting is retained. Other native status codes are not corrected
by this guard.

## Parallel cube fits

Serial and spawned workers use the same single-spectrum execution routine.
Each target receives an independent model copy, local initial values and bounds,
and its own valid-sample mask. Non-finite data remaining after masking produce
an unsuccessful target, unless `filter_non_finite=True` is explicitly requested
for a fitter that supports it. Other targets continue as before.

Parallel startup writes one temporary internal configuration snapshot and passes
only its path to children. This keeps large model/grid objects out of spawn's
startup pipe, so worker imports can overlap. Each child restores a private fitter
and model, including callable links, the global evaluation-budget default and
any Astropy pipe extension explicitly activated by the parent.
The parent owns the temporary directory until workers stop; it is removed on
success or failure. These internal snapshots are not saved fit sessions.

Completed targets are streamed individually in arrival order and stored by
spatial index. Serial and parallel execution share result collection and bin
propagation. Progress counts completed fits (unique bins with a bin map), not
chunks; tqdm may still throttle display refreshes. The `batch` option has been
removed from `multifit` and bootstrap. At most one worker per independent target
is created; a single target runs serially. `nproc` must be a positive integer.

Before starting children, unset OPENBLAS/MKL/OMP thread counts default to one;
explicit user settings are respected. The parent environment is restored even
when pool creation fails. This avoids setting defaults too late, after numerical
libraries have already loaded in the child. Spawn startup and serialization
remain real costs; very short fits can be faster with `nproc=1`.
Bin labels are grouped once. Bin maps fit the first spatial member of each
bin and distribute that result to its members; they do not coadd spectra.
Without a bin map, targets are enumerated directly without allocating a group
for each spaxel.

### Resampling and objective conventions

Bootstrap uses the same residual multipliers as fitting: `weights=1/yerr`.
Explicit weights take precedence for refitting; supplied errors still define
the noise draws. With weights alone the noise widths are `1/weights`. Zero
weights exclude samples. Asymmetric errors retain their lower/upper widths
for uniform draws and use their mean for fitting.

`statistic='poisson'` in bootstrap selects Poisson count draws, followed by
fixed-error weighted least squares; it is not a Poisson likelihood. Direct
numerical fitters accept `chi2` or `leastsq` and one-dimensional spectra.

For SciPy robust losses, `cost` is the robust optimizer objective while `stat`
is the sum of squared weighted residuals. They satisfy `stat=2*cost` only for
the linear loss. Covariance from the robust Jacobian is a local curvature
approximation, not a complete robust confidence treatment.


## Bounds on linked parameters

The current SciPy and Sherpa adapters request an independent-interval
representation of the model constraints. They preserve declared bounds and link
expressions in the model and FitResult; the compiled intervals are solver input,
not replacement parameter definitions. The prepared problem remains usable by a
future fitter with additional constraint capabilities. Batch preflight delegates
to a fitter hook shared by regular arrays and named Datasets, so a specialized
fitter need not replace the orchestration to use another representation.

For a declared affine link `b = slope*a + offset`, target bounds are transferred
exactly to the free scalar a, then intersected with a's own bounds. A negative
slope reverses the inequality. Chains and multiple targets referring to the same
free root are combined. Fixed parameters can supply coefficients and offsets;
a constant target is checked against its bounds. Compatible fixed parameters are
retained, not counted as free nodes.

```python
model.b.link = 2*model.a
model.a.bounds = (1., 8.)
model.b.bounds = (0., 10.)
# The fit searches a in [1, 5], keeping b exactly equal to 2*a.
```

Initial values must satisfy the resulting intervals. Incompatible intersections
raise before optimization. An intersection that reduces a free variable to a
single point requires fixing that variable explicitly. Model construction and
ordinary evaluation remain editable; bounds are a fitting contract, not clipping
applied to every parameter assignment. Committed fits are checked against the
original declared bounds, allowing floating-point roundoff only.

Finite target bounds on products/ratios of free variables, nontrivial powers,
affine expressions with multiple free roots, and opaque callable ties are not
compiled by these two adapters. They raise NotImplementedError before the
optimizer is called, naming the target and requesting reparameterization. Links
without finite target bounds retain their existing arbitrary-expression/callable
behavior. The compiler does not infer algebra from numerical probes or silently
relax a constraint. A mathematically redundant bound can be removed explicitly
when the remaining parameterization already guarantees it.

For example, use a free nonnegative width increment and define
`broad_width = narrow_width + increment` inside a custom model to guarantee an
ordering. This is an alternative parameterization, not a new independently
bounded two-root link. Auxiliary parameter bounds and the model formula must
together express the intended physical domain.

The same preparation is used for single fits, joint graphs and independent
batches. Array multifit checks mapped starts/bounds with array arithmetic rather
than constructing one model per spaxel. Named Dataset preflight uses the full
local source–response graph; configuration failures do not become local NaN fits.
Line covariance propagation uses the resulting driver intervals, including
bounds declared on linked targets. It retains the existing clipped-Gaussian
approximation, not sampling from a constrained posterior. Local Hessian errors
near active bounds remain approximate and need not define symmetric confidence
intervals.

## Optimizer dependencies and derivatives

`TRF` uses SciPy and selects analytic model derivatives when available; otherwise
it uses finite differences. Linked parameters retain analytic profile derivatives
with numerical link sensitivities for the chain rule. This normal path is quiet.
For non-smooth links, compare with `TRF(jac="2-point")` or reparameterize the model.

Sherpa is optional and must be installed separately. Importing Prism or choosing
`TRF` does not import Sherpa; choosing `SherpaLM` loads its optimizer.
`HAS_SHERPA` reports whether the package can be found, not whether an installation
and all its optional dependencies are functional.

## Covariance option

`TRF()` and `SherpaLM()` retain parameter covariance by default. Use
`covariance=False` to omit the estimate and its storage; large multifit arrays
otherwise scale as the number of targets times the squared free-parameter count.
The matrix is available as `result.covariance`. Astropy interoperability adapters
keep Astropy's separate `calc_uncertainties` constructor option.

## Inspecting results and bootstrap draws

`result.show()` displays fitted values, errors and diagnostics. A detached
`result.model` contains the evaluable model without result uncertainty metadata.

```python
from prism.modeling.fitting.uncertainty import bootstrap

resampled = bootstrap(result.model, TRF(covariance=False),
                      x=x, y=y, yerr=yerr, n_samples=200, seed=11)
resampled.show()
resampled.params["amplitude"]
resampled.samples["amplitude"]
```

`bootstrap` returns `BootstrapResult` and leaves its input unchanged. `.params`
contains nominal values, sample standard deviations and central intervals;
`.samples` contains correlated free-parameter refits, with NaNs for failed
realizations. `.model` returns a detached nominal model. Bootstrap currently
operates on one spectrum, not a complete joint or batch result. Save draws
explicitly with `np.savez(filename, **resampled.samples)` when needed.

New `FitResult.save` files also carry a small parameter-value table.
`load_params(filename)` reads it without reconstructing models or responses,
allowing explicit name-based initialization of a modified model. It does not
assert scientific equivalence of two model expressions. `FitResult.load`
continues to restore complete sessions.
