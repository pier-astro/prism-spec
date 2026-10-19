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
Native Model results provide `.covariance` and `.save()` directly.

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
change the next fit. SciPy copies the input model by default; `inplace=True`
updates and returns that model.

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
Simply instantiate a fitter, pass it a parameterized model and the observed data, and receive the optimized model back.

```python
from prism.modeling.fitting.sherpafit import SherpaLM
from astropy.modeling.models import Gaussian1D

fitter = SherpaLM()
model = Gaussian1D(amplitude=1, mean=5000, stddev=10)

best_fit = fitter(model, x, y, yerr=yerr)
```

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
free-parameter order; fitted models retain this order in `_fit_indices` alongside
`_param_cov` and per-parameter standard deviations. The same order is used by
batch results, covariance attachment and resampling.

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
unit. Linked values are not independently optimized or clipped to target bounds;
choose driver bounds and expressions consistent with the intended physical domain.

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
link sensitivities, with a per-fit warning; a numeric residual Jacobian remains
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
