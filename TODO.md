# Prism 1.0 interface and fitting architecture

- [x] Expose the same live parameter objects through attributes and `params`
  on native components, LineSet, adapted models and compound expressions.
- [x] Add `Dataset` with coordinates, observed values, errors/weights, mask and
  optional response; adapt the current Spectrum view without reapplying its
  original-grid mask. Share the numerical preparation with direct array fits.
- [x] Add `Dataset.predict(model, x=None)` for the observable prediction; retain
  explicit operator pipes and profile-local `instfwhm`.
- [x] Add canonical `components` access without exposing native internal leaves
  as the ordinary user interface. Keep physical components distinct from
  algebraically expanded additive terms and response-folded predictions.
- [x] Implement `jointfit(associations)` with one signature: a named mapping
  from labels to `(model, dataset)` pairs. Reusing a model shares its parameter
  nodes; distinct models have independent nodes unless linked. Do not add
  `sharedfit` or a second shorthand signature.
- [x] Prepare and copy the complete joint graph once, preserving shared nodes
  and cross-dataset links. Assemble global residuals/Jacobian, count unique free
  nodes and retain each dataset's native grid, selection and response.
- [x] Introduce the common `FitResult` contract for single and joint fitting:
  fitted source, observed prediction on original/alternative grids, immutable
  parameter estimates, diagnostics and covariance with explicit free ordering.
  Single-fit inplace updates values only; joint/batch fits copy their graphs.
- [x] Finalize linked-bound policy for the numerical adapters. Compile declared
  affine scalar links to exact free-node intervals, including chains and mapped
  batch controls. Reject incompatible or unsupported target bounds before fitting;
  retain raw constraints for future specialized fitters. Propagate the same
  intervals into line-measurement sampling without clipping the link itself.
- [x] Persist fitted associations, links and covariance, with a complete-graph
  reuse path preserving shared source/response nodes; propagate derived
  measurements using the complete covariance, including cross-dataset terms.
  FitResult.save/load and associations preserve the full graph; per-observation
  line sampling retains external drivers. Saved sessions require scalar nodes,
  declarative links and supported operator reconstruction rules.
- [x] Adapt `multifit` to regular arrays (`axis`) and named Datasets using the
  same local fit preparation. Keep states/optimizations independent; validate
  scalar or per-target initial values and bounds before starting workers.
  Named collections return dictionaries of FitResult snapshots; arrays retain
  compact maps. Source/response links stay local in serial and spawn execution.
- [x] Rename the SciPy default to `TRF`, isolate optional Sherpa loading, and
  silence the routine tied-analytic-derivative notification by default.
- [x] Migrate maintained notebooks to the final interface, then revise them as
  short scientific walkthroughs. Keep regression assertions in tests; retain
  displays that teach parameter/result inspection and meaningful diagnostics.
- [x] Validate full reference workflows from realistic constructor starts,
  including both supported optimizers, different responses/grids, masks,
  cross-dataset links, covariance, serial/parallel batch fits and persistence.
  Beta verification uses a built wheel in an isolated Python 3.11 environment;
  repeat the release checks when dependencies or supported platforms change.

- [x] Keep built-in profile physics explicit; provide analytic total fluxes on
  ready-made line sets and covariance/bootstrap propagation on selections.
- [x] Keep bootstrap summaries in BootstrapResult, add result inspection and
  lightweight parameter loading from sessions; simplify line-file/window setup.

- [x] Separate preset loading, optional filtered export and explicit local-directory
  configuration; keep paths exact and freeze each model's prepared transition table.
- [x] Keep Spectrum working arrays authoritative through copies/cutouts/rebinning,
  with descriptive frame metadata and explicit copies instead of hidden history.

# Deferred improvements

- Benchmark infrastructure time and memory on realistic multi-component fits,
  parameter links and response operators, including joint and batch workflows.
  Compare with Astropy TRF on equivalent supported problems at matched solver
  settings, accuracy and convergence; measure Prism-only capabilities separately.
  Separate preparation/result construction from repeated residual/Jacobian and
  response evaluation, and distinguish infrastructure overhead from optimizer
  evaluation counts. The small two-Gaussian timing is not evidence of scaling.
  Optimize only measured bottlenecks when the practical benefit justifies the
  added complexity, preserving scientific readability and numerical behavior.
- Evaluate the model on an independent, finer grid and integrate the response-folded prediction over observed bins; consider analytic profile integrals and selective component refinement where useful.
- Optional initialization and parameter-scaling strategies for difficult fits, benchmarked with both SciPy and Sherpa.
- Rework LMFitter for reliable bounded and linked fits.
- Optimize SeparableTRF for line models; validate reduced derivatives, active bounds and covariance.
- Evaluate an optimized XSPEC-inspired Levenberg–Marquardt implementation against the existing engines.


## Primary native 2D fitting

- Treat general 2D fitting as a primary supported Prism capability;
  retain the current, tested Astropy 2D adapters throughout the migration.
- Extend the native component/evaluation contract to two independent coordinates
  with explicit source coordinates `(x, y)` and observed values `z`.
  Distinguish these coordinates from operator inputs such as `(x, flux)`.
  Keep coordinate preparation distinct from flattening residuals/Jacobian rows.
- Reuse the common numerical problem for SciPy and Sherpa with 2D masks, errors,
  bounds, links, parameter covariance and consistent mutation/copy semantics.
- Make image-coordinate units, axis order, pixel sampling/integration and the
  relationship with PSF calibration explicit. Keep PSF-specific fitting tools
  separate from the general model-fitting contract.
- Validate known-source recovery, analytic/numeric derivatives, linked components,
  masked pixels, serialization and representative image workflows against the
  supported Astropy adapters before changing the default public 2D route.

## Generic core and domain organization

- Keep components, parameter graphs, algebra, constraints, numerical fitting and
  persistence independent of spectral physics. Generic 1D functions already use
  the same native `Component` and fitting engines.
- Keep transition sets, kinematic conventions and spectral responses in domain
  specializations. Consider a dedicated `spectral` model namespace after the
  current migration; avoid another premature directory reshuffle.
- A future `base` collection may provide elementary generic models. It is not
  required for fitting user-defined functions or for completing this review.
- Extend the generic native coordinate contract to 2D as a primary architecture
  task, retaining the supported Astropy route during implementation.


## Future specialized fitters

- Keep raw model links/bounds and prepared data/residuals independent of the
  solver's supported constraint representation. Additional linear inequalities
  or non-linear constraints may be implemented by a specialized fitter without
  redesigning Dataset, component composition or FitResult.
- When a concrete template/population workflow is adopted, support efficient
  numerical template/coefficient blocks and a validated linear/non-linear split;
  do not require one Python component object per template coefficient.
- Add regularization or kinematic penalties through an explicit numerical
  objective extension, retaining separate data-statistic diagnostics. Correlated
  data errors likewise require an explicit residual-weighting extension. These
  are future capabilities, not new requirements for the present 1.0 review.

## Follow-up from notebook review

- Add a native interface for arbitrary transformations of upstream model values,
  beyond the current matrix operators. Define coordinates, parameters, input/output
  grids and boundaries, chain-rule derivatives and session reconstruction; coordinate
  this with operator-aware line selection below. These need not be convolutions.
- Define operator-aware line selection: distinguish instrumental responses from
  physical transformations, retain additive branches and their fitted context,
  and specify the corresponding integration grids before extending select_line.
  Current selections intentionally measure intrinsic line leaves.
- Validate native 2D fitting before replacing the current Astropy spatial route;
  PSF calibration continues to use the tested Astropy adapters for 1.0.
- Evaluate explicit spline edge constraints or curvature regularization only
  when demanded by calibration data; repeated endpoint knots alone do not
  constrain endpoint derivatives.
