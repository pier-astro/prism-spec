# Spectral operators and instrumental responses

A fixed instrumental response is a linear operator:

```text
observed = matrix @ intrinsic
observed_jacobian = matrix @ intrinsic_jacobian
```

Rows index output samples; columns index input samples. Jacobian columns use
sample-major layout when applying the operator. Dense and sparse matrices share
the same application routine, preserving Quantity units. Custom operators may
still depend on parameters: their matrix must be evaluated with those parameters.

## Native expression nodes

Use `source | operator` to apply an operator to a native model expression. Each application evaluates its source on the operator's
input grid, applies it, then samples its output on the requested coordinates.
Nested operators retain their own grids. Multiplicative factors before and after
an operator describe different expressions and are never reordered.

- A fixed `LinearOperatorModel` or `SpectralResponse` prepares its matrix and
  interpolation once per fit. It applies the matrix to the source Jacobian in
  one block. Fixed operators use the existing sparse FITS session resources.
- A custom `LinearOperatorModel.get_matrix(x, *parameters)` is evaluated on
  every call. Parameterized operators use a numerical Jacobian of the complete
  expression, including changes in both source and matrix.
- There is no native function-backed interface for arbitrary whole-profile
  transformations yet. Matrix operators are linear in the incoming values, even
  when their matrices depend on coordinates or fitted parameters.
- For interoperability, a general grid-preserving profile transform can be adapted from an Astropy
  model with `evaluate(x, flux, *parameters)`, then applied with `|`. It receives
  the whole upstream profile. Such a transform need not be linear; its parameter
  derivatives alone do not describe its response to the input flux. The complete
  expression therefore uses a numerical Jacobian. Custom transforms require a
  reconstruction rule to save a session.

Processing an upstream profile differs from multiplying it by a function of
coordinates. Neither operation requires a convolution interpretation.
It does not add OGIP response handling, bin integration of the source model or
a general custom-transform serialization protocol.

## Responsibilities

- `InstrumentResponse` builds, loads, crops and saves a numerical response. It is
  directly callable on a spectrum or on Jacobian columns, without a model wrapper.
- `LinearOperatorModel` exposes a matrix operation to native expressions and the Astropy adapter.
- `SpectralResponse` is the wavelength-aware adapter: it adds the native response
  grid, sampling and archive recipe. Its `response_matrix`/`matrix` and
  `wavelength_grid`/`x` names refer to the same underlying objects.

```python
response = InstrumentResponse.from_fixed_fwhm(wave, fwhm=2.5)
blurred = response(intrinsic_flux)

rsp = SpectralResponse(instrument=response, wave=wave)
model = source | rsp
```

The built-in SciPy and Sherpa fitters prepare fixed generic operators and spectral
responses outside repeated evaluations. The same numerical operation is reused
by the Astropy adapter; analytic Jacobians are transformed in one matrix operation.
Custom and parameter-dependent operators retain their existing evaluation hooks.

## Coordinates, masks and boundaries

A spectral response evaluates the source on its native grid, applies the matrix,
then samples the result onto requested coordinates. Flexible sampling is linear
point interpolation, with zero outside the native range; it is not a flux-conserving
rebinning operation. `flexible=False` rejects a different evaluation grid.

Select or mask observed samples after convolution. Cropping away source columns
can remove flux that should contribute to retained output samples. In particular,
`crop()` is a matrix-domain restriction, not a general observation-mask operation.
A larger native model grid can retain contributions around a fitting window.

For a redshifted source expressed in rest-frame coordinates, `z` maps the input
grid to observer wavelengths when selecting the archived instrumental matrix.

## Gaussian redistribution and bin widths

Gaussian builders now integrate the LSF into destination bins. Bin edges are
midpoints between native centers, with half-step extensions at the endpoints.
For input bin j and output bin i:

```text
P[i,j] = integral of the Gaussian centered at input j over output bin i
matrix[i,j] = P[i,j] * input_width[j] / output_width[i]
```

The Gaussian width is defined at the input wavelength. This applies equally to
constant and variable resolving power and nonuniform native grids. The weighted
column sum equals input bin width times the retained probability; flux outside
the supplied response band is not moved back inside. Gaussian construction visits
only bins within eight sigma (omitted probability below 1.3e-15).

Source models are sampled at native centers, so the input integral uses a midpoint
approximation. Narrow features require sufficient native sampling, verified by
convergence; these routines are not exact integrators of arbitrary model functions.
Flexible evaluation remains point interpolation, not flux-conserving rebinning of
arbitrary observed bins. Bin integration and observation masking are distinct.

## Fitting windows and migration

With `SpectralResponse(instrument=..., wave=...)`, the default flexible path keeps
the available neighboring source bins needed by the requested output window and
its interpolation brackets. It folds on that native grid before sampling the data.
The matrix and fixed sampling weights are prepared once per fit. Each evaluation
uses a sparse multiplication for the spectrum or all Jacobian columns, followed
by the prepared point interpolation. Precomposing these operations has not been
promoted: tiny rounding differences increased iterations on a complex reference fit.

The default `renormalize` is now False in `crop()` and `SpectralResponse`.
True retains the previous cropped row-normalized behavior for compatibility; it
is not an alternative physical redistribution law. `flexible=False` keeps the
strict supplied grid and cannot retain additional source bins automatically.

No source grid is invented beyond the response supplied by the user or archive.
At a genuine response boundary, missing contributions remain missing. Supply a
response covering a wider native domain when the fitting window requires it.

Existing archived/direct matrices are used as stored. Gaussian builders do not
silently convert old files into the new convention: rebuild those responses from
instrument calibration inputs when the new Gaussian law is needed. Old serialized
archive recipes lacking the new defaults retain their historical normalization
setting during reconstruction. New recipes preserve the requested window, so
repeated loading does not expand the native grid further.

Cropped InstrumentResponse objects retain the original bin widths even across
removed intervals; FITS round-trips preserve them in a BINWIDTH extension.

## Storage

`InstrumentResponse.save()` stores the grid and matrix. Compressed output uses
lossless GZIP compression; floating values are not quantized. This FITS format
still materializes the matrix densely during storage/loading.

Fixed `LinearOperatorModel` and `SpectralResponse` objects share sparse storage:

```python
operator.save('response.fits')
operator = LinearOperatorModel.from_fits('response.fits')
model.save('fit.fits', operator_file='response.fits')
```

The generic operator accepts `matrix`, input grid `x`, and output grid `xout`.
Rectangular matrices and grid units are preserved; coefficients must be
finite, real and dimensionless. Output interpolation samples the resulting
profile; it does not integrate it over data bins.

Operator files use Prism's versioned `PRISMLIN` FITS format, with CSR arrays,
grids, units, evaluation options and FITS checksums. This is not an OGIP RMF.
A session stores relative resource paths and content fingerprints. By default,
direct matrices use files named from the session root and a content fingerprint;
identical resources can be reused. Move the session and resources together,
preserving their relative layout. `operator_file=` selects one resource path
(relative to the session directory) and requires exactly one fixed operator.

Registered responses normally retain their reconstruction recipe instead of a
matrix file. Loading verifies the reconstructed matrix fingerprint and rejects
changed calibration. Use `operator_file=` to save an independent snapshot.
Session overwrite never replaces a different existing resource: choose a new
path, or use the default content-based name. Writes use temporary files and
replace the destination only after successful completion.

## Astropy compatibility

Native `Model.from_astropy(source) | operator` needs no Astropy patch.
For an Astropy-root expression, explicitly call
`prism.modeling.interop.astropy.install_pipe_patch()` before evaluation. The
optional extension delegates to the shared numerical operation and does not
define response physics. It is no longer installed when Prism is imported.
Native Astropy pipe nodes can still require numerical Jacobians; use
`has_native_pipe(model)` to detect them.

## Saving and loading operators

A supported response recipe or fixed operator resource is required for model
round-trips. Standalone YAML can embed fixed matrices; FITS sessions use external
resources. Custom matrices are supported through `LinearOperatorModel`, but custom
operator subclasses and parameter-dependent actions require an explicit
reconstruction rule and cannot be silently frozen. Missing resources, checksum
failures and changed fingerprints raise errors rather than substituting an
unconvolved source. Choose `model_def=False` for parameter-only exports.

Callable parameter links still require explicit restoration in analysis code.
The new model-tree/resource schema does not promise compatibility with earlier
development files; regenerate saved products when migrating.
