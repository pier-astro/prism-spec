# prism.modeling

Prism provides general composable models, parameters, constraints, fitting and
model I/O. Spectral profiles, instrumental responses and line measurements build
on this shared infrastructure. Analysis decisions remain in user notebooks.

## Native model composition

```python
from prism.modeling import Model
from prism.modeling.fitting import TRF

source = Model.from_astropy(continuum + narrow + broad)
model = source | response
result = TRF()(model, x, y, yerr=yerr)
best = result.model
```

`Model.from_astropy` adapts individual components or an existing arithmetic
expression, retaining their parameter identities. Copy the complete model for
an independent parameter graph. Reusing a parameterized component twice in one
expression is rejected; parameter-free operators can be reused.

Addition, subtraction, multiplication, division and powers retain their
parentheses. `source | operator` is an explicit application node; order matters.
Responses can have distinct input/output grids, including rectangular matrices.
See `operators/README.md` for fixed, parameterized and general transforms.

The native expression shares numerical kernels, parameter mapping, fitters,
component analysis and session I/O with the existing adapters. It does not
inherit Astropy's model class or use its compound-expression evaluation.
Parameter constraints are edited explicitly through `model.parameter.fixed`,
`.bounds`, `.tied` or supported Prism `.link` properties; the aggregate constraint
mappings are read-only. Native fitting currently requires numeric arrays in
consistent native units. Ordinary evaluation also accepts Quantity coordinates.

`params` provides the same live parameter objects by name, for both components
and complete models. Its keys follow `param_names`; `parameters` remains the
numeric vector. The mapping structure is read-only, while parameter fields are
editable:

```python
narrow.params['fwhm'].bounds = (50., 800.)
broad.params['fwhm'].link = 2 * narrow.params['fwhm']
```

Attributes and mapping entries refer to the same parameter, so edits through
either form immediately affect evaluation, links and fitting. A complete model
copy has an independent parameter graph.

`model.components` is a read-only mapping of the physical components, without
distributing multiplication or applying a response to each term:

```python
model.components['narrow'].params['fwhm'].value = 300.
intrinsic_narrow = model.components['narrow'](x)
```

Source components are native Model views: their parameters remain the same
objects and linked values are resolved in the complete parent model. Editing a
view edits that parent; copy the complete model for an independent graph.
Unique component names are keys. Unnamed components or components with duplicate
names use their positions as integer keys, visible through `components.keys()`.
Explicit pipe operators remain public operator objects, with their own `params`
and action on flux arrays. They are not replaced with identity placeholders.
For algebraically expanded additive terms, use the separate `get_components`
decomposition interface.

## Supported interoperability

Astropy components remain available through the explicit adapter. SciPy, Sherpa
and separable Prism fitters support native expressions. Prism's Astropy fitter
wrappers continue to accept Astropy model roots; they reject native `Model`
roots explicitly. General custom Python models can be evaluated, but saving
requires a registered reconstruction rule; arbitrary Python code is not archived.

Astropy extensions live in `interop.astropy` and require explicit activation;
importing Prism does not modify Astropy display, I/O or compound evaluation.
Native expressions provide their own display, persistence and operator action.
This is supported interoperability, rather than complete Astropy transparency.

## Chromatic PSF fitting

`prism.psf` separates independent band fits from smooth chromatic trends.
It supports native Astropy `Gaussian2D`/`Moffat2D`, and Prism `MoffatPSF` (fwhm,
beta) / `EllipticalMoffatPSF` (fwhm_x, fwhm_y, beta, theta). All are ordinary
Astropy spatial models. `amplitude` is peak brightness, not total flux. A separate
`Const2D` or `Planar2D` models the background.

See **examples/Chromatic PSF.ipynb** for a complete executable field-star example,
including its neutral reference cube `examples/data/fieldstar.fits`.

```python
from astropy.modeling.models import Planar2D
from prism.modeling.models import MoffatPSF
from prism.psf import fitbands, fittrends, ChromaticPSF

model = MoffatPSF(amplitude=100, x_0=12, y_0=13, fwhm=3, beta=2.5)
bands = fitbands(cube, model, bands=20, background=Planar2D(), factor=5)
psf = fittrends(bands, {'fwhm': 5, 'beta': 5, 'x_0': 5, 'y_0': 5},
                basis='bspline', display=True, save='trends.pdf')
psf.save('psf.fits')
loaded = ChromaticPSF.from_fits('psf.fits')
image = loaded.get_image(6500, (26, 26), center=(13, 12))
widths = loaded.fwhm([5000, 6500, 8000])  # pixel Quantity, principal (x, y)
```

- `fitbands`: band count or explicit non-overlapping wavelength pairs; uses the
  existing valid-data mask (True = valid), channel means, and independent errors.
  Astropy initial values, fixed flags and bounds are respected. Results are a
  table with local covariances, background parameters and fit status. No outlier
  rejection is applied. Near circularity, initialize ellipse fits consistently
  (a mean-image fit is often enough); widths can swap with a 90-degree angle
  change, and theta can remain weakly constrained.
- `fittrends`: **default basis='bspline'**, with explicit coefficient counts.
  Local degree is min(3, count-1); uniformly spaced interior knots are stored.
  Thus 5 coefficients match the parameter budget of polynomial degree 4. Up to
  4 coefficients give one global segment. Spline coefficient bounds enforce
  physical/user bounds over the base domain without clipping predictions;
  locality does not guarantee absence of all oscillations. The spline domain
  spans the calibrated channel limits, while band centers locate measurements.
  The trend continues smoothly to the edges, where it is less constrained.
  Previously saved models retain their explicit constant extension when loaded.
- `basis='polynomial'` retains the original global-polynomial option; dictionary
  values are then degrees, including 0 for a constant. Set this explicitly when
  reusing earlier code that supplied degrees. Old version-1 PSF FITS files load
  with their polynomial interpretation preserved. New files store basis and
  knots explicitly (format version 2).
- `display=True`, `save=None`: show diagnostics without saving. Both options are
  independent of numerical model persistence. The FITS model stores plain JSON
  metadata, coefficients/covariances, units, conventions and band diagnostics;
  neither cube nor live fitter is saved. Custom metadata must be JSON-compatible.
- `factor=1`: evaluate pixel centers; larger integers use midpoint quadrature,
  consistently during fit and reconstruction. Test convergence for your sampling
  and accuracy needs. Observed data are never resampled.
- `fwhm(wave)`: intrinsic principal-axis (x, y) widths in pixels, with shape
  wave.shape+(2,). Circular profiles repeat the width. Conversion to arcseconds
  is external; a scalar angular pixel scale applies directly to square pixels.
  The curve is derived from the model, not stored redundantly.

Coordinates are local zero-based pixels, widths are pixels and theta is radians.
`get_image` does not renormalize a finite field: normalization is analytic on the
infinite plane. The calibrated domain spans actual channels in successful retained
bands; evaluation outside it is rejected. `get_cube` optionally materializes a
NumPy cube for external FITS export with explicit wavelength/WCS metadata.

Evaluate reconstruction quality in flux and relevant aperture fractions. Very
small formal errors can yield large reduced chi-square even for useful profiles;
chi-square alone is not a reason to add complexity. Local covariance and trend
covariance omit resampling and full cross-parameter propagation; active coefficient
bounds leave covariance unavailable. Broad-band profiles remain monochromatic
approximations. Host–AGN decomposition is outside this workflow.


## Fit a single spatial image

`fitimage` is the spatial fit used by `fitbands`, exposed independently for
centering a source or fitting a profile in one image. Supply a Prism `Image`
with errors and an optional boolean selection (`True` selects pixels):

```python
from prism.psf import fitimage

img = cube.get_image(method='mean')
initial = fitimage(img, model, mask=selection, background=background, factor=5)
assert initial['success'], initial['message']
center = (float(initial['y_0']), float(initial['x_0']))
```

The result is an Astropy table row with parameters, `*_err`, `covariance`,
`chi2`, `dof`, `redchi2`, `npix`, `success` and `message`. Its parent table
(`initial.table`) carries units; `initial.meta['covnames']` gives covariance
ordering. Gaussian centers use `x_mean`/`y_mean` instead of `x_0`/`y_0`.
Coordinates are local zero-based array pixels. Inputs are not modified.

Default `init=None` retains all supplied model/background initial values.
Optional `init='peak'` seeds only free centroid coordinates and amplitude from
the brightest valid background-subtracted selected pixel, clipped to their
bounds. It leaves fixed parameters, widths, shape and background unchanged.
Use it as a simple starting guess, not a measured centroid. Only positive
finite-error pixels enter the fit; insufficient support produces a failed row.
The covariance uses absolute errors without reduced-chi-square rescaling;
active bounds or rank deficiency leave covariance unavailable. `fitbands`
uses this same fitter after independently averaging each wavelength band.
Band images are created by `Cube.cutout_slices` and `Cube.get_image`, retaining
the cube dtype (float32 or float64). Fit-specific validity masks affect only
the band cutout; the original cube remains unchanged. Spectral support is
checked per pixel before fitting.


## Spectral extraction with a calibrated PSF

`prism.psf.extraction` provides `aperture` and `optimal`, also re-exported
from `prism.psf`. Both reuse the stellar chromatic PSF at an explicitly
supplied fixed target center `(y, x)` in local pixels. Shape is not refitted.
The PSF calibration and target must share spatial pixel scale and orientation.

```python
from astropy.modeling.models import Const2D
from prism.psf import aperture, optimal

spectrum, diagnostics = aperture(cube, psf, mask, center=(10.2, 10.0))
fitted, fitinfo = optimal(cube, psf, fitmask, center=(10.2, 10.0),
                          background=Const2D())
diagnostics.write('aperture.ecsv', overwrite=True)
fitinfo.write('optimal.ecsv', overwrite=True)
```

- **aperture** sums a fixed spatial selection (fractional weights allowed),
  then divides flux and statistical error by the identically weighted PSF
  fraction. It preserves a composite aperture spectrum, rescaling any host
  contribution too. Prefer it for unknown compact, asymmetric or chromatically
  structured extended emission, when imposing a flat host model is undesirable.
  It is not an isolated AGN or total-host flux estimator. Any invalid selected
  pixel invalidates that channel rather than changing the aperture silently.
- **optimal** solves the variance-weighted linear model `D = F*P + B` using
  whole-pixel measurements and a known normalized PSF. None means zero background;
  a free/fixed Const2D estimates/imposes a local constant. It already returns
  the integrated PSF amplitude: no additional aperture correction is needed.
  Covariance includes the fitted flux–background correlation, with chi2, dof,
  redchi2, success and messages available per channel. Invalid pixels are excluded
  and their changing support is recorded.

Why **optimal**: under a correct fixed profile/background and known independent
variances, this is the best linear unbiased estimator, minimizing amplitude
variance within that class. It is also maximum likelihood for Gaussian noise.
This does not guarantee physical AGN–host separation. Unknown compact/structured
host light can project onto the PSF, bias its amplitude and be removed in a
PSF subtraction. Prefer `aperture` in that situation. A tiny fitting region can
make the PSF indistinguishable from the fitted constant. Source-dominated noise
does not necessarily benefit from optimal weighting.

Both methods return `Spectrum` and a unit-bearing Astropy diagnostics table.
They evaluate only the required spatial bounding box, never a full PSF cube.
They do not estimate the target center or iterate a deblender. Errors are
conditional on the fixed PSF/center and independent voxel uncertainties; PSF,
centering, calibration and model-mismatch uncertainties and resampling covariance
are not propagated. The docstrings document the exact estimators and failure
policies. Treat these errors and unknown host contamination separately.


### FITS extraction products

```python
from prism.psf.extraction import save, read
from prism.data import Spectrum

write(diagnostics, 'aperture-spectrum.fits', overwrite=True)
spectrum, diagnostics = read('aperture-spectrum.fits')
# Spectral columns alone:
spectrum = Spectrum.from_fits('aperture-spectrum.fits', ext='SPECTRUM')
```

The same writer accepts aperture, optimal and constant-correction tables.
`SPECTRUM` contains only `WAVELENGTH`, `FLUX`, `ERROR`, with explicit units;
`ERROR` is a standard deviation. Numerical extra columns, including covariance,
go in `DIAGNOSTICS`. `APERTURE` preserves boolean masks or fractional weights.
Inspection-only `success` and `message` are not exported. Failed samples remain
NaN. Short PRIMARY keywords record method, PSF domain/unit, center, correction
and provenance; `WAVETYPE` records air/vacuum/unknown wavelength convention.
The compact `METADATA` JSON extension preserves the complete metadata and
exported column order. Unknown metadata is preserved or rejected explicitly,
never silently discarded. FITS checksums are written; existing files are
protected unless `overwrite=True`. This practical FITS layout does not claim
formal IVOA Spectrum Data Model or OGIP compliance.

Extraction FITS products preserve the supplied PSF centroid as RA/Dec in degrees
when the cube has a celestial WCS. Coordinates are stored in both PRIMARY and
SPECTRUM and restored by both extraction `read` and `Spectrum.from_fits`.
Without a celestial WCS, coordinates remain unknown. To override the saved
position, set `diagnostics.meta['ra']` and `diagnostics.meta['dec']` before writing.
Generic `Cube.extract_spectrum` instead uses the aperture-weighted geometric
barycenter; an explicitly supplied PSF center is not replaced by that position.


## Astropy interoperability

`Model.from_astropy(model)` imports supported models into native Prism
composition. Generic extensions of Astropy live separately in
`prism.modeling.interop.astropy`: `tie`, standalone display functions and
explicit `enable_model_display()` / `disable_model_display()` activation.
Importing `prism.modeling.display` no longer changes Astropy classes. Native
Prism display is implemented directly and does not need activation.


## Native function components

`Component(function, parameters, derivative=None, name=None)` uses scalar native
parameters and the same expression/fitting machinery as adapted Astropy models.
The parameter mapping must list every positional function argument after `x`,
in function-signature order. The optional derivative returns rows in that order,
with shape `(n_parameters, n_coordinates)`. Functions return numeric arrays with
the same shape as `x`; explicitly convert units before evaluation and fitting.
Use pure functions of the supplied inputs; changing captured state during a fit
invalidates the prepared problem. Bounds, fixed parameters and declarative links
are available on `component.parameter`.

See `examples/Custom Components.ipynb` for a function-backed continuum, analytic
parameter derivatives, fitting and saved-result reuse. Ready-made line sets,
custom profiles, weights and parameter scopes are shown in
`examples/Line Components.ipynb`.

Give custom components unique names before saving. Restore with
`load_model(path, components={name: original_component})`; supply every custom
definition, including its derivative and any external numerical resources.
Only state is stored, not Python code or dynamic imports. Supplied definitions
are not mutated. Existing Astropy models continue through `Model.from_astropy`.
Their parameters retain Astropy arithmetic semantics: use a parameter reference
for a direct native link; arithmetic on ordinary Astropy parameters computes a
number immediately. Prism-native parameters build declarative expressions.


## Native line groups

`LineSet(profile, lines, shared, parameters, scale="amplitude",
position="center", derivative=None, name=None)` applies a numerical profile to
weighted transitions without generating Astropy descriptors or model classes.

```python
from prism.modeling import LineSet
from prism.modeling.lines.profiles import gaussian

narrow = LineSet(
    gaussian, linetable,
    shared={"sigma": 2.0},
    parameters={"Hb": {"amplitude": 2.0}, "OIII": {"amplitude": 3.0}},
    scale="amplitude", position="center", name="narrow",
)
narrow.sigma.bounds = (0.3, 8.0)
narrow.amp_oiii.link = 2.98 * narrow.amp_hb
```

This Gaussian shares sigma **in the line table's coordinate unit**. It does not
implicitly impose equal velocity widths or transform rest positions. The profile
function owns those transformations; see `examples/Line Components.ipynb` for an
explicit logarithmic velocity convention and an exponential profile scaled by
integrated flux. Repeated table names identify one template even if weights are
omitted. Positions require explicit spectral units; evaluation uses numeric
coordinates in that unit. Neither global line-list filters nor hidden domain
conversions are applied by the native constructor.

`shared` and each template's independent `parameters` must partition all profile
arguments except the fixed tabulated position. The selected `scale` must be
independent per template and is multiplied by each transition's weight exactly
once. No weight normalization or peak/flux conversion is inferred. Independent
attributes are `argument_template` (`amp_template` for amplitude), using existing
cleaned lowercase template names; ambiguous names are rejected. Shared parameters
retain their original argument names. Parameter maps and membership are prepared
once; parameter values remain current during fitting.

An optional derivative returns rows in profile argument order after x, including
the position row. Position is fixed by the table: that row is omitted from the
fit vector. Derivatives of shared parameters accumulate across every transition;
the scale row receives the weight chain factor. Omit the derivative for numerical
differences through the complete model. Fixed response operators work through
ordinary `source | response` composition.

Named native groups save membership, weights, scopes and parameter state; supply
profile definitions explicitly with `load_model(path, components={name: group})`.
The saved setup, rather than the supplied definition's table, is restored. Code,
external response resources and callable-specific calibration remain explicit.

Line components use `select_line`/`measure_line`, equivalent widths and covariance
sampling. The old Astropy line subclasses and their analytic-property APIs have
been removed. AGN factories now return native `LineSet` instances; external
Astropy models remain supported through `Model.from_astropy`.

### Configurable built-in line profiles

`GaussianProfile`, `LorentzianProfile` and `VoigtProfile` are callable scientific
profiles in `lines`. They can be supplied to `Component` or `LineSet`,
with `derivative=profile.derivative` to use their analytic Jacobians. Their
`position` argument is a rest coordinate; offset and intrinsic FWHM are km/s.
Specify `position="position"` in a line group.

`GaussianProfile(instfwhm=..., position_unit=u.AA)` prepares a scalar Gaussian
instrumental FWHM or a numeric two-column observed-position/FWHM table. Widths
are km/s; table positions and numeric x use `position_unit`. A scalar velocity
Quantity is also accepted. Curve interpolation and its derivative are prepared
once; its value is evaluated at each current observed center, including redshift
and offset. The analytic Jacobian includes the curve slope. Cubic extrapolation
matches the historical convention; an invalid negative interpolated width is
rejected. Explicitly convert a line table if its unit differs from the profile.

Gaussian widths combine in quadrature; Gaussian broadening of a Lorentzian gives
a Voigt; the Gaussian part of a Voigt broadens independently of its Lorentzian
part. This local-width prescription retains peak/redshift scaling and is not a
general flux-conserving response operator. Avoid applying the same instrumental
broadening twice when also composing with a response matrix.

Saved built-in profile recipes include their calibration and analytic-derivative
choice and load without custom definitions. Custom profiles or custom derivative
functions still require explicitly supplied definitions. No executable Python
code or dynamic import is stored.

## General models and coordinates

`Component` accepts an ordinary function and named parameters; neither the
function nor the fitter requires a spectral interpretation. For example:

```python
from prism.modeling import Component
from prism.modeling.fitting import TRF

def trend(x, slope, intercept):
    return slope * x + intercept

model = Component(trend, parameters={"slope": 1., "intercept": 0.})
result = TRF()(model, x, y, yerr=yerr)
best = result.model
```

The same parameter graph, algebra, bounds and links apply to generic data.
`LineSet` adds transition positions, relative weights and shared parameters;
these conventions belong to the line specialization, not to the fitting engine.

General 2D fitting is a primary capability, currently supported through the
Astropy adapters. The native extension will accept two independent coordinates
`(x, y)` and values `z`, reusing the common fitting infrastructure. Its coordinate
contract must also distinguish source coordinates from operator inputs. This
extension is tracked in `TODO.md`; native `Component` currently takes one
independent coordinate.

## Line domain

Native line components, profiles and tables live in `lines`; post-fit selection
and measurements live in `lines.measurements`. The historical Astropy line constructors have been removed. See `lines/README.md` for the selected-transition contract and
import migration.

## Consistent plotting and saving

Use `Spectrum.plot()` and `Spectrum.save()/load()` for prepared active spectra.
The spectrum snapshot retains arrays, units, errors and JSON metadata, not its
previous cutout/reset history. External FITS inputs still use `from_fits`.
`Cube`, `Image`, `ChromaticPSF` and instrument responses expose `save()`;
`prism.psf.extraction.save` writes extraction products. Astropy tables retain
Astropy's own `write()` method.

The maintained notebooks show the 1.0 names without compatibility aliases.
