# prism.modeling

Prism provides composable spectral models, instrumental operators, fitting,
line measurements and model I/O. Analysis decisions remain in user notebooks.

## Native spectral composition

```python
from prism.modeling import Model
from prism.modeling.fitting import ScipyTRF

source = Model.from_astropy(continuum + narrow + broad)
model = source | response
best = ScipyTRF()(model, x, y, yerr=yerr, inplace=False)
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

`prism.modeling.psf` separates independent band fits from smooth chromatic trends.
It supports native Astropy `Gaussian2D`/`Moffat2D`, and Prism `MoffatPSF` (fwhm,
beta) / `EllipticalMoffatPSF` (fwhm_x, fwhm_y, beta, theta). All are ordinary
Astropy spatial models. `amplitude` is peak brightness, not total flux. A separate
`Const2D` or `Planar2D` models the background.

See **examples/Chromatic PSF.ipynb** for a complete executable field-star example,
including its neutral reference cube `examples/data/fieldstar.fits`.

```python
from astropy.modeling.models import Planar2D
from prism.modeling.models import MoffatPSF
from prism.modeling.psf import fitbands, fittrends, ChromaticPSF

model = MoffatPSF(amplitude=100, x_0=12, y_0=13, fwhm=3, beta=2.5)
bands = fitbands(cube, model, bands=20, background=Planar2D(), factor=5)
psf = fittrends(bands, {'fwhm': 5, 'beta': 5, 'x_0': 5, 'y_0': 5},
                basis='bspline', display=True, save='trends.pdf')
psf.write('psf.fits')
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
from prism.modeling.psf import fitimage

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

`prism.modeling.extraction` provides `aperture` and `optimal`, also re-exported
from `prism.modeling.psf`. Both reuse the stellar chromatic PSF at an explicitly
supplied fixed target center `(y, x)` in local pixels. Shape is not refitted.
The PSF calibration and target must share spatial pixel scale and orientation.

```python
from astropy.modeling.models import Const2D
from prism.modeling.psf import aperture, optimal

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
from prism.modeling.extraction import write, read
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
