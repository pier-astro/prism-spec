# prism.modeling

`prism.modeling` is the Astropy-facing layer of Prism: models, operators, fitting helpers, line analysis, display, and model I/O live here.

## Compatibility Policy

Prism follows a hybrid compatibility strategy:

- prefer Prism-owned subclasses and wrappers where the workflow can stay clean without mutating Astropy globally
- keep a very small number of Astropy monkey-patches only where native Astropy currently blocks transparent interoperability
- keep the resulting objects Astropy-compatible whenever possible

In practice this means Prism aims to let users combine native Astropy models and Prism models naturally, while avoiding a large parallel abstraction layer.

## Import-Time Behavior

Two convenience hooks are intentionally enabled at import time:

- `enable_model_display()`
- `install_model_io_methods()`

These are kept because Prism users typically want richer display and direct FITS save/load methods immediately.

They are convenience patches, not compatibility-critical patches.

Astropy transparency is preserved in the sense that:

- Prism models remain ordinary Astropy-style models
- native Astropy models can still be passed through Prism utilities
- free-function APIs still exist for display and I/O when explicit control is preferred

## Remaining Compatibility-Critical Patches

The only patches Prism currently treats as structurally necessary are the ones that preserve smooth mixed-model behavior for:

- operator-aware `CompoundModel` pipe expressions
- tied analytic Jacobians in those enhanced fitting paths

Those behaviors are documented in the module READMEs under `operators/` and `fitting/`.

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
