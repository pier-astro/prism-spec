# prism.data

The `prism.data` module provides the core data structures for manipulating spectral and spatial-spectral astronomical datasets.

## Core Classes
- **Spectrum**: A 1-D spectral container supporting masking, rebinning, and various coordinate transformations (wavelength, frequency, velocity, energy).
- **Image**: A 2-D spatial image container with WCS-aware operations, region measurements, and spatial binning (VorBin) support.
- **Cube**: A 3-D integral-field data cube container combining spatial and spectral information.

## Shared Functionality
All data containers provide robust error handling (variance/standard deviation) and seamlessly interface with Astropy's `NDData` objects and FITS files via `to_nddata()`, `from_nddata()`, and `from_fits()`. They also support algebraic operations with proper error propagation in quadrature.

## WCS and Physical Scales
For `Image` and `Cube`, spatial conversions are readily accessible:
- `pixel_scales()`: Get the angular size of a pixel.
- `spatial_scale()`: Convert angular size to physical dimensions (e.g. kpc) using cosmology and redshift.
- `pixel_area()`: Calculate the physical or angular area of a single pixel/spaxel.

## Examples
```python
from prism.data import Cube

# Load from FITS
cube = Cube.from_fits("data.fits", ext_values='DATA', ext_err='STAT', redshift=0.04)

# Rebin spatial pixels via binmap
binned_cube = cube.apply_binmap(method='mean')

# Extract spectrum
spec = cube.extract_spectrum(mask=my_aperture)
spec.plot()
```

## Editing a spectrum

`Spectrum.x`, `y`, `xerr` and `yerr` are the current working arrays. Direct
assignment, slice edits and in-place arithmetic survive copies, further cutouts
and rebinning. Arrays must remain shape-compatible; rebin() replaces a grid.
Spectrum has no persistent mask, reset() or hidden original arrays.
cutout(mask=...) filters coordinates, values and errors together, and works
also after rebinning. Keep original = spec.copy() if an earlier state is needed.
Incompatible array shapes are rejected before copying, cutting, rebinning or saving.
Image and Cube retain their masks and spatial geometry; their behavior is unchanged.

For wavelength F_lambda, the caller may explicitly transform to rest coordinates:

```python
spec.x /= 1 + spec.redshift
spec.y *= 1 + spec.redshift
spec.yerr *= 1 + spec.redshift
spec.frame = "rest"
```

Scale xerr too when present, and use the appropriate Jacobian for other spectral
density conventions. frame is None (unknown), "observed" or "rest" and is
preserved in copies and FITS snapshots. It is descriptive metadata, stored in
meta["frame"]; setting it never transforms arrays or configures a response.
Pass redshift explicitly when mapping rest wavelengths onto an observed response.
Sampling estimates velscale/fwhm follow the current x array; they are not an
instrumental resolution calibration. Correct Galactic extinction at observed
wavelengths before the rest-frame transformation.
