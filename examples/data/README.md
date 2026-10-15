# Demonstration data

`fieldstar.fits` contains an isolated point source (3681 × 26 × 26 samples).
`pointsource.fits` contains a compact source and surrounding emission (3681 × 21 × 21 samples).
Both include `DATA` and `VAR` extensions; `VAR` contains variance, not standard deviation.
Fluxes and variances retain their numerical values. Flux density is integrated per spatial pixel, with units recorded in `BUNIT`.

These demonstration copies omit identifying observation metadata and celestial coordinates. Spatial axes use local offsets in arcseconds, with a 0.2 arcsec pixel scale recorded as `PIXSCALE` in the primary header. The original air-wavelength sampling and flux calibration units are retained. They have no absolute sky registration. Removing identifiers does not guarantee that real spectra cannot be recognized.

`psf.fits` is a compact normalized chromatic PSF built from `fieldstar.fits` with the settings in [Chromatic PSF](../Chromatic%20PSF.ipynb). It allows [Aperture Correction](../Aperture%20Correction.ipynb) to run without first fitting the star. To use a revised calibration, rerun the star notebook and set `psfpath` to its output, `results/psf/psf.fits`.

These files illustrate the methods; they do not specify a source identity, distance or luminosity. Other files in this directory belong to the remaining examples.
