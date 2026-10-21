# Models

This collection contains specialized continuum, empirical and spatial models,
and AGN factories. The generic component architecture, parameter graph and fitting
engines are native Prism infrastructure. External Astropy models enter it through
`Model.from_astropy`; optional display extensions require explicit activation.

Line profiles, transition tables and native `LineSet` live in `modeling.lines`.
The `nlr`, `blr`, `fe`, `fe_uv` and `balmer` factories build native components from
those tables and profiles. Historical Astropy line subclasses are removed.

Existing continuum, empirical and spatial Astropy models remain interoperable;
their own migration is independent of removing the duplicate line infrastructure.

## Spatial PSF models

`MoffatPSF` and `EllipticalMoffatPSF` in `spatial.py` are ordinary Astropy
`Fittable2DModel` subclasses. They expose FWHM directly and use `beta` for the
wing exponent. `amplitude` is peak brightness. The elliptical model has
`fwhm_x`, `fwhm_y` along its rotated principal axes and `theta` in radians.
It remains centrosymmetric; axis labels do not impose major/minor ordering.

Native Astropy Gaussian2D remains supported (its `x_fwhm` and `y_fwhm`
properties already expose the derived FWHM). The chromatic workflow and
compact model persistence are documented under `prism.psf`.
