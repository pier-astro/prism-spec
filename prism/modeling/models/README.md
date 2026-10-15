# Models

`prism.modeling.models` contains Prism spectral model classes built on top of Astropy's modeling framework.

## Core Principle

Prism models are Astropy-style models first.

That means:

- Prism line, continuum, and empirical models subclass Astropy modeling base classes
- parameters, units, composition, and fitter interaction stay as close as possible to Astropy behavior
- native Astropy models can still be mixed with Prism models in compound expressions

## What Prism Adds

Prism adds domain-specific behavior around:

- spectral line families and tied templates
- component extraction and display helpers
- serialization metadata for fitted models
- uncertainty metadata attached to fitted model instances

Those additions are intended to stay explicit and readable rather than replacing Astropy with a hidden parallel model system.

## Display And Transparency

Prism enables enhanced model display at import time so that models print cleanly in terminals and notebooks.

This is a convenience patch, not a structural compatibility patch. It can be disabled when needed, but it is enabled by default because inspecting fitted models is a central Prism workflow.

The underlying model objects remain Astropy-compatible models.

## Spatial PSF models

`MoffatPSF` and `EllipticalMoffatPSF` in `spatial.py` are ordinary Astropy
`Fittable2DModel` subclasses. They expose FWHM directly and use `beta` for the
wing exponent. `amplitude` is peak brightness. The elliptical model has
`fwhm_x`, `fwhm_y` along its rotated principal axes and `theta` in radians.
It remains centrosymmetric; axis labels do not impose major/minor ordering.

Native Astropy Gaussian2D remains supported (its `x_fwhm` and `y_fwhm`
properties already expose the derived FWHM). The chromatic workflow and
compact model persistence are documented under `prism.modeling.psf`.
