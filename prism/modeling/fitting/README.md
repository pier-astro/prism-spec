# Fitting

`prism.modeling.fitting` provides Prism fitters, multi-spectrum fitting, and uncertainty helpers while staying as close as possible to Astropy's fitting interface.

## What Lives Here

- Prism-native fitters based on SciPy and Sherpa
- batch fitting through `multifit()`
- outlier-removal wrappers
- covariance and resampling uncertainty helpers
- compatibility code that lets Prism-enhanced models and Astropy-style fitting stay interoperable

## Astropy Transparency

The design goal is that Prism fitters should accept:

- native Astropy models
- Prism models
- native Astropy compound models
- Prism-enhanced operator compound models

and return fitted models that are still usable with Astropy tools.

Prism attaches result metadata such as covariance and parameter uncertainties to fitted instances, but it does not try to hide or replace Astropy's model system.

## Monkey-Patch Notes

This package has historically patched Astropy fitters broadly. The current direction is to reduce that and move standard enhancements into Prism-owned wrappers.

The one fitting-related patch Prism still treats as hard to avoid is the tied-analytic-Jacobian compatibility path used together with operator-enhanced compound models. That patch exists because native Astropy does not currently provide the public hook Prism needs to preserve the clean operator workflow with analytic derivatives.

So the fitting policy is:

- broad fitter mutation should shrink over time
- native Astropy fitting should remain available
- only the narrow compatibility fixes needed for Astropy-transparent operator fitting should remain long-term unless Astropy grows the needed hooks upstream