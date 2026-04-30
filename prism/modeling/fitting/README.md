# prism.modeling.fitting

The `prism.modeling.fitting` module implements a unified and extended fitting infrastructure for Astropy models, providing robust, high-performance optimization options.

## Features
- **MultiFitMixin**: Automatically handles broadcasting 1-D models over 2-D/3-D spatial maps via the `.multifit()` method, utilizing multiprocessing and masking.
- **Bootstrap Uncertainties**: Robust parameter uncertainties using empirical bootstrap resampling (`.bootstrap_uncertainties()`).
- **Sherpa Fitters**: Wrappers around CIAO's `sherpa` optimizers allowing chi-square optimization.
  - `SherpaLM`: Levenberg-Marquardt (fast, local gradient-based).
  - `SherpaSimplex`: Nelder-Mead (robust, gradient-free).
  - `SherpaMonCar`: Monte Carlo global optimization.

## Usage
Simply instantiate a fitter, pass it a parameterized model and the observed data, and receive the optimized model back.

```python
from prism.modeling.fitting.sherpafit import SherpaLM
from astropy.modeling.models import Gaussian1D

fitter = SherpaLM()
model = Gaussian1D(amplitude=1, mean=5000, stddev=10)

best_fit = fitter(model, x, y, yerr=yerr)
```

---

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