# Notebook guide

Run notebooks from this directory, in cell order, using a fresh kernel. Examples
use the bundled data and write products to `results/`. TRF is the default;
Sherpa is an alternative only when installed separately.

| Start here | What it shows |
| --- | --- |
| [Custom Components](Custom%20Components.ipynb) | Define a function, optional analytic derivatives and parameter bounds; fit, inspect and reload a result. |
| [Line Components](Line%20Components.ipynb) | Ready-made Gaussian/Lorentzian/Voigt sets; custom profiles, transition weights, shared parameters and line measurements. |
| [Model Single Spectrum](Model%20Single%20Spectrum.ipynb) | Correct an observed spectrum, build an AGN model, apply a response and propagate uncertainty. Includes 200 bootstrap refits. |
| [Operators and Responses](Operators%20and%20Responses.ipynb) | Define a custom linear operator and separate instrumental response from intrinsic model. |
| [Joint Fitting](Joint%20Fitting.ipynb) | Fit observations with different responses and selected parameter links; contrast joint and independent fitting. |
| [Model Data Cube](Model%20Data%20Cube.ipynb) | Fit spaxels independently and recover spatial parameter/error maps. |
| [Multi-Spectrum Line Extraction](Multi-Spectrum%20Line%20Extraction.ipynb) | Measure lines and equivalent widths across an independently fitted series. |
| [Blended Line Analysis](Blended%20Line%20Analysis.ipynb) | Measure intrinsic blends with the covariance from `results/agnfit.fits`; run Model Single Spectrum first. |
| [Empirical Subtraction](Empirical%20Subtraction.ipynb) | Reshape a reference spectrum with splines, reject outliers and inspect residuals. |
| [Chromatic PSF](Chromatic%20PSF.ipynb) | Fit an isolated star and describe its wavelength-dependent PSF. |
| [Aperture Correction](Aperture%20Correction.ipynb) | Compare aperture and PSF-based extraction; the supplied calibration makes this independent of the PSF notebook. |

Tables and plots show parameter estimates, convergence and residuals. The
notebooks describe each uncertainty approximation; optimizer success alone does
not establish that a scientific model adequately describes the data.
