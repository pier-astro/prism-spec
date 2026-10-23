![Prism header](docs/logo/header.png)

# prism

Getting a useful fit takes more than picking an optimizer. You need a model
that describes your problem, parameters that behave sensibly, and data and
uncertainties that are ready for the job. Then come the diagnostics, derived
measurements, and bookkeeping needed to keep the analysis reproducible. That is
a lot of machinery to rebuild for every project.

**Prism gives you reusable pieces for that work.** It is a Python toolbox for
component-based modeling and fitting: bring your own functions, combine models,
connect their parameters, and carry the fit into further analysis. Bounds,
fixed parameters, and parameter links let you express the relationships your
model needs. You choose the tools and keep the analysis in your hands.

The fitting tools are general; much of the surrounding toolbox grew out of
astronomical spectroscopy. Ready-made spectral profiles and tabulated line sets
help build complex emission and absorption models, including blended features
and multiple kinematic components. Instrumental responses connect intrinsic
models to observations. Measurement tools turn fitted components into fluxes,
widths, equivalent widths, and kinematic summaries, with covariance- or
bootstrap-based uncertainty propagation.

There is more around the model, too: containers for spectra, images, and
integral-field data cubes; empirical spectral subtraction; wavelength-dependent
point-spread function (PSF) calibration; and aperture and PSF-based extraction.
Numerical efficiency is a design priority, from model evaluation to parallel
fitting of many spectra. Batch tools help handle larger datasets and recover
parameter maps without writing a separate fitting loop for every spatial pixel.

## Installation

Prism requires **Python 3.11 or newer**. To install from this repository:

```bash
git clone https://github.com/pier-astro/prism-spec.git
cd prism-spec
python -m pip install .
```

For an editable installation, use `python -m pip install -e .` instead.
While the distribution is called `prism-spec`, the Python import name is `prism` for simplicity.
Dependencies are installed automatically and listed in [pyproject.toml](pyproject.toml).

Prism is under active development, feel free to [report issues](https://github.com/pier-astro/prism-spec/issues) or
[submit pull requests](https://github.com/pier-astro/prism-spec/pulls).


About the minimization algorithms, the default SciPy trust-region fitter (`TRF`) works out of the box. Install
[Sherpa](https://sherpa.readthedocs.io/) separately to use its optional robust optimizers.
See the [fitting guide](prism/modeling/fitting/README.md)
for more details on supported workflows and uncertainty conventions.

## Highlights

### Build once, connect what matters

Start with an ordinary Python function. Give its parameters names, starting
values, and bounds, then fit it to your data:

```python
from prism.modeling import Component
from prism.modeling.fitting import TRF


def continuum(x, amplitude, slope):
    return amplitude + slope * (x - 4950.0) / 100.0


component = Component(
    function=continuum,
    parameters={"amplitude": 1.0, "slope": 0.0},
    name="continuum",
)
component.amplitude.bounds = (0.0, None)
result = TRF()(model=component, x=x, y=y, yerr=yerr)
```

Here `x`, `y`, and `yerr` are your coordinate, data, and uncertainty arrays.
Add or multiply components to build richer models. Fix parameters or link them
across components; use joint fitting to connect selected parameters across
datasets with different sampling and responses. Existing Astropy models
can join in through `Model.from_astropy()`.

### A line list becomes a model

Spectral modeling often means handling many lines at once. Some share a
velocity and width; others have relative strengths that should stay fixed.
Keeping those relationships consistent by hand gets tedious fast.

In Prism, a transition table defines the line groups and their relative weights,
while a few shared parameters describe their kinematics. Here is an oxygen
doublet added to an Astropy power-law continuum:

```python
from astropy import units as u
from astropy.modeling.models import PowerLaw1D
from prism.modeling import Model
from prism.modeling.lines import GaussianLines

transitions = {
    "name": ["OIII", "OIII"],
    "position": [4958.9, 5006.8] * u.AA,
    "weight": [1.0 / 2.98, 1.0],
}
narrow = GaussianLines(transitions, flux=3.0, fwhm=250.0, name="narrow")
continuum = Model.from_astropy(
    PowerLaw1D(amplitude=1.0, x_0=5000.0, alpha=1.0, fixed={"x_0": True})
)
model = continuum + narrow
```

Here `fwhm` is in km/s, and `flux` sets the reference transition's integrated
flux in your data units; the weights set the relative transition fluxes.
Gaussian, Lorentzian, and Voigt profiles are ready to use. Bring your own profile
with `LineSet`, load external transition tables, or start from the bundled line
lists. Explore [Line Components](examples/Line%20Components.ipynb) to build
richer line models, define your own profiles, and measure individual transitions.

### Keep the source and instrument separate
An observed spectrum is the source seen through an instrument. Finite spectral
resolution broadens features and blends nearby lines, so fitting the intrinsic
profile directly can confuse instrumental broadening with source physics.
That matters when the widths or shapes are the quantities you want to measure.

Prism lets you keep the source model and instrument response separate, then
combine them for comparison with the data:

```python
from prism.modeling.operators import InstrumentResponse, SpectralResponse

response = SpectralResponse(
    instrument=InstrumentResponse.from_fixed_fwhm(wave, 2.0),
    wave=wave,
)
observed_model = model | response
```

Here `wave` is a wavelength grid in Angstrom and the instrumental FWHM is
2 Angstrom. The response broadens the model; the intrinsic profile stays
available for physical measurements. Each dataset can also carry its own
response, so the same source model can be fitted to observations from different
instruments. Explore [Operators and Responses](examples/Operators%20and%20Responses.ipynb)
to see both views of a fitted spectrum and build your own linear operators.

### The fit is just the beginning

Best-fit parameters are often a stepping stone to the quantities you actually
need. A feature may contain several overlapping lines or kinematic components;
its total flux, width, and velocity need not correspond to any single model
parameter. Computing those quantities and carrying their correlated
uncertainties through the calculation can become a project of its own.

Prism provides tools for that step, too.
Want the flux of a line, the width of a blend, or the combined profile of several
kinematic components? Select the feature you mean, then measure it:

```python
from prism.modeling import select_line

result = TRF()(model=model, x=x, y=y, yerr=yerr)
line = select_line(result, "OIII", components="narrow", position=5006.8 * u.AA)
measured = line.measure()
uncertainty = line.sample(n_samples=200, method="covariance", random_state=41)
uncertainty.to_frame(metrics=["flux", "fwhm", "moment1"])
```

Selections measure intrinsic profiles, before instrumental responses. You can
combine kinematic components, measure equivalent widths against a selected
continuum, and propagate correlated parameter uncertainties. Covariance sampling
uses a local approximation; bootstrap refits offer another route and their
samples can be saved for later summaries. Fit sessions can also be saved and
reloaded. Explore [Blended Line Analysis](examples/Blended%20Line%20Analysis.ipynb)
to combine fitted components into feature measurements with uncertainties;
the [measurement guide](prism/modeling/lines/measurements/README.md) explains
the available quantities and conventions.

### One spectrum, then a whole cube

Use the same model for independent fits across a series of spectra or an
integral-field cube, with optional process parallelism:

```python
from prism.data import Cube

cube = Cube.from_fits("examples/data/linescube.fits", redshift=0.06)
cube_result = TRF(covariance=False).multifit(
    model=model, x=cube.z, y=cube.values, yerr=cube.err,
    axis=0, nproc=4,
)
```

This illustrates the batch interface; choose the model's redshift, starting
values, and bounds for your observations before fitting. Results retain the
spatial layout, with fitted models, parameter maps, and per-spectrum diagnostics.
Omitting covariance is useful when you do not need local error estimates for
every fit. Independent batch fitting and joint fitting serve different purposes:
the former fits each spectrum separately, while the latter connects datasets
through shared parameters. Explore [Model Data Cube](examples/Model%20Data%20Cube.ipynb)
to go from an input cube to fitted spectra and spatial parameter maps.

## Explore the API through examples

Looking for a particular workflow? The notebooks below show the tools in action,
from a custom function to spectral measurements and cube fitting. Pick the one
closest to your task. Start with [Custom Components](examples/Custom%20Components.ipynb)
for the building blocks, or [Model Single Spectrum](examples/Model%20Single%20Spectrum.ipynb)
for a full astronomical spectral analysis.

- [Custom Components](examples/Custom%20Components.ipynb): turn a function into a model, set bounds, fit data, and save and reload the result.
- [Line Components](examples/Line%20Components.ipynb): build line sets from tables, share parameters, define custom profiles, and measure transitions.
- [Model Single Spectrum](examples/Model%20Single%20Spectrum.ipynb): prepare an astronomical spectrum, build an active galactic nucleus model, include the response, and explore uncertainties.
- [Operators and Responses](examples/Operators%20and%20Responses.ipynb): build a linear operator and connect an intrinsic model to an instrumental response.
- [Joint Fitting](examples/Joint%20Fitting.ipynb): fit multiple observations together with selected parameter links and different responses.
- [Model Data Cube](examples/Model%20Data%20Cube.ipynb): fit spectra across a cube and inspect spatial parameter and uncertainty maps.
- [Multi-Spectrum Line Extraction](examples/Multi-Spectrum%20Line%20Extraction.ipynb): extract line measurements and equivalent widths across a fitted series.
- [Blended Line Analysis](examples/Blended%20Line%20Analysis.ipynb): measure blended features and propagate covariance using the saved single-spectrum fit.
- [Empirical Subtraction](examples/Empirical%20Subtraction.ipynb): reshape a reference spectrum with splines, reject outliers, and inspect the residuals.
- [Chromatic PSF](examples/Chromatic%20PSF.ipynb): use a field star to calibrate a wavelength-dependent PSF.
- [Aperture Correction](examples/Aperture%20Correction.ipynb): compare aperture corrections and PSF-based spectral extraction.

The [notebook guide](examples/README.md) covers bundled data, prerequisites, and
execution notes. For more detail, see the guides to
[fitting](prism/modeling/fitting/README.md),
[line models](prism/modeling/lines/README.md), and
[line measurements](prism/modeling/lines/measurements/README.md).
Spatial image fitting is supported through the Astropy fitter adapters;
PSF calibration and extraction tools live in `prism.psf`.

## Future developments

On the horizon:

- Stellar population models.
- A wider choice of minimizers.
- Support for X-ray astronomical datasets.

These are planned directions; stay tuned!

## License

Prism is distributed under [GPLv3](LICENSE).
Found a bug or have an idea? Open an [issue](https://github.com/pier-astro/prism-spec/issues).
