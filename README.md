# prism

`prism` provides general component-based modelling and fitting, with scientific tools for spectral models, instrumental responses and line analysis from single spectra to IFU cubes.

## Installation

```bash
pip install /path/to/prism-spec
```

Python 3.11 or newer is required. Runtime dependencies are listed in
`pyproject.toml`, with minimum versions and no speculative upper bounds. TRF
works without Sherpa; install Sherpa separately to use its optional optimizers.

The dependency ranges are provisional. Full beta checks used Python 3.11 with
NumPy 2.3; brief compatibility checks also cover NumPy 1.26 and 2.4 on Python
3.11. Other allowed combinations have not all been tested.

## Main Packages

- `prism.data`: native spectral containers such as `Spectrum`, `Cube`, and `Image`.
- `prism.modeling.lines`: native line profiles, weighted components and transition tables.
- `prism.modeling.models`: continuum, empirical and spatial models, plus native AGN factories.
- `prism.modeling.fitting`: shared SciPy and Sherpa engines; Astropy wrappers are in `interop.astropy` and development solvers in `prism.experimental`.
- `prism.modeling.lines.measurements`: post-fit extraction and measurement with `select_line`, `measure_line`, and `sample_line_measurements`.
- `prism.modeling.operators`: operator-aware models such as `InstrumentResponse` and `SpectralResponse`.

## Single-Spectrum Workflow

```python
import numpy as np

from prism.data import Spectrum
from prism.modeling.fitting import TRF
from prism.modeling.lines.measurements import measure_line, select_line
from prism.modeling.lines import GaussianLines
from astropy import units as u

spec = Spectrum.from_txt(
	'examples/data/agnspec.txt',
	redshift=0.043,
	xunit='AA',
)
spec = spec.cutout(min=4300.0, max=7000.0)

model = GaussianLines(dict(name=['Hb4861'], position=[4861.333]*u.AA),
                      amplitude=1., fwhm=400., name="hb")

fitter = TRF()
result = fitter(model, x=spec.x, y=spec.y, yerr=spec.yerr)
fitted = result.model

line = select_line(result, 'Hb4861')
measurement = measure_line(line, x=np.linspace(4800.0, 4925.0, 2048))
print(measurement.flux.value, measurement.fwhm.value)
```

## Cube Fitting

Every Prism fitter supports explicit batch fitting through `multifit`, and the Astropy-style wrappers also auto-dispatch when the input data has extra spatial dimensions.

```python
from prism.data import Cube
from prism.modeling.fitting import TRF

cube = Cube.from_fits('examples/data/datacube.fits')
fitter = TRF(covariance=False)

cube_result = fitter.multifit(
	model,
	x=cube.z,
	y=cube.values,
	yerr=cube.err,
	nproc=1,
	progress=False,
)

print(cube_result.success.shape)
print(cube_result.cost.shape)
```

The batched output exposes fitted models and parameter maps through `MultiFitResult`, `SpectrumFitResult`, and `MultiFitParameter`.

## Instrumental Response Operators

```python
from prism.modeling.operators import InstrumentResponse, SpectralResponse

inst_rsp = InstrumentResponse.from_fixed_resolution(wavelength_grid=wave, R=2000)
rsp = SpectralResponse(instrument=inst_rsp, wave=wave, z=0.0)
observed_model = intrinsic_model | rsp
```

This keeps the source model explicit while applying the response matrix on the right-hand side of the pipe operator.

## Saving Results

```python
from prism.modeling.fitting import FitResult

result.save('results/agnfit.fits', overwrite=True)
restored = FitResult.load('results/agnfit.fits')
cube_result.save('results/cubemodel.fits', overwrite=True, std=True)
```

## Current scope

- Native components and joint/batch fitting use one independent coordinate;
  general 2D fitting uses the supported Astropy adapters described below.
- Matrix operators can process upstream model values. A native interface for
  arbitrary whole-profile transformations is planned, rather than required for
  this beta. Alternative-grid predictions use point interpolation and are zero
  outside the response grid; they are not flux-conserving rebinning.
- Line selection measures intrinsic line leaves, before pipe operators.
- Covariance is estimated by default; use `TRF(covariance=False)` to omit it,
  particularly for large batches. It is a local uncertainty approximation.

## Examples

The [notebook guide](examples/README.md) lists the examples and their prerequisites, from custom components and line profiles to single, joint and cube fitting.

- [Chromatic PSF](examples/Chromatic%20PSF.ipynb): build and inspect a wavelength-dependent PSF from a field star.
- [Aperture Correction](examples/Aperture%20Correction.ipynb): compare a point-source aperture correction, PSF-plus-constant extraction, and a constant correction.

## License

Distributed under GPLv3. See `LICENSE`.


## Import paths after backend organization

| Responsibility | Import |
|---|---|
| Supported numerical fitters | `from prism.modeling.fitting import TRF, SherpaLM` |
| Astropy fitter adapters, including spatial fits | `from prism.modeling.interop.astropy import TRFLSQFitter` |
| Experimental solvers | `from prism.experimental import LMFitter, SeparableTRF` |
| PSF calibration and extraction | `from prism.psf import ChromaticPSF, fitbands, aperture, optimal` |
| Extraction result I/O | `from prism.psf.extraction import read, write` |
| Instrument responses | `from prism.modeling.operators.instrument import SpectralResponse` |

The old modeling.psf, modeling.extraction and modeling.instrument modules were
removed. Batch result classes retain their public names in
`prism.modeling.fitting.multifit`; their implementation is organized into
configuration, execution and results without changing independent-fit behavior.


## Spatial fitting

General two-coordinate fitting is a primary supported capability, currently
provided through the stable Astropy adapters, independently of PSF calibration:

```python
from prism.modeling.interop.astropy import TRFLSQFitter
fitted = TRFLSQFitter()(model, x_grid, y_grid, z=image_values,
                      yerr=image_errors, mask=valid_pixels, inplace=False)
```

Coordinates, values, errors and mask must have matching shapes. True mask entries
are included; non-finite included values require filter_non_finite=True.
Uncertainties are positive symmetric per-pixel errors. The returned model still
accepts two coordinate grids. Native SciPy/Sherpa 2D integration is an explicit
architectural task in TODO.md; moving adapters to interop does not make spatial
fitting an experimental feature.
