# prism

`prism` is an Astropy-compliant package for spectral modelling, fitting, and line analysis from single spectra to IFU cubes.

## Installation

```bash
pip install /path/to/prism-spec
```

Core runtime dependencies are listed in `pyproject.toml`. Sherpa-backed fitters are optional.

## Main Packages

- `prism.data`: native spectral containers such as `Spectrum`, `Cube`, and `Image`.
- `prism.modeling.models`: analytic line models, line groups, empirical components, and AGN model factories.
- `prism.modeling.fitting`: Astropy-style wrappers (`TRFLSQFitter`, `DogBoxLSQFitter`, `LevMarLSQFitter`) together with Prism-native fitters (`ScipyTRF`, `ScipyDogBox`, `LMFitter`).
- `prism.modeling.line_analysis`: post-fit extraction and measurement with `select_line`, `measure_line`, and `sample_line_measurements`.
- `prism.modeling.operators`: operator-aware models such as `InstrumentResponse` and `SpectralResponse`.

## Single-Spectrum Workflow

```python
import numpy as np

from prism.data import Spectrum
from prism.modeling.fitting import ScipyTRF
from prism.modeling.line_analysis import measure_line, select_line
from prism.modeling.models import GaussianLines

spec = Spectrum.from_txt(
	'examples/data/agnspec.txt',
	redshift=0.043,
	xunit='AA',
)
spec.crop(bounds=(4300.0, 7000.0))

model = GaussianLines.from_arrays(
	names=['Hb4861'],
	pos=[4861.333],
	name='hb',
	amplitude=1.0,
	offset=0.0,
	fwhm=400.0,
)

fitter = ScipyTRF(calc_uncertainties=True)
fitted = fitter(model, x=spec.x, y=spec.y, yerr=spec.yerr)

line = select_line(fitted, 'Hb4861')
result = measure_line(line, x=np.linspace(4800.0, 4925.0, 2048))
print(result.flux.value, result.fwhm.value)
```

## Cube Fitting

Every Prism fitter supports explicit batch fitting through `multifit`, and the Astropy-style wrappers also auto-dispatch when the input data has extra spatial dimensions.

```python
from prism.data import Cube
from prism.modeling.fitting import TRFLSQFitter

cube = Cube.from_fits('examples/data/datacube.fits')
fitter = TRFLSQFitter(calc_uncertainties=False)

result = fitter.multifit(
	model,
	x=cube.z,
	y=cube.values,
	yerr=cube.err,
	nproc=1,
	progress=False,
)

print(result.success.shape)
print(result.cost.shape)
```

The batched output exposes fitted models and parameter maps through `MultiFitResult`, `SpectrumFitResult`, and `MultiFitParameter`.

## Instrumental Response Operators

```python
from prism.modeling.operators import InstrumentResponse, SpectralResponse

inst_rsp = InstrumentResponse.from_fixed_resolution(wavelength_grid=wave, R=2000)
rsp = SpectralResponse(instrument=inst_rsp, wave=wave, z=0.0)
observed_model = intrinsic_model | rsp
```

This keeps the source model explicit while applying the response matrix on the right-hand side of the Astropy pipe operator.

## Saving Results

```python
fitted.save('results/agnmodel.fits', overwrite=True)
result.save('results/cubemodel.fits', overwrite=True, std=True)
```

## Examples

Notebook examples live in `examples/` and cover single-spectrum fitting, cube fitting, empirical subtraction, and blended-line analysis.

## License

Distributed under GPLv3. See `LICENSE`.

