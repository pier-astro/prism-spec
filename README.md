# prism

**prism** is a spectral modeling and fitting framework for optical/NIR spectroscopy, built around **Astropy models** and a robust fitter interface.

It is designed for workflows that go from a single spectrum to full IFU cubes with minimal friction.

---

## ⚙️ Requirements

Core dependencies are listed in `pyproject.toml` (notably `astropy`, `numpy`, `scipy`, `pandas`, `matplotlib`).

Optional:
- `sherpa>=4.16` if you want to use the Sherpa-backed fitters.

---

## 🚀 Installation

Clone the repository and install locally:

```bash
pip install /path/to/prism
```

Then import with:

```python
import prism
```

---

## 📦 Data Containers (Spectrum, Cube, Image)

`prism.data` includes native containers for the most common spectral data types:

- `Spectrum`: 1D spectra with crop/rebin/redshift and utility methods.
- `Cube`: 3D IFU cubes (or flattened 2D spectral stacks), with FITS/WCS support.
- `Image`: 2D maps or images

### Example: load and prepare a single spectrum

```python
from prism.data import Spectrum

spec = Spectrum.from_txt("examples/data/agnspec.txt", z=0.043)
spec.zcorrect()                 # move to rest frame
spec.crop(wbounds=(4300, 7000))  # keep fitting window
```

### Example: load a cube

```python
from prism.data import Cube

cube = Cube.from_fits("examples/data/datacube.fits")
```

---

## 🧩 Model Setup (Astropy-Compliant)

`prism.modeling.models` provides Astropy-compatible components for:

- Emission/absorption lines (`GaussianLine`, `LorentzianLine`, `VoigtLine`)
- Tied line groups from CSV templates (`GaussianLines`, `LorentzianLines`, `VoigtLines`)
- Continua (`Powerlaw`, `BrokenPowerlaw`, `BalmerContinuum`)
- AGN convenience factories (`nlr`, `blr`, `fe`, `fe_uv`, `balmer`)

```python
from prism.modeling.models import setup_local_lines, Powerlaw, nlr, blr

# Optional: create local filtered CSV line lists
setup_local_lines(wmin=4000, wmax=8000, dirpath="./lines", overwrite=False)

continuum = Powerlaw(amplitude=1.0, x_ref=5500.0, index=-1.5)
narrow = nlr(name="nlr", instfwhm=120.0)
broad = blr(name="blr", instfwhm=120.0)

model = continuum + narrow + broad
```

---

## 🔧 Fitters (Astropy-Fitter Interface + Extensions)

`prism.modeling.fitting` exposes fitters that follow Astropy's fitter conventions while extending defaults for spectroscopy workflows.

Main families:
- Astropy wrappers: `AstroTRF`, `AstroDogBox`, `AstroLM`, `AstroSimplex`
- SciPy wrappers: `ScipyTRF`, `ScipyDogBox`
- Native LM backend: `LMFitter`
- Optional Sherpa wrappers: `SherpaLM`, `SherpaSimplex`, `SherpaMonCar`

```python
from prism.modeling.fitting import AstroTRF

fitter = AstroTRF(calc_uncertainties=True, verbose=True)
fitted_model = fitter(model, spec.wave, spec.flux, yerr=spec.err, statistic="chi2")
```

---

## 🧠 Batch / Cube Fitting (MultiFit)

Batched fitting is a core strength of prism for efficient cube analysis.

You can fit many spectra in one call and get structured outputs through:
- `MultiFitResult`
- `SpectrumFitResult`
- `MultiFitParameter`

```python
from prism.modeling.fitting import AstroTRF

fitter = AstroTRF(calc_uncertainties=False)

# flux and err can be (Nwave, Ny, Nx) or any shape with one spectral axis
result = fitter.multifit(
	model,
	x=cube.wave,
	y=cube.data,
	yerr=cube.err,
	nproc=4,
	progress=True,
)

success_map = result.success
cost_map = result.cost
```

> NOTE: `prism` fitters support cube fitting either via `fitter.multifit()`that directly via `fitter()`call.
This is especially useful for fast, clean spaxel-by-spaxel fitting on IFU cubes.

---

## 📦 Instrumental Response (Supported, but still evolving)

prism supports spectral instrumental response matrices via:
- `InstrumentResponse`
- `SpectralResponse`
- `ResponseModel` / `ConvolvedModel`

```python
from prism.modeling.operators import InstrumentResponse, SpectralResponse

# Build a response from fixed resolving power R
inst_rsp = InstrumentResponse.from_fixed_resolution(wavelength_grid=wave, R=2000)

# Wrap your intrinsic model with the response
rsp = SpectralResponse(instrument=inst_rsp, wave=wave, z=0.0)
fit_model = rsp(model)
```

> ⚠️ Current caveat:
> The instrumental-response path is working and useful, but the wrapped result is not yet a standard Astropy compound model in the strictest sense. It is functional for fitting, but this area is still being hardened and may be made more robust in future releases.

So: supported and usable, but not yet the most mature path in the package.

---

## 💾 Save Fit Results

You can serialize fitted outputs to FITS:

- Single model: `model.to_fits(...)` / `model.save(...)`
- Batch results: `MultiFitResult.to_fits(...)` / `MultiFitResult.save(...)`

```python
# Single-spectrum model
fitted_model.to_fits("results/agnmodel.fits", overwrite=True)

# Batched/cube fit maps
result.to_fits("results/cubemodel.fits", overwrite=True, std=True)
```

---

## 📚 Examples

See notebooks in `examples/`:

- `Model Single Spectrum.ipynb`
- `Model Data Cube.ipynb`
- `Empirical Subtraction.ipynb`

---

## License

This project is distributed under the GPLv3 license. See `LICENSE` for details.

