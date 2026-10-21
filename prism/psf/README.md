# PSF and spectral extraction

`prism.psf` exposes `fitimage`, `fitbands`, `fittrends`, `ChromaticPSF`, `aperture`
and `optimal`. `psf.py` implements calibration and chromatic trends;
`extraction.py` implements aperture/optimal extraction and result I/O.

```python
from prism.psf import fitbands, fittrends, ChromaticPSF, aperture, optimal
from prism.psf.extraction import read, write
```

The previous modeling.psf and modeling.extraction import paths have been removed.
Scientific algorithms, pixel geometry, sampling and uncertainty assumptions are
unchanged. See the modeling README for the existing workflow details, now using
the dedicated namespace.


## Scope of the namespace

Spatial profile definitions remain under modeling.models. This package owns the
broader workflow of PSF calibration, chromatic trends and spectral extraction,
which also operates on data and is not only model construction. General spatial
2D fitting is a primary Prism capability and is not restricted to this package.
