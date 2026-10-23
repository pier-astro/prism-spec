"""Native line components seeded from Prism's packaged AGN transition tables.

Normalization follows GaussianLines: flux by default, or explicit amplitude.
Weights multiply the selected template scale once; shared velocities use km/s.
The optical Fe II weights match published atomic intensity ratios; flux mode
expresses them directly. Explicit amplitude mode reproduces the historical
FANTASY computational convention, which used the same numbers as peak ratios.
"""
from pathlib import Path
from ..lines import GaussianLines

__all__ = ['nlr', 'blr', 'fe', 'fe_uv', 'balmer']


def _from_builtin(files, *, name, dirpath=None, **kwargs):
    files = [files] if isinstance(files, str) else files
    sources = ([Path(dirpath)/filename for filename in files] if dirpath is not None
               else [Path(filename).stem for filename in files])
    return GaussianLines(sources, name=name, **kwargs)


def nlr(name='nlr', dirpath=None, **kwargs):
    """Create a native narrow-line-region component from ``narrow_basic.ecsv``.

    Keyword arguments set normalization, flux or amplitude, shared offset/FWHM
    (km/s), redshift, instrumental FWHM and bounds. The active scale bounds
    apply to every template; redshift starts fixed. Preset names use the configured directory then packaged defaults.
    Pass window=(min, max) to select rest positions explicitly.
    """
    return _from_builtin('narrow_basic.ecsv', name=name, dirpath=dirpath, **kwargs)


def blr(name='blr', dirpath=None, **kwargs):
    """Create a native broad-line component from hydrogen and helium tables.

    Positions and relative weights come from the tables; templates have separate
    scale parameters and one shared offset and width in km/s. See ``nlr`` for the
    initialization options. Redshift starts fixed.
    """
    return _from_builtin(['hydrogen.ecsv', 'helium.ecsv'], name=name, dirpath=dirpath, **kwargs)


def fe(name='fe', dirpath=None, **kwargs):
    """Create the optical Fe II component from ``feII_model.ecsv``.

    The returned native LineSet shares kinematics across the tabulated
    multiplets. Each template retains its own scale. Use normalization="amplitude" to
    reproduce the historical FANTASY peak convention. See ``nlr`` for scale,
    width, instrumental broadening and bounds options.
    """
    return _from_builtin('feII_model.ecsv', name=name, dirpath=dirpath, **kwargs)


def fe_uv(name='fe_uv', dirpath=None, **kwargs):
    """Create the native UV Fe II component from ``uvfe.ecsv``.

    Original weights were peak multipliers; choose normalization="amplitude"
    for the original template. Flux mode uses integrated-flux ratios. Kinematics are
    shared within this component. See ``nlr`` for initialization, instrumental
    broadening and parameter-bound options.
    """
    return _from_builtin('uvfe.ecsv', name=name, dirpath=dirpath, **kwargs)


def balmer(name='balmer', dirpath=None, **kwargs):
    """Create the native Balmer-series component from ``balmer.ecsv``.

    The tabulated transitions preserve their relative weights and template
    membership, with common kinematics. See ``nlr`` for initialization,
    instrumental broadening and parameter-bound options.
    """
    return _from_builtin('balmer.ecsv', name=name, dirpath=dirpath, **kwargs)
