"""Native line components seeded from Prism's packaged AGN transition tables.

Amplitudes are profile peaks, weights multiply each template amplitude once,
and shared offsets/widths use km/s. Further constraints belong to the caller.
"""
from pathlib import Path
from astropy import units as u
from ..lines import GaussianLines
from ..lines import tables

__all__ = ['nlr', 'blr', 'fe', 'fe_uv', 'balmer']


def _from_builtin(files, *, name, dirpath=None, amplitude=1., offset=0.,
                  fwhm=1000., redshift=0., instfwhm=0., bounds=None):
    directory = Path(tables.linetable_path if dirpath is None else dirpath)
    files = [files] if isinstance(files, str) else files
    table = tables.stack_linetables([tables.read_linetable(directory/path) for path in files])
    if tables._wmin is not None or tables._wmax is not None:
        wavelength = table['position'].to_value(u.AA, equivalencies=u.spectral())
        mask = wavelength == wavelength
        if tables._wmin is not None:
            mask &= wavelength >= tables._wmin
        if tables._wmax is not None:
            mask &= wavelength <= tables._wmax
        table = table[mask]
    return GaussianLines(table, amplitude=amplitude, offset=offset, fwhm=fwhm,
                         redshift=redshift, instfwhm=instfwhm, bounds=bounds, name=name)


def nlr(name='nlr', dirpath=None, **kwargs):
    """Create a native narrow-line-region component from ``narrow_basic.ecsv``.

    Keyword arguments set peak amplitude, shared offset/FWHM (km/s), redshift,
    instrumental FWHM and parameter bounds. ``amplitude`` bounds apply to every
    template; redshift starts fixed. The current line-table directory is used.
    """
    return _from_builtin('narrow_basic.ecsv', name=name, dirpath=dirpath, **kwargs)


def blr(name='blr', dirpath=None, **kwargs):
    """Create a native broad-line component from hydrogen and helium tables.

    Positions and relative weights come from the tables; templates have separate
    peak amplitudes and one shared offset and width in km/s. See ``nlr`` for the
    initialization options. Redshift starts fixed.
    """
    return _from_builtin(['hydrogen.ecsv', 'helium.ecsv'], name=name, dirpath=dirpath, **kwargs)


def fe(name='fe', dirpath=None, **kwargs):
    """Create the optical Fe II component from ``feII_model.ecsv``.

    The returned native LineSet shares kinematics across the tabulated
    multiplets. Each template retains its own amplitude. See ``nlr`` for peak,
    width, instrumental broadening and bounds options.
    """
    return _from_builtin('feII_model.ecsv', name=name, dirpath=dirpath, **kwargs)


def fe_uv(name='fe_uv', dirpath=None, **kwargs):
    """Create the native UV Fe II component from ``uvfe.ecsv``.

    Tabulated relative weights multiply template peak amplitudes. Kinematics are
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
