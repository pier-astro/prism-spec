"""
prism.modeling.models.agn — AGN component factory functions.

Provides convenience constructors that return pre-configured
``GaussianLines`` (or related) models for the most common AGN emission
components. Each factory internally calls ``GaussianLines.from_csv()``
with the appropriate ECSV line list from the package resources.

Factory functions
-----------------
nlr()
    Narrow Line Region — ``narrow_basic.ecsv``.
blr()
    Broad Line Region — hydrogen + helium line lists.
fe()
    Optical/UV Fe ˳˳ pseudo-continuum — ``feII_model.ecsv``.
fe_uv()
    UV Fe ˳˳ emission — ``uvfe.ecsv``.
balmer()
    Balmer series with tied relative amplitudes — ``balmer.ecsv``.

All factories forward extra keyword arguments to ``GaussianLines.from_csv()``
(e.g. ``instfwhm``, ``bounds``, ``amplitude``).
"""
from .lines import GaussianLines
from . import lines as lines_module

__all__ = ['nlr', 'blr', 'fe', 'fe_uv', 'balmer']


def _from_builtin(csv_files, *, name, dirpath=None, **kwargs):
    """Build a ``GaussianLines`` model from Prism's packaged AGN line lists.

    Parameters
    ----------
    csv_files : str or sequence of str
        Built-in line-table filename or filenames.
    name : str
        Model name assigned to the returned component.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's current line
        table path.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`.

    Returns
    -------
    GaussianLines
        Pre-configured line-group model built from the requested templates.
    """
    base_dir = lines_module.linetable_path if dirpath is None else dirpath
    return GaussianLines.from_csv(csv_files, name=name, dirpath=base_dir, **kwargs)

def nlr(name='nlr', dirpath=None, **kwargs):
    """Return the default narrow-line-region emission model.

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned model. Default is ``'nlr'``.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's packaged line
        resources.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`, such as
        ``amplitude``, ``offset``, ``fwhm``, ``instfwhm``, or ``bounds``.

    Returns
    -------
    GaussianLines
        Gaussian line-group model built from ``narrow_basic.ecsv``.

    Notes
    -----
    This helper is the compact starting point for classical narrow forbidden and
    recombination features. It uses Prism's curated narrow-line template with one
    shared kinematic solution across all included transitions.

    Examples
    --------
    >>> narrow = nlr(amplitude=1.0, fwhm=250.0, instfwhm=120.0)
    >>> narrow.name
    'nlr'
    """
    return _from_builtin('narrow_basic.ecsv', name=name, dirpath=dirpath, **kwargs)

def blr(name='blr', dirpath=None, **kwargs):
    """Return the default broad-line-region emission model.

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned model. Default is ``'blr'``.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's packaged line
        resources.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`.

    Returns
    -------
    GaussianLines
        Gaussian line-group model built from the packaged hydrogen and helium
        broad-line templates.

    Notes
    -----
    The broad-line-region template combines the main hydrogen and helium series so
    one helper can seed the high-velocity permitted-line component of an AGN fit.
    Relative wavelengths remain fixed by the underlying line lists, while widths
    and offsets are fitted globally.
    """
    return _from_builtin(['hydrogen.ecsv', 'helium.ecsv'], name=name, dirpath=dirpath, **kwargs)

def fe(name='fe', dirpath=None, **kwargs):
    """Return the default optical Fe II pseudo-continuum model.

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned model. Default is ``'fe'``.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's packaged line
        resources.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`.

    Returns
    -------
    GaussianLines
        Gaussian line-group model built from ``feII_model.ecsv``.

    Notes
    -----
    The optical Fe II forest is represented as a dense template group. Repeated
    template names are tied automatically by ``GaussianLines``, which keeps the
    parameter count compact while preserving the physically tabulated multiplet
    structure.
    """
    return _from_builtin('feII_model.ecsv', name=name, dirpath=dirpath, **kwargs)

def fe_uv(name='fe_uv', dirpath=None, **kwargs):
    """Return the default ultraviolet Fe II emission model.

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned model. Default is ``'fe_uv'``.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's packaged line
        resources.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`.

    Returns
    -------
    GaussianLines
        Gaussian line-group model built from ``uvfe.ecsv``.

    Notes
    -----
    This helper targets the UV Fe II complexes with the same tied-template logic
    used for the optical Fe II component, but with a dedicated ultraviolet line
    list.
    """
    return _from_builtin('uvfe.ecsv', name=name, dirpath=dirpath, **kwargs)

def balmer(name='balmer', dirpath=None, **kwargs):
    """Return the default Balmer-series line-group model.

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned model. Default is ``'balmer'``.
    dirpath : str, optional
        Directory containing the line tables. Default uses Prism's packaged line
        resources.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`prism.modeling.models.GaussianLines.from_csv`.

    Returns
    -------
    GaussianLines
        Gaussian line-group model built from ``balmer.ecsv``.

    Notes
    -----
    The Balmer helper is useful when the relative series structure should remain
    tied while one kinematic component describes the whole Balmer blend. This is a
    good starting point for pseudo-continuum or recombination-line submodels.

    Examples
    --------
    >>> bal = balmer(amplitude=1.0, fwhm=800.0)
    >>> 'balmer' in bal.name
    True
    """
    return _from_builtin('balmer.ecsv', name=name, dirpath=dirpath, **kwargs)
