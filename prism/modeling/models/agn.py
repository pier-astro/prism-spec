"""
prism.modeling.models.agn — AGN component factory functions.

Provides convenience constructors that return pre-configured
``GaussianLines`` (or related) models for the most common AGN emission
components.  Each factory internally calls ``GaussianLines.from_csv()``
with the appropriate CSV line list from the package resources.

Factory functions
-----------------
nlr()
    Narrow Line Region — ``narrow_basic.csv``.
blr()
    Broad Line Region — hydrogen + helium line lists.
fe()
    Optical/UV Fe ˳˳ pseudo-continuum — ``feII_model.csv``.
fe_uv()
    UV Fe ˳˳ emission — ``uvfe.csv``.
balmer()
    Balmer series with tied relative amplitudes — ``balmer.csv``.

All factories forward extra keyword arguments to ``GaussianLines.from_csv()``
(e.g. ``instfwhm``, ``bounds``, ``amplitude``).
"""
from .lines import GaussianLines
from . import lines as lines_module


def _from_builtin(csv_files, *, name, dirpath=None, **kwargs):
    base_dir = lines_module.csv_lines_path if dirpath is None else dirpath
    return GaussianLines.from_csv(csv_files, name=name, dirpath=base_dir, **kwargs)

def nlr(name='nlr', dirpath=None, **kwargs):
    """
    Returns a unified GaussianLines model for the Narrow Line Region using 'narrow_basic.csv'.
    """
    return _from_builtin('narrow_basic.csv', name=name, dirpath=dirpath, **kwargs)

def blr(name='blr', dirpath=None, **kwargs):
    """
    Returns a unified GaussianLines model for the Broad Line Region using hydrogen and helium lines.
    """
    return _from_builtin(['hydrogen.csv', 'helium.csv'], name=name, dirpath=dirpath, **kwargs)

def fe(name='fe', dirpath=None, **kwargs):
    """
    Returns a unified GaussianLines model for Optical/UV Fe II emission using 'feII_model.csv'.
    It implicitly ties templates with the same name.
    """
    return _from_builtin('feII_model.csv', name=name, dirpath=dirpath, **kwargs)

def fe_uv(name='fe_uv', dirpath=None, **kwargs):
    """
    Returns a unified GaussianLines model for UV Fe II emission using 'uvfe.csv'.
    It implicitly ties templates with the same name.
    """
    return _from_builtin('uvfe.csv', name=name, dirpath=dirpath, **kwargs)

def balmer(name='balmer', dirpath=None, **kwargs):
    """
    Returns a unified GaussianLines model for Balmer series with tied lines and relative amplitudes.
    """
    return _from_builtin('balmer.csv', name=name, dirpath=dirpath, **kwargs)
