from .lines import GaussianLines
from . import lines as lines_module

def nlr(name='nlr', **kwargs):
    """
    Returns a unified GaussianLines model for the Narrow Line Region using 'narrow_basic.csv'.
    """
    return GaussianLines.from_csv('narrow_basic.csv', name=name, dirpath=lines_module.csv_lines_path, **kwargs)

def blr(name='blr', **kwargs):
    """
    Returns a unified GaussianLines model for the Broad Line Region using hydrogen and helium lines.
    """
    return GaussianLines.from_csv(['hydrogen.csv', 'helium.csv'], name=name, dirpath=lines_module.csv_lines_path, **kwargs)

def fe(name='fe', **kwargs):
    """
    Returns a unified GaussianLines model for Optical/UV Fe II emission using 'feII_model.csv'.
    It implicitly ties templates with the same name.
    """
    return GaussianLines.from_csv('feII_model.csv', name=name, dirpath=lines_module.csv_lines_path, **kwargs)

def fe_uv(name='fe_uv', **kwargs):
    """
    Returns a unified GaussianLines model for UV Fe II emission using 'uvfe.csv'.
    It implicitly ties templates with the same name.
    """
    return GaussianLines.from_csv('uvfe.csv', name=name, dirpath=lines_module.csv_lines_path, **kwargs)

def balmer(name='balmer', **kwargs):
    """
    Returns a unified GaussianLines model for Balmer series with tied lines and relative amplitudes.
    """
    return GaussianLines.from_csv('balmer.csv', name=name, dirpath=lines_module.csv_lines_path, **kwargs)
