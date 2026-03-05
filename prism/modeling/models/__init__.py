"""
FantasyLab Models Module

Provides Astropy-compliant spectral models for fitting and simulation.

Models:
-------
- Lines: GaussianLine, TiedGaussianLines, GaussianFeII, BalmerLines
- Continuum: BrokenPowerlaw, Blackbody

Utilities:
----------
- tie(): Create parameter constraints (e.g., OIII doublet ratio)
- apply_tied(): Apply tied constraints to model
- get_components(): Extract components from compound models
- ModelComponents: Container for model components with dual-key access
- show(): Display formatted model parameter summary
- count_free(): Count free parameters

Example:
--------
>>> from fantasylab.models import TiedGaussianLines, tie
>>> nlr = TiedGaussianLines.from_csv('oiii_nii.csv', name='nlr', ...)
>>> nlr.amp_OIII_4959.tied = tie('nlr', lambda m: m.amp_OIII_5007 / 3)
>>> 
>>> # Decompose compound models
>>> from fantasylab.models import get_components
>>> compound = m1 + m2 + m3
>>> comps = get_components(compound)
>>> comps[0]  # Access by index
>>> comps['line1']  # Access by name
>>> 
>>> # Inspect model parameters
>>> from fantasylab.models import show, count_free
>>> show(compound)  # Formatted table
>>> n_free = count_free(compound)
"""
from .lines import *
from .continuum import *
from .components import get_components, ModelComponents
from .convolved import ConvolvedModel
from .instrument import InstrumentResponse, SpectralResponse, ResponseOperator, ResponseModel
# Import display utilities from the new display package
from fantasylab.display import show, show_free, show_tied, show_bounds, show_fluxes, show_model_lineflux, count_free