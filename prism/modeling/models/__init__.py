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
- count_free(): Count free parameters

Example:
--------
>>> from prism.modeling.models import TiedGaussianLines, tie
>>> nlr = TiedGaussianLines.from_csv('oiii_nii.csv', name='nlr', ...)
>>> nlr.amp_OIII_4959.tied = tie('nlr', lambda m: m.amp_OIII_5007 / 3)
>>> 
>>> # Decompose compound models
>>> from prism.modeling.models import get_components
>>> compound = m1 + m2 + m3
>>> comps = get_components(compound)
>>> comps[0]  # Access by index
>>> comps['line1']  # Access by name
>>> 
>>> # Inspect model parameters
>>> from prism.modeling.models import count_free
>>> n_free = count_free(compound)
"""
from .lines import *
from .continuum import *
from .components import get_components, ModelComponents

from .agn import *
from .empirical import *
