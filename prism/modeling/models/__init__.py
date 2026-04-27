"""
prism.modeling.models — Astropy-compliant spectral models.

Single-line models
------------------
``GaussianLine``, ``LorentzianLine``, ``VoigtLine``
    Fittable1DModel subclasses parameterised by amplitude, rest-frame
    position (fixed), velocity offset [km s⁻¹], width [km s⁻¹], and
    redshift (fixed).  Instrument broadening is accepted via ``instfwhm``.

Line-group models
-----------------
``GaussianLines``, ``LorentzianLines``, ``VoigtLines``
    Dynamically generated multi-line Fittable1DModel, built from line-table
    files (preferably ECSV with ``name``, ``position``, ``weight`` columns;
    plain CSV requires an explicit ``position_unit``). Lines sharing the same
    ``name`` are amplitude-tied through a single parameter.  Constructed
    via ``from_csv()``, ``from_arrays()``, or ``from_templates()``.

Continuum models
----------------
``BalmerContinuum``
    Blackbody-based Balmer pseudo-continuum below the Balmer edge (3646 Å).

For standard power-law continua, use Astropy's built-in models from
``astropy.modeling.powerlaws``.

Empirical models
----------------
``FixedTemplate``
    Interpolation-based model with no free parameters (amplitude fixed by
    the input table).
``BSpline``
    Dynamic B-spline where the coefficients are exposed as fit parameters.

AGN factory functions
---------------------
``nlr()``, ``blr()``, ``fe()``, ``fe_uv()``, ``balmer()``
    Convenience factories that return pre-configured line-group models for
    common AGN emission components.

Utilities
---------
``get_components()`` / ``ModelComponents``
    Extract and iterate over components of a compound model.
    Handles linear-operator model nodes transparently.
``count_free(model)``
    Count the number of free (unfixed, untied) parameters.
"""
from .lines import *
from .continuum import *
from .components import get_components, ModelComponents

from .agn import *
from .empirical import *

from ..operators import instrument                               # models.instrument.*
from ..operators.instrument import InstrumentResponse, SpectralResponse

from .display import (
    enable_model_display,
    disable_model_display,
    get_model_expression,
    format_model_text,
    format_model_html,
    format_fit_text,
    format_fit_html,
    show_model,
)
