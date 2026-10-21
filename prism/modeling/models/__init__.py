"""
prism.modeling.models — Astropy-compliant spectral and spatial models.

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

Spatial models
--------------
``MoffatPSF``, ``EllipticalMoffatPSF``
    Circular and elliptical Astropy profiles with fitted FWHM and beta.

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
from .continuum import *
from .components import get_components, ModelComponents

from .agn import *
from .empirical import *
from .spatial import *

from ..operators import instrument                               # models.instrument.*
from ..operators.instrument import InstrumentResponse, SpectralResponse

from ..display import (
    get_model_expression,
    format_model_text,
    format_model_html,
    format_fit_text,
    format_fit_html,
    show_model,
)

from ..composition import Model

# Explicit YAML hooks for native components and supported external models.
from . import serialization as _serialization
