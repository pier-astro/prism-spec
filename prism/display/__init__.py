"""
Display module for model inspection and formatted output.

Provides formatted views of model parameters, constraints, and structure
optimized for both terminal and Jupyter notebooks.
"""

from .sciformat import (
    round_sci,
    round_to_sig,
    round_to_error,
    format_value
)

from .model import (
    show,
    show_free,
    show_tied,
    show_bounds,
    show_fluxes,
    show_model_lineflux,
    count_free
)

__all__ = [
    # sciformat
    'round_sci',
    'round_to_sig', 
    'round_to_error',
    'format_value',
    # model display
    'show',
    'show_free',
    'show_tied',
    'show_bounds',
    'show_fluxes',
    'show_model_lineflux',
    'count_free',
]
