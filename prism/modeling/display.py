"""Public display activation for Prism models and Astropy models."""

from .models.display import (
    enable_model_display,
    disable_model_display,
    get_model_expression,
    format_model_text,
    format_model_html,
)

__all__ = [
    'enable_model_display',
    'disable_model_display',
    'get_model_expression',
    'format_model_text',
    'format_model_html',
]

enable_model_display()
