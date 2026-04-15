"""Public display activation for Prism models and Astropy models."""

from .models.display import (
    enable_model_display,
    disable_model_display,
    get_model_expression,
    format_model_text,
    format_model_html,
    format_fit_text,
    format_fit_html,
    show_model,
)

__all__ = [
    'enable_model_display',
    'disable_model_display',
    'get_model_expression',
    'format_model_text',
    'format_model_html',
    'format_fit_text',
    'format_fit_html',
    'show_model',
]

enable_model_display()
