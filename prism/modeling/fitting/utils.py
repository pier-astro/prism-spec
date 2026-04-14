"""
Utility functions for prism.modeling.fitting
"""

import numpy as np
from astropy.modeling import CompoundModel

def tie(submodel_name, func):
    """
    Create a tie function for parameter constraints in compound models.

    This is the recommended way to tie parameters in prism models.

    Parameters
    ----------
    submodel_name : str
        Name of the submodel containing the parameter to reference.
    func : callable
        Function taking submodel, returns tied value.

    Returns
    -------
    callable
        Tie function for use with model.param.tied = ...
    """
    def _tie(model):
        if isinstance(model, CompoundModel) or hasattr(model, 'submodel_names'):
            if submodel_name not in model.submodel_names:
                raise KeyError(
                    f"Submodel '{submodel_name}' not found. "
                    f"Available: {model.submodel_names}"
                )
            sub = model[submodel_name]
        else:
            sub = model
        try:
            return func(sub)
        except AttributeError as e:
            params = list(getattr(sub, 'param_names', []))
            raise AttributeError(
                f"Tie error: {e}. Available params in '{submodel_name}': {params}"
            ) from e
    return _tie


def apply_tied(model):
    """Apply all tied parameter constraints to a model."""
    if not model.has_tied:
        return model

    param_metrics = model._param_metrics
    parameters = model.parameters.copy()

    for name in model.param_names:
        tied_func = model.tied[name]
        if tied_func:
            value = tied_func(model)
            slice_ = param_metrics[name]["slice"]
            parameters[slice_] = value

    model.parameters = parameters
    return model


def _get_tied_info(model):
    """Extract and cache tied parameter information for fast evaluation."""
    tied_info = []
    if not model.has_tied:
        return tied_info

    param_metrics = model._param_metrics
    tied_dict = model.tied

    for name in model.param_names:
        tied_func = tied_dict[name]
        if tied_func:
            slice_ = param_metrics[name]["slice"]
            tied_info.append((tied_func, slice_))

    return tied_info


def _apply_tied_fast(model, tied_info, parameters_cache):
    """Apply tied constraints using pre-cached info (fast path)."""
    if not tied_info:
        return
    for tied_func, slice_ in tied_info:
        value = tied_func(model)
        parameters_cache[slice_] = value
    model.parameters = parameters_cache
