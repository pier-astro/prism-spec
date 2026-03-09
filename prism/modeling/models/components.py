"""
Component decomposition utilities for Astropy CompoundModel instances.

This module provides tools to extract and access individual components from
compound models, with support for both all-component and additive-only extraction.

ConvolvedModel support: If model is wrapped in ConvolvedModel (e.g., rsp(A + B*C)),
the extracted components are also wrapped: rsp(A), rsp(B*C).
This ensures components are evaluated consistently with the original model.
"""

import numpy as np
from astropy.modeling import CompoundModel
from astropy.modeling.models import Const1D


def _get_source_model(model):
    """
    Unwrap ConvolvedModel to get the source model.
    
    If model is a ConvolvedModel (from instrument convolution), return the
    underlying source model. Otherwise return the model unchanged.
    """
    # Import here to avoid circular imports
    from ..operators.convolved import ConvolvedModel
    
    if isinstance(model, ConvolvedModel):
        return model._source
    return model


def _get_convolution_info(model):
    """
    Extract convolution parameters if model is wrapped in ConvolvedModel.
    
    Parameters
    ----------
    model : Model
        Input model (possibly ConvolvedModel-wrapped)
        
    Returns
    -------
    tuple or None
        (ConvolvedModelClass, operator_matrix, wave, name) if wrapped, else None
    """
    from ..operators.convolved import ConvolvedModel
    
    if isinstance(model, ConvolvedModel):
        return (type(model), model._operator_matrix, model._wave, model._name)
    return None


def _wrap_component(component, conv_info):
    """
    Wrap a component with ConvolvedModel if original was convolved.
    
    Parameters
    ----------
    component : Model
        Extracted component model
    conv_info : tuple or None
        (ConvolvedModelClass, operator_matrix, wave, name) or None
        
    Returns
    -------
    Model
        Component wrapped in ConvolvedModel (if conv_info), else unchanged
    """
    if conv_info is None:
        return component
    
    cls, matrix, wave, name = conv_info
    return cls(component, matrix, wave=wave, name=name)


def _make_unique_name(name, existing_names):
    """
    Generate unique name by appending suffix if needed.
    
    Parameters
    ----------
    name : str
        Proposed name.
    existing_names : set
        Set of already used names.
        
    Returns
    -------
    str
        Unique name (original or with _N suffix).
    """
    if name not in existing_names:
        return name
    
    # Name collision: find unique suffix
    i = 0
    while f"{name}_{i}" in existing_names:
        i += 1
    return f"{name}_{i}"


class ModelComponents(dict):
    """
    Dictionary-like container for compound model components with dual-key access.
    
    Supports access by both integer index and component name:
    - comps[0], comps[1], comps[2], ... (by index)
    - comps['continuum'], comps['blr'], comps['nlr'], ... (by name)
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model to decompose.
    additive : bool, optional
        If True, only additive components are included (multiplicative terms expanded).
        If False (default), all leaf components are included.
        
    Attributes
    ----------
    additive : bool
        Whether only additive components are included.
    indices : list of int
        List of integer indices for all components.
    names : list of str
        List of component names (same order as indices).
        
    Notes
    -----
    If the input model is wrapped in ConvolvedModel (e.g., rsp(A + B*C)),
    each extracted component is also wrapped: rsp(A), rsp(B*C).
    This ensures component evaluation is consistent with the full model.
        
    Examples
    --------
    >>> from astropy.modeling import models
    >>> m1 = models.Gaussian1D(1, 5000, 10, name='line1')
    >>> m2 = models.Gaussian1D(2, 5100, 15, name='line2')
    >>> compound = m1 + m2
    >>> 
    >>> comps = ModelComponents(compound)
    >>> comps[0]  # Access by index
    >>> comps['line1']  # Access by name
    >>> comps.names  # ['line1', 'line2']
    """
    
    def __init__(self, model, additive=False):
        super().__init__()
        # Store convolution info before unwrapping
        self._conv_info = _get_convolution_info(model)
        # Unwrap ConvolvedModel if present
        self._model = _get_source_model(model)
        self._additive_only = additive
        self._indices = []
        self._names = []
        self._components = []
        
        # Build the components
        if additive:
            self._build_additive()
        else:
            self._build_all()
    
    @property
    def additive(self):
        """Whether only additive components are included."""
        return self._additive_only
    
    @property
    def indices(self):
        """List of integer indices."""
        return self._indices.copy()
    
    @property  
    def names(self):
        """List of component names."""
        return self._names.copy()
    
    def to_list(self):
        """
        Return components as a simple list.
        
        Returns
        -------
        list
            List of model components in order.
        """
        return self._components.copy()
    
    def __repr__(self):
        """String representation showing structure."""
        mode = "additive-only" if self._additive_only else "all"
        n_comps = len(self._components)
        names_str = ", ".join([f"'{name}'" for name in self._names[:3]])
        if n_comps > 3:
            names_str += f", ... ({n_comps-3} more)"
        return f"ModelComponents({mode}, n={n_comps}, names=[{names_str}])"
    
    def _build_all(self):
        """Build list of all leaf components (depth-first traversal)."""
        idx = [0]
        used_names = set()
        
        def traverse(submodel):
            if isinstance(submodel, CompoundModel):
                traverse(submodel.left)
                traverse(submodel.right)
            else:
                # Leaf node
                name = self._get_component_name(submodel)
                unique_name = _make_unique_name(name, used_names)
                used_names.add(unique_name)
                
                # Wrap with convolution if original was convolved
                wrapped = _wrap_component(submodel, self._conv_info)
                
                # Store with dual keys
                self[idx[0]] = wrapped
                self[unique_name] = wrapped
                
                self._indices.append(idx[0])
                self._names.append(unique_name)
                self._components.append(wrapped)
                
                idx[0] += 1
        
        traverse(self._model)
    
    def _build_additive(self):
        """Build list of additive components (expand multiplicative terms)."""
        components = self._expand_additive(self._model)
        
        used_names = set()
        for idx, comp in enumerate(components):
            name = self._get_component_name(comp)
            unique_name = _make_unique_name(name, used_names)
            used_names.add(unique_name)
            
            # Wrap with convolution if original was convolved
            wrapped = _wrap_component(comp, self._conv_info)
            
            self[idx] = wrapped
            self[unique_name] = wrapped
            
            self._indices.append(idx)
            self._names.append(unique_name)
            self._components.append(wrapped)
    
    def _expand_additive(self, model):
        """
        Recursively expand model into additive components.
        
        For addition (+): keep components separate
        For subtraction (-): negate right side with Const1D(-1) * component
        For multiplication/division/power: expand all pairwise combinations
        
        Parameters
        ----------
        model : Model
            Model to expand.
            
        Returns
        -------
        list
            List of additive component models.
        """
        if not isinstance(model, CompoundModel):
            return [model]
        
        left = self._expand_additive(model.left)
        right = self._expand_additive(model.right)
        
        # Check operator (stored as string in Astropy)
        if model.op == '+':
            return left + right
        elif model.op == '-':
            # Negate right side: create (-1) * component for each right component
            neg_right = [Const1D(-1) * r for r in right]
            return left + neg_right
        elif model.op in ('*', '/', '**'):
            # Create all pairwise combinations
            products = []
            for l in left:
                for r in right:
                    # Create compound model with the appropriate operator
                    if model.op == '*':
                        products.append(l * r)
                    elif model.op == '/':
                        products.append(l / r)
                    elif model.op == '**':
                        products.append(l ** r)
            return products
        else:
            # Unknown operator, treat as single component
            return [model]
    
    def _get_component_name(self, model):
        """
        Extract or construct name from model.
        
        Priority:
        1. model.name if set
        2. Construct descriptive name for compound models (in additive mode)
        3. Fallback to class name
        
        Parameters
        ----------
        model : Model
            Model to extract name from.
            
        Returns
        -------
        str
            Component name.
        """
        if isinstance(model, CompoundModel):
            # Compound model in additive mode - construct descriptive name
            return self._construct_compound_name(model)
        
        if hasattr(model, 'name') and model.name:
            return model.name
        
        # Fallback to class name
        return type(model).__name__
    
    def _construct_compound_name(self, model):
        """
        Construct descriptive name for compound model.
        
        Parameters
        ----------
        model : CompoundModel
            Compound model.
            
        Returns
        -------
        str
            Descriptive name like "A*B" or "-C".
        """
        left_name = self._get_component_name(model.left)
        right_name = self._get_component_name(model.right)
        
        if model.op == '*':
            return f"{left_name}*{right_name}"
        elif model.op == '/':
            return f"{left_name}/{right_name}"
        elif model.op == '**':
            return f"{left_name}**{right_name}"
        elif model.op == '-':
            # Shouldn't happen if _expand_additive worked correctly
            return f"{left_name}-{right_name}"
        elif model.op == '+':
            # Shouldn't happen either, but handle it
            return f"{left_name}+{right_name}"
        else:
            # Fallback for unknown operators
            return f"compound_{id(model)}"


def get_components(model, additive=False):
    """
    Extract components from a compound model.
    
    This function decomposes compound models into their constituent parts,
    providing dual-key access (by index and by name) to each component.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model to decompose.
    additive : bool, optional
        If True, return only additive components (expand multiplicative terms).
        If False (default), return all leaf components.
        
    Returns
    -------
    ModelComponents
        Dictionary-like object with dual-key access to components.
        Supports both integer indices (0, 1, 2, ...) and component names.
        
    Examples
    --------
    Basic usage with simple compound model:
    
    >>> from astropy.modeling import models
    >>> m1 = models.Gaussian1D(1, 5000, 10, name='line1')
    >>> m2 = models.Gaussian1D(2, 5100, 15, name='line2')
    >>> compound = m1 + m2
    >>> 
    >>> # Get all components
    >>> comps = get_components(compound)
    >>> comps.names  # ['line1', 'line2']
    >>> comps[0]  # First component (by index)
    >>> comps['line1']  # By name
    
    Additive-only extraction for plotting:
    
    >>> import matplotlib.pyplot as plt
    >>> wave = np.linspace(4900, 5200, 1000)
    >>> add_comps = get_components(compound, additive=True)
    >>> for name in add_comps.names:
    ...     plt.plot(wave, add_comps[name](wave), label=name)
    >>> plt.plot(wave, compound(wave), 'k-', label='Total')
    
    Multiplicative expansion:
    
    >>> cont = models.Const1D(1.0, name='continuum')
    >>> line = models.Gaussian1D(2, 5000, 10, name='line')
    >>> compound = cont * (m1 + m2)
    >>> 
    >>> # All components: cont, m1, m2
    >>> comps = get_components(compound)
    >>> len(comps.indices)  # 3
    >>> 
    >>> # Additive components: cont*m1, cont*m2
    >>> add_comps = get_components(compound, additive=True)
    >>> add_comps.names  # ['continuum*line1', 'continuum*line2']
    """
    return ModelComponents(model, additive=additive)
