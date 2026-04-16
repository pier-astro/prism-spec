"""
Component decomposition for compound models.

Decomposes ``CompoundModel`` trees into named, indexable parts via
``get_components``, returning a ``ModelComponents`` container.

For ``LinearOperatorCompoundModel`` (convolved models, ``h = M @ f``):

* ``additive=True, deconvolve=False`` — each additive term is wrapped
  with the operator for direct observed-frame evaluation.
* ``additive=True, deconvolve=True`` — additive terms are returned
  *without* the operator, giving intrinsic (source-frame) components.
* ``additive=False`` — all leaves of the source model plus an
  ``Identity`` marker for the operator.
"""

import numpy as np
from astropy.modeling import CompoundModel
from astropy.modeling.models import Const1D, Identity

from ..operators.convolved import LinearOperatorCompoundModel


def _get_source_model(model):
    """Return the source model, unwrapping ``LinearOperatorCompoundModel``."""
    if isinstance(model, LinearOperatorCompoundModel):
        return model.left
    return model


def _get_operator_info(model):
    """Return the ``LinearOperatorCompoundModel`` node, or ``None``."""
    if isinstance(model, LinearOperatorCompoundModel):
        return model
    return None


def _wrap_component(component, op_info):
    """Wrap *component* in the same linear operator if *op_info* is set."""
    if op_info is None:
        return component
    return op_info.with_left(component)


def _make_unique_name(name, existing_names):
    """Return *name* with a ``_N`` suffix appended when necessary."""
    if name not in existing_names:
        return name
    i = 0
    while f"{name}_{i}" in existing_names:
        i += 1
    return f"{name}_{i}"


class ModelComponents(dict):
    """
    Dictionary-like container for compound model components.

    Supports access by integer index and by component name::

        comps[0]            # by index
        comps['blr']        # by name
        comps.names         # list of names
        comps.response      # LinearOperatorCompoundModel or None

    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model to decompose.
    additive : bool, optional
        If ``True``, return additive components (multiplicative terms
        are distributed).  If ``False`` (default), return all leaves.
    deconvolve : bool, optional
        Only meaningful when ``additive=True`` and the model is
        convolved.  If ``True``, strip the operator and return
        intrinsic source components.  Default ``False``.

    Notes
    -----
    For ``LinearOperatorCompoundModel`` (e.g. ``rsp(A + B)``):

    * ``additive=True, deconvolve=False`` →
      ``rsp(A), rsp(B)``  (operator-wrapped, for observed evaluation).
    * ``additive=True, deconvolve=True``  →
      ``A, B``  (source-only, for intrinsic measurements).
    * ``additive=False`` →
      ``A, B, rsp``  (all leaves + Identity marker).

    Examples
    --------
    >>> comps = ModelComponents(model, additive=True)
    >>> for name in comps.names:
    ...     plt.plot(wave, comps[name](wave), label=name)

    >>> comps = ModelComponents(model, additive=True, deconvolve=True)
    >>> comps['blr']   # intrinsic blr, no operator
    """

    def __init__(self, model, additive=False, deconvolve=False):
        super().__init__()
        self._op_info = _get_operator_info(model)
        self._model = _get_source_model(model)
        self._additive_only = additive
        self._deconvolve = bool(deconvolve)
        self._indices = []
        self._names = []
        self._components = []

        if additive:
            self._build_additive()
        else:
            self._build_all()

    @property
    def additive(self):
        """Whether only additive components are included."""
        return self._additive_only

    @property
    def deconvolved(self):
        """Whether additive components are returned without the operator."""
        return self._deconvolve

    @property
    def response(self):
        """The ``LinearOperatorCompoundModel`` wrapping the source, or ``None``."""
        return self._op_info

    @property
    def indices(self):
        """List of integer indices."""
        return self._indices.copy()

    @property
    def names(self):
        """List of component names."""
        return self._names.copy()

    def to_list(self):
        """Return components as a plain list."""
        return self._components.copy()

    def __repr__(self):
        mode = "additive-only" if self._additive_only else "all"
        n = len(self._components)
        names_str = ", ".join(f"'{n}'" for n in self._names[:3])
        if n > 3:
            names_str += f", ... ({n - 3} more)"
        extra = ""
        if self._op_info:
            extra = ", deconvolved" if self._deconvolve else ", convolved"
        return f"ModelComponents({mode}, n={n}, names=[{names_str}]{extra})"

    # ---- builders --------------------------------------------------------

    def _build_all(self):
        """Depth-first leaf extraction; operator appended as Identity marker."""
        idx = [0]
        used_names = set()

        def traverse(submodel):
            if isinstance(submodel, LinearOperatorCompoundModel):
                traverse(submodel.left)
                return
            if isinstance(submodel, CompoundModel):
                traverse(submodel.left)
                traverse(submodel.right)
            else:
                name = self._get_component_name(submodel)
                unique_name = _make_unique_name(name, used_names)
                used_names.add(unique_name)

                self[idx[0]] = submodel
                self[unique_name] = submodel
                self._indices.append(idx[0])
                self._names.append(unique_name)
                self._components.append(submodel)
                idx[0] += 1

        traverse(self._model)

        if self._op_info is not None:
            op_name = self._op_info.name or 'response'
            op_name = _make_unique_name(op_name, used_names)
            marker = Identity(1, name=op_name)

            self[idx[0]] = marker
            self[op_name] = marker
            self._indices.append(idx[0])
            self._names.append(op_name)
            self._components.append(marker)

    def _build_additive(self):
        """Additive decomposition with optional operator wrapping."""
        components = self._expand_additive(self._model)

        # Decide whether to wrap each component with the operator
        wrap = (not self._deconvolve)
        used_names = set()
        for idx, comp in enumerate(components):
            name = self._get_component_name(comp)
            unique_name = _make_unique_name(name, used_names)
            used_names.add(unique_name)

            if wrap:
                out = _wrap_component(comp, self._op_info)
            else:
                out = comp

            self[idx] = out
            self[unique_name] = out
            self._indices.append(idx)
            self._names.append(unique_name)
            self._components.append(out)
    
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

        if isinstance(model, LinearOperatorCompoundModel):
            left_expanded = self._expand_additive(model.left)
            return [model.with_left(comp) for comp in left_expanded]
        
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
        """Name from model: ``model.name``, compound expression, or class name."""
        if isinstance(model, CompoundModel):
            return self._construct_compound_name(model)
        if hasattr(model, 'name') and model.name:
            return model.name
        return type(model).__name__

    def _construct_compound_name(self, model):
        """Build a descriptive name like ``A*B`` from a compound model."""
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


def get_components(model, additive=False, deconvolve=False):
    """
    Decompose a model into named, indexable components.

    For ``LinearOperatorCompoundModel`` (convolved models):

    * ``additive=True, deconvolve=False`` — operator-wrapped additive
      terms (for observed-frame evaluation / plotting).
    * ``additive=True, deconvolve=True``  — intrinsic source additive
      terms with the operator stripped (for line measurements).
    * ``additive=False`` — all source leaves plus an ``Identity``
      marker for the operator.

    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model.
    additive : bool, optional
        If ``True``, return additive components (distribute
        multiplicative terms).  Default ``False``.
    deconvolve : bool, optional
        If ``True`` and ``additive=True``, strip the operator and
        return source-frame components.  Ignored when
        ``additive=False``.  Default ``False``.

    Returns
    -------
    ModelComponents
        Container with ``.names``, ``.indices``, ``.response``,
        and integer / name-based lookups.

    Examples
    --------
    >>> comps = get_components(model, additive=True)
    >>> comps['blr'](wave)          # convolved component

    >>> comps = get_components(model, additive=True, deconvolve=True)
    >>> comps['blr'](wave)          # intrinsic source component

    >>> comps = get_components(model, additive=False)
    >>> comps.response              # operator node or None
    """
    return ModelComponents(model, additive=additive, deconvolve=deconvolve)
