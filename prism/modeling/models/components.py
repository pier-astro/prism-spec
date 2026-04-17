"""
Component decomposition for compound models.

Decomposes ``CompoundModel`` trees into named, indexable parts via
``get_components``, returning a ``ModelComponents`` container.

For linear-operator pipe models (convolved models, ``h = M @ f``):

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

from ..operators.matop import is_linear_operator_pipe


def _get_operator_info(model):
    """Return the top-level linear-operator pipe node, or ``None``."""
    if is_linear_operator_pipe(model):
        return model
    return None


def _wrap_component(component, op_info):
    """Wrap *component* in the same linear operator if *op_info* is set."""
    if op_info is None:
        return component
    wrapped = component | op_info.right
    return wrapped


def _is_negative_const(model):
    """Return True when *model* is the synthetic Const1D(-1) sign carrier."""
    return isinstance(model, Const1D) and np.isclose(float(model.amplitude.value), -1.0)


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
        comps.response      # linear-operator pipe or None

    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model to decompose.
    additive : bool, optional
        If ``True``, return additive components (multiplicative terms
        are distributed).  If ``False`` (default), return all leaves.
    deconvolve : bool, optional
        Only meaningful when ``additive=True`` and the top-level model is
        a linear-operator pipe. If ``True``, strip only that first operator
        layer and return the additive decomposition of the left-hand source.
        Nested pipe nodes are left untouched. Default ``False``.

    Notes
    -----
        For linear-operator pipe models (e.g. ``(A + B) | rsp``):

    * ``additive=True, deconvolve=False`` →
            ``A|rsp, B|rsp``  (operator-wrapped, for observed evaluation).
    * ``additive=True, deconvolve=True``  →
            ``A, B``  (first operator layer stripped for intrinsic measurements).
    * ``additive=False`` →
            ``A, B, rsp``  (all leaves + operator markers).

        The deconvolution logic is intentionally shallow: it removes only a
        top-level linear operator. For ``(gauss | linop1) | linop2``,
        ``deconvolve=True`` removes ``linop2`` only. For
        ``(gauss | linop) + other``, ``deconvolve=True`` has no effect because
        the top-level node is not itself a linear-operator pipe.

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
        self._original_model = model
        self._op_info = _get_operator_info(model)
        self._model = model.left if self._op_info is not None else model
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
        """The linear-operator pipe wrapping the source, or ``None``."""
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
        """Depth-first leaf extraction; pipe operators become Identity markers."""
        idx = [0]
        used_names = set()

        def add_item(component, name):
            unique_name = _make_unique_name(name, used_names)
            used_names.add(unique_name)

            self[idx[0]] = component
            self[unique_name] = component
            self._indices.append(idx[0])
            self._names.append(unique_name)
            self._components.append(component)
            idx[0] += 1

        def add_operator_marker(pipe_model):
            op_name = getattr(pipe_model.right, 'name', None) or getattr(pipe_model, 'name', None) or 'linop'
            add_item(Identity(1, name=op_name), op_name)

        def traverse(submodel):
            if is_linear_operator_pipe(submodel):
                traverse(submodel.left)
                add_operator_marker(submodel)
                return
            if isinstance(submodel, CompoundModel):
                traverse(submodel.left)
                traverse(submodel.right)
            else:
                add_item(submodel, self._get_component_name(submodel))

        traverse(self._original_model)

    def _build_additive(self):
        """Additive decomposition with optional operator wrapping."""
        components = self._expand_additive(self._model)

        wrap = (self._op_info is not None and not self._deconvolve)
        used_names = set()
        for idx, comp in enumerate(components):
            if wrap:
                out = _wrap_component(comp, self._op_info)
            else:
                out = comp

            name = self._get_component_name(out)
            unique_name = _make_unique_name(name, used_names)
            used_names.add(unique_name)

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

        if is_linear_operator_pipe(model):
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
        """Name from model: ``model.name``, compound expression, or class name."""
        if isinstance(model, CompoundModel):
            return self._construct_compound_name(model)
        if hasattr(model, 'name') and model.name:
            return model.name
        return type(model).__name__

    def _flatten_product_names(self, model):
        """Return multiplicative factor names, omitting synthetic -1 factors."""
        if _is_negative_const(model):
            return []
        if isinstance(model, CompoundModel) and model.op == '*':
            return self._flatten_product_names(model.left) + self._flatten_product_names(model.right)
        return [self._get_component_name(model)]

    def _construct_compound_name(self, model):
        """Build a descriptive name like ``A*B`` from a compound model."""
        if is_linear_operator_pipe(model):
            left_name = self._get_component_name(model.left)
            if isinstance(model.left, CompoundModel):
                left_name = f"({left_name})"
            right_name = getattr(model.right, 'name', None) or getattr(model, 'name', None) or 'linop'
            return f"{left_name}|{right_name}"

        left_name = self._get_component_name(model.left)
        right_name = self._get_component_name(model.right)
        
        if model.op == '*':
            factors = self._flatten_product_names(model)
            return '*'.join(factors) if factors else 'Const1D'
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


def get_components(model, additive=True, deconvolve=False):
    """
    Decompose a model into named, indexable components.

        For linear-operator pipe models (convolved models):

        * ``additive=True, deconvolve=False`` — first-layer additive
            terms, wrapped again by a top-level linear operator when present.
        * ``additive=True, deconvolve=True``  — same first-layer additive
            terms, but only the top-level operator is stripped.
        * ``additive=False`` — all leaves in depth-first order, with an
            ``Identity`` marker for each linear operator node.

    Parameters
    ----------
    model : astropy.modeling.Model
        Single or compound model.
    additive : bool, optional
        If ``True``, return additive components (distribute
        multiplicative terms).  Default ``False``.
    deconvolve : bool, optional
        If ``True`` and ``additive=True``, strip only a top-level linear
        operator and return the additive decomposition of its left-hand
        source. Ignored when ``additive=False``. Default ``False``.

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
