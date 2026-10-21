"""Prepared numerical evaluation shared by the SciPy and Sherpa adapters."""

import numpy as np
from astropy.modeling import CompoundModel

from ..operators.instrument import SpectralResponse, _sample_response
from ..operators.matop import LinearOperatorModel, _apply_matrix
from .utils import _analytic_jacobian_parameter_major


def _contains_pipe(model):
    if not isinstance(model, CompoundModel):
        return False
    return model.op == '|' or _contains_pipe(model.left) or _contains_pipe(model.right)


class _ModelEvaluation:
    """Separate a fixed spectral response from its source for one fit.

    The numerical path supports scalar parameters and an arithmetic
    source. Other models retain their existing Astropy evaluation semantics;
    in particular, custom or parameter-dependent operators are never frozen.
    Parameter constraints and final model updates remain the fitter's concern.
    """

    def __new__(cls, model, x):
        from ..composition import Model
        if isinstance(model, Model):
            return _CompositeEvaluation(model, x)
        return super().__new__(cls)

    def __init__(self, model, x):
        self.model = model
        self.x = x
        self.source = None
        self.derivatives = None
        self.stateless = False
        if not (
            isinstance(model, CompoundModel)
            and model.op == '|'
            and type(model.right) in (LinearOperatorModel, SpectralResponse)
            and not _contains_pipe(model.left)
            and not hasattr(x, 'unit')
            and all(getattr(model, name).size == 1 for name in model.param_names)
        ):
            return

        leaves = model.left.traverse_postorder() if isinstance(model.left, CompoundModel) else [model.left]
        if any(not isinstance(leaf, CompoundModel)
               and not type(leaf).__module__.startswith(('astropy.modeling.', 'prism.modeling.'))
               for leaf in leaves):
            return

        response = model.right
        native = response.x
        if hasattr(native, 'unit'):
            return
        # Reuse the response's sampling convention: fold on the native grid,
        # then interpolate onto observed samples (including masks/subsets).
        self.matrix = response.get_matrix(x) if native is None else response.matrix
        self.sample = None
        if native is not None:
            self.x = native
            if not response._same_grid(x):
                if not response.flexible:
                    raise ValueError('Response grid differs from the fitting grid with flexible=False.')
                self.sample = response._get_interp_cache(x)
        elif self.matrix.shape[1] != len(x):
            raise ValueError('Response matrix and fitting grid have incompatible sizes.')
        self.source = model.left
        self.stateless = not model.has_tied

        # A sum needs no intermediate spectra or derivative concatenations.
        # Keep products, quotients and other compositions on the existing path.
        parts = []
        def collect(component, sign=1):
            if isinstance(component, CompoundModel):
                if component.op not in ('+', '-'):
                    return False
                right_sign = sign if component.op == '+' else -sign
                return collect(component.left, sign) and collect(component.right, right_sign)
            if component.n_inputs != 1 or component.n_outputs != 1 or not callable(component.fit_deriv):
                return False
            parts.append((component, sign))
            return True

        if collect(self.source) and len(parts) > 1:
            self.derivatives = []
            offset = 0
            for component, sign in parts:
                stop = offset + len(component.param_names)
                self.derivatives.append((component, slice(offset, stop), sign))
                offset = stop

    def _fold(self, values):
        # Apply the same matrix to a spectrum or all Jacobian columns at once.
        folded = _apply_matrix(self.matrix, values)
        if self.sample is None:
            return folded
        return _sample_response(folded, self.sample)

    def __call__(self, parameters):
        if self.source is None:
            return self.model(self.x)
        flux = self.source.evaluate(self.x, *parameters)
        return self._fold(flux)

    def fit_deriv(self, parameters):
        if self.source is None:
            return self.model.fit_deriv(self.x, *parameters)
        if self.derivatives is None:
            source_deriv = _analytic_jacobian_parameter_major(
                self.source, self.source.fit_deriv(self.x, *parameters))
        else:
            source_deriv = np.empty((len(parameters), len(self.x)))
            for component, span, sign in self.derivatives:
                derivative = component.fit_deriv(self.x, *parameters[span])
                source_deriv[span] = _analytic_jacobian_parameter_major(component, derivative)
                if sign < 0:
                    source_deriv[span] *= -1

        folded = self._fold(source_deriv.T).T
        return folded if self.model.col_fit_deriv else folded.T


from dataclasses import dataclass


@dataclass
class _EvaluationNode:
    op: str | None
    x: object
    indices: object = None
    select: object = None
    component: object = None
    left: object = None
    right: object = None
    operator: object = None
    output: object = None
    matrix: object = None
    cache: object = None
    right_indices: object = None
    linear: bool = False
    function: object = None
    terms: object = None
    slots: object = None


class _CompositeEvaluation:
    """Prepare one native expression; keep operator action explicit.

    Numeric kernels receive a parameter vector. Ordinary evaluation uses the
    public leaf adapter for units, but traverses exactly the same expression.
    Fixed linear operators transform Jacobian columns in a single block.
    Parameterized operators and general transforms use a whole-model numerical
    Jacobian rather than omitting their input or parameter sensitivities.
    """
    def __init__(self, model, x, ordinary=False):
        from ..composition import Model
        if np.ndim(x) != 1:
            raise ValueError('A native spectral model requires a one-dimensional coordinate grid.')
        self.model = model
        self.ordinary = ordinary
        self.stateless = not model.has_tied and all(
            type(c).__module__.startswith(('astropy.modeling.', 'prism.modeling.')) for c in model._components)
        self.analytic = model.fit_deriv is not None
        self.source = None
        self.derivatives = None
        identities = {id(getattr(model, name)): index for index, name in enumerate(model.param_names)}
        if len(identities) != len(model.param_names):
            raise ValueError('Repeated parameter identities are ambiguous in the native model.')
        self.root = self._prepare(model, x, identities)
        self.needs_values = self._needs_values(self.root)

    @staticmethod
    def _selector(indices):
        if len(indices) and np.array_equal(indices, np.arange(indices[0], indices[-1]+1)):
            return slice(int(indices[0]), int(indices[-1])+1)
        return indices

    def _prepare(self, model, x, identities):
        from ..composition import _OPERATIONS
        from ..operators.instrument import SpectralResponse
        from ..operators.matop import LinearOperatorModel
        node = _EvaluationNode(model.op, x)
        if model.op is None:
            component = model.component
            if not self.ordinary and hasattr(x, 'unit'):
                units = component.input_units or {}
                unit = units.get(component.inputs[0])
                equivalencies = (component.input_units_equivalencies or {}).get(component.inputs[0], [])
                node.x = x.value if unit is None else x.to_value(unit, equivalencies=equivalencies)
            indices = np.array([identities[id(getattr(component,n))] for n in component.param_names], dtype=int)
            node.indices = indices
            node.select = self._selector(indices)
            node.component = component
            return node
        if model.op == '|':
            response = model.right
            if isinstance(response, LinearOperatorModel):
                same = response._same_grid(x)
                if not same and not response.flexible:
                    raise ValueError('Operator output grid differs from the requested coordinates.')
                native = response._native_input(x)
                grid = x if native is None else native
                left = self._prepare(model.left, grid, identities)
                right = np.array([identities[id(getattr(response,n))] for n in response.param_names], dtype=int)
                fixed = type(response) in (LinearOperatorModel, SpectralResponse)
                output = response.xout if response.xout is not None else x
                matrix = response.get_matrix(output) if fixed else None
                cache = None if same else response._get_interp_cache(x)
                node.left = left
                node.operator = response
                node.output = output
                node.matrix = matrix
                node.cache = cache
                node.right_indices = right
                node.linear = True
            else:
                # General profile transforms receive coordinates and the full
                # source spectrum. This contract preserves the requested grid.
                left = self._prepare(model.left, x, identities)
                right = np.array([identities[id(getattr(response.component,n))]
                                  for n in response.component.param_names], dtype=int)
                node.left = left
                node.operator = response.component
                node.right_indices = right
                node.linear = False
            node.indices = np.sort(np.r_[left.indices, right])
            return node
        left = self._prepare(model.left, x, identities)
        right = self._prepare(model.right, x, identities)
        indices = np.sort(np.r_[left.indices, right.indices])
        node.left = left
        node.right = right
        node.indices = indices
        node.function = _OPERATIONS[model.op]
        if model.op in ('+', '-'):
            terms = []
            for child,sign in [(left,1), (right,1 if model.op == '+' else -1)]:
                if child.terms is None:
                    terms.append((child, sign))
                else:
                    terms.extend((term, sign*s) for term,s,_ in child.terms)
            node.terms = [(term,sign,self._selector(np.searchsorted(indices,term.indices))) for term,sign in terms]
        else:
            node.slots = (self._selector(np.searchsorted(indices,left.indices)),
                             self._selector(np.searchsorted(indices,right.indices)))
        return node

    @staticmethod
    def _needs_values(node):
        if node.op is None:
            return False
        if node.op == '|':
            return _CompositeEvaluation._needs_values(node.left)
        return node.op not in ('+','-') or _CompositeEvaluation._needs_values(node.left) or _CompositeEvaluation._needs_values(node.right)

    @staticmethod
    def _fold(node, values, p):
        matrix = node.matrix
        if matrix is None:
            matrix = node.operator.get_matrix(node.output, *p[node.right_indices])
        result = _apply_matrix(matrix, values)
        if node.cache is None:
            return result
        unit = getattr(result, 'unit', None)
        array = result.to_value(unit) if unit is not None else result
        sampled = _sample_response(array, node.cache)
        return sampled*unit if unit is not None else sampled

    def _value(self, node, p, cache):
        op = node.op
        if op is None:
            component = node.component
            result = component(node.x) if self.ordinary else component.evaluate(node.x, *p[node.select])
        elif op == '|':
            source = self._value(node.left, p, cache)
            if node.linear:
                result = self._fold(node, source, p)
            else:
                transform = node.operator
                result = transform(node.x, source) if self.ordinary else transform.evaluate(node.x, source, *p[node.right_indices])
                if np.shape(result) != np.shape(source):
                    raise ValueError('A general profile transform must preserve its input grid and shape.')
        else:
            left = self._value(node.left, p, cache)
            right = self._value(node.right, p, cache)
            with np.errstate(divide='raise', invalid='raise', over='raise'):
                result = node.function(left, right)
        if cache is not None:
            cache[id(node)] = result
        return result

    def __call__(self, parameters):
        return self._value(self.root, parameters, None)

    def _derivative(self, node, p, values):
        if node.op is None:
            component = node.component
            if not component.param_names:
                return np.empty((0,len(node.x)))
            return _analytic_jacobian_parameter_major(component, component.fit_deriv(node.x, *p[node.select]))
        if node.op == '|':
            return self._fold(node, self._derivative(node.left, p, values).T, p).T
        result = np.empty((len(node.indices),len(node.x)))
        if node.terms is not None:
            for term,sign,slots in node.terms:
                result[slots] = self._derivative(term, p, values)
                if sign < 0:
                    result[slots] *= -1
            return result
        left,right = node.left,node.right
        a,b = values[id(left)],values[id(right)]
        ja,jb = self._derivative(left,p,values),self._derivative(right,p,values)
        if node.op == '*':
            result[node.slots[0]] = ja*b
            result[node.slots[1]] = jb*a
        elif node.op == '/':
            result[node.slots[0]] = ja/b
            result[node.slots[1]] = -jb*a/b**2
        return result

    def fit_deriv(self, parameters):
        if not self.analytic:
            raise NotImplementedError('Use a whole-model numerical Jacobian for this transform/expression.')
        values = {}
        if self.needs_values:
            self._value(self.root, parameters, values)
        return self._derivative(self.root, parameters, values)
