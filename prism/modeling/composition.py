"""Native one-dimensional model expressions with explicit Astropy adapters.

A Model owns an expression and references to its components, not another copy of
its parameters. Copies clone the complete graph. Numerical preparation belongs
to a fit/evaluation, so workers never share mutable prepared state.
"""
import copy
import numbers
from types import MappingProxyType
import operator
import numpy as np
from astropy import units as u
from astropy.modeling import Model as AstropyModel, CompoundModel
from astropy.modeling.models import Const1D

_OPERATIONS = {'+': operator.add, '-': operator.sub, '*': operator.mul,
               '/': operator.truediv, '**': operator.pow}


class Model:
    """A native spectral expression built from adapted components.

    Use ``Model.from_astropy(component)`` or convert a complete existing
    arithmetic expression. One-input components are spectra or factors;
    two-input components implement ``evaluate(x, flux, *parameters)`` and can
    appear only on the right of ``|``. These general transforms preserve the
    grid and use a numerical Jacobian. LinearOperatorModel instances instead
    carry explicit input/output grids and may be rectangular.

    Addition, subtraction, multiplication, division and powers preserve their
    parentheses. ``source | operator`` applies the operator after evaluating
    its source. Components retain identity; assemble first and copy the whole
    model to obtain an independent graph. Repeated component instances are
    rejected. Native fitters currently expect numeric data in consistent native
    units; ordinary evaluation also accepts Quantity coordinates.
    """
    n_inputs = 1
    n_outputs = 1
    col_fit_deriv = True
    fittable = True
    linear = False

    def __init__(self, component=None, *, op=None, left=None, right=None, name=None):
        if op is None and not isinstance(component, AstropyModel):
            raise TypeError('Use Model.from_astropy(component) to construct a leaf.')
        if op is not None and (op not in (*_OPERATIONS, '|') or not isinstance(left, Model)):
            raise ValueError('Invalid native model expression.')
        object.__setattr__(self, 'n_inputs', component.n_inputs if op is None else 1)
        object.__setattr__(self, 'component', component)
        object.__setattr__(self, 'op', op)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)
        object.__setattr__(self, '_name', name)
        components = tuple(self._walk())
        parameterized = [c for c in components if c.param_names]
        if len({id(c) for c in parameterized}) != len(parameterized):
            raise ValueError('Repeated component instances are ambiguous; use independent components.')
        owners = {}
        for index, leaf in enumerate(components):
            for local in leaf.param_names:
                parameter = getattr(leaf, local)
                if parameter.size != 1:
                    raise ValueError('Native spectral composition requires scalar parameters.')
                key = local if len(components) == 1 else f'{local}_{index}'
                owners[key] = (leaf, local)
        object.__setattr__(self, '_components', components)
        object.__setattr__(self, '_owners', owners)
        object.__setattr__(self, 'param_names', tuple(owners))

    @classmethod
    def from_astropy(cls, model):
        """Adapt a leaf or arithmetic expression, retaining parameter identity.

        Parameters
        ----------
        model : astropy.modeling.Model or Model
            A one-output model with scalar parameters. Two-input leaves are
            profile transforms, not spectral source components. Unsupported
            Astropy operators are rejected instead of retained opaquely.
        """
        from .interop.astropy import _from_astropy
        return _from_astropy(model, cls)

    def _walk(self):
        if self.op is None:
            yield self.component
        else:
            yield from self.left._walk()
            if isinstance(self.right, Model):
                yield from self.right._walk()
            else:
                yield self.right

    def _expressions(self):
        yield self
        if self.op is not None:
            yield from self.left._expressions()
            if isinstance(self.right, Model):
                yield from self.right._expressions()

    def _spectrum(self):
        if self.op is None and self.component.n_inputs != 1:
            raise TypeError('A two-input transform must follow a spectrum with |.')

    @staticmethod
    def _operand(value):
        if isinstance(value, Model):
            value._spectrum()
            return value
        if isinstance(value, AstropyModel):
            result = Model.from_astropy(value)
            result._spectrum()
            return result
        if isinstance(value, (numbers.Real, u.Quantity)):
            if np.ndim(value) or not np.isfinite(value):
                raise ValueError('Model constants must be finite scalars.')
            return Model.from_astropy(Const1D(value, fixed={'amplitude': True}))
        raise TypeError('Combine adapted models or finite scalar constants.')

    def _binary(self, other, op, reverse=False):
        self._spectrum()
        other = self._operand(other)
        left, right = (other, self) if reverse else (self, other)
        return Model(op=op, left=left, right=right)

    def __add__(self, other): return self._binary(other, '+')
    def __radd__(self, other): return self._binary(other, '+', True)
    def __sub__(self, other): return self._binary(other, '-')
    def __rsub__(self, other): return self._binary(other, '-', True)
    def __mul__(self, other): return self._binary(other, '*')
    def __rmul__(self, other): return self._binary(other, '*', True)
    def __truediv__(self, other): return self._binary(other, '/')
    def __rtruediv__(self, other): return self._binary(other, '/', True)
    def __pow__(self, other): return self._binary(other, '**')
    def __rpow__(self, other): return self._binary(other, '**', True)
    def __neg__(self): return self * -1

    def __or__(self, other):
        from .operators.matop import LinearOperatorModel
        self._spectrum()
        general = isinstance(other, Model) and other.op is None and other.component.n_inputs == 2
        if not isinstance(other, LinearOperatorModel) and not general:
            raise TypeError('Use a linear operator or an adapted evaluate(x, flux, *parameters) transform.')
        return Model(op='|', left=self, right=other)

    def __getattr__(self, name):
        owners = self.__dict__.get('_owners', {})
        if name in owners:
            component, local = owners[name]
            return getattr(component, local)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        owners = self.__dict__.get('_owners', {})
        if name in owners:
            component, local = owners[name]
            setattr(component, local, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def name(self):
        return self.component.name if self.op is None else self._name

    @name.setter
    def name(self, value):
        if self.op is None:
            self.component.name = value
        else:
            self._name = value

    @property
    def submodel_names(self):
        return tuple(component.name for component in self._components)

    @property
    def n_submodels(self):
        return len(self._components)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._components[key]
        matches = [component for component in self._components if component.name == key]
        if len(matches) != 1:
            raise KeyError(f'Expected one component named {key!r}; found {len(matches)}.')
        return matches[0]

    @property
    def parameters(self):
        return np.array([getattr(self, name).value for name in self.param_names], dtype=float)

    @parameters.setter
    def parameters(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.param_names),):
            raise ValueError('Parameter vector does not match the native model.')
        for name, value in zip(self.param_names, values):
            getattr(self, name).value = value

    @property
    def fixed(self):
        return MappingProxyType({name: getattr(self, name).fixed for name in self.param_names})

    @property
    def tied(self):
        return MappingProxyType({name: getattr(self, name).tied for name in self.param_names})

    @property
    def bounds(self):
        return MappingProxyType({name: getattr(self, name).bounds for name in self.param_names})

    @property
    def has_tied(self):
        return any(getattr(self, name).tied for name in self.param_names)

    @property
    def covariance(self):
        return getattr(self, '_param_cov', None)

    @covariance.setter
    def covariance(self, value):
        self._param_cov = None if value is None else np.asarray(value)

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, x, flux=None):
        if self.op is None and self.component.n_inputs == 2:
            if flux is None:
                raise TypeError('A profile transform requires x and flux, or source | transform.')
            return self.component(x, flux)
        if flux is not None:
            raise TypeError('A spectral model accepts one coordinate array.')
        from .fitting.utils import apply_tied
        from .fitting.evaluation import _CompositeEvaluation
        apply_tied(getattr(self, '_evaluation_parent', self))
        scalar = np.ndim(x) == 0
        result = _CompositeEvaluation(self, np.atleast_1d(x), ordinary=True)(self.parameters)
        return result[0] if scalar else result

    def evaluate(self, x, *parameters):
        from .fitting.evaluation import _CompositeEvaluation
        return _CompositeEvaluation(self, x)(np.asarray(parameters))

    @property
    def fit_deriv(self):
        # An ordinary model's parameter derivatives do not describe its action
        # on an input profile. Do not silently omit operator/source derivatives.
        if any(node.op == '**' for node in self._expressions()):
            return None
        for component in self._components:
            if component.n_inputs == 2 or (hasattr(component, 'get_matrix') and component.param_names):
                return None
            if component.param_names and not callable(component.fit_deriv):
                return None
        def derivative(x, *parameters):
            from .fitting.evaluation import _CompositeEvaluation
            return _CompositeEvaluation(self, x).fit_deriv(np.asarray(parameters))
        return derivative

    @property
    def expression(self):
        from .display import get_model_expression
        return get_model_expression(self)

    def show(self, mode='auto', option=None):
        from .display import show_model
        return show_model(self, mode=mode, option=option)

    def _repr_html_(self):
        from .display import format_model_html
        return format_model_html(self)

    def save(self, filename, overwrite=False, parfilter=None, std=True, model=True,
             operator_file=None, **custom_exts):
        from .io import model_to_fits
        return model_to_fits(self, filename, overwrite=overwrite, parfilter=parfilter,
                             std=std, model_def=model, operator_file=operator_file, **custom_exts)

    to_fits = save

    def __repr__(self):
        if self.op is None:
            return f'Model({self.name or type(self.component).__name__})'
        right = repr(self.right) if isinstance(self.right, Model) else self.right.name or type(self.right).__name__
        return f'({self.left!r} {self.op} {right})'


def _is_compound(model):
    return isinstance(model, CompoundModel) or isinstance(model, Model) and model.op is not None
