"""Declarative arithmetic links for Prism parameters, without global patches.

Expressions hold parameter references in memory. Session storage replaces those
references with names in the complete model's parameter order, never Python code.
"""
import operator
import numbers
import numpy as np
from astropy import units as u
from astropy.modeling import Parameter as AstropyParameter

_OPERATIONS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
               'div': operator.truediv, 'pow': operator.pow, 'neg': operator.neg}


class _Arithmetic:
    __array_priority__ = 10000

    def __add__(self, other): return _Expression('add', self, other)
    def __radd__(self, other): return _Expression('add', other, self)
    def __sub__(self, other): return _Expression('sub', self, other)
    def __rsub__(self, other): return _Expression('sub', other, self)
    def __mul__(self, other): return _Expression('mul', self, other)
    def __rmul__(self, other): return _Expression('mul', other, self)
    def __truediv__(self, other): return _Expression('div', self, other)
    def __rtruediv__(self, other): return _Expression('div', other, self)
    def __pow__(self, other): return _Expression('pow', self, other)
    def __rpow__(self, other): return _Expression('pow', other, self)
    def __neg__(self): return _Expression('neg', self)

    @property
    def link(self):
        return self.tied if isinstance(self.tied, _Expression) else None

    @link.setter
    def link(self, value):
        if value is None:
            current = self.value
            self.tied = False
            self.value = current
            return
        expression = _Expression('ref', value)
        expression.target = self
        _check_cycle(self, expression)
        _linked_value(self, expression)  # validate before changing the constraint
        self.tied = expression



def _operand(value):
    if isinstance(value, (_Expression, AstropyParameter, _Scalar)):
        return value
    if isinstance(value, u.Quantity):
        if not value.isscalar or not np.isfinite(value.value):
            raise ValueError('Link constants must be finite scalars.')
        return value.copy()
    if isinstance(value, numbers.Real) and np.isfinite(value):
        return float(value)
    raise TypeError('Links accept parameters, finite scalar constants and arithmetic expressions.')


def _value(value):
    if isinstance(value, _Expression):
        return value.value
    if isinstance(value, (AstropyParameter, _Scalar)):
        unit = value.unit or getattr(value, '_prism_evaluation_unit', None)
        return value.value * unit if unit is not None else value.value
    return value


class _Expression(_Arithmetic):
    def __init__(self, operation, *operands):
        self.operation = operation
        self.operands = tuple(_operand(value) for value in operands)

    @property
    def value(self):
        if self.operation == 'ref':
            return _value(self.operands[0])
        with np.errstate(all='raise'):
            return _OPERATIONS[self.operation](*(_value(value) for value in self.operands))

    def __repr__(self):
        def label(value):
            return value.name if isinstance(value, (AstropyParameter, _Scalar)) else repr(value)
        if self.operation == 'ref':
            return label(self.operands[0])
        if self.operation == 'neg':
            return f'(-{label(self.operands[0])})'
        symbols = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
        left, right = self.operands
        return f'({label(left)} {symbols[self.operation]} {label(right)})'

    def references(self):
        for value in self.operands:
            if isinstance(value, (AstropyParameter, _Scalar)):
                yield value
            elif isinstance(value, _Expression):
                yield from value.references()

    def __call__(self, model):
        return _linked_value(self.target, self) if hasattr(self, 'target') else self.value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype) if copy is None else np.array(self.value, dtype=dtype, copy=copy)

    def __float__(self):
        return float(self.value)


def _linked_value(parameter, expression):
    value = expression.value
    if isinstance(value, u.Quantity):
        unit = parameter.unit or getattr(parameter, '_prism_evaluation_unit', None)
        value = value.to_value(unit or u.dimensionless_unscaled)
    array = np.asarray(value, dtype=float)
    if array.shape != parameter.shape:
        raise ValueError(f'Link for {parameter.name} must preserve parameter shape {parameter.shape}.')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'Link for {parameter.name} produced a non-finite value.')
    return np.float64(array.item()) if array.ndim == 0 else array


def _check_cycle(parameter, expression):
    def visit(current, active):
        if id(current) in active:
            raise ValueError('Cyclic parameter link.')
        link = expression if current is parameter else current.tied
        if isinstance(link, _Expression):
            for dependency in link.references():
                visit(dependency, active | {id(current)})
    visit(parameter, set())


class _Parameter(_Arithmetic, AstropyParameter):
    """Astropy adapter exposing Prism's declarative ``parameter.link`` syntax.

    Arithmetic builds expressions; use ``.value`` or ``.quantity`` for immediate
    numerical arithmetic. Setting ``link=None`` removes the constraint and keeps
    its current value. Arbitrary callable constraints remain available via tied.
    """
    @property
    def value(self):
        if isinstance(self.tied, _Expression):
            return _linked_value(self, self.tied)
        return AstropyParameter.value.fget(self)

    @value.setter
    def value(self, value):
        AstropyParameter.value.fset(self, value)


def _ordered_ties(model, items):
    """Topologically order declared dependencies; retain legacy callable order."""
    parameters = {id(getattr(model, name)): name for name in model.param_names}
    if len(parameters) != len(model.param_names) and any(isinstance(item[0], _Expression) for item in items):
        raise ValueError('Linked models cannot contain repeated component instances; use independent components.')
    by_parameter = {id(item[2]): item for item in items}
    result, active, done = [], set(), set()
    def visit(item):
        expression, _, parameter = item
        identity = id(parameter)
        if identity in done:
            return
        if identity in active:
            raise ValueError('Cyclic parameter link.')
        active.add(identity)
        if isinstance(expression, _Expression):
            for dependency in expression.references():
                if id(dependency) not in parameters:
                    raise ValueError(f'Link for {parameter.name} references a parameter outside the assembled model.')
                if id(dependency) in by_parameter:
                    visit(by_parameter[id(dependency)])
        active.remove(identity)
        done.add(identity)
        result.append(item)
    for item in items:
        visit(item)
    return result


def _encode_links(model):
    names = {id(getattr(model, name)): name for name in model.param_names}
    def encode(value):
        if isinstance(value, (AstropyParameter, _Scalar)):
            if id(value) not in names:
                raise ValueError('Cannot save a link to a parameter outside the assembled model.')
            return {'parameter': names[id(value)]}
        if isinstance(value, _Expression):
            return {'operation': value.operation, 'operands': [encode(v) for v in value.operands]}
        if isinstance(value, u.Quantity):
            return {'constant': float(value.value), 'unit': value.unit.to_string()}
        return {'constant': float(value)}
    return {name: encode(getattr(model, name).tied) for name in model.param_names
            if isinstance(getattr(model, name).tied, _Expression)}


def _restore_links(model, links):
    def decode(state):
        if set(state) == {'parameter'}:
            name = state['parameter']
            if name not in model.param_names:
                raise ValueError(f'Unknown link parameter: {name!r}.')
            return getattr(model, name)
        if set(state) in ({'constant'}, {'constant', 'unit'}):
            value = _operand(state['constant'])
            return value * u.Unit(state['unit']) if 'unit' in state else value
        if set(state) == {'operation', 'operands'}:
            operation, operands = state['operation'], state['operands']
            arity = 1 if operation in ('neg', 'ref') else 2
            if operation not in {*_OPERATIONS, 'ref'} or len(operands) != arity:
                raise ValueError('Invalid parameter-link operation.')
            return _Expression(operation, *(decode(value) for value in operands))
        raise ValueError('Invalid parameter-link expression.')
    restored = {}
    for name, state in links.items():
        if name not in model.param_names:
            raise ValueError(f'Unknown linked parameter: {name!r}.')
        expression = decode(state)
        if not isinstance(expression, _Expression):
            raise ValueError('A saved parameter link must contain an expression root.')
        restored[name] = expression
    # Install together, then validate the full graph before evaluating any value.
    for name, expression in restored.items():
        expression.target = getattr(model, name)
        getattr(model, name).tied = expression
    items = [(getattr(model, name).tied, None, getattr(model, name)) for name in model.param_names if getattr(model, name).tied]
    for expression, _, parameter in _ordered_ties(model, items):
        if isinstance(expression, _Expression):
            parameter.value = _linked_value(parameter, expression)


class _Scalar(_Arithmetic):
    """Native scalar state; shares link expressions with the Astropy adapter."""

    size = 1
    shape = ()
    unit = None

    def __init__(self, name, value):
        self.name = name
        self.fixed = False
        self.tied = False
        self.bounds = (None, None)
        self.value = value

    @property
    def value(self):
        if isinstance(self.tied, _Expression):
            return _linked_value(self, self.tied)
        return self._value

    @value.setter
    def value(self, value):
        if isinstance(value, u.Quantity):
            raise TypeError('Native parameters require numeric values in explicit, consistent units.')
        array = np.asarray(value, dtype=float)
        if array.shape != () or not np.isfinite(array):
            raise ValueError('A native parameter requires one finite scalar value.')
        self._value = float(array)

    @property
    def min(self):
        return self.bounds[0]

    @property
    def max(self):
        return self.bounds[1]
