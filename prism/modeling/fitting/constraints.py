"""Compile declared parameter bounds for solvers that accept independent boxes.

The model retains the complete link expressions and original bounds. This module
is the capability requested by the current numerical adapters, not a restriction
on what a future constrained solver may read from the same prepared problem.
"""
import numpy as np
from astropy import units as u
from astropy.modeling import Parameter

from ..parameters import _Expression, _Scalar


def _number(value, unit=None):
    if isinstance(value, u.Quantity):
        value = value.to_value(unit or u.dimensionless_unscaled)
    return np.asarray(value, dtype=float)


def _affine_form(value, nodes, initial, cache):
    """Return constant and free-node coefficients, substituting fixed values.

    Arrays are batch controls, never coordinate samples. Keeping them here lets
    multifit validate mapped bounds and fixed starts without materializing models.
    """
    if isinstance(value, (Parameter, _Scalar)):
        identity = id(value)
        if identity not in nodes:
            raise ValueError('A bounded link references a parameter outside the assembled model.')
        if identity in cache:
            return cache[identity]
        name = nodes[identity]
        unit = value.unit or getattr(value, '_prism_evaluation_unit', None)
        if value.tied:
            if not isinstance(value.tied, _Expression):
                raise NotImplementedError('opaque callable ties do not declare their algebra')
            constant, coefficients = _affine_form(value.tied, nodes, initial, cache)
            constant = _number(constant, unit)
            coefficients = {key: _number(item, unit) for key, item in coefficients.items()}
            if unit is not None:
                constant = constant*unit
                coefficients = {key: item*unit for key, item in coefficients.items()}
        elif value.fixed:
            constant = initial.get(name, value.value)
            constant = constant if unit is None else constant*unit
            coefficients = {}
        else:
            if value.size != 1:
                raise NotImplementedError('bounded links require scalar parameter nodes')
            constant, coefficients = 0., {identity: 1. if unit is None else 1.*unit}
        cache[identity] = constant, coefficients
        return cache[identity]
    if not isinstance(value, _Expression):
        return value, {}
    left, coefficients = _affine_form(value.operands[0], nodes, initial, cache)
    if value.operation == 'ref':
        return left, coefficients
    if value.operation == 'neg':
        return -left, {key: -item for key, item in coefficients.items()}
    right, other = _affine_form(value.operands[1], nodes, initial, cache)
    if value.operation in ('add', 'sub'):
        sign = 1. if value.operation == 'add' else -1.
        result = coefficients.copy()
        for key, item in other.items():
            result[key] = result.get(key, 0.)+sign*item
        return left+sign*right, {key: item for key, item in result.items() if np.any(item != 0)}
    if value.operation == 'mul':
        if coefficients and other:
            raise NotImplementedError('products of free parameters require a non-linear constraint')
        result = {key: item*right for key, item in coefficients.items()}
        result.update({key: item*left for key, item in other.items()})
        return left*right, {key: item for key, item in result.items() if np.any(item != 0)}
    if value.operation == 'div':
        if other:
            raise NotImplementedError('division by a free parameter requires a non-linear constraint')
        if np.any(right == 0):
            raise ValueError('Division by zero in a bounded link.')
        return left/right, {key: item/right for key, item in coefficients.items()}
    if value.operation == 'pow':
        if other or coefficients and not (np.all(right == 0) or np.all(right == 1)):
            raise NotImplementedError('powers of free parameters require a non-linear constraint')
        if coefficients:
            return (1., {}) if np.all(right == 0) else (left, coefficients)
        with np.errstate(all='raise'):
            return left**right, {}
    raise NotImplementedError(f'unsupported bounded-link operation {value.operation!r}')


def _box_limits(model, *, initial=None, bounds=None):
    """Intersect free bounds with exact affine linked bounds, before optimization.

    Optional overrides are scalar or matching batch arrays and use model names.
    No model or caller controls are changed. Unsupported bounded links raise;
    links without finite bounds retain their existing arbitrary-expression path.
    """
    from .utils import _get_tied_info
    _get_tied_info(model)  # validate scope and cycles even outside a fitter
    initial, bounds = initial or {}, bounds or {}
    nodes = {id(getattr(model, name)): name for name in model.param_names}
    starts, limits, cache = {}, {}, {}
    for name in model.param_names:
        parameter = getattr(model, name)
        value = np.asarray(initial.get(name, parameter.value), dtype=float)
        starts[name] = np.where(np.isfinite(value), value, parameter.value)
        lower, upper = bounds.get(name, parameter.bounds)
        lower = np.asarray(-np.inf if lower is None else lower, dtype=float)
        upper = np.asarray(np.inf if upper is None else upper, dtype=float)
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError(f'Incompatible or invalid bounds for {name!r}.')
        limits[name] = lower, upper

    for name in model.param_names:
        parameter = getattr(model, name)
        if not parameter.tied:
            continue
        lower, upper = limits[name]
        if np.all(np.isneginf(lower)) and np.all(np.isposinf(upper)):
            continue
        try:
            constant, coefficients = _affine_form(parameter, nodes, starts, cache)
            unit = parameter.unit or getattr(parameter, '_prism_evaluation_unit', None)
            constant = _number(constant, unit)
            if len(coefficients) > 1:
                raise NotImplementedError('the bound couples several free parameters')
        except NotImplementedError as error:
            raise NotImplementedError(
                f'Bounds on linked parameter {name!r} cannot be represented as independent intervals: '
                f'{error}. Reparameterize the model or remove a redundant target bound; '
                'the bound will not be ignored.') from error
        if not np.all(np.isfinite(constant)):
            raise ValueError(f'Non-finite constant in bounded link {name!r}.')
        if not coefficients:
            if np.any(constant < lower) or np.any(constant > upper):
                raise ValueError(f'Constant linked parameter {name!r} lies outside its bounds.')
            continue
        identity, coefficient = next(iter(coefficients.items()))
        coefficient = _number(coefficient, unit)
        if not np.all(np.isfinite(coefficient)):
            raise ValueError(f'Non-finite coefficient in bounded link {name!r}.')
        # lower <= coefficient*driver + constant <= upper. Reverse the
        # inequalities for negative coefficients; zero coefficients are constant.
        lower, upper, coefficient, constant = np.broadcast_arrays(lower, upper, coefficient, constant)
        zero = coefficient == 0
        if np.any(zero & ((constant < lower) | (constant > upper))):
            raise ValueError(f'Constant linked parameter {name!r} lies outside its bounds.')
        with np.errstate(over='raise', invalid='raise'):
            first = np.divide(lower-constant, coefficient, out=np.zeros_like(coefficient), where=~zero)
            second = np.divide(upper-constant, coefficient, out=np.zeros_like(coefficient), where=~zero)
        derived_lower = np.where(zero, -np.inf, np.minimum(first, second))
        derived_upper = np.where(zero, np.inf, np.maximum(first, second))
        driver = nodes[identity]
        lo, hi = limits[driver]
        limits[driver] = np.maximum(lo, derived_lower), np.minimum(hi, derived_upper)

    for name in model.param_names:
        parameter = getattr(model, name)
        if parameter.tied:
            continue
        lower, upper = limits[name]
        if not parameter.fixed:
            if np.any(lower > upper):
                raise ValueError(f'Incompatible linked bounds leave no feasible interval for {name!r}.')
            if np.any(lower == upper):
                raise ValueError(f'Parameters with equal lower and upper bounds: {name}. '
                                 'Use param.fixed = True instead.')
        value = starts[name]
        if np.any(~np.isfinite(value)) or np.any(value < lower) or np.any(value > upper):
            raise ValueError(f'Initial values for {name!r} must be finite and within all declared and linked bounds.')
    return limits


def _box_bounds(model, parameters):
    """Request independent-interval capability for a prepared numerical problem."""
    limits = _box_limits(model)
    return np.asarray([limits[parameters.names[index]] for index in parameters.free], dtype=float).reshape(-1, 2)


def _check_fitted_bounds(model):
    """Check the committed fit against declared bounds, allowing roundoff only."""
    for name in model.param_names:
        parameter = getattr(model, name)
        lower, upper = parameter.bounds
        value = np.asarray(parameter.value, dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError(f'Fitted parameter {name!r} is non-finite.')
        for limit, sign in ((lower, -1), (upper, 1)):
            if limit is None or not np.isfinite(limit):
                continue
            scale = np.maximum(np.maximum(np.abs(value), abs(limit)), np.finfo(float).tiny)
            if not np.any(sign*(value-limit) > 32*np.finfo(float).eps*scale):
                continue
            if isinstance(parameter.tied, _Expression):
                # Near zero, cancellation in a*driver+b is controlled by the
                # scale of its terms, not by the tiny final value alone.
                nodes = {id(getattr(model, key)): key for key in model.param_names}
                try:
                    constant, coefficients = _affine_form(parameter, nodes, {}, {})
                    unit = parameter.unit or getattr(parameter, '_prism_evaluation_unit', None)
                    terms = abs(_number(constant, unit))
                    for identity, coefficient in coefficients.items():
                        terms = terms+abs(_number(coefficient, unit)*getattr(model, nodes[identity]).value)
                    scale = np.maximum(scale, terms)
                except NotImplementedError:
                    pass  # A future constrained solver may commit other links.
            tolerance = 32*np.finfo(float).eps*scale
            if np.any(sign*(value-limit) > tolerance):
                raise ValueError(f'Fitted parameter {name!r} violates its declared bounds.')
