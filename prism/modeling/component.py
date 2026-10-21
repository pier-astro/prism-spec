"""Function-backed spectral components, independent of Astropy modeling."""
import inspect
from types import MappingProxyType
import numpy as np
from astropy import units as u

from .composition import Model
from .parameters import _Scalar


class _Function:
    """Numerical leaf used by the existing prepared expression evaluator."""
    n_inputs = 1
    n_outputs = 1
    inputs = ('x',)
    input_units = None
    input_units_equivalencies = None
    col_fit_deriv = True

    def __init__(self, function, parameters, derivative, name):
        if not callable(function) or derivative is not None and not callable(derivative):
            raise TypeError('Provide a callable profile and an optional callable derivative.')
        self.name = name
        self.param_names = tuple(parameters)
        signature = inspect.signature(function)
        arguments = tuple(signature.parameters.values())
        if (not arguments or any(arg.kind not in (arg.POSITIONAL_ONLY, arg.POSITIONAL_OR_KEYWORD)
                                 for arg in arguments)
                or tuple(arg.name for arg in arguments[1:]) != self.param_names):
            raise ValueError('Declare every function parameter after x in the same order as parameters; use positional arguments.')
        signature.bind(object(), *parameters.values())
        if derivative is not None:
            inspect.signature(derivative).bind(object(), *parameters.values())
        self._init_parameters(parameters, name)
        self._function = function
        self._derivative = derivative
        self.fit_deriv = None if derivative is None else self._jacobian

    def _init_parameters(self, parameters, name):
        self.name = name
        self.param_names = tuple(parameters)
        for key, value in parameters.items():
            if not isinstance(key, str) or not key.isidentifier() or hasattr(self, key) or key.startswith('_'):
                raise ValueError(f'Invalid or reserved parameter name: {key!r}.')
            setattr(self, key, _Scalar(key, value))

    def __setattr__(self, name, value):
        current = self.__dict__.get(name)
        if isinstance(current, _Scalar):
            current.value = value
        else:
            object.__setattr__(self, name, value)

    @property
    def params(self):
        """Read-only mapping to this component's live parameters."""
        return MappingProxyType({name: getattr(self, name) for name in self.param_names})

    def evaluate(self, x, *parameters):
        if isinstance(x, u.Quantity):
            raise TypeError('Convert coordinates to explicit numeric units before evaluating a native component.')
        values = self._function(x, *parameters)
        if isinstance(values, u.Quantity):
            raise TypeError('Native profiles must return numeric values in the chosen output unit.')
        result = np.asarray(values, dtype=float)
        if result.shape != np.shape(x):
            raise ValueError('A component must return one value per input coordinate.')
        return result

    def _jacobian(self, x, *parameters):
        result = np.asarray(self._derivative(x, *parameters), dtype=float)
        if result.shape != (len(self.param_names), len(x)):
            raise ValueError('Derivatives must have shape (n_parameters, n_coordinates), in declared parameter order.')
        return result

    def __call__(self, x):
        return self.evaluate(x, *(getattr(self, name).value for name in self.param_names))


class Component(Model):
    """Create a spectral component from a numeric function and scalar parameters.

    Parameters
    ----------
    function : callable
        ``function(x, *parameters)`` returns an array matching the coordinates.
        Coordinates, parameters and output use explicit, consistent numeric units.
        The function must depend only on its inputs during a fit.
    parameters : mapping
        Parameter names and initial numeric scalar values. Insertion order fixes
        both the parameter vector and the rows of the analytic derivative.
    derivative : callable, optional
        Same arguments as the function; returns ``(n_parameters, n_coordinates)``.
        Omit it to use the fitter's numerical Jacobian.
    name : str, optional
        Component label, also used to restore its function from a saved session.

    Notes
    -----
    Set ``component.params['parameter'].bounds``, ``fixed`` and ``link`` after
    construction, or use the equivalent ``component.parameter`` attribute.
    Composition preserves parameter identity; copying the complete model preserves
    links within an independent graph. Saved sessions contain state, not Python
    code: pass named Component definitions to ``load_model(..., components=...)``.
    """

    def __init__(self, function, parameters, derivative=None, name=None):
        leaf = _Function(function, parameters, derivative, name)
        collisions = set(leaf.param_names).intersection(dir(Model))
        if collisions:
            raise ValueError(f'Parameter names conflict with model attributes: {sorted(collisions)}.')
        super().__init__(component=leaf)
