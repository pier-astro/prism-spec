"""Native line groups: explicit parameter sharing and weighted template sums."""
import inspect
import numpy as np
from astropy import units as u

from ..component import Component, _Function
from ..composition import Model


class _LineGroup(_Function):
    """Prepared argument maps; each profile still contains its own physics."""

    def __init__(self, profile, lines, shared, parameters, scale, position, derivative, name):
        from .tables import normalize_linetable, _clean_name
        table = normalize_linetable(lines)
        if hasattr(profile, 'position_unit') and profile.position_unit != table['position'].unit:
            raise ValueError('Profile and line table must use the same numeric spectral unit; convert the table explicitly.')
        if not len(table):
            raise ValueError('A line component requires at least one transition.')
        if not callable(profile) or derivative is not None and not callable(derivative):
            raise TypeError('Provide a profile and an optional callable derivative.')
        signature = inspect.signature(profile)
        arguments = tuple(signature.parameters.values())
        positional = tuple(arg for arg in arguments if arg.kind in
                           (arg.POSITIONAL_ONLY, arg.POSITIONAL_OR_KEYWORD))
        if (not positional or any(arg.kind in (arg.VAR_POSITIONAL, arg.VAR_KEYWORD)
                                  or arg.kind == arg.KEYWORD_ONLY and arg.default is arg.empty
                                  for arg in arguments)):
            raise ValueError('Profile arguments must be explicit positional parameters after x.')
        names = tuple(arg.name for arg in positional[1:])
        if position == scale or position not in names or scale not in names:
            raise ValueError('Position and scale must name distinct profile arguments.')
        templates = tuple(dict.fromkeys(str(value) for value in table['name']))
        if set(parameters) != set(templates):
            raise ValueError('Provide independent parameters for every template, without extra templates.')
        if position in shared or scale in shared:
            raise ValueError('Positions are tabulated and the scale must be independent per template.')
        values = dict(shared)
        scopes = []
        for template in templates:
            local = dict(parameters[template])
            if set(local).intersection(shared):
                raise ValueError('A profile parameter cannot be both shared and independent.')
            if position in local or scale not in local or set(local) | set(shared) != set(names)-{position}:
                raise ValueError('Every profile argument must be tabulated, shared, or independent; each template needs its scale.')
            slots = {}
            for argument, value in local.items():
                prefix = 'amp' if argument == 'amplitude' else argument
                key = f'{prefix}_{_clean_name(template)}'
                if key in values:
                    raise ValueError(f'Ambiguous parameter name {key!r}; rename the template.')
                values[key] = value
                slots[argument] = key
            scopes.append(slots)
        self._init_parameters(values, name)
        collisions = set(values).intersection(dir(Model))
        if collisions:
            raise ValueError(f'Parameter names conflict with model attributes: {sorted(collisions)}.')
        order = {key: index for index, key in enumerate(values)}
        self._profile = profile
        self._derivative = derivative
        self.fit_deriv = None if derivative is None else self._jacobian
        self._position_slot = names.index(position)
        self._scale_slot = names.index(scale)
        self._argument_count = len(names)
        self._groups = []
        for template, scope in zip(templates, scopes):
            selection = np.asarray(table['name'], dtype=str) == template
            # -1 is overwritten by the fixed transition position before calling.
            slots = np.array([-1 if key == position else order[scope.get(key, key)] for key in names])
            self._groups.append((table['position'][selection].value.copy(),
                                 np.asarray(table['weight'][selection], dtype=float).copy(), slots))
        self._setup = dict(lines=dict(name=list(map(str, table['name'])),
                                     position=table['position'].value.tolist(),
                                     weight=np.asarray(table['weight']).tolist()),
                           unit=table['position'].unit.to_string(),
                           shared=list(shared.items()),
                           parameters=[(key, list(parameters[key].items())) for key in templates],
                           scale=scale, position=position)

    def evaluate(self, x, *parameters):
        if isinstance(x, u.Quantity):
            raise TypeError('Convert coordinates to the explicit line-table unit before evaluation.')
        values = np.asarray(parameters)
        total = np.zeros_like(x, dtype=float)
        shape = np.shape(x)
        for positions, weights, slots in self._groups:
            for position, weight in zip(positions, weights):
                arguments = values[slots]
                arguments[self._position_slot] = position
                arguments[self._scale_slot] *= weight
                profile = self._profile(x, *arguments)
                if isinstance(profile, u.Quantity) or np.shape(profile) != shape:
                    raise ValueError('A line profile must return numeric values matching the coordinate grid.')
                total += profile
        return total

    def _jacobian(self, x, *parameters):
        values = np.asarray(parameters)
        result = np.zeros((len(parameters), len(x)))
        shape = np.shape(x)
        for positions, weights, slots in self._groups:
            for position, weight in zip(positions, weights):
                arguments = values[slots]
                arguments[self._position_slot] = position
                arguments[self._scale_slot] *= weight
                derivatives = self._derivative(x, *arguments)
                if len(derivatives) != self._argument_count:
                    raise ValueError('Profile derivatives must follow its positional argument order, including position.')
                for argument, parameter in enumerate(slots):
                    if argument == self._position_slot:
                        continue
                    row = np.asarray(derivatives[argument], dtype=float)
                    if row.shape != shape:
                        raise ValueError('Each profile derivative must match the coordinate grid.')
                    # Chain rule: the profile receives scale_template * weight.
                    if argument == self._scale_slot:
                        result[parameter] += row*weight
                    else:
                        result[parameter] += row
        return result


class LineSet(Component):
    """Sum a custom profile over weighted transitions with explicit parameter scopes.

    Parameters
    ----------
    profile : callable
        Numeric function ``profile(x, ...)``. The function owns all coordinate,
        width, normalization and velocity transformations.
    lines : astropy.table.Table or mapping
        Columns ``name``, ``position`` (spectral Quantity), and optional ``weight``.
        Repeated names form one template. Weights multiply the selected scale
        argument without normalization. Use unit-bearing positions explicitly.
    shared : mapping
        Profile argument names and initial values common to every template.
    parameters : mapping
        ``{template: {argument: initial_value}}`` for independent parameters.
        Every template must supply its scale argument. Every profile argument
        after x must be either position, shared or independent, exactly once.
    scale : str, optional
        Profile argument multiplied by each transition's weight; default amplitude.
    position : str, optional
        Profile argument receiving the tabulated position; default center.
    derivative : callable, optional
        Same arguments as profile; returns parameter-major derivatives in profile
        signature order, including the fixed position row. No derivative is
        required to fit the component with numerical differences.
    name : str, optional
        Label for composition and explicit custom-definition restoration.

    Notes
    -----
    Independent parameter attributes use ``argument_template``; amplitude uses
    the existing ``amp_template`` convention. Template names are cleaned with
    Prism's table naming utility; collisions are rejected. Shared parameters keep
    their argument names. Units are converted explicitly by the caller; evaluation
    is numeric in the line table's unit. Negative and zero weights are retained.
    Table membership and argument maps are prepared once and are not mutable.
    """

    def __init__(self, profile, lines, shared, parameters, scale='amplitude',
                 position='center', derivative=None, name=None):
        leaf = _LineGroup(profile, lines, shared, parameters, scale, position, derivative, name)
        Model.__init__(self, component=leaf)
