"""Normalize selected members without changing their profile or parameter graph.

Each member retains its live component. Numerical measurements do not inspect
model classes.
"""
import numpy as np
from astropy import units as u
from ...composition import Model as NativeModel
from ...component import _Function
from ..component import _LineGroup
from .. import profiles

def _native_line(component):
    if isinstance(component, NativeModel) and component.op is None:
        component = component.component
    return isinstance(component, _LineGroup) or (
        isinstance(component, _Function) and isinstance(component._function, profiles._LineProfile))


def _position_unit(component):
    return (u.Unit(component._setup['unit']) if isinstance(component, _LineGroup)
            else component._function.position_unit)


def _group_data(component):
    names = [item[0] for item in component._setup['parameters']]
    return names, [item[0] for item in component._groups], [item[1] for item in component._groups]


class _LineMember:
    """One transition: live parameters, evaluation, units and integration extent."""

    def __init__(self, entry, component):
        self.entry = entry
        self.component = component
        self.position_unit = _position_unit(component)
        self.output_unit = getattr(component, entry.amplitude_name).unit
        profile = component._profile if isinstance(component, _LineGroup) else component._function
        if self.output_unit is not None and getattr(profile, '_flux_normalized', False):
            self.output_unit /= self.position_unit
        if isinstance(component, _LineGroup):
            slots = component._groups[entry.template_index][2]
            self.parameter_names = tuple(dict.fromkeys(
                component.param_names[index] for index in slots if index >= 0))
        else:
            self.parameter_names = component.param_names

    @property
    def position(self):
        return (self.component.position.value if self.entry.template_position is None
                else self.entry.template_position)

    def profile_arguments(self):
        """Use the same physical kernel and live parameters as model evaluation."""
        component, entry = self.component, self.entry
        values = np.array([getattr(component, name).value for name in component.param_names])
        if isinstance(component, _LineGroup):
            arguments = values[component._groups[entry.template_index][2]]
            arguments[component._position_slot] = entry.template_position
            arguments[component._scale_slot] *= entry.weight
            return component._profile, arguments
        return component._function, values

    def evaluate(self, x):
        profile, arguments = self.profile_arguments()
        values = profile(x, *arguments)
        return values if self.output_unit is None else values*self.output_unit

    def integral(self):
        """Analytic integral and local parameter gradient, using evaluation's weights."""
        profile, arguments = self.profile_arguments()
        integral = getattr(profile, '_integral', None)
        gradient = getattr(profile, '_integral_gradient', None)
        if integral is None or gradient is None:
            raise NotImplementedError('This profile has no analytic total integral; use measure with an explicit grid/window.')
        value = float(integral(*arguments))
        derivatives = np.asarray(gradient(*arguments), dtype=float)
        component = self.component
        if isinstance(component, _LineGroup):
            slots = component._groups[self.entry.template_index][2]
            pairs = [(getattr(component, component.param_names[index]), derivatives[arg]*
                      (self.entry.weight if arg == component._scale_slot else 1.))
                     for arg, index in enumerate(slots) if arg != component._position_slot]
        else:
            pairs = [(getattr(component, name), derivative)
                     for name, derivative in zip(component.param_names, derivatives)]
        return value, pairs

    def center_width(self):
        """Effective width for a grid extent; the measured FWHM remains numerical."""
        profile, arguments = self.profile_arguments()
        if isinstance(profile, profiles._LineProfile):
            center, width = profile._center_width(*arguments)
        elif profile is profiles.gaussian:
            _, center, sigma = arguments
            width = float(profiles._profile_fwhm(sigma))
        elif profile is profiles.lorentzian:
            _, center, gamma = arguments
            width = 2.*gamma
        elif profile is profiles.voigt:
            _, center, sigma, gamma = arguments
            width = float(profiles._profile_fwhm(sigma, gamma))
        else:
            raise ValueError('Custom profiles require an explicit measurement x grid or window.')
        if not np.isfinite(width) or width <= 0:
            raise ValueError('Zero-width native profiles require an explicit measurement x grid or window.')
        return center, width
