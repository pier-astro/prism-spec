"""
prism.modeling.models.continuum — spectral continuum models.

Provides Astropy ``Fittable1DModel`` subclasses for common AGN/galaxy
spectral continuum shapes.  All models expose analytic ``fit_deriv``
methods for efficient Jacobian-based fitting.
"""
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.physical_models import BlackBody
from astropy import units as u


BALMER_EDGE = 3646.0 * u.AA


class ContinuumModelBase(Fittable1DModel):
    """Shared Astropy-style unit handling for continuum models."""

    input_units_allow_dimensionless = {'x': True}

    def __init__(self, *args, **kwargs):
        param_names = list(type(self).param_names)
        coerced_args = list(args)
        for index, value in enumerate(coerced_args[:len(param_names)]):
            coerced_args[index] = type(self)._coerce_init_parameter(param_names[index], value)
        coerced_kwargs = {
            key: type(self)._coerce_init_parameter(key, value) if key in param_names else value
            for key, value in kwargs.items()
        }
        super().__init__(*coerced_args, **coerced_kwargs)

    @classmethod
    def _coerce_init_parameter(cls, name, value):
        param = getattr(cls, name, None)
        unit = getattr(param, 'unit', None)
        if unit is None or isinstance(value, (u.Quantity, Parameter)):
            return value
        return value * unit

    @staticmethod
    def _wavelength_quantity(x):
        if isinstance(x, u.Quantity):
            return x.to(u.AA)
        return np.asarray(x, dtype=float) * u.AA

    @staticmethod
    def _wavelength_values(x):
        return ContinuumModelBase._wavelength_quantity(x).to_value(u.AA)

    @staticmethod
    def _matched_scale_values(x, x0):
        if isinstance(x, u.Quantity):
            unit = x.unit
            x_val = np.asarray(x.to_value(unit), dtype=float)
            if isinstance(x0, u.Quantity):
                x0_val = np.asarray(x0.to_value(unit), dtype=float)
            else:
                x0_val = np.asarray(x0, dtype=float)
            return x_val, x0_val

        if isinstance(x0, u.Quantity):
            unit = x0.unit
            x_val = np.asarray(x, dtype=float)
            x0_val = np.asarray(x0.to_value(unit), dtype=float)
            return x_val, x0_val

        return np.asarray(x, dtype=float), np.asarray(x0, dtype=float)

class Powerlaw(ContinuumModelBase):
    """
    Simple power-law model: amp * (x / x0)^index
    
    Parameters
    ----------
    amplitude : float
        Amplitude at the reference wavelength.
    x0 : float or `~astropy.units.Quantity`
        Reference coordinate for the continuum normalisation.
    index : float
        Power-law index (typically negative for continuum slopes).
    
    Notes
    -----
    Astropy has `PowerLaw1D` but with different parameterization.
    This version uses a generic scale coordinate. If `x0` is given as a
    quantity, plain numeric `x` values are interpreted in the same unit.
    """
    amplitude = Parameter(default=1.0, bounds=(1e-5, 1e4))
    x0 = Parameter(default=1.0, bounds=(1e-10, None))
    index = Parameter(default=-1.7, bounds=(-3, 0))
    
    @staticmethod
    def evaluate(x, amplitude, x0, index):
        x_val, x0_val = ContinuumModelBase._matched_scale_values(x, x0)
        return amplitude * np.power(x_val / x0_val, index)

    def fit_deriv(self, x, amplitude, x0, index):
        x_val, x0_val = self._matched_scale_values(x, x0)
        ratio = x_val / x0_val
        val = amplitude * np.power(ratio, index)
        
        d_amplitude = np.power(ratio, index)
        d_x0 = -index * val / x0_val
        d_index = val * np.log(ratio)
        
        return [d_amplitude, d_x0, d_index]

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            'x0': inputs_unit[self.inputs[0]],
            'amplitude': outputs_unit[self.outputs[0]],
        }

class BrokenPowerlaw(ContinuumModelBase):
    """
    Broken power-law model with different slopes below and above break point.
    
    Parameters
    ----------
    amplitude : float
        Amplitude at the reference wavelength.
    x0 : float or `~astropy.units.Quantity`
        Break coordinate of the continuum.
    index1 : float
        Power-law index below the break (typically negative).
    index2 : float
        Additional index above the break (total index = index1 + index2).
    
    Notes
    -----
    Below x0: amp * (x / x0)^index1
    Above x0: amp * (x / x0)^(index1 + index2)
    """
    amplitude = Parameter(default=1.0, bounds=(1e-5, 1e4))
    x0 = Parameter(default=1.0, bounds=(1e-10, None))
    index1 = Parameter(default=-1.7, bounds=(-3, 0))
    index2 = Parameter(default=0.0, bounds=(-1, 1))
    
    @staticmethod
    def evaluate(x, amplitude, x0, index1, index2):
        x_val, x0_val = ContinuumModelBase._matched_scale_values(x, x0)
        ratio = x_val / x0_val
        # Below break: use index1, above break: use index1 + index2
        exponent = np.where(ratio > 1.0, index1 + index2, index1)
        return amplitude * np.power(ratio, exponent)

    def fit_deriv(self, x, amplitude, x0, index1, index2):
        x_val, x0_val = self._matched_scale_values(x, x0)
        ratio = x_val / x0_val
        exponent = np.where(ratio > 1.0, index1 + index2, index1)
        val = amplitude * np.power(ratio, exponent)
        
        d_amplitude = np.power(ratio, exponent)
        d_x0 = -exponent * val / x0_val
        d_index1 = val * np.log(ratio)
        d_index2 = np.where(ratio > 1.0, val * np.log(ratio), 0.0)
        
        return [d_amplitude, d_x0, d_index1, d_index2]

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            'x0': inputs_unit[self.inputs[0]],
            'amplitude': outputs_unit[self.outputs[0]],
        }
    
class BalmerContinuum(ContinuumModelBase):
    """
    Balmer continuum emission model.
    
    Models the pseudo-continuum emission from higher-order Balmer lines
    using a blackbody spectrum with optical depth effects.
    
    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`
        Continuum amplitude below the Balmer edge.
    temperature : float
        Temperature in Kelvin.
    tau : float
        Optical depth parameter.
    
    Notes
    -----
    The model is only evaluated for wavelengths below the Balmer edge (3646 Å).
    """
    amplitude = Parameter(default=1.0, bounds=(0, None))
    temperature = Parameter(default=10000.0, unit=u.K, bounds=(5000, 50000))
    tau = Parameter(default=1.0, bounds=(0.01, 2))

    @property
    def input_units(self):
        return {self.inputs[0]: u.AA}
    
    @staticmethod
    def evaluate(x, amplitude, temperature, tau):
        x_q = ContinuumModelBase._wavelength_quantity(x)
        temperature_q = temperature if isinstance(temperature, u.Quantity) else temperature * u.K
        bbflux = BlackBody(temperature=temperature_q)
        bb_shape = (bbflux(x_q) / bbflux(BALMER_EDGE)).to_value(u.dimensionless_unscaled)
        tau_lambda = tau * (x_q / BALMER_EDGE).to_value(u.dimensionless_unscaled) ** 3
        result = amplitude * bb_shape * (1.0 - np.exp(-tau_lambda))
        mask = np.asarray(x_q <= BALMER_EDGE)

        if isinstance(result, u.Quantity):
            values = np.where(mask, result.to_value(result.unit), 0.0)
            return values * result.unit

        return np.where(mask, result, 0.0)

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            'amplitude': outputs_unit[self.outputs[0]],
            'temperature': u.K,
        }