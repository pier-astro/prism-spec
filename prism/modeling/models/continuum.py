"""prism.modeling.models.continuum — custom continuum models."""
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.physical_models import BlackBody
from astropy import units as u


BALMER_EDGE = 3646.0 * u.AA


class BalmerContinuum(Fittable1DModel):
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

    @property
    def input_units(self):
        return {self.inputs[0]: u.AA}
    
    @staticmethod
    def evaluate(x, amplitude, temperature, tau):
        x_q = BalmerContinuum._wavelength_quantity(x)
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