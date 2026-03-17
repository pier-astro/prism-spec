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

class Powerlaw(Fittable1DModel):
    """
    Simple power-law model: amp * (x / x_ref)^index
    
    Parameters
    ----------
    amplitude : float
        Amplitude at the reference wavelength.
    x_ref : float
        Reference wavelength in Angstroms.
    index : float
        Power-law index (typically negative for continuum slopes).
    
    Notes
    -----
    Astropy has `PowerLaw1D` but with different parameterization.
    This version uses wavelength-based parametrization common in spectroscopy.
    """
    amplitude = Parameter(default=1.0, bounds=(1e-5, 1e4))
    x_ref = Parameter(default=5500.0, bounds=(1e-5, 1e5))
    index = Parameter(default=-1.7, bounds=(-3, 0))
    
    @staticmethod
    def evaluate(x, amplitude, x_ref, index):
        return amplitude * np.power(x / x_ref, index)

    def fit_deriv(self, x, amplitude, x_ref, index):
        ratio = x / x_ref
        val = amplitude * np.power(ratio, index)
        
        d_amplitude = np.power(ratio, index)
        d_x_ref = -index * val / x_ref
        d_index = val * np.log(ratio)
        
        return [d_amplitude, d_x_ref, d_index]

class BrokenPowerlaw(Fittable1DModel):
    """
    Broken power-law model with different slopes below and above break point.
    
    Parameters
    ----------
    amplitude : float
        Amplitude at the reference wavelength.
    x_ref : float
        Break wavelength in Angstroms.
    index1 : float
        Power-law index below the break (typically negative).
    index2 : float
        Additional index above the break (total index = index1 + index2).
    
    Notes
    -----
    Below x_ref: amp * (x / x_ref)^index1
    Above x_ref: amp * (x / x_ref)^(index1 + index2)
    """
    amplitude = Parameter(default=1.0, bounds=(1e-5, 1e4))
    x_ref = Parameter(default=5500.0, bounds=(1e-10, 1e5))
    index1 = Parameter(default=-1.7, bounds=(-3, 0))
    index2 = Parameter(default=0.0, bounds=(-1, 1))
    
    @staticmethod
    def evaluate(x, amplitude, x_ref, index1, index2):
        ratio = x / x_ref
        # Below break: use index1, above break: use index1 + index2
        exponent = np.where(ratio > 1.0, index1 + index2, index1)
        return amplitude * np.power(ratio, exponent)

    def fit_deriv(self, x, amplitude, x_ref, index1, index2):
        ratio = x / x_ref
        exponent = np.where(ratio > 1.0, index1 + index2, index1)
        val = amplitude * np.power(ratio, exponent)
        
        d_amplitude = np.power(ratio, exponent)
        d_x_ref = -exponent * val / x_ref
        d_index1 = val * np.log(ratio)
        d_index2 = np.where(ratio > 1.0, val * np.log(ratio), 0.0)
        
        return [d_amplitude, d_x_ref, d_index1, d_index2]
    
class BalmerContinuum(Fittable1DModel):
    """
    Balmer continuum emission model.
    
    Models the pseudo-continuum emission from higher-order Balmer lines
    using a blackbody spectrum with optical depth effects.
    
    Parameters
    ----------
    amplitude : float
        Amplitude scaling factor.
    temperature : float
        Temperature in Kelvin.
    tau : float
        Optical depth parameter.
    
    Notes
    -----
    The model is only evaluated for wavelengths below the Balmer edge (3646 Å).
    """
    amplitude = Parameter(default=1.0, bounds=(0, None))
    temperature = Parameter(default=10000.0, bounds=(5000, 50000))
    tau = Parameter(default=1.0, bounds=(0.01, 2))
    
    @staticmethod
    def evaluate(x, amplitude, temperature, tau):
        
        lambda_BE = 3646.0  # Balmer edge in Angstroms
        
        # Create blackbody with the given temperature
        bbflux = BlackBody(temperature=temperature * u.K, scale=10000)
        
        # Calculate optical depth
        tau_lambda = tau * (x / lambda_BE) ** 3
        
        # Calculate blackbody flux
        bb = bbflux(x * u.AA)
        
        # Apply optical depth correction
        result = amplitude * bb * (1.0 - np.exp(-tau_lambda))
        
        # Zero out values above Balmer edge
        result = np.where(x > lambda_BE, 0.0, result)
        
        return result.value