import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import re

import astropy.units as u
import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter

from . import profiles
from .flux import (
    Flux,
    flux_from_samples,
    extract_fluxes,
    get_flux_with_uncertainty,
    compute_flux_std,
    compute_flux_limits,
    compute_feii_flux_std,
    compute_feii_flux_limits,
)

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

script_dir = os.path.dirname(__file__) # get the directory of the current script
input_path = os.path.join(script_dir, "..", "..", "..", "resources", "lines")
csv_lines_path = input_path

def setup_local_lines(wmin=4000, wmax=7000, dirpath=input_path, overwrite=False):
    """
    Initializes the lines by reading the csv files from the input folder and filtering them based on the wavelength range.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        is_created = True
        print(f"Directory {dirpath} created.")
    else:
        is_created = False
    global csv_lines_path
    csv_lines_path = dirpath
    if overwrite or is_created:
        for files in glob.glob(input_path + "/*.csv"):
            df = pd.read_csv(files)
            if 'pos' in df.columns:
                df = df[df.pos > wmin]
                df = df[df.pos < wmax]
            name = os.path.join(dirpath, Path(files).name)
            df.to_csv(name, index=False)

# ------
# MODELS
# ------

class LineModelBase(Fittable1DModel):
    """Base class for line models providing common derivative logic."""
    pass

# -------------
### LINE MODELS
# -------------

class GaussianLine(LineModelBase):
    """
    Gaussian line model for spectra, using rest wavelength and velocity offset.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm = Parameter(default=1000.0)

    @staticmethod
    def evaluate(x, amplitude, position, offset, fwhm):
        center = position * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        return profiles.gaussian(x, amplitude, center, sigma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm):
        center = position * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        
        val, d_amp, d_center, d_sigma = profiles.gaussian_deriv(x, amplitude, center, sigma)
        
        d_center_d_pos = 1 + offset / c_kms
        d_center_d_off = position / c_kms
        
        d_sigma_d_pos = (fwhm / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_d_fwhm = (center / c_kms) / sigma2fwhm
        
        d_position = d_center * d_center_d_pos + d_sigma * d_sigma_d_pos
        d_offset = d_center * d_center_d_off + d_sigma * d_sigma_d_off
        d_fwhm = d_sigma * d_sigma_d_fwhm
        
        return [d_amp, d_position, d_offset, d_fwhm]

    @property
    def flux(self):
        """Return Flux object with value and uncertainties if available."""
        return get_flux_with_uncertainty(self)

class LorentzianLine(LineModelBase):
    """
    Lorentzian line model for spectra, using rest wavelength and velocity offset.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm = Parameter(default=1000.0)

    @staticmethod
    def evaluate(x, amplitude, position, offset, fwhm):
        center = position * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        return profiles.lorentzian(x, amplitude, center, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm):
        center = position * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        
        val, d_amp, d_center, d_gamma = profiles.lorentzian_deriv(x, amplitude, center, gamma)
        
        d_center_d_pos = 1 + offset / c_kms
        d_center_d_off = position / c_kms
        
        d_gamma_d_pos = (fwhm / c_kms * d_center_d_pos) / 2.0
        d_gamma_d_off = (fwhm / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhm = (center / c_kms) / 2.0
        
        d_position = d_center * d_center_d_pos + d_gamma * d_gamma_d_pos
        d_offset = d_center * d_center_d_off + d_gamma * d_gamma_d_off
        d_fwhm = d_gamma * d_gamma_d_fwhm
        
        return [d_amp, d_position, d_offset, d_fwhm]

    @property
    def flux(self):
        """Return Flux object with value and uncertainties if available."""
        return get_flux_with_uncertainty(self)

class VoigtLine(LineModelBase):
    """
    Voigt line model for spectra, using rest wavelength and velocity offset.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm_G = Parameter(default=1000.0)
    fwhm_L = Parameter(default=1000.0)

    @staticmethod
    def evaluate(x, amplitude, position, offset, fwhm_G, fwhm_L):
        center = position * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        return profiles.voigt(x, amplitude, center, sigma, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm_G, fwhm_L):
        center = position * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        
        val, d_amp, d_center, d_sigma, d_gamma = profiles.voigt_deriv(x, amplitude, center, sigma, gamma)
        
        d_center_d_pos = 1 + offset / c_kms
        d_center_d_off = position / c_kms
        
        d_sigma_d_pos = (fwhm_G / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_d_fwhmG = (center / c_kms) / sigma2fwhm
        
        d_gamma_d_pos = (fwhm_L / c_kms * d_center_d_pos) / 2.0
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_position = d_center * d_center_d_pos + d_sigma * d_sigma_d_pos + d_gamma * d_gamma_d_pos
        d_offset = d_center * d_center_d_off + d_sigma * d_sigma_d_off + d_gamma * d_gamma_d_off
        d_fwhmG = d_sigma * d_sigma_d_fwhmG
        d_fwhmL = d_gamma * d_gamma_d_fwhmL
        
        return [d_amp, d_position, d_offset, d_fwhmG, d_fwhmL]

    @property
    def flux(self):
        """Return Flux object with value and uncertainties if available."""
        return get_flux_with_uncertainty(self)

# -----------------
### UNIFIED CLASSES
# -----------------

def _clean_name(name):
    """Convert scientific name to pythonic attribute name."""
    name = str(name).strip()
    # Remove brackets and common non-pythonic characters
    name = re.sub(r'[\[\]<>]', '', name)
    # Replace spaces and dashes with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    return name.lower()

def _make_unique(names):
    """
    Make a list of unique names by appending suffixes to duplicates, starting from '_b'.
    """
    import string
    counts = {}
    result = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            suffix = '_' + string.ascii_lowercase[counts[name]]
            result.append(name + suffix)
    return result

class LineGroupBase(Fittable1DModel):
    """
    Base class for models defined from CSVs with columns: name, pos, weight.
    Lines with the same 'name' are tied together under a single amplitude parameter.
    """
    @classmethod
    def from_csv(cls, csv_files, name=None, dirpath='./lines', bounds=None, amplitude=None, **init_kwargs):
        if not isinstance(csv_files, (list, tuple)):
            csv_files = [csv_files]
        
        dfs = []
        for f in csv_files:
            path = f if os.path.isabs(f) else os.path.join(dirpath, f)
            dfs.append(pd.read_csv(path))
            
        df = pd.concat(dfs, ignore_index=True)
        return cls.from_templates(df, name=name, bounds=bounds, amplitude=amplitude, **init_kwargs)

    @classmethod
    def from_arrays(cls, names, pos, weights=None, name=None, bounds=None, amplitude=None, **init_kwargs):
        """
        Initialize the line group directly from arrays.
        """
        names = np.atleast_1d(names)
        pos = np.atleast_1d(pos)
        if weights is None:
            weights = np.ones_like(pos)
        else:
            weights = np.atleast_1d(weights)
            
        df = pd.DataFrame({'name': names, 'pos': pos, 'weight': weights})
        return cls.from_templates(df, name=name, bounds=bounds, amplitude=amplitude, **init_kwargs)

    @classmethod
    def from_templates(cls, df, name=None, bounds=None, amplitude=None, **init_kwargs):
        # Identify unique template names
        templates = pd.unique(df['name'])
        
        # Generate pythonic parameter names
        _raw_param_names = [f"amp_{_clean_name(tmpl)}" for tmpl in templates]
        param_names = _make_unique(_raw_param_names)
        n_templates = len(templates)
        
        # Handle initial values for amplitudes
        for pname in param_names:
            if amplitude is not None and pname not in init_kwargs:
                init_kwargs[pname] = amplitude

        # Handle bounds for amplitudes
        bounds = dict(bounds) if bounds else {}
        amp_bounds = bounds.pop('amplitude', None)
        param_bounds = {}
        for pname in param_names:
            if amp_bounds is not None:
                param_bounds[pname] = amp_bounds
            if pname in bounds:
                param_bounds[pname] = bounds[pname]
        for k, v in bounds.items():
            if k not in param_bounds:
                param_bounds[k] = v

        params = {pname: Parameter(default=1.0) for pname in param_names}
        for pname, default in cls._shared_params.items():
            params[pname] = Parameter(default=default)

        def evaluate(x, *args):
            amplitudes = args[:n_templates]
            shared = args[n_templates:]
            
            x_arr = np.atleast_1d(x)
            total = np.zeros_like(x_arr)
            x_col = x_arr[:, np.newaxis]
            
            for i, tmpl in enumerate(templates):
                df_tmpl = df[df['name'] == tmpl]
                positions = df_tmpl['pos'].values
                weights = df_tmpl['weight'].values
                
                pos_row = positions[np.newaxis, :]
                wt_row = weights[np.newaxis, :]
                
                profile_args = cls._profile_args(pos_row, amplitudes[i], wt_row, *shared)
                components = cls._profile_func(x_col, *profile_args)
                total += np.sum(components, axis=1)
                
            if np.ndim(x) == 0:
                return total[0]
            return total
        
        def fit_deriv(x, *args):
            amplitudes = args[:n_templates]
            shared = args[n_templates:]
            
            x_arr = np.atleast_1d(x)
            n_params = len(args)
            n_x = len(x_arr)
            grad = np.zeros((n_params, n_x))
            
            d_shared = [np.zeros(n_x) for _ in shared]
            x_col = x_arr[:, np.newaxis]
            
            for i, tmpl in enumerate(templates):
                df_tmpl = df[df['name'] == tmpl]
                positions = df_tmpl['pos'].values
                weights = df_tmpl['weight'].values
                
                pos_row = positions[np.newaxis, :]
                wt_row = weights[np.newaxis, :]
                
                derivs = cls._profile_deriv_func(x_col, pos_row, amplitudes[i], wt_row, *shared)
                
                d_amp_tmpl = np.sum(derivs[1] * wt_row, axis=1)
                grad[i] = d_amp_tmpl
                
                for j, d in enumerate(derivs[2:]):
                    d_shared[j] += np.sum(d, axis=1)
            
            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d
                
            return list(grad)

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': staticmethod(evaluate),
            'fit_deriv': staticmethod(fit_deriv),
            'n_inputs': 1,
            'n_outputs': 1,
            '_df': df,
            '_templates': templates,
            '_n_templates': n_templates,
            '_param_names_list': param_names,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(bounds=param_bounds, **init_kwargs)

    @property
    def lines(self):
        """
        Return the DataFrame containing the physical lines included in this model group.
        """
        return self._df

    @property
    def fluxes(self):
        """
        Get theoretical fluxes for all fitted templates with uncertainties.
        """
        fluxes = {}
        shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
        
        for i, (tmpl, pname) in enumerate(zip(self._templates, self._param_names_list)):
            amp = getattr(self, pname)
            
            df_tmpl = self._df[self._df['name'] == tmpl]
            positions = df_tmpl['pos'].values
            weights = df_tmpl['weight'].values
            
            total_flux = 0.0
            for pos, wt in zip(positions, weights):
                amp_eff = amp.value * wt
                total_flux += self._calc_flux(pos, amp_eff, *shared_values)
                
            try:
                flux_std = compute_feii_flux_std(self, tmpl)
            except Exception:
                flux_std = None
                
            flux_lolim, flux_uplim = None, None
            try:
                if hasattr(amp, 'lolim') or any(hasattr(getattr(self, p), 'lolim') for p in self._shared_params.keys()):
                    flux_lolim, flux_uplim = compute_feii_flux_limits(self, tmpl)
            except Exception:
                pass
                
            method = None
            if flux_std is not None:
                method = 'covariance'
            elif flux_lolim is not None and flux_uplim is not None:
                method = 'limits'
                
            fluxes[tmpl] = Flux(
                value=total_flux,
                std=flux_std,
                lolim=flux_lolim,
                uplim=flux_uplim,
                method=method
            )
        return fluxes

class GaussianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0}
    _profile_func = staticmethod(profiles.gaussian)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        amplitude = amp_template * weight
        return (amplitude, center, sigma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        amplitude = amp_template * weight
        
        val, d_amp, d_center, d_sigma = profiles.gaussian_deriv(x, amplitude, center, sigma)
        
        d_center_d_off = pos / c_kms
        d_sigma_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_d_fwhm = (center / c_kms) / sigma2fwhm
        
        d_offset = d_center * d_center_d_off + d_sigma * d_sigma_d_off
        d_fwhm = d_sigma * d_sigma_d_fwhm
        
        return val, d_amp, d_offset, d_fwhm

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        return profiles.gaussian_flux(amp, sigma)

class LorentzianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0}
    _profile_func = staticmethod(profiles.lorentzian)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude = amp_template * weight
        return (amplitude, center, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude = amp_template * weight
        
        val, d_amp, d_center, d_gamma = profiles.lorentzian_deriv(x, amplitude, center, gamma)
        
        d_center_d_off = pos / c_kms
        d_gamma_d_off = (fwhm / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhm = (center / c_kms) / 2.0
        
        d_offset = d_center * d_center_d_off + d_gamma * d_gamma_d_off
        d_fwhm = d_gamma * d_gamma_d_fwhm
        
        return val, d_amp, d_offset, d_fwhm

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        return profiles.lorentzian_flux(amp, gamma)


class VoigtLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0, 'fwhm_L': 1000.0}
    _profile_func = staticmethod(profiles.voigt)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        amplitude = amp_template * weight
        return (amplitude, center, sigma, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        amplitude = amp_template * weight
        
        val, d_amp, d_center, d_sigma, d_gamma = profiles.voigt_deriv(x, amplitude, center, sigma, gamma)
        
        d_center_d_off = pos / c_kms
        d_sigma_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_d_fwhmG = (center / c_kms) / sigma2fwhm
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_offset = d_center * d_center_d_off + d_sigma * d_sigma_d_off + d_gamma * d_gamma_d_off
        d_fwhmG = d_sigma * d_sigma_d_fwhmG
        d_fwhmL = d_gamma * d_gamma_d_fwhmL
        
        return val, d_amp, d_offset, d_fwhmG, d_fwhmL

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        return profiles.voigt_flux(amp, sigma, gamma)
