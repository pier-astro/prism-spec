import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path

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

def init_lines_csv(wmin=4000, wmax=7000, dirpath=input_path, overwrite=False):
    """
    The init_lines function initializes the lines by reading the csv files from the input folder and filtering them based on the wavelength range.
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
            try:
                df = df[df.position > wmin]
                df = df[df.position < wmax]
            except:
                df = df[df.wav > wmin]
                df = df[df.wav < wmax]

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


# -------------------
### TIED LINES MODELS
# -------------------

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

class _TiedLinesBase(Fittable1DModel):
    """
    Base class for tied lines models with shared offset and width parameters.
    """
    @classmethod
    def from_csv(cls, csv_files, name=None, dirpath='./lines', bounds=None, amplitude=None, **init_kwargs):
        import pandas as pd
        import os
        dfs = [pd.read_csv(os.path.join(dirpath, f)) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df['safe_line'] = df['line'].str.replace(r'[\[\]<>]', '', regex=True).str.replace(' ', '_')
        df['param_name'] = 'amp_' + df['safe_line'] + '_' + df['position'].round(0).astype(int).astype(str)
        df['param_name'] = _make_unique(df['param_name'].tolist())
        positions = df['position'].values
        param_names = df['param_name'].tolist()
        return cls.from_arrays(
            positions, param_names=param_names, name=name, bounds=bounds, amplitude=amplitude, **init_kwargs
        )

    @classmethod
    def from_arrays(cls, positions, param_names=None, name=None, bounds=None, amplitude=None, **init_kwargs):
        positions = np.asarray(positions)
        n_lines = len(positions)
        if param_names is None:
            param_names = [f'amp_{int(pos)}' for pos in positions]

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
            amplitudes = args[:n_lines]
            shared = args[n_lines:]
            total = 0.0
            for i, pos in enumerate(positions):
                profile_args = cls._profile_args(pos, amplitudes[i], *shared)
                total += cls._profile_func(x, *profile_args)
            return total
        
        def fit_deriv(x, *args):
            amplitudes = args[:n_lines]
            shared = args[n_lines:]
            
            n_params = len(args)
            n_x = len(x)
            grad = np.zeros((n_params, n_x))
            
            d_shared = [np.zeros(n_x) for _ in shared]
            
            for i, pos in enumerate(positions):
                derivs = cls._profile_deriv_func(x, pos, amplitudes[i], *shared)
                d_amp = derivs[1]
                d_shared_line = derivs[2:]
                
                grad[i] = d_amp
                
                for j, d in enumerate(d_shared_line):
                    d_shared[j] += d
            
            for j, d in enumerate(d_shared):
                grad[n_lines + j] = d
                
            return list(grad)  # Return list for Astropy compound model compatibility

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': staticmethod(evaluate),
            'fit_deriv': staticmethod(fit_deriv),
            'n_inputs': 1,
            'n_outputs': 1,
            '_positions': positions,
            '_param_names_list': param_names,
            '_n_lines': n_lines,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(bounds=param_bounds, **init_kwargs)

    @property
    def fluxes(self):
        """Return dict of Flux objects (one per line) with uncertainties if available."""
        fluxes = {}
        shared_values = [getattr(self, pname) for pname in self._shared_params.keys()]
        shared_param_names = list(self._shared_params.keys())
        
        for i, pname in enumerate(self._param_names_list):
            amp = getattr(self, pname)
            pos = self._positions[i]
            flux_value = self._calc_flux(pos, amp, *shared_values)
            
            # Try to compute uncertainties for this line
            flux_std = None
            flux_lolim = None
            flux_uplim = None
            method = None
            
            # Option 1: From covariance (propagate from amplitude and shared parameters)
            if hasattr(amp, 'std') and amp.std is not None:
                # For tied lines, uncertainty propagation is more complex
                # For now, propagate only amplitude uncertainty (conservative)
                # Full covariance propagation would require knowing derivatives
                # of flux w.r.t. shared parameters for each line
                try:
                    flux_std = self._propagate_tied_flux_std(i, pname, pos, amp, shared_values, shared_param_names)
                    if flux_std is not None:
                        method = 'covariance'
                except Exception:
                    flux_std = None
            
            # Option 2: From bootstrap limits
            if hasattr(amp, 'lolim'):
                try:
                    flux_lolim, flux_uplim = self._flux_limits_tied(i, pname, pos, amp, shared_values, shared_param_names)
                    if flux_lolim is not None or flux_uplim is not None:
                        method = 'limits'
                except Exception:
                    flux_lolim, flux_uplim = None, None
            
            fluxes[pname] = Flux(flux_value, std=flux_std, lolim=flux_lolim, 
                               uplim=flux_uplim, method=method)
        return fluxes
    
    def _propagate_tied_flux_std(self, line_idx, pname, pos, amp, shared_values, shared_param_names):
        """
        Propagate covariance-based uncertainties to flux for a tied line.
        
        Conservative approach: propagate only from parameters with available uncertainties.
        """
        # This is a simplified version - just propagate amplitude uncertainty
        # Full implementation would require derivatives w.r.t. all shared parameters
        if not hasattr(amp, 'std') or amp.std is None or not np.isfinite(amp.std):
            return None
        
        # For now, return None (implement per-profile in subclasses if needed)
        # Or use a simple amplitude-only propagation
        flux_value = self._calc_flux(pos, amp, *shared_values)
        if amp.value != 0:
            # Simple linear propagation: flux ∝ amplitude
            flux_std = flux_value / amp.value * amp.std
            return flux_std
        return None
    
    def _flux_limits_tied(self, line_idx, pname, pos, amp, shared_values, shared_param_names):
        """
        Estimate flux limits from parameter limits for a tied line.
        
        Uses monotonicity: evaluate at (amp_lo, shared_lo) and (amp_hi, shared_hi).
        """
        # Check which parameters have finite limits
        has_amp_lo = hasattr(amp, 'lolim') and amp.lolim is not None and np.isfinite(amp.lolim)
        has_amp_hi = hasattr(amp, 'uplim') and amp.uplim is not None and np.isfinite(amp.uplim)
        
        # Get amplitude limits
        amp_lo = amp.lolim if has_amp_lo else amp.value
        amp_hi = amp.uplim if has_amp_hi else amp.value
        
        # Get shared parameter limits
        shared_lo = []
        shared_hi = []
        has_shared_lo = []
        has_shared_hi = []
        for i, spname in enumerate(shared_param_names):
            sparam = getattr(self, spname)
            sp_has_lo = hasattr(sparam, 'lolim') and sparam.lolim is not None and np.isfinite(sparam.lolim)
            sp_has_hi = hasattr(sparam, 'uplim') and sparam.uplim is not None and np.isfinite(sparam.uplim)
            sp_lo = sparam.lolim if sp_has_lo else sparam.value
            sp_hi = sparam.uplim if sp_has_hi else sparam.value
            shared_lo.append(sp_lo)
            shared_hi.append(sp_hi)
            has_shared_lo.append(sp_has_lo)
            has_shared_hi.append(sp_has_hi)
        
        # Check if any limits available
        if not (has_amp_lo or has_amp_hi or any(has_shared_lo) or any(has_shared_hi)):
            return None, None
        
        # Flux is monotonic: min at (amp_lo, shared_lo), max at (amp_hi, shared_hi)
        flux_lo = self._calc_flux(pos, amp_lo, *shared_lo)
        flux_hi = self._calc_flux(pos, amp_hi, *shared_hi)
        
        # If any parameter is unbounded on a side, flux is unbounded on that side (use nan like parameters do)
        if not (has_amp_lo and all(has_shared_lo)):
            flux_lo = np.nan
        if not (has_amp_hi and all(has_shared_hi)):
            flux_hi = np.nan
        
        return flux_lo, flux_hi

# --- Subclasses for each profile ---
class TiedGaussianLines(_TiedLinesBase):
    _shared_params = {'offset': 0.0, 'fwhm': 100.0}
    _profile_func = staticmethod(profiles.gaussian)
    
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        return (amp, center, sigma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        
        val, d_amp, d_center, d_sigma = profiles.gaussian_deriv(x, amp, center, sigma)
        
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

class TiedLorentzianLines(_TiedLinesBase):
    _shared_params = {'offset': 0.0, 'fwhm': 100.0}
    _profile_func = staticmethod(profiles.lorentzian)
    
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        return (amp, center, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        
        val, d_amp, d_center, d_gamma = profiles.lorentzian_deriv(x, amp, center, gamma)
        
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

class TiedVoigtLines(_TiedLinesBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 100.0, 'fwhm_L': 100.0}
    _profile_func = staticmethod(profiles.voigt)
    
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        return (amp, center, sigma, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        
        val, d_amp, d_center, d_sigma, d_gamma = profiles.voigt_deriv(x, amp, center, sigma, gamma)
        
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


# --------------
### FE II MODELS
# --------------

class _BaseFeII(Fittable1DModel):
    """
    Base class for Fe II models with multiple templates.
    """
    @classmethod
    def from_csv(cls, csv_path=None, name=None, dirpath='./lines', bounds=None, amplitude=None, **init_kwargs):
        if csv_path is None:
            csv_path = os.path.join(dirpath, "feII_model.csv")
        elif not os.path.isabs(csv_path):
            csv_path = os.path.join(dirpath, csv_path)

        df = pd.read_csv(csv_path)
        templates = pd.unique(df['ime'])

        param_names = [f'amp_{tmpl}' for tmpl in templates]

        return cls.from_templates(df, templates, param_names=param_names, name=name, bounds=bounds, amplitude=amplitude, **init_kwargs)

    @classmethod
    def from_templates(cls, df, templates, param_names=None, name=None, bounds=None, amplitude=None, **init_kwargs):
        n_templates = len(templates)
        if param_names is None:
            param_names = [f'amp_{tmpl}' for tmpl in templates]

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

        params = {pname: Parameter(default=2.0) for pname in param_names}
        for pname, default in cls._shared_params.items():
            params[pname] = Parameter(default=default)

        def evaluate(x, *args):
            amplitudes = args[:n_templates]
            shared = args[n_templates:]
            
            # Ensure x is at least 1D array
            x_arr = np.atleast_1d(x)
            total = np.zeros_like(x_arr)
            
            # Prepare for broadcasting: x -> (N_x, 1)
            x_col = x_arr[:, np.newaxis]
            
            for i, tmpl in enumerate(templates):
                df_tmpl = df[df['ime'] == tmpl]
                positions = df_tmpl['wav'].values
                intensities = df_tmpl['Int'].values
                
                # positions, intensities -> (1, N_lines)
                pos_row = positions[np.newaxis, :]
                int_row = intensities[np.newaxis, :]
                
                # Calculate profile args (broadcasted)
                profile_args = cls._profile_args(pos_row, amplitudes[i], int_row, *shared)
                
                # Calculate profile values (N_x, N_lines)
                components = cls._profile_func(x_col, *profile_args)
                
                # Sum over lines and add to total
                total += np.sum(components, axis=1)
            
            # If input was scalar, return scalar
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
                df_tmpl = df[df['ime'] == tmpl]
                positions = df_tmpl['wav'].values
                intensities = df_tmpl['Int'].values
                
                pos_row = positions[np.newaxis, :]
                int_row = intensities[np.newaxis, :]
                
                # Calculate derivatives (N_x, N_lines)
                derivs = cls._profile_deriv_func(x_col, pos_row, amplitudes[i], int_row, *shared)
                
                # derivs[1] is d_amp (w.r.t profile amplitude)
                # We need d_amp_template = d_amp * intensity
                d_amp_tmpl = np.sum(derivs[1] * int_row, axis=1)
                grad[i] = d_amp_tmpl
                
                # Accumulate shared derivatives
                for j, d in enumerate(derivs[2:]):
                    d_shared[j] += np.sum(d, axis=1)
            
            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d
                
            return list(grad)  # Return list for Astropy compound model compatibility

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': staticmethod(evaluate),
            'fit_deriv': staticmethod(fit_deriv),
            'n_inputs': 1,
            'n_outputs': 1,
            '_df': df,
            '_templates': templates,
            '_n_templates': n_templates,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(bounds=param_bounds, **init_kwargs)

    @property
    def fluxes(self):
        """
        Get fluxes for all fitted templates with uncertainties.
        
        Returns
        -------
        fluxes : dict
            Dictionary mapping template names to Flux objects
        """
        fluxes = {}
        shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
        
        for i, tmpl in enumerate(self._templates):
            pname = f'amp_{tmpl}'
            amp = getattr(self, pname)
            
            df_tmpl = self._df[self._df['ime'] == tmpl]
            positions = df_tmpl['wav'].values
            intensities = df_tmpl['Int'].values
            
            # Compute total flux value
            total_flux = 0.0
            for pos, intensity in zip(positions, intensities):
                amp_eff = amp.value * intensity
                total_flux += self._calc_flux(pos, amp_eff, *shared_values)
            
            # Try covariance-based uncertainty first
            flux_std = compute_feii_flux_std(self, tmpl)
            
            # Try bootstrap limits
            flux_lolim = None
            flux_uplim = None
            if hasattr(amp, 'lolim') or any(hasattr(getattr(self, p), 'lolim') for p in self._shared_params.keys()):
                flux_lolim, flux_uplim = compute_feii_flux_limits(self, tmpl)
            
            # Determine method
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


class GaussianFeII(_BaseFeII):
    _shared_params = {'offset': 0.0, 'fwhm': 2000.0}
    _profile_func = staticmethod(profiles.gaussian)
    
    @staticmethod
    def _profile_args(pos, amp_template, intensity, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        amplitude = amp_template * intensity
        return (amplitude, center, sigma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, intensity, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        amplitude = amp_template * intensity
        
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


class LorentzFeII(_BaseFeII):
    _shared_params = {'offset': 0.0, 'fwhm': 2000.0}
    _profile_func = staticmethod(profiles.lorentzian)
    
    @staticmethod
    def _profile_args(pos, amp_template, intensity, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude = amp_template * intensity
        return (amplitude, center, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, intensity, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude = amp_template * intensity
        
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

class VoigtFeII(_BaseFeII):
    _shared_params = {'offset': 0.0, 'fwhm_G': 2000.0, 'fwhm_L': 2000.0}
    _profile_func = staticmethod(profiles.voigt)
    
    @staticmethod
    def _profile_args(pos, amp_template, intensity, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        amplitude = amp_template * intensity
        return (amplitude, center, sigma, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, intensity, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        amplitude = amp_template * intensity
        
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


# --------------
### UV Fe MODELS
# --------------

class GaussianUVFeII(GaussianFeII):
    """Gaussian UV Fe II model using uvfe.csv template."""
    @classmethod
    def from_csv(cls, csv_path=None, dirpath='./lines', **init_kwargs):
        if csv_path is None:
            csv_path = "uvfe.csv"
        return super().from_csv(csv_path=csv_path, dirpath=dirpath, **init_kwargs)


class LorentzUVFeII(LorentzFeII):
    """Lorentzian UV Fe II model using uvfe.csv template."""
    @classmethod
    def from_csv(cls, csv_path=None, dirpath='./lines', **init_kwargs):
        if csv_path is None:
            csv_path = "uvfe.csv"
        return super().from_csv(csv_path=csv_path, dirpath=dirpath, **init_kwargs)


class VoigtUVFeII(VoigtFeII):
    """Voigt UV Fe II model using uvfe.csv template."""
    @classmethod
    def from_csv(cls, csv_path=None, dirpath='./lines', **init_kwargs):
        if csv_path is None:
            csv_path = "uvfe.csv"
        return super().from_csv(csv_path=csv_path, dirpath=dirpath, **init_kwargs)



# --------------
### BALMER LINES
# --------------

class _TiedBalmerBase(Fittable1DModel):
    """
    Base class for tied Balmer lines models with shared offset and width parameters.
    """
    @classmethod
    def from_csv(cls, csv_path=None, name=None, dirpath='./lines', bounds=None, amplitude=None, **init_kwargs):
        if csv_path is None:
            csv_path = os.path.join(dirpath, "balmer.csv")
        elif not os.path.isabs(csv_path):
            csv_path = os.path.join(dirpath, csv_path)
        df = pd.read_csv(csv_path)
        positions = df['position'].values
        ratios = df['int'].values
        param_names = [f'amp_{int(pos)}' for pos in positions]
        return cls.from_arrays(positions, ratios, param_names=param_names, name=name, bounds=bounds, amplitude=amplitude, **init_kwargs)

    @classmethod
    def from_arrays(cls, positions, ratios, param_names=None, name=None, bounds=None, amplitude=None, **init_kwargs):
        positions = np.asarray(positions)
        ratios = np.asarray(ratios)
        n_lines = len(positions)
        if param_names is None:
            param_names = [f'amp_{int(pos)}' for pos in positions]

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

        params = {name: Parameter(default=1.0) for name in param_names}
        for pname, default in cls._shared_params.items():
            params[pname] = Parameter(default=default)

        def evaluate(x, *args):
            amplitudes = args[:n_lines]
            shared = args[n_lines:]
            total = 0.0
            for i, (pos, ratio) in enumerate(zip(positions, ratios)):
                profile_args = cls._profile_args(pos, amplitudes[i] * ratio, *shared)
                total += cls._profile_func(x, *profile_args)
            # Zero out values at or below Balmer edge
            lambda_BE = 3646.0
            total = np.where(x <= lambda_BE, 0.0, total)
            return total
        
        def fit_deriv(x, *args):
            amplitudes = args[:n_lines]
            shared = args[n_lines:]
            
            n_params = len(args)
            n_x = len(x)
            grad = np.zeros((n_params, n_x))
            
            d_shared = [np.zeros(n_x) for _ in shared]
            
            lambda_BE = 3646.0
            mask = x > lambda_BE
            
            for i, (pos, ratio) in enumerate(zip(positions, ratios)):
                # Effective amplitude
                amp_eff = amplitudes[i] * ratio
                
                derivs = cls._profile_deriv_func(x, pos, amp_eff, *shared)
                # derivs: val, d_amp_eff, d_shared...
                
                d_amp = derivs[1] * ratio
                d_shared_line = derivs[2:]
                
                grad[i] = np.where(mask, d_amp, 0.0)
                
                for j, d in enumerate(d_shared_line):
                    d_shared[j] += np.where(mask, d, 0.0)
            
            for j, d in enumerate(d_shared):
                grad[n_lines + j] = d
                
            return list(grad)  # Return list for Astropy compound model compatibility

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': staticmethod(evaluate),
            'fit_deriv': staticmethod(fit_deriv),
            'n_inputs': 1,
            'n_outputs': 1,
            '_positions': positions,
            '_ratios': ratios,
            '_param_names_list': param_names,
            '_n_lines': n_lines,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(bounds=param_bounds, **init_kwargs)

    @property
    def fluxes(self):
        fluxes = {}
        shared_values = [getattr(self, pname) for pname in self._shared_params.keys()]
        
        for i, pname in enumerate(self._param_names_list):
            amp = getattr(self, pname)
            pos = self._positions[i]
            ratio = self._ratios[i]
            
            amp_eff = amp * ratio
            fluxes[pname] = self._calc_flux(pos, amp_eff, *shared_values)
        return fluxes

class GaussianBalmerLines(_TiedBalmerBase):
    _shared_params = {'offset': 0.0, 'fwhm': 3000.0}
    _profile_func = staticmethod(profiles.gaussian)
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        return (amp, center, sigma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        sigma = fwhm_A / sigma2fwhm
        
        val, d_amp, d_center, d_sigma = profiles.gaussian_deriv(x, amp, center, sigma)
        
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

class LorentzianBalmerLines(_TiedBalmerBase):
    _shared_params = {'offset': 0.0, 'fwhm': 3000.0}
    _profile_func = staticmethod(profiles.lorentzian)
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        return (amp, center, gamma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm):
        center = pos * (1 + offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        
        val, d_amp, d_center, d_gamma = profiles.lorentzian_deriv(x, amp, center, gamma)
        
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

class VoigtBalmerLines(_TiedBalmerBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 3000.0, 'fwhm_L': 3000.0}
    _profile_func = staticmethod(profiles.voigt)
    @staticmethod
    def _profile_args(pos, amp, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        return (amp, center, sigma, gamma)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp, offset, fwhm_G, fwhm_L):
        center = pos * (1 + offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        sigma = fwhm_G_A / sigma2fwhm
        gamma = fwhm_L_A / 2.0
        
        val, d_amp, d_center, d_sigma, d_gamma = profiles.voigt_deriv(x, amp, center, sigma, gamma)
        
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

