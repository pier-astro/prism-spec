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
import pandas as pd

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

script_dir = os.path.dirname(__file__) # get the directory of the current script
resource_path = os.path.join(script_dir, "..", "..", "..", "resources", "lines")
csv_lines_path = resource_path

def setup_local_lines(wmin=4000, wmax=7000, dirpath='./lines', overwrite=False):
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
        for files in glob.glob(resource_path + "/*.csv"):
            df = pd.read_csv(files)
            if 'pos' in df.columns:
                df = df[df.pos > wmin]
                df = df[df.pos < wmax]
            name = os.path.join(dirpath, Path(files).name)
            df.to_csv(name, index=False)

# ------
# MODELS
# ------

def _has_param_std(param) -> bool:
    """Check if parameter has valid std."""
    return hasattr(param, 'std') and param.std is not None and np.isfinite(param.std)

def _get_param_limits(param):
    """
    Extract limits from a parameter.
    Returns (lo_val, hi_val, has_finite_lo, has_finite_hi)
    """
    has_lo = hasattr(param, 'lolim') and param.lolim is not None and np.isfinite(param.lolim)
    has_hi = hasattr(param, 'uplim') and param.uplim is not None and np.isfinite(param.uplim)
    lo = param.lolim if has_lo else param.value
    hi = param.uplim if has_hi else param.value
    return lo, hi, has_lo, has_hi

class LineModelBase(Fittable1DModel):
    """Base class for line models providing common derivative logic."""
    def __init__(self, *args, instfwhm=0.0, **kwargs):
        self.instfwhm = instfwhm
        super().__init__(*args, **kwargs)
        
    @property
    def flux(self):
        """Return a pandas.Series containing flux value, uncertainty, and bounds."""
        flux_val = self._calc_flux()
        flux_std = self._compute_flux_std(flux_val)
        flux_limits = self._compute_flux_limits()
        
        data = {'value': flux_val}
        if flux_std is not None:
            data['std'] = flux_std
        if flux_limits[0] is not None:
            data['lolim'] = flux_limits[0]
        if flux_limits[1] is not None:
            data['uplim'] = flux_limits[1]
        
        return pd.Series(data)

    def _compute_flux_std(self, flux_val) -> float:
        """
        Compute flux standard deviation from covariance using numerical parameter perturbation.
        var(flux) = sum(df/dp_i)^2 * var(p_i)
        """
        eps = 1e-6
        var_flux = 0.0
        has_any_std = False
        
        for pname in self.param_names:
            param = getattr(self, pname)
            if _has_param_std(param):
                has_any_std = True
                orig_val = param.value
                delta = orig_val * eps if orig_val != 0 else eps
                
                # Perturb parameter forward
                param.value = orig_val + delta
                flux_plus = self._calc_flux()
                # Restore parameter
                param.value = orig_val
                
                d_flux_dp = (flux_plus - flux_val) / delta
                var_flux += (d_flux_dp * param.std) ** 2
                
        return np.sqrt(var_flux) if has_any_std and var_flux > 0 else None

    def _compute_flux_limits(self):
        """
        Compute flux limits from parameter limits for single line model.
        Evaluates flux at parameter limit corners locally.
        """
        import itertools
        param_limits = []
        has_any_limit = False
        
        # We find combinations only for relevant active parameters that govern strength
        for pname in self.param_names:
            if pname in ['position', 'redshift', 'offset']:
                continue
            param = getattr(self, pname)
            lo, hi, has_lo, has_hi = _get_param_limits(param)
            if has_lo or has_hi:
                has_any_limit = True
            param_limits.append((lo, hi))
            
        if not has_any_limit:
            return None, None
            
        test_params = [p for p in self.param_names if p not in ['position', 'redshift', 'offset']]
        
        combos = list(itertools.product(*param_limits))
        flux_values = []
        
        for combo in combos:
            orig = {}
            for pname, pval in zip(test_params, combo):
                param = getattr(self, pname)
                orig[pname] = param.value
                param.value = pval
                
            flux_values.append(self._calc_flux())
            
            for pname, oval in orig.items():
                getattr(self, pname).value = oval
                
        return np.min(flux_values), np.max(flux_values)

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
    redshift = Parameter(default=0.0, fixed=True)

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = amplitude / (1.0 + redshift)
        return profiles.gaussian(x, amplitude_eff, center, sigma_eff)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = amplitude / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_sigma_eff = profiles.gaussian_deriv(x, amplitude_eff, center, sigma_eff)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_pos = (1.0 + redshift) * np.exp(offset / c_kms)
        d_center_d_off = center / c_kms
        
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / np.where(sigma_eff != 0, sigma_eff, 1.0), 0.0)
        
        d_sigma_intrinsic_d_pos = (fwhm / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_intrinsic_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhm = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_pos = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_pos
        d_sigma_eff_d_off = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
        d_sigma_eff_d_fwhm = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhm
        
        d_position = d_center * d_center_d_pos + d_sigma_eff * d_sigma_eff_d_pos
        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off
        d_fwhm = d_sigma_eff * d_sigma_eff_d_fwhm
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_eff_d_redshift = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
        
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift
        
        return [d_amp, d_position, d_offset, d_fwhm, d_redshift]

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_A = self.fwhm.value / c_kms * center
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = self.amplitude.value / (1.0 + self.redshift.value)
        return profiles.gaussian_flux(amplitude_eff, sigma_eff)

class LorentzianLine(LineModelBase):
    """
    Lorentzian line model for spectra, using rest wavelength and velocity offset.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm = Parameter(default=1000.0)
    redshift = Parameter(default=0.0, fixed=True)

    @staticmethod
    def evaluate(x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = amplitude / (1.0 + redshift)
        return profiles.lorentzian(x, amplitude_eff, center, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = amplitude / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_gamma = profiles.lorentzian_deriv(x, amplitude_eff, center, gamma)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_pos = (1.0 + redshift) * np.exp(offset / c_kms)
        d_center_d_off = center / c_kms
        
        d_gamma_d_pos = (fwhm / c_kms * d_center_d_pos) / 2.0
        d_gamma_d_off = (fwhm / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhm = (center / c_kms) / 2.0
        
        d_position = d_center * d_center_d_pos + d_gamma * d_gamma_d_pos
        d_offset = d_center * d_center_d_off + d_gamma * d_gamma_d_off
        d_fwhm = d_gamma * d_gamma_d_fwhm
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_gamma_d_redshift = gamma / (1.0 + redshift)
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_gamma * d_gamma_d_redshift
        
        return [d_amp, d_position, d_offset, d_fwhm, d_redshift]

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_A = self.fwhm.value / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = self.amplitude.value / (1.0 + self.redshift.value)
        return profiles.lorentzian_flux(amplitude_eff, gamma)

class VoigtLine(LineModelBase):
    """
    Voigt line model for spectra, using rest wavelength and velocity offset.
    """
    amplitude = Parameter(default=1.0)
    position = Parameter(default=5000.0, fixed=True)
    offset = Parameter(default=0.0)
    fwhm_G = Parameter(default=1000.0)
    fwhm_L = Parameter(default=1000.0)
    redshift = Parameter(default=0.0, fixed=True)

    def evaluate(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = amplitude / (1.0 + redshift)
        return profiles.voigt(x, amplitude_eff, center, sigma_eff, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = amplitude / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_sigma_eff, d_gamma = profiles.voigt_deriv(x, amplitude_eff, center, sigma_eff, gamma)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_pos = (1.0 + redshift) * np.exp(offset / c_kms)
        d_center_d_off = center / c_kms
        
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / np.where(sigma_eff != 0, sigma_eff, 1.0), 0.0)
        
        d_sigma_intrinsic_d_pos = (fwhm_G / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_intrinsic_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhmG = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_pos = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_pos
        d_sigma_eff_d_off = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
        d_sigma_eff_d_fwhmG = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhmG
        
        d_gamma_d_pos = (fwhm_L / c_kms * d_center_d_pos) / 2.0
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_position = d_center * d_center_d_pos + d_sigma_eff * d_sigma_eff_d_pos + d_gamma * d_gamma_d_pos
        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off + d_gamma * d_gamma_d_off
        d_fwhmG = d_sigma_eff * d_sigma_eff_d_fwhmG
        d_fwhmL = d_gamma * d_gamma_d_fwhmL
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_eff_d_redshift = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
        d_gamma_d_redshift = gamma / (1.0 + redshift)
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift + d_gamma * d_gamma_d_redshift
        
        return [d_amp, d_position, d_offset, d_fwhmG, d_fwhmL, d_redshift]

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_G_A = self.fwhm_G.value / c_kms * center
        fwhm_L_A = self.fwhm_L.value / c_kms * center
        
        instfwhm_val = self.instfwhm(center) if callable(self.instfwhm) else self.instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = self.amplitude.value / (1.0 + self.redshift.value)
        return profiles.voigt_flux(amplitude_eff, sigma_eff, gamma)

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
    def from_csv(cls, csv_files, name=None, dirpath=None, bounds=None, amplitude=None, instfwhm=0.0, **init_kwargs):
        if not isinstance(csv_files, (list, tuple)):
            csv_files = [csv_files]
        
        if dirpath is None:
            dirpath = csv_lines_path

        dfs = []
        for f in csv_files:
            path = f if os.path.isabs(f) else os.path.join(dirpath, f)
            dfs.append(pd.read_csv(path))
            
        df = pd.concat(dfs, ignore_index=True)
        return cls.from_templates(df, name=name, bounds=bounds, amplitude=amplitude, instfwhm=instfwhm, **init_kwargs)

    @classmethod
    def from_arrays(cls, names, pos, weights=None, name=None, bounds=None, amplitude=None, instfwhm=0.0, **init_kwargs):
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
        return cls.from_templates(df, name=name, bounds=bounds, amplitude=amplitude, instfwhm=instfwhm, **init_kwargs)

    @classmethod
    def from_templates(cls, df, name=None, bounds=None, amplitude=None, instfwhm=0.0, **init_kwargs):
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
            is_fixed = True if pname == 'redshift' else False
            params[pname] = Parameter(default=default, fixed=is_fixed)

        def evaluate(self, x, *args):
            amplitudes = args[:n_templates]
            shared = args[n_templates:]
            
            x_arr = np.atleast_1d(x)
            total = np.zeros_like(x_arr, dtype=float)
            x_col = x_arr[:, np.newaxis]
            
            for i, tmpl in enumerate(templates):
                df_tmpl = df[df['name'] == tmpl]
                positions = df_tmpl['pos'].values
                weights = df_tmpl['weight'].values
                
                pos_row = positions[np.newaxis, :]
                wt_row = weights[np.newaxis, :]
                
                profile_args = cls._profile_args(pos_row, amplitudes[i], wt_row, *shared, instfwhm=self.instfwhm)
                components = cls._profile_func(x_col, *profile_args)
                total += np.sum(components, axis=1)
                
            if np.ndim(x) == 0:
                return total[0]
            return total
        
        def fit_deriv(self, x, *args):
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
                
                derivs = cls._profile_deriv_func(x_col, pos_row, amplitudes[i], wt_row, *shared, instfwhm=self.instfwhm)
                
                d_amp_tmpl = np.sum(derivs[1] * wt_row, axis=1)
                grad[i] = d_amp_tmpl
                
                for j, d in enumerate(derivs[2:]):
                    d_shared[j] += np.sum(d, axis=1)
            
            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d
                
            return list(grad)

        def __init__(self, *args, instfwhm=0.0, **kwargs):
            self.instfwhm = instfwhm
            super(model_class, self).__init__(*args, **kwargs)

        model_class = type(cls.__name__, (cls,), {
            **params,
            '__init__': __init__,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            'n_inputs': 1,
            'n_outputs': 1,
            '_df': df,
            '_templates': templates,
            '_n_templates': n_templates,
            '_param_names_list': param_names,
        })
        if name is not None:
            init_kwargs['name'] = name
        return model_class(bounds=param_bounds, instfwhm=instfwhm, **init_kwargs)

    @property
    def lines(self):
        """
        Return the DataFrame containing the physical lines included in this model group.
        """
        return self._df

    @property
    def flux(self) -> pd.DataFrame:
        """
        Get theoretical fluxes for all fitted templates with uncertainties.
        Returns a pandas DataFrame where index is template name.
        """
        data = []
        shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
        
        for i, (tmpl, pname) in enumerate(zip(self._templates, self._param_names_list)):
            amp = getattr(self, pname)
            df_tmpl = self._df[self._df['name'] == tmpl]
            positions = df_tmpl['pos'].values
            weights = df_tmpl['weight'].values
            
            # 1) Calculate Value
            total_flux = 0.0
            for pos, wt in zip(positions, weights):
                amp_eff = amp.value * wt
                total_flux += self._calc_flux(pos, amp_eff, *shared_values, instfwhm=self.instfwhm)
                
            # 2) Calculate Std (numerical parameter perturbation)
            flux_std = self._compute_group_flux_std(tmpl, pname, total_flux, positions, weights)
            
            # 3) Calculate limits
            flux_limits = self._compute_group_flux_limits(pname, positions, weights)
            
            row = {'value': total_flux}
            if flux_std is not None:
                row['std'] = flux_std
            if flux_limits[0] is not None:
                row['lolim'] = flux_limits[0]
            if flux_limits[1] is not None:
                row['uplim'] = flux_limits[1]
                
            data.append((tmpl, row))
            
        df = pd.DataFrame([r[1] for r in data], index=[r[0] for r in data])
        return df

    def _compute_group_flux_std(self, tmpl, amp_pname, flux_val, positions, weights) -> float:
        """Numerical parameter perturbation for a single template group."""
        eps = 1e-6
        var_flux = 0.0
        has_any_std = False
        
        # We need to perturb the specific amplitude and all shared params
        relevant_params = [amp_pname] + list(self._shared_params.keys())
        
        for p_name in relevant_params:
            param = getattr(self, p_name)
            if _has_param_std(param):
                has_any_std = True
                orig_val = param.value
                delta = orig_val * eps if orig_val != 0 else eps
                
                # Perturb parameter forward
                param.value = orig_val + delta
                
                # Recalculate flux
                flux_plus = 0.0
                shared_values_plus = [getattr(self, sp).value for sp in self._shared_params.keys()]
                amp_plus = getattr(self, amp_pname)
                
                for pos, wt in zip(positions, weights):
                    amp_eff = amp_plus.value * wt
                    flux_plus += self._calc_flux(pos, amp_eff, *shared_values_plus, instfwhm=self.instfwhm)
                
                # Restore parameter
                param.value = orig_val
                
                d_flux_dp = (flux_plus - flux_val) / delta
                var_flux += (d_flux_dp * param.std) ** 2
                
        return np.sqrt(var_flux) if has_any_std and var_flux > 0 else None

    def _compute_group_flux_limits(self, amp_pname, positions, weights):
        """Compute flux limits by iterating bound corners."""
        import itertools
        param_limits = []
        has_any_limit = False
        relevant_params = [amp_pname] + [sp for sp in self._shared_params.keys() if sp not in ['offset', 'redshift']]
        
        for p_name in relevant_params:
            param = getattr(self, p_name)
            lo, hi, has_lo, has_hi = _get_param_limits(param)
            if has_lo or has_hi:
                has_any_limit = True
            param_limits.append((lo, hi))
            
        if not has_any_limit:
            return None, None
            
        combos = list(itertools.product(*param_limits))
        flux_values = []
        
        for combo in combos:
            orig = {}
            for pname, pval in zip(relevant_params, combo):
                param = getattr(self, pname)
                orig[pname] = param.value
                param.value = pval
                
            flux_plus = 0.0
            shared_values_plus = [getattr(self, sp).value for sp in self._shared_params.keys()]
            amp_plus = getattr(self, amp_pname)
            
            for pos, wt in zip(positions, weights):
                amp_eff = amp_plus.value * wt
                flux_plus += self._calc_flux(pos, amp_eff, *shared_values_plus, instfwhm=self.instfwhm)
                
            flux_values.append(flux_plus)
            
            for pname, oval in orig.items():
                getattr(self, pname).value = oval
                
        return np.min(flux_values), np.max(flux_values)

class GaussianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _profile_func = staticmethod(profiles.gaussian)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, sigma_eff)
    
    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_sigma_eff = profiles.gaussian_deriv(x, amplitude_eff, center, sigma_eff)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_off = center / c_kms
        
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / np.where(sigma_eff != 0, sigma_eff, 1.0), 0.0)
        
        d_sigma_intrinsic_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhm = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_off = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
        d_sigma_eff_d_fwhm = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhm
        
        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off
        d_fwhm = d_sigma_eff * d_sigma_eff_d_fwhm
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_eff_d_redshift = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
        
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift
        
        return val, d_amp, d_offset, d_fwhm, d_redshift

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        amplitude_eff = amp / (1.0 + redshift)
        return profiles.gaussian_flux(amplitude_eff, sigma_eff)

class LorentzianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _profile_func = staticmethod(profiles.lorentzian)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_gamma = profiles.lorentzian_deriv(x, amplitude_eff, center, gamma)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_off = center / c_kms
        d_gamma_d_off = (fwhm / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhm = (center / c_kms) / 2.0
        
        d_offset = d_center * d_center_d_off + d_gamma * d_gamma_d_off
        d_fwhm = d_gamma * d_gamma_d_fwhm
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_gamma_d_redshift = gamma / (1.0 + redshift)
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_gamma * d_gamma_d_redshift
        
        return val, d_amp, d_offset, d_fwhm, d_redshift

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        amplitude_eff = amp / (1.0 + redshift)
        return profiles.lorentzian_flux(amplitude_eff, gamma)


class VoigtLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0, 'fwhm_L': 1000.0, 'redshift': 0.0}
    _profile_func = staticmethod(profiles.voigt)
    
    @staticmethod
    def _profile_args(pos, amp_template, weight, offset, fwhm_G, fwhm_L, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, sigma_eff, gamma)

    @staticmethod
    def _profile_deriv_func(x, pos, amp_template, weight, offset, fwhm_G, fwhm_L, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        
        val, d_amp_eff, d_center, d_sigma_eff, d_gamma = profiles.voigt_deriv(x, amplitude_eff, center, sigma_eff, gamma)
        
        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_off = center / c_kms
        
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / np.where(sigma_eff != 0, sigma_eff, 1.0), 0.0)
        
        d_sigma_intrinsic_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhmG = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_off = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
        d_sigma_eff_d_fwhmG = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhmG
        
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off + d_gamma * d_gamma_d_off
        d_fwhmG = d_sigma_eff * d_sigma_eff_d_fwhmG
        d_fwhmL = d_gamma * d_gamma_d_fwhmL
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_eff_d_redshift = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
        
        d_gamma_d_redshift = gamma / (1.0 + redshift)
        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift + d_gamma * d_gamma_d_redshift
        
        return val, d_amp, d_offset, d_fwhmG, d_fwhmL, d_redshift

    @staticmethod
    def _calc_flux(pos, amp, offset, fwhm_G, fwhm_L, redshift, instfwhm=0.0):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = instfwhm(center) if callable(instfwhm) else instfwhm
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = amp / (1.0 + redshift)
        return profiles.voigt_flux(amplitude_eff, sigma_eff, gamma)
