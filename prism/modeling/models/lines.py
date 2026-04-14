"""
prism.modeling.models.lines — emission-line spectral models.

Provides three families of Astropy-compliant emission-line models:

Single-line models
------------------
``GaussianLine``, ``LorentzianLine``, ``VoigtLine``
    Each wraps the corresponding profile (Gaussian, Lorentzian, Voigt) and
    parameterises the line by rest-frame ``position`` (fixed), velocity
    ``offset`` [km s⁻¹], ``fwhm`` [km s⁻¹], and ``amplitude``.
    Instrumental broadening is accepted via the ``instfwhm`` argument
    (scalar or Nx2 wavelength-FWHM table); the instrument FWHM is added
    in quadrature to the intrinsic width.

Line-group models
-----------------
``GaussianLines``, ``LorentzianLines``, ``VoigtLines``
    Dynamically created subclasses of ``LineGroupBase`` that collect many
    lines from a CSV table.  Lines with the same ``name`` column share a
    single amplitude parameter (weighted relative to each other by the
    ``weight`` column).  Constructed via:
    - ``from_csv(csv_files, ...)`` — load from package or local CSV files.
    - ``from_arrays(names, pos, weights, ...)`` — build from arrays.
    - ``from_templates(df, ...)`` — build from a pandas DataFrame.

Global filters
--------------
``set_wavelength_range(wmin, wmax)``
    Set a global wavelength filter applied when loading CSV line lists.
``setup_local_lines(wmin, wmax, dirpath, ...)``
    Create a filtered local copy of the package line CSV files.
"""
import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import re
from scipy.interpolate import CubicSpline

import astropy.units as u
import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter

from . import profiles

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))

script_dir = os.path.dirname(__file__) # get the directory of the current script
resource_path = os.path.join(script_dir, "..", "..", "..", "resources", "lines")
csv_lines_path = resource_path

# Global wavelength range for filtering
_wmin = 0.0
_wmax = np.inf

def set_wavelength_range(wmin=None, wmax=None):
    """
    Sets the global wavelength range for line filtering.
    """
    global _wmin, _wmax
    if wmin is not None:
        _wmin = wmin
    if wmax is not None:
        _wmax = wmax
    print(f"Wavelength range set to: [{_wmin}, {_wmax}]")

def setup_local_lines(wmin=4000, wmax=7000, dirpath='./lines', overwrite=False):
    """
    Initializes the lines by reading the csv files from the input folder and filtering them based on the wavelength range.
    Also updates the global wavelength range.
    """
    # Sync global limits with provided ones
    set_wavelength_range(wmin=wmin, wmax=wmax)

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
            if not {'name', 'pos'}.issubset(df.columns):
                raise ValueError(f"CSV format not recognized in {files}. Required columns: name, pos.")
            if 'weight' not in df.columns:
                df['weight'] = 1.0
                df['name'] = _make_unique(df['name'])
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


def _voigt_eval_and_deriv(x, amplitude, center, sigma, gamma):
    """Return Voigt value and derivatives with a stable 5-tuple signature."""
    out = profiles.voigt_deriv(x, amplitude, center, sigma, gamma)
    if len(out) == 5:
        return out
    val, d_amp, d_center, d_gamma = out
    d_sigma = np.zeros_like(val)
    return val, d_amp, d_center, d_sigma, d_gamma

class LineModelBase(Fittable1DModel):
    """Base class for line models providing common derivative logic."""

    @staticmethod
    def _normalize_instfwhm(instfwhm):
        if np.isscalar(instfwhm):
            value = float(instfwhm)
            return value, 'constant', value, None

        arr = np.asarray(instfwhm, dtype=float)
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(
                    "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array."
                )
            arr = arr.reshape(1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array."
            )

        if arr.shape[0] == 1:
            return arr.copy(), 'constant', float(arr[0, 1]), None

        order = np.argsort(arr[:, 0])
        arr = arr[order]
        wl = arr[:, 0]
        fwhm = arr[:, 1]

        if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(fwhm)):
            raise ValueError("instfwhm wavelength/FWHM values must be finite.")
        if np.any(np.diff(wl) <= 0):
            raise ValueError("instfwhm wavelengths must be strictly increasing.")

        spline = CubicSpline(wl, fwhm, extrapolate=True)
        return arr.copy(), 'spline', spline, spline.derivative()

    def __init__(self, *args, instfwhm=0.0, **kwargs):
        raw, mode, value_or_spline, deriv = self._normalize_instfwhm(instfwhm)
        self._instfwhm_raw = raw
        self._instfwhm_mode = mode
        if mode == 'constant':
            self._instfwhm_const = float(value_or_spline)
            self._instfwhm_spline = None
            self._instfwhm_spline_deriv = None
        else:
            self._instfwhm_const = None
            self._instfwhm_spline = value_or_spline
            self._instfwhm_spline_deriv = deriv
        self.instfwhm = raw
        super().__init__(*args, **kwargs)

    def instfwhm_val(self, center):
        if self._instfwhm_mode == 'constant':
            center_arr = np.asarray(center, dtype=float)
            return center_arr * 0.0 + self._instfwhm_const
        return self._instfwhm_spline(center)

    def instfwhm_deriv(self, center):
        if self._instfwhm_mode == 'constant':
            center_arr = np.asarray(center, dtype=float)
            return center_arr * 0.0
        return self._instfwhm_spline_deriv(center)
        
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
    
    _parameter_units = {'position': u.AA, 'offset': u.km/u.s, 'fwhm': u.km/u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)

        amplitude_eff = amplitude / (1.0 + redshift)
        return profiles.gaussian(x, amplitude_eff, center, sigma_eff)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_deriv = self.instfwhm_deriv(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)

        amplitude_eff = amplitude / (1.0 + redshift)

        val, d_amp_eff, d_center, d_sigma_eff = profiles.gaussian_deriv(x, amplitude_eff, center, sigma_eff)

        d_amp = d_amp_eff / (1.0 + redshift)

        d_center_d_pos = (1.0 + redshift) * np.exp(offset / c_kms)
        d_center_d_off = center / c_kms

        safe_sigma_eff = np.where(sigma_eff != 0, sigma_eff, 1.0)
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / safe_sigma_eff, 0.0)
        d_sigma_eff_d_sigma_inst = np.where(sigma_eff != 0, sigma_inst / safe_sigma_eff, 0.0)
        d_sigma_inst_d_center = (instfwhm_val + instfwhm_deriv * center) / (c_kms * sigma2fwhm)

        d_sigma_intrinsic_d_pos = (fwhm / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_intrinsic_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhm = (center / c_kms) / sigma2fwhm

        d_sigma_eff_d_pos = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_pos
                             + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_pos)
        d_sigma_eff_d_off = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
                             + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_off)
        d_sigma_eff_d_fwhm = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhm

        d_position = d_center * d_center_d_pos + d_sigma_eff * d_sigma_eff_d_pos
        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off
        d_fwhm = d_sigma_eff * d_sigma_eff_d_fwhm

        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_inst_d_redshift = d_sigma_inst_d_center * d_center_d_redshift
        d_sigma_eff_d_redshift = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
                                  + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_redshift)

        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift
        
        return [d_amp, d_position, d_offset, d_fwhm, d_redshift]

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_A = self.fwhm.value / c_kms * center
        instfwhm_val = self.instfwhm_val(center)
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
    
    _parameter_units = {'position': u.AA, 'offset': u.km/u.s, 'fwhm': u.km/u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        instfwhm_val = self.instfwhm_val(center)
        sigma_inst = (instfwhm_val / c_kms * center) / sigma2fwhm

        amplitude_eff = amplitude / (1.0 + redshift)
        if sigma_inst == 0.0:
            return profiles.lorentzian(x, amplitude_eff, center, gamma)

        return profiles.voigt(x, amplitude_eff, center, sigma_inst, gamma)

    def fit_deriv(self, x, amplitude, position, offset, fwhm, redshift):
        params = [amplitude, position, offset, fwhm, redshift]
        base = self.evaluate(x, *params)
        derivs = []
        eps = 1e-6

        for i, pval in enumerate(params):
            delta = abs(pval) * eps if pval != 0 else eps
            shifted = list(params)
            shifted[i] = pval + delta
            derivs.append((self.evaluate(x, *shifted) - base) / delta)

        return derivs

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_A = self.fwhm.value / c_kms * center
        gamma = fwhm_A / 2.0
        instfwhm_val = self.instfwhm_val(center)
        sigma_inst = (instfwhm_val / c_kms * center) / sigma2fwhm

        amplitude_eff = self.amplitude.value / (1.0 + self.redshift.value)
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amplitude_eff, gamma)

        return profiles.voigt_flux(amplitude_eff, sigma_inst, gamma)

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
    
    _parameter_units = {'position': u.AA, 'offset': u.km/u.s, 'fwhm_G': u.km/u.s, 'fwhm_L': u.km/u.s}

    def evaluate(self, x, amplitude, position, offset, fwhm_G, fwhm_L, redshift):
        center = position * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
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
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_deriv = self.instfwhm_deriv(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = amplitude / (1.0 + redshift)

        val, d_amp_eff, d_center, d_profile_sigma_eff, d_profile_gamma = _voigt_eval_and_deriv(
            x,
            amplitude_eff,
            center,
            sigma_eff,
            gamma,
        )

        d_amp = d_amp_eff / (1.0 + redshift)
        
        d_center_d_pos = (1.0 + redshift) * np.exp(offset / c_kms)
        d_center_d_off = center / c_kms
        
        safe_sigma_eff = np.where(sigma_eff != 0, sigma_eff, 1.0)
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / safe_sigma_eff, 0.0)
        d_sigma_eff_d_sigma_inst = np.where(sigma_eff != 0, sigma_inst / safe_sigma_eff, 0.0)
        d_sigma_inst_d_center = (instfwhm_val + instfwhm_deriv * center) / (c_kms * sigma2fwhm)
        
        d_sigma_intrinsic_d_pos = (fwhm_G / c_kms * d_center_d_pos) / sigma2fwhm
        d_sigma_intrinsic_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhmG = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_pos = (
            d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_pos
            + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_pos
        )
        d_sigma_eff_d_off = (
            d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
            + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_off
        )
        d_sigma_eff_d_fwhmG = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhmG
        
        d_gamma_d_pos = (fwhm_L / c_kms * d_center_d_pos) / 2.0
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_position = (
            d_center * d_center_d_pos
            + d_profile_sigma_eff * d_sigma_eff_d_pos
            + d_profile_gamma * d_gamma_d_pos
        )
        d_offset = (
            d_center * d_center_d_off
            + d_profile_sigma_eff * d_sigma_eff_d_off
            + d_profile_gamma * d_gamma_d_off
        )
        d_fwhmG = d_profile_sigma_eff * d_sigma_eff_d_fwhmG
        d_fwhmL = d_profile_gamma * d_gamma_d_fwhmL
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_inst_d_redshift = d_sigma_inst_d_center * d_center_d_redshift
        d_sigma_eff_d_redshift = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
                                  + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_redshift)
        d_gamma_d_redshift = gamma / (1.0 + redshift)

        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = (
            d_amp_eff * d_amp_eff_d_redshift
            + d_center * d_center_d_redshift
            + d_profile_sigma_eff * d_sigma_eff_d_redshift
            + d_profile_gamma * d_gamma_d_redshift
        )
        
        return [d_amp, d_position, d_offset, d_fwhmG, d_fwhmL, d_redshift]

    def _calc_flux(self):
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        fwhm_G_A = self.fwhm_G.value / c_kms * center
        fwhm_L_A = self.fwhm_L.value / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
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
    # Remove all whitespace
    name = re.sub(r'\s+', '', name)
    # Replace dashes with underscores
    name = name.replace('-', '_')
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

class LineGroupBase(LineModelBase):
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
            df_curr = pd.read_csv(path)
            if not {'name', 'pos'}.issubset(df_curr.columns):
                raise ValueError(f"CSV format not recognized in {path}. Required columns: name, pos.")
            if 'weight' not in df_curr.columns:
                df_curr['weight'] = 1.0
                df_curr['name'] = _make_unique(df_curr['name'])
            dfs.append(df_curr)
            
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
            names = _make_unique(names)
        else:
            weights = np.atleast_1d(weights)
            
        df = pd.DataFrame({'name': names, 'pos': pos, 'weight': weights})
        return cls.from_templates(df, name=name, bounds=bounds, amplitude=amplitude, instfwhm=instfwhm, **init_kwargs)

    @classmethod
    def from_templates(cls, df, name=None, bounds=None, amplitude=None, instfwhm=0.0, **init_kwargs):
        # Filter based on global wavelength range
        if 'pos' in df.columns:
            df = df[(df.pos >= _wmin) & (df.pos <= _wmax)]
        
        if df.empty:
            raise ValueError(f"No lines found in the range [{_wmin}, {_wmax}]")

        templates = pd.unique(df['name'])

        _raw_param_names = [f"amp_{_clean_name(tmpl)}" for tmpl in templates]
        param_names = _make_unique(_raw_param_names)
        n_templates = len(templates)

        # Pre-cache per-template positions and weights as numpy arrays
        _tmpl_positions = []
        _tmpl_weights = []
        for tmpl in templates:
            df_tmpl = df[df['name'] == tmpl]
            _tmpl_positions.append(df_tmpl['pos'].values)
            _tmpl_weights.append(df_tmpl['weight'].values)

        for pname in param_names:
            if amplitude is not None and pname not in init_kwargs:
                init_kwargs[pname] = amplitude

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

        shared_units = getattr(cls, '_shared_units', {})
        params = {pname: Parameter(default=1.0) for pname in param_names}
        for pname, default in cls._shared_params.items():
            is_fixed = True if pname == 'redshift' else False
            params[pname] = Parameter(default=default, fixed=is_fixed)

        def evaluate(self, x, *args):
            amplitudes = args[:n_templates]
            shared = args[n_templates:]

            x_arr = np.atleast_1d(x)
            total = np.zeros_like(x_arr, dtype=float)

            for i in range(n_templates):
                positions = _tmpl_positions[i]
                weights = _tmpl_weights[i]
                for j in range(len(positions)):
                    profile_args = self._single_profile_args(
                        positions[j], amplitudes[i], weights[j], *shared)
                    total += cls._profile_func(x_arr, *profile_args)

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

            for i in range(n_templates):
                positions = _tmpl_positions[i]
                weights = _tmpl_weights[i]
                for j in range(len(positions)):
                    derivs = self._single_profile_deriv(
                        x_arr, positions[j], amplitudes[i], weights[j], *shared)
                    grad[i] += derivs[1] * weights[j]
                    for k, d in enumerate(derivs[2:]):
                        d_shared[k] += d

            for j, d in enumerate(d_shared):
                grad[n_templates + j] = d

            return list(grad)

        model_class = type(cls.__name__, (cls,), {
            **params,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            'n_inputs': 1,
            'n_outputs': 1,
            '_df': df,
            '_templates': templates,
            '_n_templates': n_templates,
            '_param_names_list': param_names,
            '_tmpl_positions': _tmpl_positions,
            '_tmpl_weights': _tmpl_weights,
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
                total_flux += self._calc_flux(pos, amp_eff, *shared_values)
                
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
                    flux_plus += self._calc_flux(pos, amp_eff, *shared_values_plus)
                
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
                flux_plus += self._calc_flux(pos, amp_eff, *shared_values_plus)
                
            flux_values.append(flux_plus)
            
            for pname, oval in orig.items():
                getattr(self, pname).value = oval
                
        return np.min(flux_values), np.max(flux_values)

class GaussianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km/u.s, 'fwhm': u.km/u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.gaussian)
    
    def _single_profile_args(self, pos, amp_template, weight, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)

        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, sigma_eff)
    
    def _single_profile_deriv(self, x, pos, amp_template, weight, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_deriv = self.instfwhm_deriv(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)

        amplitude_eff = (amp_template * weight) / (1.0 + redshift)

        val, d_amp_eff, d_center, d_sigma_eff = profiles.gaussian_deriv(x, amplitude_eff, center, sigma_eff)

        d_amp = d_amp_eff * (weight / (1.0 + redshift))

        d_center_d_off = center / c_kms

        safe_sigma_eff = np.where(sigma_eff != 0, sigma_eff, 1.0)
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / safe_sigma_eff, 0.0)
        d_sigma_eff_d_sigma_inst = np.where(sigma_eff != 0, sigma_inst / safe_sigma_eff, 0.0)
        d_sigma_inst_d_center = (instfwhm_val + instfwhm_deriv * center) / (c_kms * sigma2fwhm)

        d_sigma_intrinsic_d_off = (fwhm / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhm = (center / c_kms) / sigma2fwhm

        d_sigma_eff_d_off = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
                             + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_off)
        d_sigma_eff_d_fwhm = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhm

        d_offset = d_center * d_center_d_off + d_sigma_eff * d_sigma_eff_d_off
        d_fwhm = d_sigma_eff * d_sigma_eff_d_fwhm

        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_inst_d_redshift = d_sigma_inst_d_center * d_center_d_redshift
        d_sigma_eff_d_redshift = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
                                  + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_redshift)

        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = d_amp_eff * d_amp_eff_d_redshift + d_center * d_center_d_redshift + d_sigma_eff * d_sigma_eff_d_redshift
        
        return val, d_amp, d_offset, d_fwhm, d_redshift

    def _calc_flux(self, pos, amp, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)

        amplitude_eff = amp / (1.0 + redshift)
        return profiles.gaussian_flux(amplitude_eff, sigma_eff)

class LorentzianLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km/u.s, 'fwhm': u.km/u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)

    def _single_profile_args(self, pos, amp_template, weight, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        instfwhm_val = self.instfwhm_val(center)
        sigma_inst = (instfwhm_val / c_kms * center) / sigma2fwhm

        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, sigma_inst, gamma)

    def _single_profile_deriv(self, x, pos, amp_template, weight, offset, fwhm, redshift):
        eps = 1e-6

        def _eval(amp_t, off, fw, z):
            args = self._single_profile_args(pos, amp_t, weight, off, fw, z)
            return self._profile_func(x, *args)

        val = _eval(amp_template, offset, fwhm, redshift)

        d_amp_step = abs(amp_template) * eps if amp_template != 0 else eps
        d_off_step = abs(offset) * eps if offset != 0 else eps
        d_fwhm_step = abs(fwhm) * eps if fwhm != 0 else eps
        d_z_step = abs(redshift) * eps if redshift != 0 else eps

        d_amp_true = (_eval(amp_template + d_amp_step, offset, fwhm, redshift) - val) / d_amp_step
        d_offset = (_eval(amp_template, offset + d_off_step, fwhm, redshift) - val) / d_off_step
        d_fwhm = (_eval(amp_template, offset, fwhm + d_fwhm_step, redshift) - val) / d_fwhm_step
        d_redshift = (_eval(amp_template, offset, fwhm, redshift + d_z_step) - val) / d_z_step

        if weight != 0:
            d_amp = d_amp_true / weight
        else:
            d_amp = np.zeros_like(val)

        return val, d_amp, d_offset, d_fwhm, d_redshift

    def _calc_flux(self, pos, amp, offset, fwhm, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_A = fwhm / c_kms * center
        gamma = fwhm_A / 2.0
        instfwhm_val = self.instfwhm_val(center)
        sigma_inst = (instfwhm_val / c_kms * center) / sigma2fwhm

        amplitude_eff = amp / (1.0 + redshift)
        if sigma_inst == 0.0:
            return profiles.lorentzian_flux(amplitude_eff, gamma)

        return profiles.voigt_flux(amplitude_eff, sigma_inst, gamma)


class VoigtLines(LineGroupBase):
    _shared_params = {'offset': 0.0, 'fwhm_G': 1000.0, 'fwhm_L': 1000.0, 'redshift': 0.0}
    _shared_units = {'offset': u.km/u.s, 'fwhm_G': u.km/u.s, 'fwhm_L': u.km/u.s, 'redshift': None}
    _profile_func = staticmethod(profiles.voigt)
    
    def _single_profile_args(self, pos, amp_template, weight, offset, fwhm_G, fwhm_L, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)
        return (amplitude_eff, center, sigma_eff, gamma)

    def _single_profile_deriv(self, x, pos, amp_template, weight, offset, fwhm_G, fwhm_L, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_deriv = self.instfwhm_deriv(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = (amp_template * weight) / (1.0 + redshift)

        val, d_amp_eff, d_center, d_profile_sigma_eff, d_profile_gamma = _voigt_eval_and_deriv(
            x,
            amplitude_eff,
            center,
            sigma_eff,
            gamma,
        )

        d_amp = d_amp_eff * (weight / (1.0 + redshift))
        
        d_center_d_off = center / c_kms
        
        safe_sigma_eff = np.where(sigma_eff != 0, sigma_eff, 1.0)
        d_sigma_eff_d_sigma_intrinsic = np.where(sigma_eff != 0, sigma_intrinsic / safe_sigma_eff, 0.0)
        d_sigma_eff_d_sigma_inst = np.where(sigma_eff != 0, sigma_inst / safe_sigma_eff, 0.0)
        d_sigma_inst_d_center = (instfwhm_val + instfwhm_deriv * center) / (c_kms * sigma2fwhm)
        
        d_sigma_intrinsic_d_off = (fwhm_G / c_kms * d_center_d_off) / sigma2fwhm
        d_sigma_intrinsic_d_fwhmG = (center / c_kms) / sigma2fwhm
        
        d_sigma_eff_d_off = (
            d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_off
            + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_center * d_center_d_off
        )
        d_sigma_eff_d_fwhmG = d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_fwhmG
        
        d_gamma_d_off = (fwhm_L / c_kms * d_center_d_off) / 2.0
        d_gamma_d_fwhmL = (center / c_kms) / 2.0
        
        d_offset = (
            d_center * d_center_d_off
            + d_profile_sigma_eff * d_sigma_eff_d_off
            + d_profile_gamma * d_gamma_d_off
        )
        d_fwhmG = d_profile_sigma_eff * d_sigma_eff_d_fwhmG
        d_fwhmL = d_profile_gamma * d_gamma_d_fwhmL
        
        d_center_d_redshift = center / (1.0 + redshift)
        d_sigma_intrinsic_d_redshift = sigma_intrinsic / (1.0 + redshift)
        d_sigma_inst_d_redshift = d_sigma_inst_d_center * d_center_d_redshift
        d_sigma_eff_d_redshift = (d_sigma_eff_d_sigma_intrinsic * d_sigma_intrinsic_d_redshift
                                  + d_sigma_eff_d_sigma_inst * d_sigma_inst_d_redshift)
        
        d_gamma_d_redshift = gamma / (1.0 + redshift)

        d_amp_eff_d_redshift = -amplitude_eff / (1.0 + redshift)
        d_redshift = (
            d_amp_eff * d_amp_eff_d_redshift
            + d_center * d_center_d_redshift
            + d_profile_sigma_eff * d_sigma_eff_d_redshift
            + d_profile_gamma * d_gamma_d_redshift
        )
        
        return val, d_amp, d_offset, d_fwhmG, d_fwhmL, d_redshift

    def _calc_flux(self, pos, amp, offset, fwhm_G, fwhm_L, redshift):
        center = pos * (1.0 + redshift) * np.exp(offset / c_kms)
        fwhm_G_A = fwhm_G / c_kms * center
        fwhm_L_A = fwhm_L / c_kms * center
        
        instfwhm_val = self.instfwhm_val(center)
        instfwhm_A = instfwhm_val / c_kms * center
        
        sigma_intrinsic = fwhm_G_A / sigma2fwhm
        sigma_inst = instfwhm_A / sigma2fwhm
        sigma_eff = np.sqrt(sigma_intrinsic**2 + sigma_inst**2)
        
        gamma = fwhm_L_A / 2.0
        amplitude_eff = amp / (1.0 + redshift)
        return profiles.voigt_flux(amplitude_eff, sigma_eff, gamma)
