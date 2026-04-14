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
from dataclasses import dataclass
from pathlib import Path
import re
from scipy.interpolate import CubicSpline

import astropy.units as u
import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter

from . import profiles

c_kms = const.c.to(u.km/u.s).value # Speed of light in km/s
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


@dataclass(frozen=True)
class Metric:
    """A single measured quantity with optional uncertainty bounds.

    Attributes
    ----------
    value  : nominal (best-fit or integrated) measurement.
    std    : standard deviation (NaN when unavailable).
    lolim  : lower confidence limit (NaN when unavailable).
    uplim  : upper confidence limit (NaN when unavailable).
    """
    value: float
    std: float = float('nan')
    lolim: float = float('nan')
    uplim: float = float('nan')

    def __repr__(self) -> str:
        if not np.isfinite(self.std):
            return f"{self.value:.4g}"
        return f"{self.value:.4g} \u00b1 {self.std:.4g}  [{self.lolim:.4g}, {self.uplim:.4g}]"

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
        
    # Subclasses override to list params that govern flux magnitude.
    _flux_governing_params = ('amplitude',)

    @property
    def flux(self) -> Metric:
        """Return a Metric with flux value, uncertainty, and confidence limits."""
        flux_val = self._calc_flux()
        flux_std = self._propagate_flux_std(flux_val)
        flux_lo, flux_hi = self._propagate_flux_limits()
        return Metric(
            value=float(flux_val),
            std=float(flux_std) if flux_std is not None else float('nan'),
            lolim=float(flux_lo) if flux_lo is not None else float('nan'),
            uplim=float(flux_hi) if flux_hi is not None else float('nan'),
        )

    def eqw(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096) -> Metric:
        """Compute and store the equivalent width of the line.

        The result is stored on the model as the ``.ew`` attribute for
        subsequent access via ``model.ew.value``, ``model.ew.std``, etc.

        Parameters
        ----------
        continuum : float, callable, or None
            Local continuum level.  For ``method='constant-continuum'``, a
            scalar float is required and uncertainty is propagated from the
            flux ``Metric``.  For ``method='integration'``, a scalar float
            or any callable ``continuum(x)`` (e.g. an astropy model) is
            accepted; no uncertainty is propagated in this case.

            ``None`` (default) — auto-detect: the model is evaluated at
            the line centre (accounting for redshift and velocity offset)
            and the result is used as a constant continuum level.  For a
            line sitting on a zero baseline this equals the peak amplitude.
            This is a single-point estimate; pass an explicit value when a
            more accurate continuum level is needed.
        method : {'constant-continuum', 'integration'}
            ``'constant-continuum'`` (default): EW = flux / continuum.
            Analytic; fully propagates std, lolim, uplim.
            ``'integration'``: EW = ∫ line(x)/continuum(x) dx numerically.
            Requires a wavelength grid (x or window + num); no uncertainty
            is propagated.
        x : array-like, optional
            Wavelength grid for ``method='integration'``.
        window : (float, float), optional
            Wavelength window ``(min, max)`` used to build a grid when *x* is
            not provided (``method='integration'`` only).  Defaults to
            ``±6 × FWHM_Å`` around the line centre.
        num : int
            Number of grid points when *window* is used.

        Returns
        -------
        Metric
            EW with uncertainties (``method='constant-continuum'`` only).
            Positive for emission, negative for absorption.
            Also stored as ``self.ew``.
        """
        if continuum is None:
            center, _ = self._infer_center_fwhm_aa()
            continuum = float(np.asarray(self(np.array([center])), dtype=float).ravel()[0])
        if method == 'constant-continuum':
            continuum_val = float(continuum)
            if not np.isfinite(continuum_val) or continuum_val == 0.0:
                result = Metric(value=float('nan'))
                self.ew = result
                return result
            f = self.flux
            ew_val = f.value / continuum_val
            ew_std = float(f.std / abs(continuum_val)) if np.isfinite(f.std) else float('nan')
            ew_lo = float(f.lolim / continuum_val) if np.isfinite(f.lolim) else float('nan')
            ew_hi = float(f.uplim / continuum_val) if np.isfinite(f.uplim) else float('nan')
            result = Metric(value=ew_val, std=ew_std, lolim=ew_lo, uplim=ew_hi)

        elif method == 'integration':
            if x is None:
                center, fwhm_aa = self._infer_center_fwhm_aa()
                if window is None:
                    hw = 6.0 * fwhm_aa
                    window = (center - hw, center + hw)
                x_arr = np.linspace(float(window[0]), float(window[1]), int(num), dtype=float)
            else:
                x_arr = np.asarray(x, dtype=float)
            y_line = np.asarray(self(x_arr), dtype=float)
            if callable(continuum):
                y_cont = np.asarray(continuum(x_arr), dtype=float)
            else:
                y_cont = np.full_like(x_arr, float(continuum))
            safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
            try:
                ew_val = float(np.trapezoid(y_line / safe, x_arr))
            except AttributeError:
                ew_val = float(np.trapz(y_line / safe, x_arr))
            result = Metric(value=ew_val)

        else:
            raise ValueError("method must be 'constant-continuum' or 'integration'.")

        self.ew = result
        return result

    def _infer_center_fwhm_aa(self):
        """Return (center_Å, fwhm_Å) from current parameter values."""
        center = self.position.value * (1.0 + self.redshift.value) * np.exp(self.offset.value / c_kms)
        if hasattr(self, 'fwhm'):
            fwhm_aa = self.fwhm.value / c_kms * center
        elif hasattr(self, 'fwhm_G'):
            fwhm_aa = self.fwhm_G.value / c_kms * center
        else:
            fwhm_aa = 10.0
        return center, max(fwhm_aa, 0.1)

    def _flux_jacobian(self, flux_val):
        """Numerical Jacobian dF/dp for every free parameter."""
        from astropy.modeling.fitting import model_to_fit_params
        _, fit_indices, _ = model_to_fit_params(self)
        eps = 1e-6
        jac = np.zeros(len(fit_indices))
        all_params = self.parameters

        for j, idx in enumerate(fit_indices):
            orig = float(all_params[idx])
            delta = abs(orig) * eps if orig != 0 else eps
            all_params[idx] = orig + delta
            self.parameters = all_params
            jac[j] = (self._calc_flux() - flux_val) / delta
            all_params[idx] = orig
        self.parameters = all_params
        return jac, fit_indices

    def _propagate_flux_std(self, flux_val) -> float | None:
        """Propagate parameter uncertainties to flux std.

        Uses full covariance matrix when available (J C J^T),
        otherwise falls back to diagonal propagation.
        """
        cov = getattr(self, '_param_cov', None)
        if cov is not None:
            jac, _ = self._flux_jacobian(flux_val)
            var = float(jac @ cov @ jac)
            return np.sqrt(var) if var > 0 else None

        # Diagonal fallback: sum (dF/dp_i)^2 * std_i^2
        eps = 1e-6
        var_flux = 0.0
        has_any = False
        for pname in self.param_names:
            param = getattr(self, pname)
            if not _has_param_std(param):
                continue
            has_any = True
            orig = param.value
            delta = abs(orig) * eps if orig != 0 else eps
            param.value = orig + delta
            df = (self._calc_flux() - flux_val) / delta
            param.value = orig
            var_flux += (df * param.std) ** 2

        return np.sqrt(var_flux) if has_any and var_flux > 0 else None

    def _propagate_flux_limits(self):
        """Propagate parameter confidence limits to flux limits.

        Flux is monotonically increasing in amplitude and width parameters.
        For each limit direction (lo/hi), we use the parameter's limit if
        available, otherwise its best-fit value.  A flux limit is reported
        only when at least one governing parameter has that limit defined.
        """
        governing = list(self._flux_governing_params)
        amp_pname = governing[0]  # amplitude always gates flux magnitude
        amp_param = getattr(self, amp_pname)
        _, _, amp_has_lo, amp_has_hi = _get_param_limits(amp_param)

        if not amp_has_lo and not amp_has_hi:
            return None, None

        def _eval_at(overrides):
            orig = {}
            for p, v in overrides.items():
                orig[p] = getattr(self, p).value
                getattr(self, p).value = v
            f = self._calc_flux()
            for p, v in orig.items():
                getattr(self, p).value = v
            return f

        def _compute(use_lo):
            has_amp = amp_has_lo if use_lo else amp_has_hi
            if not has_amp:
                return None
            overrides = {amp_pname: amp_param.lolim if use_lo else amp_param.uplim}
            for p in governing[1:]:
                _, _, has_lo, has_hi = _get_param_limits(getattr(self, p))
                if use_lo and has_lo:
                    overrides[p] = getattr(self, p).lolim
                elif not use_lo and has_hi:
                    overrides[p] = getattr(self, p).uplim
            return _eval_at(overrides)

        return _compute(True), _compute(False)

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
    _flux_governing_params = ('amplitude', 'fwhm')
    
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
    _flux_governing_params = ('amplitude', 'fwhm')
    
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
    _flux_governing_params = ('amplitude', 'fwhm_G', 'fwhm_L')
    
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
        """Theoretical fluxes for all templates with uncertainties.

        Returns a DataFrame indexed by template name with columns
        ``value``, ``std``, ``lolim``, ``uplim``.
        """
        shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
        rows = []

        for tmpl, pname in zip(self._templates, self._param_names_list):
            amp = getattr(self, pname)
            positions = self._tmpl_positions[self._param_names_list.index(pname)]
            weights = self._tmpl_weights[self._param_names_list.index(pname)]

            total_flux = sum(
                self._calc_flux(pos, amp.value * wt, *shared_values)
                for pos, wt in zip(positions, weights)
            )

            flux_std = self._propagate_template_flux_std(
                pname, total_flux, positions, weights)
            flux_lo, flux_hi = self._propagate_template_flux_limits(
                pname, positions, weights)

            rows.append({
                'value': total_flux,
                'std': float(flux_std) if flux_std is not None else float('nan'),
                'lolim': float(flux_lo) if flux_lo is not None else float('nan'),
                'uplim': float(flux_hi) if flux_hi is not None else float('nan'),
            })

        return pd.DataFrame(rows, index=list(self._templates))

    def eqw(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096) -> 'pd.DataFrame':
        """Compute and store the equivalent width for all templates.

        The result is stored as the ``.ew`` attribute (a DataFrame) for
        subsequent access via ``model.ew``, ``model.ew.loc['Ha', 'value']``, etc.

        Parameters
        ----------
        continuum : float, callable, or None
            Continuum level.  For ``method='constant-continuum'``, a scalar
            float is required and uncertainty is propagated from the flux
            DataFrame.  For ``method='integration'``, a scalar float or any
            callable ``continuum(x)`` (e.g. an astropy model) is accepted;
            no uncertainty is propagated in this case.

            ``None`` (default) — auto-detect: each template is evaluated at
            its own mean observed centre and that peak value is used as the
            per-template continuum level.  This is a single-point estimate;
            pass an explicit value when a more accurate continuum level is
            needed.
        method : {'constant-continuum', 'integration'}
            ``'constant-continuum'`` (default): EW = flux / continuum per
            template.  Propagates std, lolim, uplim.
            ``'integration'``: EW = ∫ template(x)/continuum(x) dx
            numerically.  No uncertainty propagated.
        x : array-like, optional
            Wavelength grid for ``method='integration'``.
        window : (float, float), optional
            Wavelength window used when *x* is not provided.  Defaults to
            a range spanning all template positions ± 6 × FWHM.
        num : int
            Number of grid points when *window* is used.

        Returns
        -------
        pd.DataFrame
            Indexed by template name; columns ``value``, ``std``,
            ``lolim``, ``uplim``.  Positive values indicate emission.
            Also stored as ``self.ew``.
        """
        if continuum is None:
            shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
            redshift = shared_values[-1]
            flux_df = self.flux
            rows = []
            for idx, (tmpl, pname) in enumerate(zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude = getattr(self, pname).value
                center_aa = float(np.mean(positions)) * (1.0 + redshift)
                x_c = np.array([center_aa])
                peak = 0.0
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(pos, amplitude, weight, *shared_values)
                    peak += float(self._profile_func(x_c, *args).ravel()[0])
                if np.isfinite(peak) and peak > 0.0:
                    f_row = flux_df.loc[tmpl]
                    rows.append({
                        'value': float(f_row['value']) / peak,
                        'std': float(f_row['std']) / peak if np.isfinite(f_row['std']) else float('nan'),
                        'lolim': float(f_row['lolim']) / peak if np.isfinite(f_row['lolim']) else float('nan'),
                        'uplim': float(f_row['uplim']) / peak if np.isfinite(f_row['uplim']) else float('nan'),
                    })
                else:
                    rows.append({'value': float('nan'), 'std': float('nan'),
                                 'lolim': float('nan'), 'uplim': float('nan')})
            result = pd.DataFrame(rows, index=list(self._templates))
            self.ew = result
            return result
        if method == 'constant-continuum':
            continuum_val = float(continuum)
            if not np.isfinite(continuum_val) or continuum_val == 0.0:
                rows = [{'value': float('nan'), 'std': float('nan'),
                         'lolim': float('nan'), 'uplim': float('nan')}
                        for _ in self._templates]
                result = pd.DataFrame(rows, index=list(self._templates))
                self.ew = result
                return result

            flux_df = self.flux
            result = flux_df / continuum_val
            self.ew = result
            return result

        elif method == 'integration':
            shared_values = [getattr(self, pname).value for pname in self._shared_params.keys()]
            rows = []
            for idx, (tmpl, pname) in enumerate(zip(self._templates, self._param_names_list)):
                positions = self._tmpl_positions[idx]
                weights = self._tmpl_weights[idx]
                amplitude = getattr(self, pname).value

                if x is None:
                    centers = [pos * (1.0 + shared_values[-1]) for pos in positions]
                    mean_c = float(np.mean(centers))
                    if window is None:
                        shared_fwhm = shared_values[list(self._shared_params.keys()).index('fwhm')
                                                    if 'fwhm' in self._shared_params else 0] if self._shared_params else 10.0
                        fwhm_aa = (shared_fwhm / c_kms * mean_c) if self._shared_params else 10.0
                        hw = 6.0 * max(fwhm_aa, 0.1)
                        _window = (mean_c - hw, mean_c + hw)
                    else:
                        _window = window
                    x_arr = np.linspace(float(_window[0]), float(_window[1]), int(num), dtype=float)
                else:
                    x_arr = np.asarray(x, dtype=float)

                y_line = np.zeros_like(x_arr)
                for pos, weight in zip(positions, weights):
                    args = self._single_profile_args(pos, amplitude, weight, *shared_values)
                    y_line += self._profile_func(x_arr, *args)

                if callable(continuum):
                    y_cont = np.asarray(continuum(x_arr), dtype=float)
                else:
                    y_cont = np.full_like(x_arr, float(continuum))
                safe = np.where(np.abs(y_cont) > 0.0, y_cont, np.nan)
                try:
                    ew_val = float(np.trapezoid(y_line / safe, x_arr))
                except AttributeError:
                    ew_val = float(np.trapz(y_line / safe, x_arr))
                rows.append({'value': ew_val, 'std': float('nan'),
                             'lolim': float('nan'), 'uplim': float('nan')})

            result = pd.DataFrame(rows, index=list(self._templates))
            self.ew = result
            return result

        else:
            raise ValueError("method must be 'constant-continuum' or 'integration'.")

    def _template_flux(self, amp_pname, positions, weights):
        """Total flux for one template at current parameter values."""
        shared = [getattr(self, sp).value for sp in self._shared_params.keys()]
        amp = getattr(self, amp_pname).value
        return sum(
            self._calc_flux(pos, amp * wt, *shared)
            for pos, wt in zip(positions, weights)
        )

    def _propagate_template_flux_std(self, amp_pname, flux_val, positions, weights):
        """Propagate parameter std to template flux.

        Uses full covariance when available, else diagonal std.
        """
        cov = getattr(self, '_param_cov', None)
        if cov is not None:
            return self._propagate_template_flux_std_cov(
                amp_pname, flux_val, positions, weights, cov)

        eps = 1e-6
        var_flux = 0.0
        has_any = False
        relevant = [amp_pname] + list(self._shared_params.keys())

        for p_name in relevant:
            param = getattr(self, p_name)
            if not _has_param_std(param):
                continue
            has_any = True
            orig = param.value
            delta = abs(orig) * eps if orig != 0 else eps
            param.value = orig + delta
            df = (self._template_flux(amp_pname, positions, weights) - flux_val) / delta
            param.value = orig
            var_flux += (df * param.std) ** 2

        return np.sqrt(var_flux) if has_any and var_flux > 0 else None

    def _propagate_template_flux_std_cov(self, amp_pname, flux_val, positions, weights, cov):
        """Covariance-aware flux std for one template."""
        from astropy.modeling.fitting import model_to_fit_params
        _, fit_indices, _ = model_to_fit_params(self)
        eps = 1e-6
        jac = np.zeros(len(fit_indices))
        all_params = self.parameters

        for j, idx in enumerate(fit_indices):
            orig = float(all_params[idx])
            delta = abs(orig) * eps if orig != 0 else eps
            all_params[idx] = orig + delta
            self.parameters = all_params
            jac[j] = (self._template_flux(amp_pname, positions, weights) - flux_val) / delta
            all_params[idx] = orig
        self.parameters = all_params

        var = float(jac @ cov @ jac)
        return np.sqrt(var) if var > 0 else None

    def _propagate_template_flux_limits(self, amp_pname, positions, weights):
        """Propagate parameter limits to template flux using monotonicity.

        Amplitude gates whether each direction's limit is computed.
        Width parameters contribute their CI if available, else best-fit.
        """
        width_params = [sp for sp in self._shared_params
                        if sp not in ('offset', 'redshift')]
        amp_param = getattr(self, amp_pname)
        _, _, amp_has_lo, amp_has_hi = _get_param_limits(amp_param)

        if not amp_has_lo and not amp_has_hi:
            return None, None

        def _eval_at(overrides):
            orig = {}
            for p, v in overrides.items():
                orig[p] = getattr(self, p).value
                getattr(self, p).value = v
            f = self._template_flux(amp_pname, positions, weights)
            for p, v in orig.items():
                getattr(self, p).value = v
            return f

        def _compute(use_lo):
            has_amp = amp_has_lo if use_lo else amp_has_hi
            if not has_amp:
                return None
            overrides = {amp_pname: amp_param.lolim if use_lo else amp_param.uplim}
            for p in width_params:
                _, _, has_lo, has_hi = _get_param_limits(getattr(self, p))
                if use_lo and has_lo:
                    overrides[p] = getattr(self, p).lolim
                elif not use_lo and has_hi:
                    overrides[p] = getattr(self, p).uplim
            return _eval_at(overrides)

        return _compute(True), _compute(False)

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
