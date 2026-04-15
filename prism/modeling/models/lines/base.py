"""
prism.modeling.models.lines.base — base classes, Metric, and CSV utilities.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy.interpolate import CubicSpline

import astropy.units as u
import astropy.constants as const
from astropy.modeling import Fittable1DModel, Parameter

from . import profiles

c_kms = profiles.C_KMS
sigma2fwhm = profiles.SIGMA2FWHM

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    """A single measured quantity with optional uncertainty bounds."""
    value: float
    std: float = float('nan')
    lolim: float = float('nan')
    uplim: float = float('nan')

    def __repr__(self) -> str:
        if not np.isfinite(self.std):
            return f"{self.value:.4g}"
        return f"{self.value:.4g} \u00b1 {self.std:.4g}  [{self.lolim:.4g}, {self.uplim:.4g}]"

# ---------------------------------------------------------------------------
# CSV / wavelength-range globals
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(__file__)
resource_path = os.path.join(script_dir, "..", "..", "..", "..", "resources", "lines")
csv_lines_path = resource_path

_wmin = 0.0
_wmax = np.inf


def set_wavelength_range(wmin=None, wmax=None):
    """Set the global wavelength range for line filtering."""
    global _wmin, _wmax
    if wmin is not None:
        _wmin = wmin
    if wmax is not None:
        _wmax = wmax
    print(f"Wavelength range set to: [{_wmin}, {_wmax}]")


def setup_local_lines(wmin=4000, wmax=7000, dirpath='./lines', overwrite=False):
    """Read CSV line lists, filter by wavelength range, and write local copies."""
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_name(name):
    """Convert scientific name to pythonic attribute name."""
    name = str(name).strip()
    name = re.sub(r'[\[\]<>]', '', name)
    name = re.sub(r'\s+', '', name)
    name = name.replace('-', '_')
    return name.lower()


def _make_unique(names):
    """Make a list of unique names by appending suffixes to duplicates."""
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


def _has_param_std(param) -> bool:
    return hasattr(param, 'std') and param.std is not None and np.isfinite(param.std)


def _get_param_limits(param):
    has_lo = hasattr(param, 'lolim') and param.lolim is not None and np.isfinite(param.lolim)
    has_hi = hasattr(param, 'uplim') and param.uplim is not None and np.isfinite(param.uplim)
    lo = param.lolim if has_lo else param.value
    hi = param.uplim if has_hi else param.value
    return lo, hi, has_lo, has_hi

# ---------------------------------------------------------------------------
# LineModelBase
# ---------------------------------------------------------------------------

class LineModelBase(Fittable1DModel):
    """Base class for line models providing instfwhm handling and flux/EW/propagation."""

    @staticmethod
    def _normalize_instfwhm(instfwhm):
        if np.isscalar(instfwhm):
            value = float(instfwhm)
            return value, 'constant', value, None

        arr = np.asarray(instfwhm, dtype=float)
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError(
                    "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array.")
            arr = arr.reshape(1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "instfwhm must be a scalar or a 2-column [wavelength, fwhm] array.")

        if arr.shape[0] == 1:
            return arr.copy(), 'constant', float(arr[0, 1]), None

        order = np.argsort(arr[:, 0])
        arr = arr[order]
        wl, fwhm = arr[:, 0], arr[:, 1]
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
            return np.asarray(center, dtype=float) * 0.0 + self._instfwhm_const
        return self._instfwhm_spline(center)

    def instfwhm_deriv(self, center):
        if self._instfwhm_mode == 'constant':
            return np.asarray(center, dtype=float) * 0.0
        return self._instfwhm_spline_deriv(center)

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
        """Compute the equivalent width of the line.

        Parameters
        ----------
        continuum : float, callable, or None
            Local continuum level.
        method : {'constant-continuum', 'integration'}
        x : array-like, optional
        window : (float, float), optional
        num : int

        Returns
        -------
        Metric
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
        center = profiles.observed_center(
            self.position.value, self.offset.value, self.redshift.value)
        if hasattr(self, 'fwhm'):
            fwhm_aa = self.fwhm.value / c_kms * center
        elif hasattr(self, 'fwhm_G'):
            fwhm_aa = self.fwhm_G.value / c_kms * center
        else:
            fwhm_aa = 10.0
        return center, max(fwhm_aa, 0.1)

    def _flux_jacobian(self, flux_val):
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
        cov = getattr(self, '_param_cov', None)
        if cov is not None:
            jac, _ = self._flux_jacobian(flux_val)
            var = float(jac @ cov @ jac)
            return np.sqrt(var) if var > 0 else None

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
        governing = list(self._flux_governing_params)
        amp_pname = governing[0]
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
