"""
Flux computation and uncertainty propagation for line models.

Refactored module with unified profile handling via registry pattern.

ConvolvedModel support: Functions automatically unwrap ConvolvedModel to access
the underlying source model for flux extraction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Callable
from . import profiles

# Constants
C_KMS = 299792.458  # Speed of light in km/s
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))

__all__ = ['Flux', 'flux_from_samples', 'extract_fluxes', 'get_flux_params',
           'get_flux_with_uncertainty', 'compute_flux_value', 'compute_flux_std', 
           'compute_flux_limits', 'compute_feii_flux_std', 'compute_feii_flux_limits']


# ============================================================================
# Flux dataclass
# ============================================================================

@dataclass
class Flux:
    """
    Container for flux value with uncertainties.
    
    Attributes
    ----------
    value : float
        Flux value
    std : float or None
        Standard deviation (from covariance propagation)
    lolim : float or None
        Lower confidence limit (from bootstrap or parameter limits)
    uplim : float or None  
        Upper confidence limit (from bootstrap or parameter limits)
    method : str or None
        How uncertainty was computed: 'covariance', 'limits', 'samples', None
    """
    value: float
    std: Optional[float] = None
    lolim: Optional[float] = None
    uplim: Optional[float] = None
    method: Optional[str] = None
    
    @property
    def has_uncertainty(self) -> bool:
        """True if any uncertainty information is available."""
        return self.std is not None or self.lolim is not None or self.uplim is not None
    
    @property
    def is_constrained(self) -> bool:
        """True if both limits are finite (not hitting parameter boundaries)."""
        if self.lolim is None and self.uplim is None:
            return True
        lower_ok = self.lolim is None or np.isfinite(self.lolim)
        upper_ok = self.uplim is None or np.isfinite(self.uplim)
        return lower_ok and upper_ok
    
    @property
    def boundary_flags(self) -> Optional[str]:
        """Return which boundaries are hit: 'lower', 'upper', 'both', or None."""
        lower_hit = self.lolim is not None and not np.isfinite(self.lolim)
        upper_hit = self.uplim is not None and not np.isfinite(self.uplim)
        if lower_hit and upper_hit:
            return 'both'
        elif lower_hit:
            return 'lower'
        elif upper_hit:
            return 'upper'
        return None
    
    def __repr__(self) -> str:
        if self.lolim is not None and self.uplim is not None:
            return f"{self.value:.4g} [{self.lolim:.4g}, {self.uplim:.4g}]"
        elif self.std is not None:
            return f"{self.value:.4g} ± {self.std:.4g}"
        return f"{self.value:.4g}"
    
    def __float__(self) -> float:
        return float(self.value)


# ============================================================================
# Profile Registry - unified handling of different profile types
# ============================================================================

def _get_profile_info(model):
    """
    Identify profile type and return relevant info.
    
    Returns
    -------
    dict with keys:
        'type': 'gaussian', 'lorentzian', 'voigt', or None
        'width_params': list of width parameter names
        'flux_func': function(amp, *width_values) -> flux
        'width_to_physical': function(model) -> dict of physical widths (sigma, gamma)
    """
    model_class = model.__class__.__name__
    
    if 'Gaussian' in model_class and 'Voigt' not in model_class:
        return {
            'type': 'gaussian',
            'width_params': ['fwhm'],
            'flux_func': profiles.gaussian_flux,
            'get_physical_width': lambda m: _gaussian_physical_width(m),
        }
    elif 'Lorentz' in model_class:
        return {
            'type': 'lorentzian', 
            'width_params': ['fwhm'],
            'flux_func': profiles.lorentzian_flux,
            'get_physical_width': lambda m: _lorentzian_physical_width(m),
        }
    elif 'Voigt' in model_class:
        return {
            'type': 'voigt',
            'width_params': ['fwhm_G', 'fwhm_L'],
            'flux_func': profiles.voigt_flux,
            'get_physical_width': lambda m: _voigt_physical_width(m),
        }
    return None


def _gaussian_physical_width(model):
    """Convert model fwhm to physical sigma."""
    center = model.position * (1 + model.offset / C_KMS)
    fwhm_A = model.fwhm / C_KMS * center
    sigma = fwhm_A / SIGMA2FWHM
    return {'sigma': sigma, 'center': center}


def _lorentzian_physical_width(model):
    """Convert model fwhm to physical gamma."""
    center = model.position * (1 + model.offset / C_KMS)
    fwhm_A = model.fwhm / C_KMS * center
    gamma = fwhm_A / 2.0
    return {'gamma': gamma, 'center': center}


def _voigt_physical_width(model):
    """Convert model fwhm_G, fwhm_L to physical sigma, gamma."""
    center = model.position * (1 + model.offset / C_KMS)
    fwhm_G_A = model.fwhm_G / C_KMS * center
    fwhm_L_A = model.fwhm_L / C_KMS * center
    sigma = fwhm_G_A / SIGMA2FWHM
    gamma = fwhm_L_A / 2.0
    return {'sigma': sigma, 'gamma': gamma, 'center': center}


# ============================================================================
# Parameter limit utilities
# ============================================================================

def _get_param_limits(param) -> Tuple[Optional[float], Optional[float], bool, bool]:
    """
    Extract limits from a parameter.
    
    Returns
    -------
    (lo_val, hi_val, has_finite_lo, has_finite_hi)
    """
    has_lo = hasattr(param, 'lolim') and param.lolim is not None and np.isfinite(param.lolim)
    has_hi = hasattr(param, 'uplim') and param.uplim is not None and np.isfinite(param.uplim)
    lo = param.lolim if has_lo else param.value
    hi = param.uplim if has_hi else param.value
    return lo, hi, has_lo, has_hi


def _has_param_std(param) -> bool:
    """Check if parameter has valid std."""
    return hasattr(param, 'std') and param.std is not None and np.isfinite(param.std)


# ============================================================================
# Flux computation for single line models
# ============================================================================

def compute_flux_value(model) -> float:
    """Compute flux value for a single line model."""
    info = _get_profile_info(model)
    if info is None:
        raise ValueError(f"Unknown model type: {model.__class__.__name__}")
    
    physical = info['get_physical_width'](model)
    amp = model.amplitude.value if hasattr(model.amplitude, 'value') else model.amplitude
    
    if info['type'] == 'gaussian':
        return profiles.gaussian_flux(amp, physical['sigma'])
    elif info['type'] == 'lorentzian':
        return profiles.lorentzian_flux(amp, physical['gamma'])
    elif info['type'] == 'voigt':
        return profiles.voigt_flux(amp, physical['sigma'], physical['gamma'])


def compute_flux_std(model) -> Optional[float]:
    """
    Compute flux standard deviation from covariance for single line model.
    
    Uses linear error propagation: var(flux) = sum(df/dp_i)^2 * var(p_i)
    """
    info = _get_profile_info(model)
    if info is None:
        return None
    
    amp = model.amplitude
    if not _has_param_std(amp):
        # Check if any width param has std
        has_any_std = any(_has_param_std(getattr(model, p)) for p in info['width_params'])
        if not has_any_std:
            return None
    
    physical = info['get_physical_width'](model)
    flux_val = compute_flux_value(model)
    
    # Compute derivatives numerically (simpler and more maintainable than analytic)
    eps = 1e-6
    var_flux = 0.0
    
    # Amplitude contribution (linear: flux = amp * f(width))
    if _has_param_std(amp):
        d_flux_d_amp = flux_val / amp.value if amp.value != 0 else 0
        var_flux += (d_flux_d_amp * amp.std) ** 2
    
    # Width parameter contributions
    for wpname in info['width_params']:
        wparam = getattr(model, wpname)
        if _has_param_std(wparam):
            # Numerical derivative
            orig_val = wparam.value
            wparam.value = orig_val * (1 + eps)
            flux_plus = compute_flux_value(model)
            wparam.value = orig_val
            
            delta = orig_val * eps if orig_val != 0 else eps
            d_flux_d_w = (flux_plus - flux_val) / delta
            var_flux += (d_flux_d_w * wparam.std) ** 2
    
    return np.sqrt(var_flux) if var_flux > 0 else None


def compute_flux_limits(model) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute flux limits from parameter limits for single line model.
    
    Evaluates flux at parameter limit corners and takes min/max.
    """
    info = _get_profile_info(model)
    if info is None:
        return None, None
    
    amp = model.amplitude
    amp_lo, amp_hi, has_amp_lo, has_amp_hi = _get_param_limits(amp)
    
    # Collect width parameter limits
    width_limits = []
    has_all_lo = has_amp_lo
    has_all_hi = has_amp_hi
    
    for wpname in info['width_params']:
        wparam = getattr(model, wpname)
        lo, hi, has_lo, has_hi = _get_param_limits(wparam)
        width_limits.append((lo, hi))
        has_all_lo = has_all_lo and has_lo
        has_all_hi = has_all_hi and has_hi
    
    # No limits at all
    if not (has_amp_lo or has_amp_hi or any(w[0] != w[1] for w in width_limits)):
        return None, None
    
    # Generate corner combinations and evaluate flux
    import itertools
    
    amp_vals = [amp_lo, amp_hi]
    width_combos = list(itertools.product(*[(lo, hi) for lo, hi in width_limits]))
    
    flux_values = []
    center = model.position * (1 + model.offset / C_KMS)
    
    for a in amp_vals:
        for widths in width_combos:
            if info['type'] == 'gaussian':
                fwhm_A = widths[0] / C_KMS * center
                sigma = fwhm_A / SIGMA2FWHM
                flux_values.append(profiles.gaussian_flux(a, sigma))
            elif info['type'] == 'lorentzian':
                fwhm_A = widths[0] / C_KMS * center
                gamma = fwhm_A / 2.0
                flux_values.append(profiles.lorentzian_flux(a, gamma))
            elif info['type'] == 'voigt':
                fwhm_G_A = widths[0] / C_KMS * center
                fwhm_L_A = widths[1] / C_KMS * center
                sigma = fwhm_G_A / SIGMA2FWHM
                gamma = fwhm_L_A / 2.0
                flux_values.append(profiles.voigt_flux(a, sigma, gamma))
    
    flux_lo = np.min(flux_values)
    flux_hi = np.max(flux_values)
    
    # If unbounded, use nan
    if not has_all_lo:
        flux_lo = np.nan
    if not has_all_hi:
        flux_hi = np.nan
    
    return flux_lo, flux_hi


def get_flux_with_uncertainty(model) -> Flux:
    """
    Compute flux with uncertainties for a single line model.
    
    This is the main function called by model.flux property.
    """
    flux_val = compute_flux_value(model)
    flux_std = compute_flux_std(model)
    flux_lo, flux_hi = compute_flux_limits(model)
    
    method = None
    if flux_std is not None:
        method = 'covariance'
    elif flux_lo is not None or flux_hi is not None:
        method = 'limits'
    
    return Flux(flux_val, std=flux_std, lolim=flux_lo, uplim=flux_hi, method=method)


# ============================================================================
# Flux from bootstrap samples
# ============================================================================

def flux_from_samples(model, samples: Dict[str, np.ndarray], confidence: float = 68, 
                      return_samples: bool = False):
    """
    Compute flux distributions from bootstrap parameter samples.
    
    Works with single line models, tied line models, compound models.
    Automatically unwraps ConvolvedModel if present.
    
    Parameters
    ----------
    model : Line model or CompoundModel
        Fitted model with uncertainties attached
    samples : dict
        Parameter samples from bootstrap {param_name: array}
    confidence : float
        Confidence level in percent (default: 68)
    return_samples : bool
        If True, also return flux sample arrays
        
    Returns
    -------
    Flux or dict of Flux (optionally with sample arrays)
    """
    from astropy.modeling.core import CompoundModel
    from .convolved import ConvolvedModel
    
    # Unwrap ConvolvedModel
    if isinstance(model, ConvolvedModel):
        model = model._source
    
    if isinstance(model, CompoundModel):
        return _flux_from_samples_compound(model, samples, confidence, return_samples)
    
    # Determine model type
    if hasattr(model, 'flux') and not hasattr(model, 'fluxes'):
        return _flux_from_samples_single(model, samples, confidence, return_samples)
    elif hasattr(model, 'fluxes'):
        return _flux_from_samples_multi(model, samples, confidence, return_samples)
    
    raise ValueError("Model must have .flux or .fluxes property")


def _flux_from_samples_single(model, samples, confidence, return_samples):
    """Compute flux samples for single line model."""
    n_samples = len(next(iter(samples.values())))
    alpha = (100 - confidence) / 2
    
    flux_samples = np.zeros(n_samples)
    
    # Get relevant parameter names
    param_map = {p: samples[p] for p in model.param_names if p in samples}
    
    for i in range(n_samples):
        try:
            # Store and set
            orig = {p: getattr(model, p).value for p in param_map}
            for p, arr in param_map.items():
                getattr(model, p).value = arr[i]
            
            flux_obj = model.flux
            flux_samples[i] = flux_obj.value if isinstance(flux_obj, Flux) else flux_obj
            
            # Restore
            for p, v in orig.items():
                getattr(model, p).value = v
        except Exception:
            flux_samples[i] = np.nan
    
    valid = flux_samples[~np.isnan(flux_samples)]
    if len(valid) == 0:
        flux_obj = model.flux
        flux_val = flux_obj.value if isinstance(flux_obj, Flux) else flux_obj
        result = Flux(flux_val)
    else:
        flux_val = np.median(valid)
        flux_std = np.std(valid)
        flux_lo = np.percentile(valid, alpha)
        flux_hi = np.percentile(valid, 100 - alpha)
        
        # Check boundary flags
        flux_lo, flux_hi = _apply_boundary_flags_single(model, flux_lo, flux_hi)
        result = Flux(flux_val, std=flux_std, lolim=flux_lo, uplim=flux_hi, method='samples')
    
    return (result, flux_samples) if return_samples else result


def _flux_from_samples_multi(model, samples, confidence, return_samples):
    """Compute flux samples for tied line or FeII model."""
    n_samples = len(next(iter(samples.values())))
    alpha = (100 - confidence) / 2
    
    # Get flux keys from model
    current_fluxes = model.fluxes
    flux_keys = list(current_fluxes.keys())
    
    flux_samples_dict = {k: np.zeros(n_samples) for k in flux_keys}
    
    for i in range(n_samples):
        try:
            # Store and set all params
            orig = {}
            for key, arr in samples.items():
                if '[' in key:
                    # Array param: param[idx]
                    base = key.split('[')[0]
                    idx = int(key.split('[')[1].rstrip(']'))
                    if hasattr(model, base):
                        param = getattr(model, base)
                        if base not in orig:
                            orig[base] = param.value.copy() if hasattr(param.value, 'copy') else param.value
                        if hasattr(param.value, '__setitem__'):
                            param.value[idx] = arr[i]
                else:
                    if hasattr(model, key):
                        param = getattr(model, key)
                        orig[key] = param.value
                        param.value = arr[i]
            
            fluxes = model.fluxes
            for k in flux_keys:
                flux_obj = fluxes[k]
                flux_samples_dict[k][i] = flux_obj.value if isinstance(flux_obj, Flux) else flux_obj
            
            # Restore
            for p, v in orig.items():
                getattr(model, p).value = v
        except Exception:
            for k in flux_keys:
                flux_samples_dict[k][i] = np.nan
    
    # Compute statistics
    results = {}
    for k in flux_keys:
        valid = flux_samples_dict[k][~np.isnan(flux_samples_dict[k])]
        if len(valid) == 0:
            flux_val = current_fluxes[k].value if isinstance(current_fluxes[k], Flux) else current_fluxes[k]
            results[k] = Flux(flux_val)
        else:
            flux_val = np.median(valid)
            flux_std = np.std(valid)
            flux_lo = np.percentile(valid, alpha)
            flux_hi = np.percentile(valid, 100 - alpha)
            
            # Apply boundary flags
            flux_lo, flux_hi = _apply_boundary_flags_multi(model, k, flux_lo, flux_hi)
            results[k] = Flux(flux_val, std=flux_std, lolim=flux_lo, uplim=flux_hi, method='samples')
    
    return (results, flux_samples_dict) if return_samples else results


def _apply_boundary_flags_single(model, flux_lo, flux_hi):
    """Apply boundary flags for single line model."""
    amp = model.amplitude
    if hasattr(amp, 'lolim') and not np.isfinite(amp.lolim):
        flux_lo = np.nan
    if hasattr(amp, 'uplim') and not np.isfinite(amp.uplim):
        flux_hi = np.nan
    
    # Check width params
    for wpname in ['fwhm', 'fwhm_G', 'fwhm_L']:
        if hasattr(model, wpname):
            wp = getattr(model, wpname)
            if hasattr(wp, 'lolim') and not np.isfinite(wp.lolim):
                flux_lo = np.nan
            if hasattr(wp, 'uplim') and not np.isfinite(wp.uplim):
                flux_hi = np.nan
    
    return flux_lo, flux_hi


def _apply_boundary_flags_multi(model, flux_key, flux_lo, flux_hi):
    """Apply boundary flags for tied/FeII model."""
    # Get amplitude param for this flux
    if hasattr(model, '_param_names_list'):
        # Tied line model
        amp = getattr(model, flux_key)
    elif hasattr(model, '_templates'):
        # FeII model
        amp = getattr(model, flux_key) if hasattr(model, flux_key) else getattr(model, f'amp_{flux_key}', None)
    else:
        amp = None
    
    if amp is not None:
        if hasattr(amp, 'lolim') and not np.isfinite(amp.lolim):
            flux_lo = np.nan
        if hasattr(amp, 'uplim') and not np.isfinite(amp.uplim):
            flux_hi = np.nan
    
    # Check shared params
    shared_params = getattr(model, '_shared_params', {})
    for pname in shared_params.keys():
        param = getattr(model, pname)
        if hasattr(param, 'lolim') and not np.isfinite(param.lolim):
            flux_lo = np.nan
        if hasattr(param, 'uplim') and not np.isfinite(param.uplim):
            flux_hi = np.nan
    
    return flux_lo, flux_hi


# ============================================================================
# Compound model support
# ============================================================================

def extract_fluxes(model) -> Dict[str, Flux]:
    """
    Extract all fluxes from a model (single or compound).
    
    Returns dict {name: Flux} for all line components.
    Automatically unwraps ConvolvedModel if present.
    """
    from astropy.modeling.core import CompoundModel
    from .convolved import ConvolvedModel
    
    # Unwrap ConvolvedModel
    if isinstance(model, ConvolvedModel):
        model = model._source
    
    if isinstance(model, CompoundModel):
        return _extract_fluxes_compound(model)
    
    # Single model
    if hasattr(model, 'fluxes'):
        fluxes = model.fluxes
        # Strip 'amp_' prefix for cleaner naming
        return {k.replace('amp_', '', 1) if k.startswith('amp_') else k: v 
                for k, v in fluxes.items()}
    elif hasattr(model, 'flux'):
        return {'flux': model.flux}
    
    return {}


def _iter_components(model):
    """
    Iterate over compound model components depth-first.
    
    Yields (index, component) matching Astropy's numbering.
    """
    from astropy.modeling.core import CompoundModel
    
    idx = [0]
    
    def traverse(m):
        if isinstance(m, CompoundModel):
            if hasattr(m, 'left') and m.left is not None:
                yield from traverse(m.left)
            if hasattr(m, 'right') and m.right is not None:
                yield from traverse(m.right)
        else:
            yield (idx[0], m)
            idx[0] += 1
    
    yield from traverse(model)


def _extract_fluxes_compound(model) -> Dict[str, Flux]:
    """Extract fluxes from compound model with component indices."""
    fluxes = {}
    
    for comp_idx, comp in _iter_components(model):
        if hasattr(comp, 'fluxes'):
            for name, flux in comp.fluxes.items():
                clean = name.replace('amp_', '', 1) if name.startswith('amp_') else name
                fluxes[f"{clean}_{comp_idx}"] = flux
        elif hasattr(comp, 'flux'):
            try:
                fluxes[f"flux_{comp_idx}"] = comp.flux
            except Exception:
                pass
    
    return fluxes


def _flux_from_samples_compound(model, samples, confidence, return_samples):
    """Compute flux samples for compound model."""
    all_fluxes = {}
    all_samples = {}
    
    for comp_idx, comp in _iter_components(model):
        if not (hasattr(comp, 'flux') or hasattr(comp, 'fluxes')):
            continue
        
        # Filter samples for this component
        suffix = f"_{comp_idx}"
        comp_samples = {}
        for key, val in samples.items():
            if key.endswith(suffix):
                comp_samples[key[:-len(suffix)]] = val
        
        if not comp_samples:
            continue
        
        try:
            result = flux_from_samples(comp, comp_samples, confidence, return_samples=return_samples)
            
            if return_samples and isinstance(result, tuple):
                fluxes_dict, samples_dict = result
                if isinstance(fluxes_dict, dict):
                    for name, flux in fluxes_dict.items():
                        clean = name.replace('amp_', '', 1) if name.startswith('amp_') else name
                        key = f"{clean}_{comp_idx}"
                        all_fluxes[key] = flux
                        all_samples[key] = samples_dict[name]
                else:
                    all_fluxes[f"flux_{comp_idx}"] = fluxes_dict
                    all_samples[f"flux_{comp_idx}"] = samples_dict
            else:
                if isinstance(result, dict):
                    for name, flux in result.items():
                        clean = name.replace('amp_', '', 1) if name.startswith('amp_') else name
                        all_fluxes[f"{clean}_{comp_idx}"] = flux
                else:
                    all_fluxes[f"flux_{comp_idx}"] = result
        except Exception:
            pass
    
    return (all_fluxes, all_samples) if return_samples else all_fluxes


# ============================================================================
# FeII-specific flux computation (templates with multiple lines)
# ============================================================================

def compute_feii_flux_std(model, template_name: str) -> Optional[float]:
    """
    Compute flux std for an FeII template using error propagation.
    
    Parameters
    ----------
    model : FeII model instance
    template_name : str
        Name of the template (multiplet)
        
    Returns
    -------
    float or None
        Standard deviation of template flux
    """
    amp_param_name = f'amp_{template_name}'
    if not hasattr(model, amp_param_name):
        return None
    
    amp = getattr(model, amp_param_name)
    if not _has_param_std(amp):
        # Check shared params
        has_any_std = any(_has_param_std(getattr(model, p)) for p in model._shared_params.keys())
        if not has_any_std:
            return None
    
    # Get template data
    df_tmpl = model._df[model._df['ime'] == template_name]
    positions = df_tmpl['wav'].values
    intensities = df_tmpl['Int'].values
    shared_values = [getattr(model, pname).value for pname in model._shared_params.keys()]
    
    # Current flux
    total_flux = 0.0
    for pos, intensity in zip(positions, intensities):
        amp_eff = amp.value * intensity
        total_flux += model._calc_flux(pos, amp_eff, *shared_values)
    
    # Simple amplitude-proportional propagation (conservative)
    if _has_param_std(amp) and amp.value != 0:
        flux_std = total_flux / amp.value * amp.std
        return flux_std
    
    return None


def compute_feii_flux_limits(model, template_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute flux limits for an FeII template from parameter limits.
    
    Parameters
    ----------
    model : FeII model instance
    template_name : str
        Name of the template
        
    Returns
    -------
    (flux_lo, flux_hi) : tuple
        Lower and upper flux limits
    """
    import itertools
    
    amp_param_name = f'amp_{template_name}'
    if not hasattr(model, amp_param_name):
        return np.nan, np.nan
    
    amp = getattr(model, amp_param_name)
    amp_lo, amp_hi, has_amp_lo, has_amp_hi = _get_param_limits(amp)
    
    # Get shared parameter limits
    shared_param_names = list(model._shared_params.keys())
    shared_limits = []
    has_all_lo = has_amp_lo
    has_all_hi = has_amp_hi
    
    for pname in shared_param_names:
        param = getattr(model, pname)
        lo, hi, has_lo, has_hi = _get_param_limits(param)
        shared_limits.append((lo, hi))
        has_all_lo = has_all_lo and has_lo
        has_all_hi = has_all_hi and has_hi
    
    # No limits at all
    if not (has_amp_lo or has_amp_hi or any(lo != hi for lo, hi in shared_limits)):
        return None, None
    
    # Get template data
    df_tmpl = model._df[model._df['ime'] == template_name]
    positions = df_tmpl['wav'].values
    intensities = df_tmpl['Int'].values
    
    # Evaluate flux at all corner combinations
    flux_values = []
    shared_combos = list(itertools.product(*[(lo, hi) for lo, hi in shared_limits]))
    
    for amp_val in [amp_lo, amp_hi]:
        for shared_vals in shared_combos:
            total_flux = 0.0
            for pos, intensity in zip(positions, intensities):
                amp_eff = amp_val * intensity
                total_flux += model._calc_flux(pos, amp_eff, *shared_vals)
            flux_values.append(total_flux)
    
    if not flux_values:
        return None, None
    
    flux_lo = np.min(flux_values)
    flux_hi = np.max(flux_values)
    
    # Apply boundary flags
    if not has_all_lo:
        flux_lo = np.nan
    if not has_all_hi:
        flux_hi = np.nan
    
    return flux_lo, flux_hi


# ============================================================================
# Helper for models - replaces individual propagation functions
# ============================================================================

def get_flux_params(model) -> Dict[str, Any]:
    """
    Get flux-related parameters for a model.
    
    Returns dict with flux value, std, limits - for use in model.flux property.
    """
    return {
        'value': compute_flux_value(model),
        'std': compute_flux_std(model),
        'limits': compute_flux_limits(model),
    }

