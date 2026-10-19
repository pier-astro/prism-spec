"""Numerical line-shape measurements on sampled arrays; no model adapters."""
import numpy as np

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

def _crossings_at_fraction(x, y, fraction):
    if x.size < 3:
        return float('nan'), float('nan')
    peak_index = int(np.argmax(y))
    peak_value = float(y[peak_index])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        return float('nan'), float('nan')
    level = fraction * peak_value

    left_x = float('nan')
    for i in range(peak_index, 0, -1):
        if y[i - 1] < level:
            denom = y[i] - y[i - 1]
            if denom != 0.0:
                left_x = float(
                    x[i - 1] + (level - y[i - 1]) / denom * (x[i] - x[i - 1]))
            break

    right_x = float('nan')
    for i in range(peak_index, x.size - 1):
        if y[i + 1] < level:
            denom = y[i + 1] - y[i]
            if denom != 0.0:
                right_x = float(
                    x[i] + (level - y[i]) / denom * (x[i + 1] - x[i]))
            break

    return left_x, right_x


def _compute_profile_metrics(x_arr, y_arr) -> dict:
    """Integrate the signed profile on an increasing, finite spectral grid.

    Moments are normalized by the finite-window flux, not the analytic integral
    over an infinite profile. Crossing widths use the nearest crossings around
    the highest positive peak; they do not span all peaks in a blend.
    """
    # Integrated flux and the highest sampled peak.
    flux = float(_trapezoid(y_arr, x_arr))
    peak_index = int(np.argmax(y_arr))
    peak_value = float(y_arr[peak_index])
    peak_position = float(x_arr[peak_index])

    # Flux-weighted centroid and central moments within this window.
    moment1 = moment2 = sigma = skewness = kurtosis = float('nan')
    if np.isfinite(flux) and flux != 0.0:
        moment1 = float(_trapezoid(x_arr * y_arr, x_arr) / flux)
        residuals = x_arr - moment1
        moment2 = float(_trapezoid(residuals ** 2 * y_arr, x_arr) / flux)
        if moment2 >= 0.0:
            sigma = float(np.sqrt(moment2))
            if sigma > 0.0:
                skewness = float(
                    _trapezoid(residuals ** 3 * y_arr, x_arr)
                    / (flux * sigma ** 3))
                kurtosis = float(
                    _trapezoid(residuals ** 4 * y_arr, x_arr)
                    / (flux * sigma ** 4)) - 3.0

    # Missing crossings are NaN, so widths and bisectors propagate them.
    fwhm_l, fwhm_r = _crossings_at_fraction(x_arr, y_arr, 0.5)
    fwhm = float(fwhm_r - fwhm_l)
    hw_blue = float(peak_position - fwhm_l)
    hw_red = float(fwhm_r - peak_position)
    asymmetry = float('nan')
    if np.isfinite(fwhm) and fwhm > 0.0:
        asymmetry = float((hw_red - hw_blue) / fwhm)

    fw10m_l, fw10m_r = _crossings_at_fraction(x_arr, y_arr, 0.10)
    fw20m_l, fw20m_r = _crossings_at_fraction(x_arr, y_arr, 0.20)
    fw80m_l, fw80m_r = _crossings_at_fraction(x_arr, y_arr, 0.80)
    fw10m = float(fw10m_r - fw10m_l)
    fw20m = float(fw20m_r - fw20m_l)
    fw80m = float(fw80m_r - fw80m_l)

    bisector_20 = 0.5 * (fw20m_l + fw20m_r)
    bisector_80 = 0.5 * (fw80m_l + fw80m_r)
    bisector_span = float(bisector_80 - bisector_20)

    return {
        'flux': flux, 'peak_value': peak_value, 'peak_position': peak_position,
        'fwhm': fwhm, 'hw_blue': hw_blue, 'hw_red': hw_red,
        'asymmetry': asymmetry,
        'moment1': moment1, 'moment2': moment2, 'sigma': sigma,
        'skewness': skewness, 'kurtosis': kurtosis,
        'fw10m': fw10m, 'fw20m': fw20m, 'fw80m': fw80m,
        'bisector_span': bisector_span,
    }


def _compute_ew_numerical(x_arr, y_line, y_continuum) -> float:
    # Preserve signed scalar-continuum behavior; sampled continua must be positive.
    if np.ndim(y_continuum) == 0:
        if not np.isfinite(y_continuum) or y_continuum == 0.:
            return float("nan")
        return float(_trapezoid(y_line, x_arr) / y_continuum)
    safe = np.where(y_continuum > 0.0, y_continuum, np.nan)
    return float(_trapezoid(y_line / safe, x_arr))

