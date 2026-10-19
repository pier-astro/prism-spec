"""Validation and sample selection for one-dimensional fitting data."""
import numpy as np


def _prepare_fit_data(x, y, yerr=None, weights=None, mask=None, *, filter_non_finite=False):
    """Return full-grid arrays and a validity mask (True means included).

    Masked-array masks and zero residual weights exclude samples. Invalid values
    outside the selection are harmless; non-finite included values require
    explicit filtering. Included errors must be positive and weights nonnegative.
    Quantities retain their units; numerical backends enforce their own unit
    boundary before calling this routine. No automatic unit conversion occurs.
    """
    arrays = []
    inferred = None
    for value in (x, y, yerr, weights):
        if value is None:
            arrays.append(None)
            continue
        if np.ma.isMaskedArray(value):
            excluded = np.ma.getmaskarray(value)
            inferred = excluded if inferred is None else inferred | excluded
            value = np.ma.getdata(value)
        arrays.append(np.asanyarray(value))
    x, y, yerr, weights = arrays
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError('x and y must be one-dimensional arrays with the same shape.')
    for name, array in [('yerr', yerr), ('weights', weights)]:
        if array is not None and array.shape != y.shape:
            if name == 'yerr' and array.ndim != 1:
                raise ValueError('Asymmetric yerr is not supported by direct fitters. Use symmetric yerr or bootstrap(..., noise_dist="uniform").')
            raise ValueError(f'{name} must have the same shape as y.')
    if inferred is not None and inferred.shape != y.shape:
        raise ValueError('Masked inputs must have the same shape as y.')
    if mask is None:
        valid = np.ones(y.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool).copy()
        if valid.shape != y.shape:
            raise ValueError('Per-spectrum mask must have the same shape as y.')
    if inferred is not None:
        valid &= ~inferred
    if weights is not None:
        valid &= weights != 0
    finite = np.isfinite(x) & np.isfinite(y)
    for array in (yerr, weights):
        if array is not None:
            finite &= np.isfinite(array)
    if filter_non_finite:
        valid &= finite
    elif np.any(valid & ~finite):
        raise ValueError('Non-finite values detected in included fitting data. Set filter_non_finite=True.')
    if yerr is not None and np.any(yerr[valid] <= 0):
        raise ValueError('Included yerr values must be strictly positive.')
    if weights is not None and np.any(weights[valid] < 0):
        raise ValueError('Included weights must be nonnegative.')
    if not np.any(valid):
        raise ValueError('No valid spectral samples remain after masking.')
    return x, y, yerr, weights, valid
