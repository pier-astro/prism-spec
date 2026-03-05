"""
Scientific Value and Error Formatting Utilities

Format scientific values with errors for publication-ready output.

Features:
- Scientific rounding (≥5 rounds up)
- Symmetric/asymmetric error handling
- Array support with preserved input/output types
- Multiple formats: LaTeX, HTML, text
- Confidence limits formatting
"""

import numpy as np
from typing import Union, Tuple, List


def round_sci(value: Union[float, np.ndarray], decimals: int = 0) -> Union[float, np.ndarray]:
    """
    Round using scientific rounding (≥0.5 rounds up).
    
    Parameters
    ----------
    value : float or array-like
        Value(s) to round.
    decimals : int
        Number of decimal places.
        
    Returns
    -------
    float or ndarray
        Rounded value(s), matching input type.
    """
    input_was_scalar = np.isscalar(value)
    value_arr = np.asarray(value)
    multiplier = 10.0 ** decimals
    mask = ~(np.isnan(value_arr) | np.isinf(value_arr))
    result = np.copy(value_arr)
    result[mask] = np.floor(value_arr[mask] * multiplier + 0.5) / multiplier
    return result.item() if input_was_scalar else result


def round_to_sig(value: Union[float, np.ndarray], sig_digits: int) -> Union[float, np.ndarray]:
    """
    Round to specified significant figures.
    
    Parameters
    ----------
    value : float or array-like
        Value(s) to round.
    sig_digits : int
        Number of significant figures (must be >= 1).
        
    Returns
    -------
    float or ndarray
        Rounded value(s), matching input type.
    """
    if sig_digits < 1:
        raise ValueError("sig_digits must be >= 1")
    
    input_was_scalar = np.isscalar(value)
    value_arr = np.asarray(value)
    result = np.zeros_like(value_arr)
    mask = value_arr != 0
    
    if np.any(mask):
        with np.errstate(invalid='ignore'):
            log_vals = np.floor(np.log10(np.abs(value_arr[mask])))
            valid_log = ~(np.isnan(log_vals) | np.isinf(log_vals))
            if np.any(valid_log):
                decimals = np.where(valid_log, sig_digits - log_vals.astype(int) - 1, 0)
                result[mask] = round_sci(value_arr[mask], decimals.astype(int))
            else:
                result[mask] = value_arr[mask]
    
    return result.item() if input_was_scalar else result


def round_to_error(value: Union[float, np.ndarray], error: Union[float, np.ndarray], 
                   sig_digits: int = 2) -> Union[float, np.ndarray]:
    """
    Round value to precision implied by error.
    
    Parameters
    ----------
    value : float or array-like
        Value(s) to round.
    error : float or array-like
        Error(s) determining precision.
    sig_digits : int
        Significant figures for error (default: 2).
        
    Returns
    -------
    float or ndarray
        Rounded value(s), matching input type.
    """
    if sig_digits < 1:
        raise ValueError("sig_digits must be >= 1")
    
    value_was_scalar = np.isscalar(value)
    value_arr = np.atleast_1d(np.asarray(value))
    error_arr = np.atleast_1d(np.asarray(error))
    
    # Broadcast arrays
    if error_arr.size == 1 and value_arr.size > 1:
        error_arr = np.full_like(value_arr, error_arr.item())
    elif value_arr.size == 1 and error_arr.size > 1:
        value_arr = np.full_like(error_arr, value_arr.item())
    
    result = np.zeros_like(value_arr)
    
    # Handle invalid errors
    invalid_mask = (error_arr == 0) | np.isnan(error_arr) | np.isinf(error_arr)
    result[invalid_mask] = value_arr[invalid_mask]
    
    valid_mask = ~invalid_mask
    if np.any(valid_mask):
        precision = np.floor(np.log10(np.abs(error_arr[valid_mask]))) - sig_digits + 1
        valid_indices = np.where(valid_mask)[0]
        for i, idx in enumerate(valid_indices):
            result[idx] = round_sci(value_arr[idx], -int(precision[i]))
    
    return result.item() if value_was_scalar else result


def format_value(value: Union[float, np.ndarray], 
                 error: Union[float, np.ndarray, Tuple[float, float]] = None, 
                 sig_digits: int = 2, 
                 output: str = "latex", 
                 use_dollars: bool = True) -> Union[str, List[str]]:
    """
    Format value with error for publication output.
    
    Parameters
    ----------
    value : float or array-like
        Value(s) to format.
    error : float, tuple, or array-like, optional
        Error(s). Can be:
        - None: no error
        - float: symmetric error
        - tuple (lo, hi): asymmetric errors
    sig_digits : int
        Significant figures (default: 2).
    output : {'latex', 'html', 'text'}
        Output format.
    use_dollars : bool
        Wrap LaTeX in $ (default: True).
        
    Returns
    -------
    str or list of str
        Formatted string(s).
        
    Examples
    --------
    >>> format_value(1.234, 0.056)
    '$1.23 \\pm 0.056$'
    >>> format_value(1.234, (0.05, 0.08), output='html')
    '1.23 [&minus;0.05 +0.08]'
    """
    # Handle array inputs recursively
    if isinstance(value, (np.ndarray, list)):
        value = np.asarray(value)
        if error is not None:
            error = np.asarray(error)
        return [
            format_value(v, error[i] if error is not None and error.shape == value.shape else error, 
                        sig_digits, output, use_dollars)
            for i, v in enumerate(value)
        ]
    
    # No error - just format value
    if error is None:
        val_str = f"{round_to_sig(value, sig_digits)}"
        return f"${val_str}$" if output == "latex" and use_dollars else val_str
    
    # Asymmetric error (tuple)
    if isinstance(error, (tuple, list, np.ndarray)) and len(error) == 2:
        err_lo, err_up = error
        
        # Determine rounding error (use valid one, or both)
        err_lo_valid = not (np.isnan(err_lo) or np.isinf(err_lo))
        err_up_valid = not (np.isnan(err_up) or np.isinf(err_up))
        
        if err_lo_valid and err_up_valid:
            rounding_error = min(err_lo, err_up)
        elif err_lo_valid:
            rounding_error = err_lo
        elif err_up_valid:
            rounding_error = err_up
        else:
            rounding_error = None
        
        val_str = (f"{round_to_error(value, rounding_error, sig_digits)}" 
                   if rounding_error else f"{round_to_sig(value, sig_digits)}")
        err_lo_str = f"{round_to_sig(err_lo, sig_digits)}"
        err_up_str = f"{round_to_sig(err_up, sig_digits)}"
        
        if output == "latex":
            s = f"{val_str}_{{-{err_lo_str}}}^{{+{err_up_str}}}"
            return f"${s}$" if use_dollars else s
        elif output == "html":
            return f"{val_str} [&minus;{err_lo_str} +{err_up_str}]"
        else:
            return f"{val_str} [-{err_lo_str} +{err_up_str}]"
    
    # Symmetric error
    err_str = f"{round_to_sig(error, sig_digits)}"
    val_str = f"{round_to_error(value, error, sig_digits)}"
    
    if output == "latex":
        s = f"{val_str} \\pm {err_str}"
        return f"${s}$" if use_dollars else s
    elif output == "html":
        return f"{val_str} &plusmn; {err_str}"
    else:
        return f"{val_str} ± {err_str}"
