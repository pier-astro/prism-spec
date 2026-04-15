"""Scientific rounding helpers for values and uncertainties."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math

import numpy as np

__all__ = ['round_sig', 'fmt', 'tex', 'format_value', 'format_tex']


def _scalar(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError('sciformat expects scalar inputs.')
    return float(arr)


def _finite(value):
    try:
        return bool(np.isfinite(value))
    except Exception:
        return False


def _round_half_up(value, decimals=0):
    value = _scalar(value)
    if not _finite(value):
        return value
    quantum = Decimal('1').scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def round_sig(value, sig=3):
    """Round a scalar using scientific half-up rounding."""
    if sig < 1:
        raise ValueError('sig must be >= 1')

    value = _scalar(value)
    if not _finite(value) or value == 0:
        return float(value)

    exponent = int(math.floor(math.log10(abs(value))))
    return _round_half_up(value, sig - exponent - 1)


def _format_number(value, decimals=None, sig=3, keep_zeros=False):
    value = _scalar(value)
    if value is None or np.isnan(value):
        return 'nan'
    if np.isposinf(value):
        return '∞'
    if np.isneginf(value):
        return '-∞'

    if decimals is None:
        rounded = round_sig(value, sig=sig)
        if rounded == 0:
            decimals = max(sig - 1, 0)
        else:
            exponent = int(math.floor(math.log10(abs(rounded))))
            decimals = sig - exponent - 1
    else:
        rounded = _round_half_up(value, decimals)

    text = f"{rounded:.{max(decimals, 0)}f}"
    if decimals < 0:
        text = f"{rounded:.0f}"
    if not keep_zeros and '.' in text:
        text = text.rstrip('0').rstrip('.')
    if text.startswith('-') and float(text) == 0.0:
        text = text[1:]
    return text


def _error_decimals(error, esig=1, add_if_one=False):
    error = _scalar(error)
    if not _finite(error) or error <= 0:
        return None
    if esig < 1:
        raise ValueError('esig must be >= 1')

    exponent = int(math.floor(math.log10(abs(error))))
    mantissa = abs(error) / (10 ** exponent)
    first_digit = int(f"{mantissa:.12g}"[0])
    digits = int(esig) + int(bool(add_if_one and first_digit == 1))
    return digits - exponent - 1


def _parts(value, err=None, lo=None, hi=None, sig=3, esig=1, add_if_one=False):
    value = _scalar(value)
    err = _scalar(err)
    lo = _scalar(lo)
    hi = _scalar(hi)

    has_lo = _finite(lo)
    has_hi = _finite(hi)

    if has_lo and not has_hi:
        decimals = _error_decimals(abs(lo), esig, add_if_one)
        return {'kind': 'lower', 'value': _format_number(lo, decimals=0 if decimals is None else decimals)}

    if has_hi and not has_lo:
        decimals = _error_decimals(abs(hi), esig, add_if_one)
        return {'kind': 'upper', 'value': _format_number(hi, decimals=0 if decimals is None else decimals)}

    if has_lo and has_hi:
        err_lo = value - lo
        err_hi = hi - value

        if err_lo <= 0 < err_hi:
            decimals = _error_decimals(abs(hi), esig, add_if_one)
            return {'kind': 'upper', 'value': _format_number(hi, decimals=0 if decimals is None else decimals)}
        if err_hi <= 0 < err_lo:
            decimals = _error_decimals(abs(lo), esig, add_if_one)
            return {'kind': 'lower', 'value': _format_number(lo, decimals=0 if decimals is None else decimals)}

        finite_errs = [item for item in (err_lo, err_hi) if _finite(item) and item > 0]
        if finite_errs:
            decimals = _error_decimals(min(finite_errs), esig, add_if_one)
            decimals = 0 if decimals is None else decimals
            value_text = _format_number(value, decimals=decimals, keep_zeros=True)
            lo_text = _format_number(abs(err_lo), decimals=decimals, keep_zeros=True)
            hi_text = _format_number(abs(err_hi), decimals=decimals, keep_zeros=True)
            if lo_text == hi_text:
                return {'kind': 'sym', 'value': value_text, 'err': lo_text}
            return {'kind': 'asym', 'value': value_text, 'lo': lo_text, 'hi': hi_text}

    if _finite(err) and err > 0:
        decimals = _error_decimals(err, esig, add_if_one)
        decimals = 0 if decimals is None else decimals
        value_text = _format_number(value, decimals=decimals, keep_zeros=True)
        err_text = _format_number(abs(err), decimals=decimals, keep_zeros=True)
        return {'kind': 'sym', 'value': value_text, 'err': err_text}

    return {'kind': 'value', 'value': _format_number(value, sig=sig)}


def fmt(value, err=None, lo=None, hi=None, sig=3, esig=1, add_if_one=False):
    """Format a scalar value as compact plain text."""
    parts = _parts(value, err=err, lo=lo, hi=hi, sig=sig, esig=esig, add_if_one=add_if_one)
    if parts['kind'] == 'lower':
        return f"> {parts['value']}"
    if parts['kind'] == 'upper':
        return f"< {parts['value']}"
    if parts['kind'] == 'sym':
        return f"{parts['value']} ± {parts['err']}"
    if parts['kind'] == 'asym':
        return f"{parts['value']} [-{parts['lo']}, +{parts['hi']}]"
    return parts['value']


def _tex_body(parts):
    if parts['kind'] == 'lower':
        return f"> {parts['value']}"
    if parts['kind'] == 'upper':
        return f"< {parts['value']}"
    if parts['kind'] == 'sym':
        return f"{parts['value']} \\pm {parts['err']}"
    if parts['kind'] == 'asym':
        return f"{parts['value']}_{{-{parts['lo']}}}^{{+{parts['hi']}}}"
    return parts['value']


def tex(value, err=None, lo=None, hi=None, sig=3, esig=1, add_if_one=False,
        use_sci=False, maxexp=4, dollars=False):
    """Format a scalar value as LaTeX, with optional scientific grouping."""
    value = _scalar(value)
    err = _scalar(err)
    lo = _scalar(lo)
    hi = _scalar(hi)

    exponent = None
    if use_sci and _finite(value) and value != 0:
        exponent = int(math.floor(math.log10(abs(value))))
        if -int(maxexp) < exponent < int(maxexp):
            exponent = None

    if exponent is not None:
        scale = 10.0 ** exponent
        parts = _parts(
            value / scale,
            err=None if not _finite(err) else err / scale,
            lo=None if not _finite(lo) else lo / scale,
            hi=None if not _finite(hi) else hi / scale,
            sig=sig,
            esig=esig,
            add_if_one=add_if_one,
        )
        body = _tex_body(parts)
        if parts['kind'] in {'sym', 'asym'}:
            body = f"({body}) \\times 10^{{{exponent}}}"
        else:
            body = f"{body} \\times 10^{{{exponent}}}"
    else:
        parts = _parts(value, err=err, lo=lo, hi=hi, sig=sig, esig=esig, add_if_one=add_if_one)
        body = _tex_body(parts)

    return f"${body}$" if dollars else body


format_value = fmt
format_tex = tex
format_measurement = fmt

