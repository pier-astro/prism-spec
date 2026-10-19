"""Discretized linear operators for native Prism spectral expressions.

Matrix application, grids and derivatives live here. The optional extension for
Astropy CompoundModel pipes lives separately in interop.astropy.
"""

from __future__ import annotations

import numpy as np

from astropy.modeling import CompoundModel, Fittable1DModel
from scipy.sparse import csr_matrix

__all__ = [
    'LinearOperatorModel',
    'is_linear_operator_pipe',
    'has_native_pipe',
]


def _apply_matrix(matrix, values):
    """Apply one operator to a spectrum or sample-major Jacobian columns."""
    unit = getattr(values, 'unit', None)
    array = values.to_value(unit) if unit is not None else np.asarray(values)
    array = np.asarray(array)
    if array.ndim not in (1, 2) or array.shape[0] != matrix.shape[1]:
        raise ValueError(
            f'Operator expects {matrix.shape[1]} input samples along the first axis; '
            f'got shape {array.shape}.')
    result = np.asarray(matrix.dot(array))
    return result * unit if unit is not None else result


def _sample_response(values, cache):
    """Interpolate a spectrum or Jacobian columns; zero outside the native grid."""
    inside = cache['inside']
    left, right = cache['left'][inside], cache['right'][inside]
    fraction = cache['frac'][inside]
    if values.ndim == 2:
        fraction = fraction[:, None]
    result = np.zeros((len(inside),) + values.shape[1:], dtype=values.dtype)
    result[inside] = (1 - fraction)*values[left] + fraction*values[right]
    return result


class LinearOperatorModel(Fittable1DModel):
    """Fixed discretized linear action, optionally composed with a source model.

    Parameters
    ----------
    matrix : array-like or scipy.sparse matrix
        Finite real coefficients with shape (n_output, n_input). No implicit
        normalization is applied; the matrix acts on sampled values.
    x : array-like or astropy.units.Quantity, optional
        Increasing native input coordinates. Used to evaluate a source before
        applying the matrix. The supplied array is copied.
    recipe : dict, optional
        Reconstruction provenance. Archived SpectralResponse instances use it
        to locate a calibration; custom fixed matrices are stored separately.
    name : str, optional
        Model component name. Default is 'linop'.
    xout : array-like or astropy.units.Quantity, optional
        Increasing native output coordinates. Defaults to x for square
        operators; supply both grids when composing a rectangular operator.
    flexible : bool, optional
        Interpolate output samples onto requested coordinates, with zero beyond
        the native output grid. Default is True. This is point sampling, not
        flux-conserving integration over new bins.
    **kwargs
        Astropy model options, also used by parameter-dependent subclasses.

    Notes
    -----
    The fixed action preserves the units of its input values. Dimensional
    matrix coefficients are not supported. Subclasses may implement parameter-
    dependent matrices and Jacobians, but cannot be saved as frozen snapshots.
    """

    n_inputs = 1
    n_outputs = 1

    def __init__(self, matrix, x=None, recipe=None, name='linop', xout=None,
                 flexible=True, **kwargs):
        if hasattr(matrix, 'unit'):
            from astropy import units as u
            matrix = matrix.to_value(u.dimensionless_unscaled)
        self._matrix = csr_matrix(matrix)
        self._matrix.check_format(full_check=True)
        if min(self._matrix.shape) == 0:
            raise ValueError('An operator must have at least one input and output sample.')
        if self._matrix.dtype.kind not in 'fiu' or not np.all(np.isfinite(self._matrix.data)):
            raise ValueError('Operator coefficients must be finite real numbers.')
        self._x = self._grid(x, self._matrix.shape[1], 'input')
        self._xout = self._grid(xout, self._matrix.shape[0], 'output')
        self.flexible = bool(flexible)
        self._interp_cache = None
        self._recipe = {} if recipe is None else dict(recipe)
        super().__init__(name=name, **kwargs)

    @staticmethod
    def _grid(x, size, name):
        if x is None:
            return None
        values = np.asarray(x)
        if values.ndim != 1 or len(values) != size or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise ValueError(f'Operator {name} grid must be finite, increasing and contain {size} coordinates.')
        return x.copy() if hasattr(x, 'unit') else np.array(x, copy=True)

    @property
    def xout(self):
        """Output coordinates; default to x for a square same-grid operator."""
        if self._xout is not None:
            return self._xout
        return self._x if self._x is not None and len(self._x) == self.matrix.shape[0] else None

    def save(self, filename, overwrite=False):
        """Save a fixed operator as lossless sparse FITS, including grid units."""
        from .storage import _write_operator
        _write_operator(self, filename, overwrite=overwrite)

    @classmethod
    def from_fits(cls, filename):
        """Reconstruct the built-in fixed operator recorded in a FITS file."""
        from .storage import _read_operator
        result = _read_operator(filename)
        if cls is not LinearOperatorModel and not isinstance(result, cls):
            raise ValueError(f'The file does not contain a {cls.__name__}.')
        return result

    @property
    def matrix(self):
        return self._matrix

    @property
    def x(self):
        return self._x

    @property
    def recipe(self):
        return self._recipe

    def get_matrix(self, x, *right_params):
        """Return the operator matrix for input grid ``x``.

        The default implementation returns the stored fixed matrix and validates
        that its output dimension matches the supplied grid length.
        """
        x = np.asarray(x).ravel()
        if self._matrix.shape[0] != len(x):
            raise ValueError(
                f"LinearOperatorModel expects {self._matrix.shape[0]} samples, got {len(x)}."
            )
        return self._matrix

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        mapping = {}
        for pname in self.param_names:
            param = getattr(self, pname)
            unit = getattr(param, 'unit', None)
            if unit is not None:
                mapping[pname] = unit
        return mapping

    def evaluate(self, flux, *right_params):
        """Apply the operator directly to a flux array.

        Native Astropy pipe evaluation does not expose the original input grid,
        so this direct path is intended only for same-grid operators.
        Prism's patched pipe path calls ``_prism_pipe_evaluate`` instead.
        """
        x = self.xout if self.xout is not None else np.arange(self.matrix.shape[0], dtype=float)
        return _apply_matrix(self.get_matrix(x, *right_params), flux)

    @staticmethod
    def fit_deriv(flux, *right_params):
        """Derivative with respect to operator parameters.

        The base class has no fittable parameters, so this returns an empty
        derivative list. Subclasses with parameters may override this or provide
        Prism-specific derivatives through ``_prism_pipe_fit_deriv``.
        """
        return []

    @staticmethod
    def _coordinates(x, native):
        if hasattr(native, 'unit') and hasattr(x, 'unit'):
            x = x.to_value(native.unit)
        return np.asarray(x, dtype=float).ravel()

    def _input_matches(self, x):
        if self.x is None:
            return True
        try:
            values = self._coordinates(x, self.x)
        except ValueError:
            return False
        return values.shape == self.x.shape and np.array_equal(values, np.asarray(self.x))

    def _same_grid(self, x):
        if self.xout is None:
            return True
        values = self._coordinates(x, self.xout)
        native = np.asarray(self.xout)
        return values.shape == native.shape and (np.array_equal(values, native) or np.allclose(
            values, native, rtol=0., atol=np.finfo(float).eps*max(1., np.abs(native).max())))

    def _get_interp_cache(self, x):
        x = self._coordinates(x, self.xout)
        if self.xout is None:
            raise ValueError("Flexible response evaluation requires a wavelength grid.")
        if self._interp_cache is not None and np.array_equal(x, self._interp_cache["x"]):
            return self._interp_cache

        native = np.asarray(self.xout)
        indices = np.searchsorted(native, x)
        inside = (x >= native[0]) & (x <= native[-1])
        left = np.clip(indices - 1, 0, len(native) - 1)
        right = np.clip(indices, 0, len(native) - 1)

        same = inside & (left == right)
        frac = np.zeros_like(x, dtype=float)

        span = native[right] - native[left]
        valid = inside & (~same)
        frac[valid] = (x[valid] - native[left[valid]]) / span[valid]
        frac[same] = 0.0

        cache = {
            "x": x.copy(),
            "inside": inside,
            "left": left,
            "right": right,
            "frac": frac,
        }
        self._interp_cache = cache
        return cache

    def _resample_from_native(self, values, x):
        unit = values.unit if hasattr(values, "unit") else None
        arr = values.to_value(unit) if unit is not None else np.asarray(values)
        arr = np.asarray(arr)
        if self._same_grid(x):
            return values
        result = _sample_response(arr, self._get_interp_cache(x))
        return result * unit if unit is not None else result

    def _native_response(self, values):
        return _apply_matrix(self.matrix, values)

    def _native_input(self, requested):
        native = self.x
        if native is not None and not hasattr(native, 'unit') and hasattr(requested, 'unit'):
            native = native * requested.unit
        return native

    def _prism_pipe_evaluate(self, leftval, left_inputs, right_params,
                             left_model=None, left_params=None, **kwargs):
        requested = left_inputs[0]
        same_output = self._same_grid(requested)
        if not same_output and not self.flexible:
            raise ValueError(f'LinearOperatorModel expects {self.matrix.shape[0]} samples on its output grid, got {len(requested)}.')
        if self._input_matches(requested):
            values = leftval
        else:
            if left_model is None:
                raise ValueError('Native-grid operator evaluation requires the source model.')
            native = self._native_input(requested)
            values = left_model(native, **kwargs) if left_params is None else left_model.evaluate(native, *left_params)
        matrix = self.get_matrix(self.xout if self.xout is not None else requested, *right_params)
        result = _apply_matrix(matrix, values)
        return result if same_output else self._resample_from_native(result, requested)

    def _prism_pipe_fit_deriv(self, left_deriv, left_inputs, left_params, right_params, left_model):
        requested = left_inputs[0]
        same_output = self._same_grid(requested)
        if not same_output and not self.flexible:
            raise ValueError(f'LinearOperatorModel expects {self.matrix.shape[0]} samples on its output grid, got {len(requested)}.')
        if not self._input_matches(requested):
            left_deriv = left_model.fit_deriv(self._native_input(requested), *left_params)
        if left_deriv is None:
            return None
        derivs = np.asanyarray(left_deriv)
        if not left_model.col_fit_deriv:
            derivs = np.moveaxis(derivs, -1, 0)
        derivs = derivs.reshape((derivs.shape[0], -1))
        matrix = self.get_matrix(self.xout if self.xout is not None else requested, *right_params)
        result = _apply_matrix(matrix, derivs.T)
        return np.asarray(result if same_output else self._resample_from_native(result, requested)).T

    def __repr__(self):
        shape = self._matrix.shape
        grid = f", x={len(self._x)}" if self._x is not None else ""
        name = f", name='{self.name}'" if self.name else ""
        return f"<LinearOperatorModel({shape[0]}x{shape[1]}{grid}{name})>"




def is_linear_operator_pipe(model):
    """Return ``True`` for a CompoundModel using a Prism linear operator pipe."""
    from ..composition import Model
    return (
        isinstance(model, (CompoundModel, Model))
        and getattr(model, 'op', None) == '|'
        and hasattr(model.right, '_prism_pipe_evaluate')
    )


def has_native_pipe(model):
    """Return ``True`` when any subtree contains a native Astropy pipe."""
    if not isinstance(model, CompoundModel):
        return False
    if getattr(model, 'op', None) == '|' and not hasattr(model.right, '_prism_pipe_evaluate'):
        return True
    return has_native_pipe(model.left) or has_native_pipe(model.right)
