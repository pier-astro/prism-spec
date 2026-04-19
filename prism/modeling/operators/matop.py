"""Generic linear-operator models for native Astropy pipe expressions.

This module provides a Prism-only extension point for Astropy's ``|`` operator.
A right-hand model that exposes Prism pipe hooks can receive both the left-hand
model output and the original input grid, allowing expressions of the form

    source | operator

to evaluate as

    h(x, θ) = M(x, φ) @ f(x, θ)

while preserving analytic derivatives from the source model.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from astropy.modeling import CompoundModel, Fittable1DModel

__all__ = [
    'LinearOperatorModel',
    'install_pipe_patch',
    'is_linear_operator_pipe',
    'has_native_pipe',
]


class LinearOperatorModel(Fittable1DModel):
    """Model that applies a matrix operator to the left-hand pipe output.

    The base class stores a fixed matrix together with optional wavelength-grid
    metadata and a reconstruction recipe used by Prism serialization.

    Subclasses may override ``get_matrix(x, *right_params)`` to build an
    x-dependent operator matrix, and may override ``_prism_pipe_fit_deriv`` to
    append derivatives with respect to operator parameters.
    """

    n_inputs = 1
    n_outputs = 1

    def __init__(self, matrix, x=None, recipe=None, name='linop', **kwargs):
        self._matrix = matrix
        self._x = None if x is None else np.asarray(x)
        self._recipe = {} if recipe is None else dict(recipe)
        super().__init__(name=name, **kwargs)

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
        flux_unit = flux.unit if isinstance(flux, u.Quantity) else None
        flux_values = flux.to_value(flux_unit) if flux_unit is not None else flux
        flux = np.asarray(flux_values).ravel()
        if self._x is not None:
            x = self._x
        else:
            x = np.arange(len(flux), dtype=float)
        matrix = self.get_matrix(x, *right_params)
        if matrix.shape[1] != flux.size:
            raise ValueError(
                f"LinearOperatorModel expected {matrix.shape[1]} flux samples, got {flux.size}."
            )
        result = np.asarray(matrix.dot(flux)).ravel()
        return result * flux_unit if flux_unit is not None else result

    @staticmethod
    def fit_deriv(flux, *right_params):
        """Derivative with respect to operator parameters.

        The base class has no fittable parameters, so this returns an empty
        derivative list. Subclasses with parameters may override this or provide
        Prism-specific derivatives through ``_prism_pipe_fit_deriv``.
        """
        return []

    def _prism_pipe_evaluate(self, leftval, left_inputs, right_params, **kwargs):
        """Evaluate a Prism-enhanced pipe expression.

        Parameters
        ----------
        leftval : array-like
            Output of the left-hand source model.
        left_inputs : tuple
            Original inputs supplied to the left-hand model.
        right_params : tuple
            Current operator parameters extracted from the compound model.
        """
        x = np.asarray(left_inputs[0]).ravel()
        flux_unit = leftval.unit if isinstance(leftval, u.Quantity) else None
        flux_values = leftval.to_value(flux_unit) if flux_unit is not None else leftval
        flux = np.asarray(flux_values).ravel()
        matrix = self.get_matrix(x, *right_params)
        if matrix.shape[1] != flux.size:
            raise ValueError(
                f"LinearOperatorModel expected {matrix.shape[1]} flux samples, got {flux.size}."
            )
        result = np.asarray(matrix.dot(flux)).ravel()
        return result * flux_unit if flux_unit is not None else result

    def _prism_pipe_fit_deriv(
        self,
        left_deriv,
        left_inputs,
        left_params,
        right_params,
        left_model,
    ):
        """Transform the left-hand analytic Jacobian through the operator.

        Subclasses whose matrices depend on right-hand parameters may override
        this method and append derivatives with respect to those parameters.
        """
        x = np.asarray(left_inputs[0]).ravel()
        matrix = self.get_matrix(x, *right_params)

        derivs = np.asanyarray(left_deriv)
        if not left_model.col_fit_deriv:
            derivs = np.moveaxis(derivs, -1, 0)
        derivs = derivs.reshape((derivs.shape[0], -1))
        return np.asarray([np.asarray(matrix.dot(dparam)).ravel() for dparam in derivs])

    def __repr__(self):
        shape = self._matrix.shape
        grid = f", x={len(self._x)}" if self._x is not None else ""
        name = f", name='{self.name}'" if self.name else ""
        return f"<LinearOperatorModel({shape[0]}x{shape[1]}{grid}{name})>"


_ORIGINAL_EVALUATE = None
_ORIGINAL_STATIC_EVALUATE = None
_ORIGINAL_FIT_DERIV = None


def is_linear_operator_pipe(model):
    """Return ``True`` for a CompoundModel using a Prism linear operator pipe."""
    return (
        isinstance(model, CompoundModel)
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


def install_pipe_patch():
    """Install Prism's opt-in ``CompoundModel`` pipe patch exactly once."""
    global _ORIGINAL_EVALUATE, _ORIGINAL_STATIC_EVALUATE, _ORIGINAL_FIT_DERIV

    if getattr(CompoundModel, '_prism_linear_operator_patch', False):
        return

    _ORIGINAL_EVALUATE = CompoundModel._evaluate
    _ORIGINAL_STATIC_EVALUATE = CompoundModel.evaluate
    _ORIGINAL_FIT_DERIV = CompoundModel.fit_deriv

    def _patched_evaluate(self, *args, **kw):
        if is_linear_operator_pipe(self):
            left_inputs = args[: self.left.n_inputs]
            leftval = self.left(*left_inputs, **kw)
            right_params = self._get_right_params_from_args(args)
            return self.right._prism_pipe_evaluate(leftval, left_inputs, right_params, **kw)
        return _ORIGINAL_EVALUATE(self, *args, **kw)

    def _patched_static_evaluate(self, *args, **kw):
        if is_linear_operator_pipe(self):
            args, kw = self._get_kwarg_model_parameters_as_positional(args, kw)
            left_inputs = self._get_left_inputs_from_args(args)
            left_params = self._get_left_params_from_args(args)
            leftval = self.left.evaluate(*left_inputs, *left_params)
            right_params = self._get_right_params_from_args(args)
            return self.right._prism_pipe_evaluate(leftval, left_inputs, right_params, **kw)
        return _ORIGINAL_STATIC_EVALUATE(self, *args, **kw)

    @property
    def _patched_fit_deriv(self):
        if is_linear_operator_pipe(self):
            if self.left.fit_deriv is None:
                return None

            def _calc(*args, **kwargs):
                args, kwargs = self._get_kwarg_model_parameters_as_positional(args, kwargs)
                left_inputs = self._get_left_inputs_from_args(args)
                left_params = self._get_left_params_from_args(args)
                right_params = self._get_right_params_from_args(args)
                left_deriv = self.left.fit_deriv(*left_inputs, *left_params)
                return self.right._prism_pipe_fit_deriv(
                    left_deriv,
                    left_inputs,
                    left_params,
                    right_params,
                    self.left,
                )

            return _calc

        return _ORIGINAL_FIT_DERIV.fget(self)

    CompoundModel._evaluate = _patched_evaluate
    CompoundModel.evaluate = _patched_static_evaluate
    CompoundModel.fit_deriv = _patched_fit_deriv
    CompoundModel._prism_linear_operator_patch = True


install_pipe_patch()
