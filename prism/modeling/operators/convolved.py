"""
Linear-operator model nodes.

This module provides a minimal protocol for linear operators and a
CompoundModel-based node that applies a fixed linear transform to the output
of another model while preserving analytic derivatives.
"""

import numpy as np
from collections import defaultdict

from astropy.modeling.core import CompoundModel
from astropy.modeling.models import Identity


class LinearOperator:
    """Minimal protocol for linear operators used by LinearOperatorCompoundModel."""

    def get_matrix(self, x):
        """Return the linear operator matrix for the input grid x."""
        raise NotImplementedError


class MatrixLinearOperator(LinearOperator):
    """Linear operator backed by a fixed dense or sparse matrix."""

    def __init__(self, matrix, wave=None):
        self._matrix = matrix
        self._wave = None if wave is None else np.asarray(wave)

    @property
    def matrix(self):
        return self._matrix

    @property
    def wave(self):
        return self._wave

    def get_matrix(self, x):
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.ravel()
        if self._matrix.shape[0] != len(x):
            raise ValueError(
                f"Linear operator expects {self._matrix.shape[0]} samples, got {len(x)}."
            )
        return self._matrix

    def copy(self):
        wave_copy = None if self._wave is None else self._wave.copy()
        return MatrixLinearOperator(self._matrix, wave=wave_copy)


class LinearOperatorCompoundModel(CompoundModel):
    """
    CompoundModel node for h(x, p) = M(x) @ f(x, p).

    Parameters
    ----------
    left : Model
        Source model.
    right : LinearOperator
        Linear operator object implementing get_matrix(x).
    name : str, optional
        Model name.
    """

    def __init__(self, left, right, name=None):
        if not isinstance(right, LinearOperator):
            raise TypeError(
                f"right must implement LinearOperator, got {type(right).__name__}"
            )
        if left.n_inputs != 1 or left.n_outputs != 1:
            raise ValueError("LinearOperatorCompoundModel currently supports 1D->1D models only.")

        object.__setattr__(self, "_right_operator", right)
        super().__init__("|", left, Identity(1, name="linear_operator"), name=name)

    @property
    def right_operator(self):
        return self._right_operator

    @property
    def source_model(self):
        return self.left

    def _map_parameters(self):
        """Map parameters from the source model without adding operator suffixes."""
        if self._parameters is not None:
            return
        if self._leaflist is None:
            self._make_leaflist()

        self._parameters_ = {}
        self._param_names = []
        param_map = {}

        if isinstance(self.left, CompoundModel):
            if self.left._parameters is None:
                self.left._map_parameters()
            for pname in self.left.param_names:
                param = getattr(self.left, pname)
                self.__dict__[pname] = param
                self._parameters_[pname] = param
                self._param_names.append(pname)
                lindex, leaf_pname = self.left._param_map[pname]
                param_map[pname] = (lindex, leaf_pname)
        else:
            for pname in self.left.param_names:
                param = getattr(self.left, pname)
                self.__dict__[pname] = param
                self._parameters_[pname] = param
                self._param_names.append(pname)
                param_map[pname] = (0, pname)

        self._param_metrics = defaultdict(dict)
        self._param_map = param_map
        self._param_map_inverse = {v: k for k, v in param_map.items()}
        self._initialize_slices()
        self._param_names = tuple(self._param_names)

    @property
    def col_fit_deriv(self):
        return True

    def evaluate(self, *args, **kwargs):
        args, kwargs = self._get_kwarg_model_parameters_as_positional(args, kwargs)
        left_inputs = self._get_left_inputs_from_args(args)
        left_params = self._get_left_params_from_args(args)

        x = np.asarray(left_inputs[0]).ravel()
        flux = np.asarray(self.left.evaluate(*left_inputs, *left_params)).ravel()
        matrix = self._right_operator.get_matrix(x)
        return np.asarray(matrix.dot(flux)).ravel()

    @property
    def fit_deriv(self):
        source_deriv = self.left.fit_deriv
        if source_deriv is None:
            return None

        def _linear_deriv(*args, **kwargs):
            args, kwargs = self._get_kwarg_model_parameters_as_positional(args, kwargs)
            left_inputs = self._get_left_inputs_from_args(args)
            left_params = self._get_left_params_from_args(args)

            x = np.asarray(left_inputs[0]).ravel()
            matrix = self._right_operator.get_matrix(x)

            derivs = np.asanyarray(source_deriv(*left_inputs, *left_params))
            if not self.left.col_fit_deriv:
                derivs = np.moveaxis(derivs, -1, 0)
            derivs = derivs.reshape((derivs.shape[0], -1))

            return np.asarray([np.asarray(matrix.dot(d)).ravel() for d in derivs])

        return _linear_deriv

    def with_left(self, new_left, name=None):
        new_name = self.name if name is None else name
        return LinearOperatorCompoundModel(new_left, self._right_operator, name=new_name)

    def copy(self):
        op = self._right_operator.copy() if hasattr(self._right_operator, "copy") else self._right_operator
        return LinearOperatorCompoundModel(self.left.copy(), op, name=self.name)

    def __repr__(self):
        return f"<LinearOperatorCompoundModel[{self.name}]({self.left!r})>"


__all__ = ["LinearOperator", "MatrixLinearOperator", "LinearOperatorCompoundModel"]
