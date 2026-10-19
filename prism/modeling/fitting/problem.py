"""Prepared least-squares problem shared by numerical optimizer adapters."""

import numpy as np

from .evaluation import _ModelEvaluation
from .utils import _ParameterMap, _apply_tied_fast, _reduce_tied_analytic_jacobian


class _FitProblem:
    """Own one fit's parameter vector, evaluation and residual weighting.

    The caller owns data preparation and optimizer-specific choices. This object
    does not copy the model, alter bounds or select a statistic. Weights multiply
    residuals directly; absent weights mean unweighted least squares.
    Prepared matrices/grids belong to this fit, never to a shared worker cache.
    """

    def __init__(self, model, x, y, weights=None, valid=None):
        self.model = model
        valid = np.ones(np.shape(y), dtype=bool) if valid is None else valid
        self.y = y if np.all(valid) else y[valid]
        self.weights = weights if weights is None or np.all(valid) else weights[valid]
        self.ndata = len(self.y)
        # Keep known coordinates for profile-wide transforms; select residuals
        # afterwards. Unknown coordinates cannot be evaluated on any model grid.
        coordinates = np.isfinite(x)
        evaluation_x = x if np.all(coordinates) else x[coordinates]
        selected = valid[coordinates]
        self.select = slice(None) if np.all(selected) else selected
        self.parameters = _ParameterMap(model)
        self.cache = self.parameters.initial.copy()
        self.evaluation = _ModelEvaluation(model, evaluation_x)

    def update(self, values):
        """Expand free values and update links before evaluating the model."""
        self.cache[self.parameters.free] = values
        if not self.evaluation.stateless:
            self.model.parameters = self.cache
        if self.parameters.tied:
            _apply_tied_fast(self.model, self.parameters.tied, self.cache)
        return self.cache

    def residual(self, values):
        """Return (prediction - data) times residual weights."""
        parameters = self.update(values)
        residual = self.evaluation(parameters)[self.select] - self.y
        if self.weights is not None:
            residual = residual * self.weights
        return residual

    def jacobian(self, values):
        """Return sample-major residual derivatives in free-parameter order."""
        parameters = self.update(values)
        full = _reduce_tied_analytic_jacobian(
            self.model, self.evaluation.fit_deriv(parameters),
            warn=bool(self.parameters.tied), stacklevel=3)
        if np.any(~np.isfinite(full)):
            raise ValueError('Invalid derivatives encountered')
        jacobian = full[self.parameters.free][:, self.select].T
        if self.weights is not None:
            jacobian = jacobian * self.weights[:, np.newaxis]
        return jacobian

    def finish(self, values):
        """Commit final values even when evaluation could remain stateless."""
        self.cache[self.parameters.free] = values
        self.model.parameters = self.cache
        if self.parameters.tied:
            _apply_tied_fast(self.model, self.parameters.tied, self.cache)
        return self.model
