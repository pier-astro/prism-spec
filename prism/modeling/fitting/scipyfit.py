"""
SciPy-based fitters for prism.modeling.fitting

This module provides wrappers around scipy.optimize.least_squares with
standardized interfaces and robust error handling.
"""

import warnings
import numpy as np
from .extension import _validate_spectral_inputs
from scipy import optimize
from numpy.linalg import LinAlgError
from astropy.modeling.fitting import Fitter
from .extension import (
    _attach_covariance,
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    _resolve_max_evaluations,
    get_max_evaluations,
)
from .multifit import MultiFitMixin
from .joint import _JointFitMixin
from .single import _SingleFitMixin
from .problem import _FitProblem
from .constraints import _box_bounds
from .data import _prepare_fit_data, _resolve_fit_inputs
from .utils import (
    _analytic_jacobian_parameter_major,

)

__all__ = ['ScipyFitter', 'TRF', 'ScipyDogBox']


class ScipyFitter(_SingleFitMixin, _JointFitMixin, MultiFitMixin, Fitter):
    """
    Wrapper around ``scipy.optimize.least_squares``.

    Parameters
    ----------
    method : {'trf', 'dogbox'}, optional
        ``least_squares`` backend. Default is ``'trf'``.
    covariance : bool, optional
        If ``True``, estimate a parameter covariance matrix from the final
        Jacobian. Default is ``True``.
    verbose : bool, optional
        If ``True``, print fallback and diagnostic messages. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to
        ``scipy.optimize.least_squares``.

    Notes
    -----
    Prism keeps the Astropy fitter calling convention while delegating the actual
    optimisation to SciPy. If analytic derivatives are available they are used by
    default; otherwise the fitter falls back to finite differences. Optional
    covariance estimation is derived from the normal-equation approximation
    ``(J^T J)^{-1}`` or its pseudo-inverse when the Jacobian is ill conditioned.
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)
    
    def __init__(self, method='trf', covariance=True, verbose=False, **kwargs):
        self.method = method
        self._estimate_covariance = bool(covariance)
        self.verbose = verbose
        self.fit_kwargs = kwargs
        # Normalize constructor controls once; keep optimizer options separate.
        controls = {name: self.fit_kwargs.pop(name) for name in
                    ('max_evaluations', 'max_nfev', 'maxiter', 'maxfev', 'maxfun')
                    if name in self.fit_kwargs}
        budget = None
        if any(value is not None for value in controls.values()):
            budget = _resolve_max_evaluations(self, controls)
        if budget is not None:
            self.fit_kwargs['max_nfev'] = budget
        if method not in ('trf', 'dogbox'):
            raise ValueError(f"method must be 'trf' or 'dogbox', got {method}")

    @property
    def max_evaluations(self):
        value = self.fit_kwargs.get('max_nfev')
        if value is not None:
            return value
        return get_max_evaluations()

    @max_evaluations.setter
    def max_evaluations(self, value):
        coerced = _coerce_max_evaluations(value)
        if coerced is None:
            self.fit_kwargs.pop('max_nfev', None)
        else:
            self.fit_kwargs['max_nfev'] = coerced

    @staticmethod
    def _is_numeric_jacobian_request(jac):
        return isinstance(jac, str) and jac in {'2-point', '3-point', 'cs'}

    def _build_jacobian(self, problem, requested_jac):
        """Build Jacobian function with proper error handling."""
        model = problem.model
        evaluation = problem.evaluation
        use_analytic = True

        if callable(requested_jac) or self._is_numeric_jacobian_request(requested_jac):
            return requested_jac
        
        try:
            # Test if fit_deriv works and check for NaNs/Infs
            test_deriv = _analytic_jacobian_parameter_major(
                model,
                evaluation.fit_deriv(model.parameters),
            )
            
            if np.any(~np.isfinite(test_deriv)):
                if self.verbose:
                    print("Warning: Model has invalid derivatives (NaN/Inf), using numeric Jacobian")
                use_analytic = False
            
            if not use_analytic:
                return '2-point'
            
            return problem.jacobian
            
        except Exception as e:
            if self.verbose:
                print(f"Note: Using numeric Jacobian due to: {type(e).__name__}")
            return '2-point'

    def _fit_single(self, model, x, y=None, z=None, weights=None, max_nfev=None,
                 yerr=None, statistic='chi2', inplace=False, **kwargs):
        """Optimize one observation without constructing a public snapshot."""
        model, x, y, yerr, weights, mask = _resolve_fit_inputs(
            model, x, y, z, yerr, weights, kwargs.pop('mask', None))
        _validate_spectral_inputs(model, x, y, z, yerr, weights, statistic=statistic)
        x, y, yerr, weights, valid = _prepare_fit_data(
            x, y, yerr, weights, mask,
            filter_non_finite=kwargs.pop('filter_non_finite', False))
        self._fit_mask = valid
        if yerr is not None and weights is None:
            weights = np.zeros_like(np.asarray(yerr), dtype=float)
            np.divide(1.0, np.asarray(yerr), out=weights, where=valid)
        if not inplace:
            model = model.copy()
        problem = _FitProblem(model, x, y, weights, valid)
        return self._fit_problem(problem, statistic=statistic,
                                 weighted=weights is not None or yerr is not None,
                                 max_nfev=max_nfev, **kwargs)

    def _fit_problem(self, problem, *, statistic='chi2', weighted=False,
                     max_nfev=None, **kwargs):
        """Optimize one prepared problem, independent of its number of datasets."""
        model = problem.model
        parameters = problem.parameters
        init_values, fit_indices = parameters.values, parameters.free
        param_bounds = _box_bounds(problem.model, parameters)

        residual = problem.residual

        if max_nfev is not None:
            kwargs['max_nfev'] = max_nfev
        budget = _resolve_max_evaluations(self, kwargs)

        # Call options override constructor options without changing later fits.
        merged_kwargs = self.fit_kwargs.copy()
        merged_kwargs.update(kwargs)

        requested_jac = merged_kwargs.pop('jac', 'auto')

        # Build Jacobian
        jac = self._build_jacobian(problem, requested_jac)

        # Prepare bounds
        min_vals = param_bounds[:, 0]
        max_vals = param_bounds[:, 1]
        bounds = (min_vals, max_vals)
        
        # Filter out prism-specific or problematic kwargs that scipy shouldn't see
        fit_kwargs = {k: v for k, v in merged_kwargs.items() 
                      if k not in ['inplace', 'max_nfev', 'yerr']}
        
        fit_kwargs.update({
            'method': self.method,
            'bounds': bounds,
            'jac': jac,
        })
        
        if budget is not None:
            fit_kwargs['max_nfev'] = budget
        
        try:
            result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
        except (LinAlgError, ValueError) as e:
            # Fallback strategy: retry with TRF and numeric Jacobian
            if self.method == 'dogbox':
                if self.verbose:
                    print(f"Warning: DogBox failed ({type(e).__name__}), retrying with TRF...")
                fit_kwargs.update({'method': 'trf', 'jac': '2-point'})
                result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
            else:
                # For TRF, just use numeric Jacobian
                if self.verbose:
                    print(f"Warning: Optimization failed ({type(e).__name__}), using numeric Jacobian...")
                fit_kwargs['jac'] = '2-point'
                result = optimize.least_squares(residual, np.array(init_values), **fit_kwargs)
        
        # Compute covariance if requested
        native_cov = None
        if self._estimate_covariance:
            try:
                J = result.jac
                if J is None or J.shape[0] == 0:
                    if self.verbose:
                        print("⚠ No Jacobian available for covariance calculation.")
                else:
                    # Compute Hessian approximation: H = J.T @ J
                    try:
                        H = np.dot(J.T, J)
                        cond = np.linalg.cond(H)
                        if cond > 1e12:
                            if self.verbose:
                                print(f"⚠ Jacobian is ill-conditioned (κ={cond:.2e}), using pseudo-inverse")
                        native_cov = np.linalg.pinv(H)
                        # The inverse of this symmetric Hessian is symmetric.
                        # Remove SVD roundoff before downstream covariance sampling.
                        native_cov = (native_cov + native_cov.T) * 0.5
                    except LinAlgError as e:
                        if self.verbose:
                            print(f"⚠ Covariance calculation failed: {e}")
                        native_cov = None
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Failed to calculate covariance: {e}")
        
        n_data = problem.ndata
        self.fit_info = {
            'success': result.success,
            'nfev': result.nfev,
            'message': getattr(result, 'message', ''),
            'native_result': result,
            'param_cov': native_cov,
            'cost': result.cost,
            'stat': float(np.dot(result.fun, result.fun)),
            'statmethod': statistic if weighted else 'leastsq',
            'nfree': len(init_values),
            'ndata': n_data,
            'dof': n_data - len(init_values),
            'optimality': getattr(result, 'optimality', np.nan),
            'status': result.status,
        }
        problem.finish(result.x)
        return _attach_covariance(model, native_cov)


class TRF(ScipyFitter):
    """SciPy trust-region reflective fitter.

    Parameters
    ----------
    ftol, xtol, gtol : float, optional
        Convergence tolerances passed to ``least_squares``. Defaults are
        ``1e-10``.
    loss : str, optional
        Robust loss function passed to SciPy. Default is ``'linear'``.
    jac : {'auto', '2-point', '3-point', 'cs'} or callable, optional
        Jacobian strategy. ``'auto'`` uses Prism analytic derivatives when
        available. Default is ``'auto'``.
    covariance : bool, optional
        Estimate parameter covariance from the final Jacobian. Default is True.
    **kwargs
        Additional ``least_squares`` keyword arguments.
    """
    def __init__(self, ftol=1e-10, xtol=1e-10, gtol=1e-10, loss='linear', jac='auto', covariance=True, **kwargs):
        super().__init__(method='trf', covariance=covariance,
        ftol=ftol,     # Force stricter gradient/cost progression
        xtol=xtol,
        gtol=gtol,
        loss=loss,  # options: 'linear', 'huber', 'soft_l1', 'cauchy', 'arctan'
        jac=jac,  # 'auto' uses analytic derivatives when available; numeric strings force finite differences
        **kwargs)


class ScipyDogBox(ScipyFitter):
    """SciPy dogbox fitter with rectangular trust regions.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments forwarded to ``least_squares``.

    Notes
    -----
    ``dogbox`` can be effective for tightly box-constrained problems with a modest
    number of free parameters. Prism keeps the same diagnostics and covariance
    handling as :class:`TRF`.
    """
    def __init__(self, **kwargs):
        super().__init__(method='dogbox', **kwargs)
