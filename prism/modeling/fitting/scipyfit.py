"""
SciPy-based fitters for prism.modeling.fitting

This module provides wrappers around scipy.optimize.least_squares with
standardized interfaces and robust error handling.
"""

import warnings
import numpy as np
from scipy import optimize
from numpy.linalg import LinAlgError
from astropy.modeling.fitting import Fitter, model_to_fit_params
from .extension import (
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    get_max_evaluations,
    validate_symmetric_yerr,
)
from .multifit import MultiFitMixin
from .utils import (
    _analytic_jacobian_parameter_major,
    _apply_tied_fast,
    _get_tied_info,
    _prime_tied_analytic_jacobian_warning,
    _reduce_tied_analytic_jacobian,
)

__all__ = ['ScipyFitter', 'ScipyTRF', 'ScipyDogBox']


class ScipyFitter(MultiFitMixin, Fitter):
    """
    Wrapper around ``scipy.optimize.least_squares``.

    Supports ``'trf'`` and ``'dogbox'`` methods with native bounds.
    Covariance is estimated from the Jacobian (``J.T @ J`` inverse/pinv)
    when ``calc_uncertainties=True``.
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)
    
    def __init__(self, method='trf', calc_uncertainties=False, verbose=False, **kwargs):
        self.method = method
        self.calc_uncertainties = calc_uncertainties
        self.verbose = verbose
        self.fit_kwargs = kwargs
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

    def _build_jacobian(self, model, x, fit_indices, tied_info, params_cache, weights, requested_jac):
        """Build Jacobian function with proper error handling."""
        use_analytic = True

        if callable(requested_jac) or self._is_numeric_jacobian_request(requested_jac):
            return requested_jac
        
        if not hasattr(model, 'fit_deriv'):
            return '2-point'
        
        try:
            # Test if fit_deriv works and check for NaNs/Infs
            test_deriv = _analytic_jacobian_parameter_major(
                model,
                model.fit_deriv(x, *model.parameters),
            )
            
            if np.any(~np.isfinite(test_deriv)):
                if self.verbose:
                    print("Warning: Model has invalid derivatives (NaN/Inf), using numeric Jacobian")
                use_analytic = False
            
            if not use_analytic:
                return '2-point'
            
            def jacobian(p):
                params_cache[fit_indices] = p
                model.parameters = params_cache
                if tied_info:
                    _apply_tied_fast(model, tied_info, params_cache)
                
                # Get full jacobian
                J_all = _reduce_tied_analytic_jacobian(
                    model,
                    model.fit_deriv(x, *model.parameters),
                    warn=bool(tied_info),
                    stacklevel=3,
                )
                
                # Check for NaNs/infs
                if np.any(~np.isfinite(J_all)):
                    raise ValueError("Invalid derivatives encountered")
                
                # Select only free parameters
                J = J_all[fit_indices].T
                
                if weights is not None:
                    return J * weights[:, np.newaxis]
                return J
                
            return jacobian
            
        except Exception as e:
            if self.verbose:
                print(f"Note: Using numeric Jacobian due to: {type(e).__name__}")
            return '2-point'

    def __call__(self, model, x, y, z=None, weights=None, max_nfev=None,
                 yerr=None, statistic='chi2', **kwargs):
        """
        Implement SciPy least_squares fitting.
        """
        if yerr is not None:
            yerr = validate_symmetric_yerr(yerr)
        if yerr is not None and weights is None:
            weights = 1.0 / np.asarray(yerr)
        model = model.copy()
        _prime_tied_analytic_jacobian_warning(model)
        init_values, fit_indices, _ = model_to_fit_params(model)
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                b[1] if b[1] is not None else np.inf)
                               for b in bounds_list])
        param_bounds = all_bounds[fit_indices]

        # Reject parameters with equal bounds (lower == upper): scipy requires
        # strict inequality.  Use param.fixed = True instead.
        equal_mask = param_bounds[:, 0] == param_bounds[:, 1]
        if np.any(equal_mask):
            bad_names = [model.param_names[fi] for fi, m
                         in zip(fit_indices, equal_mask) if m]
            raise ValueError(
                f"Parameters with equal lower and upper bounds: "
                f"{', '.join(bad_names)}. Use `param.fixed = True` instead."
            )

        tied_info = _get_tied_info(model)
        params_cache = model.parameters.copy()
        
        has_tied = bool(tied_info)
        
        # Build residual function
        def residual(p):
            params_cache[fit_indices] = p
            model.parameters = params_cache
            if has_tied:
                _apply_tied_fast(model, tied_info, params_cache)
            diff = model(x) - y
            if weights is not None:
                return diff * weights
            return diff
        
        # Prepare final kwargs, merging init and call values
        # Call kwargs take precedence over init kwargs
        merged_kwargs = self.fit_kwargs.copy()
        merged_kwargs.update(kwargs)

        requested_jac = merged_kwargs.pop('jac', 'auto')

        # Build Jacobian
        jac = self._build_jacobian(
            model,
            x,
            fit_indices,
            tied_info,
            params_cache,
            weights,
            requested_jac,
        )
        
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
        
        if max_nfev is not None:
            fit_kwargs['max_nfev'] = max_nfev
        
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
        if self.calc_uncertainties:
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
                    except LinAlgError as e:
                        if self.verbose:
                            print(f"⚠ Covariance calculation failed: {e}")
                        native_cov = None
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Failed to calculate covariance: {e}")
        
        n_data = len(y) if z is None else int(np.prod(np.shape(y)))
        self.fit_info = {
            'success': result.success,
            'nfev': result.nfev,
            'message': getattr(result, 'message', ''),
            'native_result': result,
            'param_cov': native_cov,
            'cost': result.cost,
            'stat': 2.0 * result.cost,
            'statmethod': statistic if (weights is not None or yerr is not None) else 'leastsq',
            'nfree': len(init_values),
            'ndata': n_data,
            'dof': n_data - len(init_values),
            'optimality': getattr(result, 'optimality', np.nan),
            'status': result.status,
        }
        params_cache[fit_indices] = result.x
        model.parameters = params_cache
        if tied_info:
            _apply_tied_fast(model, tied_info, params_cache)
        return model


class ScipyTRF(ScipyFitter):
    """SciPy trust-region reflective fitter."""
    def __init__(self, ftol=1e-10, xtol=1e-10, gtol=1e-10, loss='linear', jac='auto', **kwargs):
        super().__init__(method='trf',
        ftol=ftol,     # Force stricter gradient/cost progression
        xtol=xtol,
        gtol=gtol,
        loss=loss,  # options: 'linear', 'huber', 'soft_l1', 'cauchy', 'arctan'
        jac=jac,  # 'auto' uses analytic derivatives when available; numeric strings force finite differences
        **kwargs)


class ScipyDogBox(ScipyFitter):
    """SciPy dogbox fitter with rectangular trust regions."""
    def __init__(self, **kwargs):
        super().__init__(method='dogbox', **kwargs)