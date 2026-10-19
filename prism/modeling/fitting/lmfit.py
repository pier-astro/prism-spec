"""
Levenberg-Marquardt fitter with parameter transformations for bounds support.
"""

import numpy as np
from .extension import _validate_spectral_inputs
from .data import _prepare_fit_data
from scipy.optimize import leastsq
from numpy.linalg import LinAlgError
from astropy.modeling.fitting import Fitter
from .extension import (
    _attach_covariance,
    _coerce_max_evaluations,
    _fitter_covariance,
    _fitter_stdevs,
    get_max_evaluations,
)
from .multifit import MultiFitMixin
from .problem import _FitProblem

__all__ = ['LMFitter']


class LMFitter(MultiFitMixin, Fitter):
    """
    Levenberg-Marquardt fitter with internal bound transforms.

    Parameters
    ----------
    calc_uncertainties : bool, optional
        If ``True``, estimate parameter covariance after convergence. Default is
        ``False``.
    verbose : bool, optional
        If ``True``, print basic progress diagnostics. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.optimize.leastsq``.

    Notes
    -----
    Classical Levenberg-Marquardt does not natively support bounds. Prism maps the
    external bounded parameter space to an unconstrained internal space before each
    optimisation step, then transforms both best-fit values and covariance back to
    the user-facing parameterisation.
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)
    
    def __init__(self, calc_uncertainties=False, verbose=False, **kwargs):
        self.calc_uncertainties = calc_uncertainties
        self.verbose = verbose
        self.leastsq_kwargs = kwargs

    @property
    def max_evaluations(self):
        value = self.leastsq_kwargs.get('maxfev')
        if value is not None:
            return value
        return get_max_evaluations()

    @max_evaluations.setter
    def max_evaluations(self, value):
        coerced = _coerce_max_evaluations(value)
        if coerced is None:
            self.leastsq_kwargs.pop('maxfev', None)
        else:
            self.leastsq_kwargs['maxfev'] = coerced

    @staticmethod
    def _normalize_bound_pair(bounds):
        lower, upper = bounds if isinstance(bounds, (list, tuple, np.ndarray)) else (bounds[0], bounds[1])
        lower = None if lower is None or np.isneginf(lower) else lower
        upper = None if upper is None or np.isposinf(upper) else upper
        return lower, upper

    def _transform_params(self, external_values, bounds_list):
        internal = np.zeros(len(external_values))
        self._param_transforms = {}
        
        for i, (value, bounds) in enumerate(zip(external_values, bounds_list)):
            lower, upper = self._normalize_bound_pair(bounds)
            
            if lower is None and upper is None:
                internal[i] = value
                self._param_transforms[i] = {'type': 'none'}
            elif lower is not None and upper is not None:
                if not (lower < value < upper):
                    value = (lower + upper) / 2
                internal[i] = np.log((value - lower) / (upper - value))
                self._param_transforms[i] = {'type': 'two-sided', 'lower': lower, 'upper': upper}
            elif lower is not None:
                if value <= lower:
                    value = lower + 0.1 * abs(lower) if lower != 0 else 0.1
                internal[i] = np.log(value - lower)
                self._param_transforms[i] = {'type': 'lower', 'lower': lower}
            else:
                if value >= upper:
                    value = upper - 0.1 * abs(upper) if upper != 0 else -0.1
                internal[i] = np.log(upper - value)
                self._param_transforms[i] = {'type': 'upper', 'upper': upper}
                
        return internal

    def _untransform_params(self, internal_values):
        external = np.zeros(len(internal_values))
        for i, internal_val in enumerate(internal_values):
            transform = self._param_transforms[i]
            
            if transform['type'] == 'none':
                external[i] = internal_val
            elif transform['type'] == 'two-sided':
                lower, upper = transform['lower'], transform['upper']
                exp_val = np.exp(internal_val)
                external[i] = (lower + upper * exp_val) / (1 + exp_val)
            elif transform['type'] == 'lower':
                lower = transform['lower']
                external[i] = lower + np.exp(internal_val)
            elif transform['type'] == 'upper':
                upper = transform['upper']
                external[i] = upper - np.exp(internal_val)
        return external

    def _transform_covariance(self, cov_internal, internal_values):
        n_params = len(internal_values)
        jacobian = np.zeros((n_params, n_params))
        for i, internal_val in enumerate(internal_values):
            transform = self._param_transforms[i]
            if transform['type'] == 'none':
                jacobian[i, i] = 1.0
            elif transform['type'] == 'two-sided':
                lower, upper = transform['lower'], transform['upper']
                exp_val = np.exp(internal_val)
                jacobian[i, i] = (upper - lower) * exp_val / (1 + exp_val)**2
            elif transform['type'] == 'lower':
                jacobian[i, i] = np.exp(internal_val)
            elif transform['type'] == 'upper':
                jacobian[i, i] = -np.exp(internal_val)
        return jacobian @ cov_internal @ jacobian.T

    def __call__(self, model, x, y, z=None, weights=None, max_nfev=None,
                 yerr=None, statistic='chi2', **kwargs):
        _validate_spectral_inputs(model, x, y, z, yerr, weights, statistic=statistic)
        x, y, yerr, weights, valid = _prepare_fit_data(
            x, y, yerr, weights, kwargs.pop('mask', None),
            filter_non_finite=kwargs.pop('filter_non_finite', False))
        if yerr is not None and weights is None:
            weights = np.zeros_like(np.asarray(yerr), dtype=float)
            np.divide(1.0, np.asarray(yerr), out=weights, where=valid)
        if 'uncertainties' in kwargs:
            raise TypeError("The 'uncertainties' parameter is no longer supported.")

        model = model.copy()
        problem = _FitProblem(model, x, y, weights, valid)
        parameters = problem.parameters
        init_values = parameters.values
        param_bounds = parameters.bounds

        n_free = len(init_values)
        
        if n_free == 0:
            residual = problem.residual(init_values)
            stat = float(np.dot(residual, residual))
            self.fit_info = {
                'success': True, 'nfev': 0, 'message': 'No free parameters',
                'param_cov': None, 'stat': stat, 'cost': 0.5 * stat,
                'statmethod': statistic if weights is not None else 'leastsq',
                'nfree': 0, 'ndata': problem.ndata, 'dof': problem.ndata,
            }
            return _attach_covariance(problem.finish(init_values), None)

        def residuals(internal_params):
            external_params = self._untransform_params(internal_params)
            return problem.residual(external_params)

        x0_internal = self._transform_params(init_values, param_bounds)
        
        # Merge call-time kwargs with init-time leastsq_kwargs
        combined_kwargs = self.leastsq_kwargs.copy()
        combined_kwargs.update(kwargs)
        
        leastsq_kws = {k: v for k, v in combined_kwargs.items() 
                       if k not in ['inplace', 'max_nfev', 'yerr']}
        if max_nfev is not None:
            leastsq_kws['maxfev'] = max_nfev
        
        try:
            result = leastsq(residuals, x0_internal, full_output=True, **leastsq_kws)
            xopt_internal, cov_internal, infodict, mesg, ier = result
            
            if cov_internal is None and self.calc_uncertainties:
                try:
                    fjac = infodict['fjac']
                    ipvt = infodict['ipvt']
                    n_params = len(xopt_internal)
                    R = np.triu(fjac[:n_params, :n_params])
                    try:
                        RTR = R.T @ R
                        cond = np.linalg.cond(RTR)
                        if cond < 1e12:
                            cov_internal = np.linalg.inv(RTR)
                        else:
                            cov_internal = np.linalg.pinv(RTR)
                        perm = ipvt - 1
                        cov_internal = cov_internal[perm, :][:, perm]
                    except (LinAlgError, np.linalg.LinAlgError):
                        cov_internal = None
                except Exception:
                    cov_internal = None
        except Exception as e:
            raise RuntimeError(f"LMFitter failed: {str(e)}")
        
        success = ier in [1, 2, 3, 4]
        xopt_external = self._untransform_params(xopt_internal)
        
        # Compute cost/chi2 for fit_info
        resid_final = residuals(xopt_internal)
        chi2 = np.sum(resid_final**2)
        
        native_cov = None
        if cov_internal is not None and success:
            try:
                dof = max(1, len(resid_final) - n_free)
                red_chi2 = chi2 / dof
                cov_scaled = cov_internal * red_chi2
                native_cov = self._transform_covariance(cov_scaled, xopt_internal)
                diag = np.diag(native_cov)
                if np.any(diag < 0):
                    diag[diag < 0] = np.nan
                    native_cov[np.diag_indices_from(native_cov)] = diag
            except Exception:
                native_cov = None
        
        n_data = problem.ndata
        cost_val = chi2 / 2.0
        
        self.fit_info = {
            'success': success,
            'nfev': infodict['nfev'],
            'message': mesg,
            'param_cov': native_cov,
            'stat': chi2,
            'cost': cost_val,
            'statmethod': statistic if (weights is not None or yerr is not None) else 'leastsq',
            'nfree': n_free,
            'ndata': n_data,
            'dof': n_data - n_free,
        }
        
        problem.finish(xopt_external)

        return _attach_covariance(model, native_cov)