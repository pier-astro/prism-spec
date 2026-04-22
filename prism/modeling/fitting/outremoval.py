import warnings
import numpy as np
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling.fitting import Fitter, model_to_fit_params

from .extension import _fitter_covariance, _fitter_stdevs
from .multifit import MultiFitMixin
from .utils import _apply_tied_fast, _get_tied_info
import copy

__all__ = ['FittingWithOutlierRemoval']

class FittingWithOutlierRemoval(MultiFitMixin, Fitter):
    """
    Iterative outlier-rejection wrapper around any prism correlation fitter.
    """

    covariance = property(_fitter_covariance)
    stdevs = property(_fitter_stdevs)
    std = property(_fitter_stdevs)

    def __init__(self, fitter, outlier_func=sigma_clip, niter=3, **outlier_kwargs):
        self._fitter = fitter
        self.outlier_func = outlier_func
        self.niter = niter
        self.outlier_kwargs = outlier_kwargs
        self.outlier_mask = None

    @property
    def calc_uncertainties(self):
        return getattr(self._fitter, 'calc_uncertainties', False)

    @calc_uncertainties.setter
    def calc_uncertainties(self, value):
        self._fitter.calc_uncertainties = value

    @property
    def verbose(self):
        return getattr(self._fitter, 'verbose', False)

    @verbose.setter
    def verbose(self, value):
        self._fitter.verbose = value

    @property
    def max_evaluations(self):
        return getattr(self._fitter, 'max_evaluations', None)

    @max_evaluations.setter
    def max_evaluations(self, value):
        self._fitter.max_evaluations = value

    def _multifit_spawn_fitter(self):
        # Override to ensure the nested fitter is safely cloned for parallel workers
        import copy
        worker_fitter = copy.copy(self)
        if hasattr(self._fitter, '_multifit_spawn_fitter'):
            worker_fitter._fitter = self._fitter._multifit_spawn_fitter()
        else:
            worker_fitter._fitter = copy.copy(self._fitter)
        return worker_fitter

    def __call__(self, model, x, y, z=None, weights=None, **kwargs):
        # We assume _prepare_fitting logic has run or isn't needed here 
        # because the internal Fitter will get the wrapped call.
        if weights is None:
            weights = np.ones_like(y, dtype=float)
        else:
            weights = np.copy(weights)
            
        best_model = None
        last_mask = np.zeros_like(y, dtype=bool)
        
        for i in range(self.niter + 1):
            
            # Pass to inner fitter
            call_weights = np.copy(weights)
            
            try:
                # The underlying fitter might be patched with wrapped_call
                # and accept filter_non_finite, yerr etc.
                result_model = self._fitter(model, x, y, z=z, weights=call_weights, **kwargs)
            except Exception as e:
                if self.verbose:
                    print(f"FittingWithOutlierRemoval iteration {i} failed: {e}")
                break
                
            best_model = result_model
            fit_info = getattr(self._fitter, 'fit_info', {})
            
            # If things went pear-shaped, jump out
            if not fit_info.get('success', False):
                break
                
            if i == self.niter:
                break
                
            # Calculate residuals
            y_fit = result_model(x)
            residuals = y - y_fit
            
            # Isolate valid elements
            valid_mask = (weights > 0)
            if not np.any(valid_mask):
                break 
                
            masked_residuals = np.ma.array(residuals, mask=~valid_mask)
            
            clipped = self.outlier_func(masked_residuals, **self.outlier_kwargs)
            
            if hasattr(clipped, 'mask'):
                current_mask = clipped.mask
                if np.isscalar(current_mask):
                    current_mask = np.ones_like(residuals, dtype=bool) if current_mask else np.zeros_like(residuals, dtype=bool)
            else:
                current_mask = np.zeros_like(residuals, dtype=bool)
                
            # Has mask stabilized?
            if np.array_equal(current_mask, last_mask):
                break
            
            last_mask = current_mask.copy()
            
            # Force weights of newly identified outliers to zero
            weights[current_mask] = 0.0

        self.outlier_mask = last_mask
        self.fit_info = getattr(self._fitter, 'fit_info', {})
        
        return best_model
