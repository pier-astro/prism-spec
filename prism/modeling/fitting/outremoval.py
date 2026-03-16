import warnings
import numpy as np
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

from .base import FitterBase, _apply_tied_fast
import copy

__all__ = ['FittingWithOutlierRemoval']

class FittingWithOutlierRemoval(FitterBase):
    """
    Fitter wrapper to perform iterative fitting with outlier removal.
    
    Compatible with any `prism` fitter (e.g. `AstroFitter`). Provides automatic 
    batched operations across `MultiFitResult` arrays while maintaining 
    full Astropy model compliance.
    
    Parameters
    ----------
    fitter : FitterBase
        Instantiated FitterBase object to wrap.
    outlier_func : callable, optional
        Function to compute outliers (must return a masked array).
        Defaults to `astropy.stats.sigma_clip`.
    niter : int, optional
        Maximum number of outlier rejection iterations. Defaults to 3.
    **outlier_kwargs
        Additional kwargs forwarded to `outlier_func`.
    """
    def __init__(self, fitter, outlier_func=sigma_clip, niter=3, **outlier_kwargs):
        # We steal the base fitter's flags dynamically so the wrapper integrates seamlessly
        calc_unc = getattr(fitter, 'calc_uncertainties', False)
        force_num_cov = getattr(fitter, 'force_numerical_covariance', False)
        verbose = getattr(fitter, 'verbose', False)
        filter_non_finite = getattr(fitter, 'filter_non_finite', False)

        super().__init__(
            calc_uncertainties=calc_unc,
            force_numerical_covariance=force_num_cov,
            verbose=verbose,
            filter_non_finite=filter_non_finite
        )
        
        self._fitter = fitter
        self.outlier_func = outlier_func
        self.niter = niter
        self.outlier_kwargs = outlier_kwargs
        
        # Store mask explicitly for single target fits
        self.outlier_mask = None

    def _multifit_spawn_fitter(self):
        # Override to ensure the nested fitter is safely cloned for parallel workers
        worker_fitter = super()._multifit_spawn_fitter()
        if hasattr(self._fitter, '_multifit_spawn_fitter'):
            worker_fitter._fitter = self._fitter._multifit_spawn_fitter()
        else:
            worker_fitter._fitter = copy.copy(self._fitter)
        return worker_fitter

    def _fit_impl(self, prep_data, **kwargs):
        # Create a transient prep_data to manipulate weights locally
        prep_data_iter = prep_data.copy()
        
        # If no weights, create default array to hold iteration modifications
        if prep_data_iter['weights'] is None:
            prep_data_iter['weights'] = np.ones_like(prep_data['y'], dtype=float)
        else:
            prep_data_iter['weights'] = np.copy(prep_data_iter['weights'])
            
        best_result = None
        last_mask = np.zeros_like(prep_data['y'], dtype=bool)
        
        # Fit logic over n interactions
        for i in range(self.niter + 1):
            
            # Handle the wrapped fitter. We pass down exact kwargs.
            result = self._fitter._fit_impl(prep_data_iter, **kwargs)
            best_result = result
            
            # If things went pear-shaped or no free parameters, just abort loop
            if not result['success'] or len(prep_data['fit_indices']) == 0:
                break
                
            # Stop if iterations complete
            if i == self.niter:
                break
                
            # Re-apply parameters safely to evaluate our residual model
            fit_model = prep_data['model'].copy()
            
            fit_model.parameters[prep_data['fit_indices']] = result['fitted_params']
            _apply_tied_fast(fit_model, prep_data['tied_info'], fit_model.parameters)
            
            # Calculate residuals
            y_fit = fit_model(prep_data['x'])
            residuals = prep_data['y'] - y_fit
            
            # Isolate valid elements for correct statistics evaluation!
            valid_mask = (prep_data_iter['weights'] > 0)
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
            
            # Force weights of newly identified outliers instantly to zero!
            prep_data_iter['weights'][current_mask] = 0.0

        # Output the outlier mask on our wrapper level for single-fit inquiries
        self.outlier_mask = last_mask
        
        return best_result
