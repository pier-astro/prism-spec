import re

with open("prism/modeling/fitting/scipyfit.py", "r") as f:
    text = f.read()

# Replace imports
text = text.replace("from .base import FitterBase, _apply_tied_fast", "from astropy.modeling.fitting import Fitter, model_to_fit_params\nfrom .utils import _get_tied_info, _apply_tied_fast")
text = text.replace("class ScipyFitter(FitterBase):", "class ScipyFitter(Fitter):")
text = text.replace("""        super().__init__(calc_uncertainties=calc_uncertainties,
                        force_numerical_covariance=force_numerical_covariance,
                        verbose=verbose,
                        filter_non_finite=filter_non_finite)""", "        self.calc_uncertainties = calc_uncertainties\n        self.verbose = verbose")

# Replace _fit_impl with __call__
text = text.replace("def _fit_impl(self, prep_data, max_nfev=None, **kwargs):", """def __call__(self, model, x, y, z=None, weights=None, max_nfev=None, **kwargs):""")

# Transform prep_data unpacking
prep_data_str = """        model = prep_data['model']
        x = prep_data['x']
        y = prep_data['y']
        weights = prep_data['weights']
        init_values = prep_data['init_values']
        fit_indices = prep_data['fit_indices']
        param_bounds = prep_data['param_bounds']
        tied_info = prep_data['tied_info']
        params_cache = prep_data['params_cache']"""

new_prep = """        model = model.copy()
        init_values, fit_indices, _ = model_to_fit_params(model)
        bounds_list = [getattr(model, n).bounds for n in model.param_names]
        all_bounds = np.array([(b[0] if b[0] is not None else -np.inf,
                                b[1] if b[1] is not None else np.inf)
                               for b in bounds_list])
        param_bounds = all_bounds[fit_indices]

        tied_info = _get_tied_info(model)
        params_cache = model.parameters.copy()"""

text = text.replace(prep_data_str, new_prep)

# At the end, instead of returning a dict, we need to populate self.fit_info and return model
return_dict_str = """        return {
            'fitted_params': result.x,
            'success': result.success,
            'nfev': result.nfev,
            'message': getattr(result, 'message', ''),
            'native_result': result,
            'native_cov': native_cov
        }"""

new_return = """        self.fit_info = {
            'success': result.success,
            'nfev': result.nfev,
            'message': getattr(result, 'message', ''),
            'native_result': result,
            'param_cov': native_cov
        }
        params_cache[fit_indices] = result.x
        model.parameters = params_cache
        if tied_info:
            _apply_tied_fast(model, tied_info, params_cache)
        return model"""

text = text.replace(return_dict_str, new_return)

with open("prism/modeling/fitting/scipyfit.py", "w") as f:
    f.write(text)

print("scipyfit rewritten.")
