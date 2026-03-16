import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from scipy.interpolate import interp1d, BSpline as _ScipyBSpline

__all__ = ['FixedTemplate', 'BSpline']

class FixedTemplate(Fittable1DModel):
    """
    Interpolation-based template model with NO free parameters.
    
    Initializes a spline interpolator from given x and y arrays. Since it has 
    no parameters, astropy will automatically bypass numerical derivates and 
    instead safely apply the chain rule when combined with other models
    (e.g., `FixedTemplate * BSpline`).
    """
    n_inputs = 1
    n_outputs = 1
    
    def __init__(self, x, y, interp_kind='linear', fill_value=0.0, bounds_error=False, **kwargs):
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._interp = interp1d(
            self._x, self._y, 
            kind=interp_kind, 
            bounds_error=bounds_error, 
            fill_value=fill_value
        )
        super().__init__(**kwargs)
        
    def evaluate(self, x):
        return self._interp(np.asarray(x))


def BSpline(knots, degree=3, name="BSpline", **kwargs):
    """
    Factory function for a dynamic BSpline Fittable1DModel.
    
    Attaches coefficients as independent Fittable1DModel parameters. Evaluates
    a scipy BSpline using the fixed knots and degrees but completely free 
    coefficients. Also provides full analytical derivatives.
    """
    knots = np.asarray(knots)
    ncoeffs = len(knots) - degree - 1
    
    # Define parameters (c0, c1, ...)
    params = {f'c{i}': Parameter(default=1.0) for i in range(ncoeffs)}
    
    def evaluate(self, x, *coeffs):
        spline = _ScipyBSpline(knots, coeffs, degree, extrapolate=True)
        return spline(x)
        
    def fit_deriv(self, x, *coeffs):
        # We can extract exact analytical derivatives w.r.t our coefficients
        # by evaluating scipy's design matrix for the BSpline basis functions.
        # This yields exactly (dBSpline / dc_i) = b_i(x)
        design_matrix = _ScipyBSpline.design_matrix(x, knots, degree, extrapolate=True)
        dense_matrix = design_matrix.toarray()
        
        # Astropy derivatives must be returned as a list of arrays mapping to each parameter
        return [dense_matrix[:, i] for i in range(ncoeffs)]
        
    model_class = type(name, (Fittable1DModel,), {
        **params,
        'evaluate': evaluate,
        'fit_deriv': fit_deriv,
        'n_inputs': 1,
        'n_outputs': 1,
        '_knots': knots,
        '_degree': degree,
    })
    
    return model_class(**kwargs)
