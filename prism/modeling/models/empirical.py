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

class BSpline(Fittable1DModel):
    """
    Dynamic BSpline model with coefficients exposed as fit parameters.

    Usage remains compact and familiar:
    `BSpline(knots, degree=3, c0=..., c1=..., ...)`

    Internally this base class creates a specialized subclass where each
    coefficient is attached as a `Parameter` descriptor.
    """
    n_inputs = 1
    n_outputs = 1

    def __new__(cls, knots, degree=3, name="BSpline", **kwargs):
        knots = np.asarray(knots)
        ncoeffs = len(knots) - degree - 1
        if ncoeffs <= 0:
            raise ValueError("Invalid knot vector/degree: no spline coefficients available.")

        params = {f'c{i}': Parameter(default=1.0) for i in range(ncoeffs)}

        def evaluate(self, x, *coeffs):
            spline = _ScipyBSpline(self._knots, coeffs, self._degree, extrapolate=True)
            return spline(x)

        def fit_deriv(self, x, *coeffs):
            design_matrix = _ScipyBSpline.design_matrix(
                np.asarray(x), self._knots, self._degree, extrapolate=True
            )
            dense_matrix = design_matrix.toarray()
            return [dense_matrix[:, i] for i in range(self._ncoeffs)]

        model_class = type(name, (Fittable1DModel,), {
            **params,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            'n_inputs': 1,
            'n_outputs': 1,
            '_knots': knots,
            '_degree': degree,
            '_ncoeffs': ncoeffs,
        })

        return model_class(**kwargs)
