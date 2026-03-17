"""
prism.modeling.models.empirical — empirical/template spectral models.

Provides data-driven models that interpolate or parameterise arbitrary
continuum or emission features without imposing an analytic shape.

Classes
-------
FixedTemplate
    Interpolation wrapper with no free parameters.  Useful as a fixed
    continuum or line template multiplied by a free amplitude.
BSpline
    B-spline whose control-point coefficients are exposed as fit
    parameters.  Supports analytic ``fit_deriv`` via the B-spline
    design matrix (efficient Jacobian for fitters).
"""
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from scipy.interpolate import interp1d, BSpline as _ScipyBSpline

__all__ = ['FixedTemplate', 'BSpline']

class FixedTemplate(Fittable1DModel):
    """
    Interpolation-based template with no free parameters.

    Wraps ``scipy.interpolate.interp1d`` in an Astropy ``Fittable1DModel``
    so that a fixed spectral template can be combined with other models in
    compound expressions (e.g. ``FixedTemplate(wave, flux) * amplitude_model``).
    Because it has no free parameters Astropy applies the chain rule when
    computing the compound Jacobian, so analytic derivatives are preserved.

    Parameters
    ----------
    x, y : array-like
        Template wavelength and flux arrays.
    interp_kind : str
        Interpolation kind passed to ``interp1d`` (default ``'linear'``).
    fill_value : float
        Value assigned outside ``x`` range (default ``0.0``).
    bounds_error : bool
        Raise if evaluated outside ``x`` range (default ``False``).
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
    B-spline model whose coefficients are exposed as fit parameters.

    A new parametrised subclass is created dynamically on each instantiation,
    with each coefficient ``c0, c1, ...`` as an independent ``Parameter``.
    Supports analytic ``fit_deriv`` via the B-spline design matrix.

    Parameters
    ----------
    knots : array-like
        Knot vector (not including repeated boundary knots; must satisfy
        ``len(knots) > degree + 1``).
    degree : int
        Spline degree (default 3 = cubic).
    name : str
        Name of the generated model class.
    **kwargs
        Initial coefficient values passed as ``c0=..., c1=..., ...``.

    Examples
    --------
    >>> import numpy as np
    >>> from prism.modeling.models import BSpline
    >>> knots = np.linspace(4500, 7000, 10)
    >>> bsp = BSpline(knots, degree=3, name='cont')
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
