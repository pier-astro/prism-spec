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
from astropy import units as u
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
        Template coordinate and value arrays.
    interp_kind : str, optional
        Interpolation kind passed to ``interp1d``. Default is ``'linear'``.
    fill_value : float, optional
        Value assigned outside the sampled template range. Default is ``0.0``.
    bounds_error : bool, optional
        If ``True``, raise when evaluated outside the template range. Default is
        ``False``.

    Returns
    -------
    FixedTemplate
        Unit-aware interpolation model with no free parameters.

    Notes
    -----
    ``FixedTemplate`` is useful for empirical continua, iron templates, or any
    externally tabulated component that should remain fixed while participating in
    Astropy compound models. If the input arrays carry units, evaluation preserves
    them and Prism cooperates with Astropy's unit-stripping hooks during fitting.

    Examples
    --------
    >>> tmpl = FixedTemplate([5000.0, 5005.0, 5010.0], [0.0, 1.0, 0.0])
    >>> tmpl(5005.0)
    """
    n_inputs = 1
    n_outputs = 1
    
    def __init__(self, x, y, interp_kind='linear', fill_value=0.0, bounds_error=False, **kwargs):
        self._x_unit = x.unit if isinstance(x, u.Quantity) else None
        self._y_unit = y.unit if isinstance(y, u.Quantity) else None
        self._x = np.asarray(x.to_value(self._x_unit), dtype=float) if self._x_unit is not None else np.asarray(x)
        self._y = np.asarray(y.to_value(self._y_unit), dtype=float) if self._y_unit is not None else np.asarray(y)
        self._interp_kind = interp_kind
        self._fill_value = fill_value
        self._bounds_error = bounds_error
        self._build_interp()
        super().__init__(**kwargs)

    def _build_interp(self):
        self._interp = interp1d(
            self._x, self._y, 
            kind=self._interp_kind,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value,
        )
        
    def evaluate(self, x):
        """Evaluate the template on a new coordinate grid.

        Parameters
        ----------
        x : array-like or astropy.units.Quantity
            Sampling coordinates.

        Returns
        -------
        numpy.ndarray or astropy.units.Quantity
            Interpolated template values.
        """
        if isinstance(x, u.Quantity):
            x = x.to_value(self._x_unit) if self._x_unit is not None else x.value
        result = self._interp(np.asarray(x))
        return result * self._y_unit if self._y_unit is not None else result

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {}

    def without_units_for_data(self, **kwargs):
        """Return a copy adapted to Astropy's unit-stripped fitting context.

        Parameters
        ----------
        **kwargs
            Mapping from model input/output names to the current fitting data.

        Returns
        -------
        FixedTemplate
            Copy of the template with units converted away when Astropy requests
            a unitless model.
        """
        model = self.copy()
        x_data = kwargs.get(self.inputs[0])
        y_data = kwargs.get(self.outputs[0])
        if model._x_unit is not None and isinstance(x_data, u.Quantity):
            model._x = np.asarray((model._x * model._x_unit).to_value(x_data.unit), dtype=float)
            model._x_unit = None
        if model._y_unit is not None and isinstance(y_data, u.Quantity):
            model._y = np.asarray((model._y * model._y_unit).to_value(y_data.unit), dtype=float)
            model._y_unit = None
        model._build_interp()
        return model

    def with_units_from_data(self, **kwargs):
        """Return a copy re-attached to the units of the provided data.

        Parameters
        ----------
        **kwargs
            Mapping from model input/output names to the current fitting data.

        Returns
        -------
        FixedTemplate
            Copy of the template carrying the input and output units inferred from
            the supplied data objects.
        """
        model = self.copy()
        x_data = kwargs.get(self.inputs[0])
        y_data = kwargs.get(self.outputs[0])
        model._x_unit = getattr(x_data, 'unit', None)
        model._y_unit = getattr(y_data, 'unit', None)
        model._build_interp()
        return model

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
    degree : int, optional
        Spline degree. Default is ``3`` for a cubic spline.
    name : str, optional
        Name of the generated model class. Default is ``'BSpline'``.
    **kwargs
        Initial coefficient values passed as ``c0=..., c1=..., ...``.

    Returns
    -------
    BSpline
        Dynamically generated Astropy model whose free parameters are the spline
        coefficients.

    Notes
    -----
    Prism exposes the B-spline coefficients directly as model parameters, which is
    convenient for flexible continua and empirical backgrounds. The analytic
    derivative is computed from SciPy's spline design matrix, so least-squares
    fitters can use an efficient exact Jacobian instead of finite differences.
    Prism also marks every coefficient as a separable linear parameter, so
    ``prism.modeling.fitting.SeparableTRF`` can solve them exactly inside its
    inner linear step.

    Examples
    --------
    >>> import numpy as np
    >>> from prism.modeling.models import BSpline
    >>> knots = np.linspace(4500, 7000, 10)
    >>> bsp = BSpline(knots, degree=3, name='cont')
    >>> y = bsp(np.linspace(4500, 7000, 50))
    """
    n_inputs = 1
    n_outputs = 1
    linear = True

    def __new__(cls, knots, degree=3, name="BSpline", **kwargs):
        knot_unit = knots.unit if isinstance(knots, u.Quantity) else None
        knots = np.asarray(knots.to_value(knot_unit), dtype=float) if knot_unit is not None else np.asarray(knots)
        ncoeffs = len(knots) - degree - 1
        if ncoeffs <= 0:
            raise ValueError("Invalid knot vector/degree: no spline coefficients available.")

        params = {f'c{i}': Parameter(default=1.0) for i in range(ncoeffs)}

        def evaluate(self, x, *coeffs):
            output_unit = None
            coeff_values = []
            for coeff in coeffs:
                if isinstance(coeff, u.Quantity):
                    if output_unit is None:
                        output_unit = coeff.unit
                    coeff_values.append(coeff.to_value(output_unit))
                else:
                    coeff_values.append(coeff)
            x_values = x.to_value(self._knot_unit) if isinstance(x, u.Quantity) and self._knot_unit is not None else np.asarray(x)
            spline = _ScipyBSpline(self._knots, coeff_values, self._degree, extrapolate=True)
            result = spline(x_values)
            return result * output_unit if output_unit is not None else result

        def fit_deriv(self, x, *coeffs):
            if isinstance(x, u.Quantity):
                x = x.to_value(self._knot_unit) if self._knot_unit is not None else x.value
            design_matrix = _ScipyBSpline.design_matrix(
                np.asarray(x), self._knots, self._degree, extrapolate=True
            )
            dense_matrix = design_matrix.toarray()
            return [dense_matrix[:, i] for i in range(self._ncoeffs)]

        def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
            return {
                f'c{i}': outputs_unit[self.outputs[0]]
                for i in range(self._ncoeffs)
            }

        model_class = type(name, (Fittable1DModel,), {
            **params,
            'evaluate': evaluate,
            'fit_deriv': fit_deriv,
            '_parameter_units_for_data_units': _parameter_units_for_data_units,
            'n_inputs': 1,
            'n_outputs': 1,
            'linear': True,
            '_separable_linear_params': tuple(params),
            '_knots': knots,
            '_knot_unit': knot_unit,
            '_degree': degree,
            '_ncoeffs': ncoeffs,
        })

        return model_class(**kwargs)
