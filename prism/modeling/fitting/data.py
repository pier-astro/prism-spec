"""Observed datasets, model predictions and fitting sample selection."""
import numpy as np


class Dataset:
    """Associate one observed series with its errors, selection and response.

    Parameters
    ----------
    x : array-like or prism.data.Spectrum
        One-dimensional coordinates, or a Spectrum whose current arrays are used.
        Spectrum cutouts already contain only the selected samples.
    y : array-like, optional
        Observed values matching x. Omit only when x is a Spectrum.
    yerr : array-like, optional
        Symmetric standard deviations, matching y. Inferred from Spectrum when
        omitted. Finite, positive errors are required for included fitting data.
    weights : array-like, optional
        Residual multipliers, matching y; take precedence over yerr in fitting.
        Zero weights exclude samples. These are not inverse variances.
    mask : array-like of bool, optional
        True includes a sample. NumPy masked-array exclusions are also respected.
        A mask supplied with Spectrum refers to its current working arrays.
    response : LinearOperatorModel or Model, optional
        Operator applied after the source, using the existing pipe algebra.
        A native two-input profile transform is also accepted. Omit when the
        supplied model already includes the desired instrumental action.

    Notes
    -----
    Arrays and Spectrum metadata are referenced without an intentional copy.
    Replacing a Spectrum working array later does not update an existing Dataset;
    construct it after selecting or rebinning the spectrum. Numeric arrays are
    not converted using xunit or yunit. Quantity arrays retain their units, and
    the fitter enforces its normal unit boundary.

    Construction checks shapes; sample validity is checked when fitting, so
    filter_non_finite remains a fit-time choice. Responses are prepared once per
    fit by the existing evaluator. Masks select residuals, not the source grid.
    This initial Dataset contract describes 1D data; existing spatial adapters
    remain available independently.
    """

    def __init__(self, x, y=None, *, yerr=None, weights=None, mask=None, response=None):
        from ...data import Spectrum
        from ..composition import Model
        from ..operators.matop import LinearOperatorModel

        spectrum = x if isinstance(x, Spectrum) else None
        if spectrum is not None:
            if y is not None:
                raise TypeError('Supply either a Spectrum or x and y arrays.')
            x, y = spectrum.x, spectrum.y
            if yerr is None:
                yerr = spectrum.yerr
        elif y is None:
            raise TypeError('Supply y alongside the coordinate array x.')

        transform = (isinstance(response, Model) and response.op is None
                     and response.component.n_inputs == 2)
        if response is not None and not isinstance(response, LinearOperatorModel) and not transform:
            raise TypeError('response must be a linear operator or a native two-input profile transform.')

        self.x, self.y = np.asanyarray(x), np.asanyarray(y)
        self.yerr = None if yerr is None else np.asanyarray(yerr)
        self.weights = None if weights is None else np.asanyarray(weights)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        _validate_fit_shapes(self.x, self.y, self.yerr, self.weights, self.mask)
        self.response = response
        self.meta = {} if spectrum is None else spectrum.meta
        self.xunit = getattr(self.x, 'unit', None) if spectrum is None else spectrum.xunit
        self.yunit = getattr(self.y, 'unit', None) if spectrum is None else spectrum.yunit

    def _observed_model(self, model):
        """Compose the declared response, retaining parameter identity."""
        if self.response is None:
            return model
        from ..composition import Model
        observed = Model.from_astropy(model) | self.response
        parent = getattr(model, '_evaluation_parent', None)
        if parent is not None:
            scope = {id(getattr(parent, name)) for name in parent.param_names}
            if all(id(getattr(observed, name)) in scope for name in observed.param_names):
                observed._evaluation_parent = parent
        return observed

    def predict(self, model, x=None):
        """Evaluate a source through this response on x, or the observation grid.

        x is a one-dimensional coordinate array in the model units. Spectral
        responses fold on their native grids, then interpolate point values onto
        x, with zero outside their output domain; strict responses reject a
        different grid. Without a response, evaluate the model directly.

        Returns the full prediction, including masked samples. Non-finite
        coordinates receive NaN; finite coordinates are evaluated together so
        profile-wide operations see their complete grid. No fitting selection
        or error weights are applied. A response already in model is retained:
        supplying another response explicitly composes both operators.
        """
        x = np.ma.getdata(self.x if x is None else np.asanyarray(x))
        if x.ndim != 1:
            raise ValueError('Prediction coordinates must be a one-dimensional array.')
        finite = np.isfinite(x)
        if not np.any(finite):
            raise ValueError('No finite coordinates remain for prediction.')
        observed = self._observed_model(model)
        values = observed(x if np.all(finite) else x[finite])
        if np.shape(values) != (np.count_nonzero(finite),):
            raise ValueError('A model prediction must match the observation coordinates.')
        if np.all(finite):
            return values
        prediction = np.full(x.shape, np.nan)
        if hasattr(values, 'unit'):
            prediction = prediction * values.unit
        prediction[finite] = values
        return prediction


def _resolve_fit_inputs(model, x, y, z, yerr, weights, mask):
    """Resolve Dataset or array input without competing data definitions."""
    if not isinstance(x, Dataset):
        return model, x, y, yerr, weights, mask
    if y is not None or z is not None:
        raise TypeError('A Dataset replaces coordinate and observed-value arguments.')
    if yerr is not None or weights is not None or mask is not None:
        raise TypeError('Set yerr, weights and mask on the Dataset when fitting a Dataset.')
    return x._observed_model(model), x.x, x.y, x.yerr, x.weights, x.mask


def _validate_fit_shapes(x, y, yerr, weights, mask=None):
    """Validate a single observed series without selecting or copying samples."""
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError('x and y must be one-dimensional arrays with the same shape.')
    for name, array in [('yerr', yerr), ('weights', weights), ('mask', mask)]:
        if array is not None and array.shape != y.shape:
            if name == 'yerr' and array.ndim != 1:
                raise ValueError('Asymmetric yerr is not supported by direct fitters. Use symmetric yerr or bootstrap(..., noise_dist="uniform").')
            raise ValueError(f'{name} must have the same shape as y.')


def _prepare_fit_data(x, y, yerr=None, weights=None, mask=None, *, filter_non_finite=False):
    """Return full-grid arrays and a validity mask (True means included).

    Masked-array masks and zero residual weights exclude samples. Invalid values
    outside the selection are harmless; non-finite included values require
    explicit filtering. Included errors must be positive and weights nonnegative.
    Quantities retain their units; numerical backends enforce their own unit
    boundary before calling this routine. No automatic unit conversion occurs.
    """
    arrays = []
    inferred = None
    for value in (x, y, yerr, weights):
        if value is None:
            arrays.append(None)
            continue
        if np.ma.isMaskedArray(value):
            excluded = np.ma.getmaskarray(value)
            inferred = excluded if inferred is None else inferred | excluded
            value = np.ma.getdata(value)
        arrays.append(np.asanyarray(value))
    x, y, yerr, weights = arrays
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    _validate_fit_shapes(x, y, yerr, weights, mask)
    if inferred is not None and inferred.shape != y.shape:
        raise ValueError('Masked inputs must have the same shape as y.')
    if mask is None:
        valid = np.ones(y.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool).copy()
    if inferred is not None:
        valid &= ~inferred
    if weights is not None:
        valid &= weights != 0
    finite = np.isfinite(x) & np.isfinite(y)
    for array in (yerr, weights):
        if array is not None:
            finite &= np.isfinite(array)
    if filter_non_finite:
        valid &= finite
    elif np.any(valid & ~finite):
        raise ValueError('Non-finite values detected in included fitting data. Set filter_non_finite=True.')
    if yerr is not None and np.any(yerr[valid] <= 0):
        raise ValueError('Included yerr values must be strictly positive.')
    if weights is not None and np.any(weights[valid] < 0):
        raise ValueError('Included weights must be nonnegative.')
    if not np.any(valid):
        raise ValueError('No valid spectral samples remain after masking.')
    return x, y, yerr, weights, valid


def _prepare_spatial_data(x, y, z, yerr=None, weights=None, mask=None, *, filter_non_finite=False):
    """Prepare paired 2D coordinates using the same residual/error mask contract.

    Coordinates, image values and optional errors/weights have matching shapes.
    Flattening is confined to fitting; the returned model retains two-coordinate
    evaluation. Quantities keep their units through selection.
    """
    arrays = []
    inferred = None
    for value in (x, y, z, yerr, weights):
        if value is None:
            arrays.append(None)
            continue
        if np.ma.isMaskedArray(value):
            excluded = np.ma.getmaskarray(value)
            inferred = excluded if inferred is None else inferred | excluded
            value = np.ma.getdata(value)
        arrays.append(np.asanyarray(value))
    x, y, z, yerr, weights = arrays
    if x.shape != z.shape or y.shape != z.shape:
        raise ValueError('Spatial x, y and z must have the same shape; supply explicit coordinate grids.')
    for name, array in [('yerr', yerr), ('weights', weights), ('mask', mask)]:
        if array is not None and np.shape(array) != z.shape:
            raise ValueError(f'Spatial {name} must have the same shape as z.')
    valid = np.ones(z.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if inferred is not None:
        valid &= ~inferred
    x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    yerr = None if yerr is None else yerr.reshape(-1)
    weights = None if weights is None else weights.reshape(-1)
    # The shared validator sees invalidity of either coordinate, while the
    # original coordinate arrays remain unchanged for evaluation and units.
    coordinates = x.astype(float, copy=True)
    coordinates[~np.isfinite(y)] = np.nan
    _, _, yerr, weights, included = _prepare_fit_data(
        coordinates, z, yerr, weights, valid.reshape(-1),
        filter_non_finite=filter_non_finite)
    return (x[included], y[included], z[included],
            None if yerr is None else yerr[included],
            None if weights is None else weights[included])
