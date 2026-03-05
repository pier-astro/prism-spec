"""
Convolved Model Module

Provides a base class for models that apply a linear operator (matrix multiplication)
to an underlying source model while preserving analytic derivatives.

This is useful for:
- Instrumental response convolution (LSF/PSF)
- Velocity broadening
- Any linear transformation of spectral models

The key insight is that for h(x) = M @ f(x, θ) where M is a fixed matrix,
the Jacobian follows the chain rule: J_h = M @ J_f

This preserves analytic derivatives, avoiding the overhead of numeric jacobian
estimation which scales poorly with the number of free parameters.

Why This Custom Class Exists
----------------------------
Astropy's native pipe operator `|` does NOT propagate analytic derivatives:

    # In astropy/modeling/core.py, CompoundModel.fit_deriv:
    if op not in ["-", "+", "*", "/"]:
        return None  # Pipe | returns None!

This forces fitters to use numeric jacobian estimation, adding significant
overhead that scales with parameter count. ConvolvedModel solves this by
implementing the chain rule directly.

Classes
-------
- ConvolvedModel: Base class for matrix-convolved models with derivative propagation
- ResponseModel (in instrument.py): Specialized subclass for instrumental response

Accessing the Source Model
--------------------------
The unconvolved model is accessible via the `source_model` property:

    >>> rsp_model = lsf(compound_model)
    >>> original = rsp_model.source_model  # Returns the original CompoundModel

Package Modification Log
------------------------
The following files contain logic to handle ConvolvedModel instances:

1. fantasylab/models/convolved.py (THIS FILE)
   - Defines base ConvolvedModel class
   - Implements fit_deriv with chain rule: J_new = M @ J_source
   - Delegates all model properties to source via __getattr__

2. fantasylab/models/instrument.py
   - Defines ResponseModel(ConvolvedModel) for instrumental response
   - SpectralResponse.__call__() returns ResponseModel by default
   - ResponseModel has default name='rsp' for display

3. fantasylab/models/components.py
   - _get_source_model(): Unwraps ConvolvedModel → source for component extraction
   - _get_convolution_info(): Extracts convolution params for re-wrapping
   - _wrap_component(): Re-wraps extracted components with convolution
   - ModelComponents: Stores conv_info and wraps extracted components
   - Result: get_components(rsp(A+B)) returns [rsp(A), rsp(B)] not [A, B]

4. fantasylab/models/flux.py
   - extract_fluxes(): Unwraps ConvolvedModel before extracting line fluxes
   - flux_from_samples(): Unwraps ConvolvedModel before processing samples

5. fantasylab/display/model.py
   - _unwrap_convolved(): Returns (source, wrapper_name) tuple
   - _build_model_expression(): Shows "name(source_expr)" format
   - _get_components(): Unwraps before extracting component list

6. fantasylab/models/__init__.py
   - Exports ConvolvedModel and ResponseModel for public API

Display Format
--------------
When wrapped in ResponseModel (or ConvolvedModel), show() displays:

    Model: rsp(continuum (0) + blr_h (1) + blr_he (2) + nlr (3))

Instead of the old format:

    Model: continuum (0) + blr_h (1) + blr_he (2) + nlr (3) [convolved]
"""
import numpy as np
from scipy.interpolate import interp1d


class ConvolvedModel:
    """
    A model wrapper that applies a linear operator while preserving analytic derivatives.
    
    This class wraps a source model and linear operator (matrix), providing:
    1. Proper model behavior (via delegation to source)
    2. Analytic derivatives via chain rule: d(M @ f)/d(params) = M @ df/d(params)
    3. Grid flexibility through interpolation (with caching)
    
    This avoids the pipe operator's limitation of not propagating derivatives,
    which causes numeric jacobian estimation overhead.
    
    Parameters
    ----------
    source_model : Model
        The underlying model to transform
    operator_matrix : sparse matrix or array
        The linear operator matrix (n_out × n_in)
    wave : array-like, optional
        Wavelength grid for interpolation support
    name : str, optional
        Name for the convolved model (default: 'convolved')
        
    Attributes
    ----------
    source_model : Model
        The underlying source model (read-only access)
    operator_matrix : array
        The linear transformation matrix
    wave : array
        Wavelength grid used for interpolation
    name : str
        Name of this convolved model wrapper
        
    Notes
    -----
    The `fit_deriv` property implements the chain rule:
    
        For h(x) = M @ f(x, θ), where M is a fixed linear operator:
        ∂h/∂θ_i = M @ ∂f/∂θ_i
    
    This is mathematically exact and much faster than numeric differentiation
    for models with many parameters.
    
    Example
    -------
    >>> from astropy.modeling.models import Gaussian1D
    >>> import numpy as np
    >>> 
    >>> # Create a simple smoothing matrix
    >>> wave = np.arange(4500, 5500, 1.0)
    >>> matrix = np.eye(len(wave))  # identity for this example
    >>> 
    >>> # Wrap a model
    >>> source = Gaussian1D(amplitude=100, mean=5000, stddev=50)
    >>> convolved = ConvolvedModel(source, matrix, wave=wave, name='smoothed')
    >>> 
    >>> # Access source model
    >>> print(convolved.source_model)  # Original Gaussian1D
    >>> 
    >>> # Evaluate (applies matrix)
    >>> flux = convolved(wave)
    """
    
    def __init__(self, source_model, operator_matrix, wave=None, name=None):
        self._source = source_model
        self._operator_matrix = operator_matrix
        self._wave = np.asarray(wave) if wave is not None else None
        self._name = name or 'convolved'
        self._n_wave = operator_matrix.shape[0]
        
        # Cache for interpolation (keyed by parameter values)
        self._cache_params = None
        self._cache_result = None
    
    # --- Public Properties ---
    
    @property
    def source_model(self):
        """
        The underlying source model (read-only).
        
        Returns
        -------
        Model
            The unconvolved source model
        """
        return self._source
    
    @property
    def operator_matrix(self):
        """The linear operator matrix."""
        return self._operator_matrix
    
    @property
    def wave(self):
        """Wavelength grid for interpolation."""
        return self._wave
    
    @property 
    def name(self):
        """Name of this convolved model."""
        return self._name
    
    # --- Evaluation ---
    
    def _evaluate_on_grid(self, params_tuple):
        """
        Evaluate convolved model on the original wavelength grid.
        
        Results are cached for repeated evaluations with same parameters.
        """
        if self._cache_params == params_tuple and self._cache_result is not None:
            return self._cache_result
        
        # Evaluate source model on original grid
        flux = self._source(self._wave)
        result = np.asarray(self._operator_matrix.dot(flux)).ravel()
        
        # Cache result
        self._cache_params = params_tuple
        self._cache_result = result
        return result
    
    def __call__(self, x, *args, **kwargs):
        """
        Evaluate convolved model with grid flexibility.
        
        - If x matches the original grid length: direct matrix multiplication
        - If x differs and wave is provided: interpolate from cached result
        - Outside wavelength bounds: returns zero (operator is undefined)
        """
        x = np.atleast_1d(x)
        
        # Fast path: x matches original grid exactly
        if len(x) == self._n_wave:
            flux = self._source(x, *args, **kwargs)
            return np.asarray(self._operator_matrix.dot(flux)).ravel()
        
        # Need interpolation
        if self._wave is None:
            raise ValueError(
                f"Input has {len(x)} points but operator matrix expects {self._n_wave}. "
                "Provide 'wave' parameter to enable grid interpolation."
            )
        
        # Get cached result on original grid
        params_tuple = tuple(self._source.parameters)
        full_result = self._evaluate_on_grid(params_tuple)
        
        # Interpolate to requested grid
        interpolator = interp1d(
            self._wave, full_result, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0  # Zero outside bounds
        )
        return interpolator(x)
    
    def evaluate(self, x, *params):
        """Evaluate with explicit parameters."""
        flux = self._source.evaluate(x, *params)
        return np.asarray(self._operator_matrix.dot(flux)).ravel()
    
    # --- Derivative Support ---
    
    @property
    def fit_deriv(self):
        """
        Return derivative function that applies chain rule through operator matrix.
        
        For h(x) = M @ f(x, params), the derivative is:
        dh/d(param_i) = M @ df/d(param_i)
        """
        source_deriv = self._source.fit_deriv
        if source_deriv is None:
            return None
        
        operator_matrix = self._operator_matrix
        
        def _convolved_deriv(x, *params):
            # Get derivatives from source model
            derivs = source_deriv(x, *params)
            # Apply operator matrix to each derivative (chain rule)
            return [np.asarray(operator_matrix.dot(d)).ravel() for d in derivs]
        
        return _convolved_deriv
    
    # --- Delegate model properties to source ---
    
    @property
    def param_names(self):
        return self._source.param_names
    
    @property
    def parameters(self):
        return self._source.parameters
    
    @parameters.setter
    def parameters(self, value):
        self._source.parameters = value
    
    @property
    def n_inputs(self):
        return self._source.n_inputs
    
    @property
    def n_outputs(self):
        return self._source.n_outputs
    
    @property
    def bounds(self):
        return self._source.bounds
    
    @property
    def fixed(self):
        return self._source.fixed
    
    @property
    def tied(self):
        return self._source.tied
    
    @property
    def col_fit_deriv(self):
        """Return same as source model for consistency."""
        return self._source.col_fit_deriv
    
    @property
    def _has_units(self):
        return self._source._has_units
    
    @property
    def _param_metrics(self):
        return self._source._param_metrics
    
    @property
    def _model_set_axis(self):
        return self._source._model_set_axis
    
    @property
    def standard_broadcasting(self):
        return self._source.standard_broadcasting
    
    @property
    def eqcons(self):
        return getattr(self._source, 'eqcons', None)
    
    @property
    def ineqcons(self):
        return getattr(self._source, 'ineqcons', None)
    
    def __getattr__(self, name):
        """Delegate attribute access to source model."""
        # Don't delegate our own internal attributes
        if name in ('_source', '_operator_matrix', '_wave', '_name', 
                    '_n_wave', '_cache_params', '_cache_result'):
            raise AttributeError(name)
        return getattr(self._source, name)
    
    def __len__(self):
        """Return number of model sets (always 1 for this wrapper)."""
        return len(self._source) if hasattr(self._source, '__len__') else 1
    
    def __getitem__(self, key):
        """
        Delegate subscript access to source model.
        
        This is essential for tied parameters that access submodels by name.
        """
        return self._source[key]
    
    @property
    def submodel_names(self):
        """Return submodel names from source model."""
        return self._source.submodel_names
    
    @property
    def n_submodels(self):
        """Return number of submodels from source model."""
        return self._source.n_submodels
    
    def __setattr__(self, name, value):
        """Set attributes, delegating parameter access to source."""
        if name.startswith('_') or name in ('fit_deriv', 'name'):
            super().__setattr__(name, value)
        elif hasattr(self, '_source') and hasattr(self._source, name):
            setattr(self._source, name, value)
        else:
            super().__setattr__(name, value)
    
    def copy(self):
        """Return a copy of this model."""
        return ConvolvedModel(
            self._source.copy(),
            self._operator_matrix,
            wave=self._wave.copy() if self._wave is not None else None,
            name=self._name
        )
    
    def __repr__(self):
        return f"<ConvolvedModel[{self._name}]({self._source!r})>"


# Public exports
__all__ = ['ConvolvedModel']
