"""
MultiFit functionality for prism.modeling.fitting

This module isolates the batched spectral fitting functionality, allowing
fitters to efficiently optimize a model across multiple spectra (1D, 2D, 3D data)
with a clean, independent API.

Both `.multifit()` (explicit) and `fitter(model, x, y)` with multi-dimensional
`y` (transparent) are supported. All standard single-spectrum options are
preserved: `yerr`, `weights`, `statistic`, `calc_uncertainties`/`stdevs`,
`progress`, `nproc`, `initpars`, `bounds`, `fixed`, `tied`.
"""

import time
import warnings
import numpy as np
import abc
import os
import copy
import multiprocess as mp
from tqdm.auto import tqdm
from astropy.modeling.fitting import model_to_fit_params
from ...data.core import parse_binmap

_MULTIFIT_STAT_KEYS = (
    'cost', 'optimality', 'ndata', 'nparam', 'nfree', 'dof', 'status'
)

__all__ = [
    'MultiFitParameter', 'SpectrumFitResult', 'MultiFitResult', 'MultiFitMixin'
]


_MULTIFIT_WORKER_STATE = {}


def _multifit_worker_initializer(fitter=None, template_model=None, x=None,
                                 statistic=None, kwargs=None):
    """Configure worker runtime and preload shared multifit state."""
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    if fitter is not None:
        global _MULTIFIT_WORKER_STATE
        _MULTIFIT_WORKER_STATE = {
            'fitter': fitter,
            'template_model': template_model,
            'x': x,
            'statistic': statistic,
            'kwargs': {} if kwargs is None else kwargs,
        }


def _multifit_worker_task(args):
    """Run one multifit target using preloaded worker state."""
    idx, y_1d, yerr_1d, weights_1d, initpars_1d, bounds_1d = args
    state = _MULTIFIT_WORKER_STATE
    fitter = state['fitter']
    model = state['template_model']
    x = state['x']
    statistic = state['statistic']
    kwargs = state['kwargs']

    try:
        local_model = model.copy()
        fitter._apply_scalar_multifit_overrides(local_model, initpars=initpars_1d, bounds=bounds_1d)
        fitted = fitter(
            model=local_model, x=x, y=y_1d, yerr=yerr_1d,
            statistic=statistic, weights=weights_1d,
            inplace=True, **kwargs
        )
        return (idx, fitted.parameters, fitter.stdevs,
                fitter._multifit_entry_from_fit_info(), fitter.covariance)
    except Exception as e:
        if getattr(fitter, 'verbose', False):
            print(f"Worker {idx} failed: {e}")
        return (
            idx,
            np.full(len(model.parameters), np.nan),
            None,
            {
                'success': False, 'nfev': 0, 'message': str(e),
                **{k: np.nan for k in _MULTIFIT_STAT_KEYS},
            },
            None,
        )


# =============================================================================
# MultiFitParameter
# =============================================================================

class MultiFitParameter:
    """
    Array-like view of a single parameter across all fitted spectra.

    Exposes `.value`, `.std`, `.bounds`, `.min`, `.max`, `.fixed`, `.tied`,
    and `.unit` as shaped arrays matching the spatial shape of the result.
    Supports numpy indexing and iteration transparently.
    """

    __slots__ = ('_parent', 'name')

    def __init__(self, parent, name):
        self._parent = parent
        self.name = name

    @property
    def value(self):
        return self._parent._param_array(self.name)

    @property
    def std(self):
        return self._parent._std_array(self.name)

    @property
    def bounds(self):
        return self._parent.get_bounds(self.name)

    @property
    def min(self):
        return self._parent.get_min(self.name)

    @property
    def max(self):
        return self._parent.get_max(self.name)

    @property
    def fixed(self):
        return self._parent.get_fixed(self.name)

    @property
    def tied(self):
        return self._parent.get_tied(self.name)

    @property
    def unit(self):
        return self._parent.get_unit(self.name)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, index):
        return self.value[index]

    def __iter__(self):
        return iter(self.value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"MultiFitParameter(name={self.name!r}, shape={self.shape}, dtype={self.value.dtype})"


class _MultiFitComponentView:
    """
    Callable view of one component across all fitted spectra.

    Instances are returned by ``MultiFitResult.comps[...]`` and
    ``MultiFitResult.addcomps[...]``. Calling the view evaluates that
    component over all fitted spaxels and returns a model cube.
    """

    __slots__ = ('_parent', '_key', '_additive', 'name')

    def __init__(self, parent, key, additive, name):
        self._parent = parent
        self._key = key
        self._additive = additive
        self.name = name

    def evaluate(self, x=None, spectral_axis=None):
        return self._parent._result._evaluate_component(
            self._key,
            x=x,
            spectral_axis=spectral_axis,
            additive=self._additive,
        )

    def __call__(self, x=None, spectral_axis=None):
        return self.evaluate(x=x, spectral_axis=spectral_axis)

    def get_model(self, index):
        return self._parent._result.get_component_model(
            index,
            self._key,
            additive=self._additive,
        )

    def __repr__(self):
        mode = 'additive' if self._additive else 'components'
        return f"MultiFitComponentView(name={self.name!r}, mode={mode})"


class _MultiFitComponentsAccessor:
    """
    Batch component accessor for MultiFitResult.

    Supports key access by index or name. Returned values are callable views
    that evaluate one selected component over all fitted spectra.
    """

    __slots__ = ('_result', '_additive', '_template_components', '_cache')

    def __init__(self, result, additive=False):
        self._result = result
        self._additive = additive
        self._template_components = result._get_template_components(additive=additive)
        self._cache = {}

    @property
    def additive(self):
        return self._additive

    @property
    def names(self):
        return self._template_components.names

    @property
    def indices(self):
        return self._template_components.indices

    def to_list(self):
        return [self[name] for name in self.names]

    def get_model(self, key):
        return self._template_components[key]

    def __getitem__(self, key):
        if key not in self._cache:
            model = self._template_components[key]
            component_name = getattr(model, 'name', None)
            if not component_name:
                component_name = str(key)
            self._cache[key] = _MultiFitComponentView(
                parent=self,
                key=key,
                additive=self._additive,
                name=component_name,
            )
        return self._cache[key]

    def __iter__(self):
        for name in self.names:
            yield self[name]

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        mode = 'additive-only' if self._additive else 'all'
        return f"MultiFitComponents({mode}, n={len(self.names)}, names={self.names})"


# =============================================================================
# MultiFitResult
# =============================================================================

class MultiFitResult:
    """
    Stores the result of a batched fit over multiple spectra.

    Provides shaped numpy arrays for every fitted parameter and diagnostic
    statistic, as well as per-spaxel model reconstruction and evaluation.

    Parameters
    ----------
    shape : tuple
        Spatial shape of the dataset (e.g. ``(20, 30)`` for a 2-D cube slice).
    param_names : sequence of str
        Parameter names in model order.
    template_model : astropy Model, optional
        Copy of the initial model used for reconstruction.
    x : array_like, optional
        Wavelength / spectral grid; stored for ``evaluate()``.
    has_cov : bool
        Whether to allocate storage for per-spaxel covariance matrices.
    spectral_axis : int
        Original spectral axis index before it was moved to the last position.
    fit_indices : array_like, optional
        Indices of free parameters (used to distribute stdevs correctly).
    parameter_metadata : dict, optional
        Per-parameter metadata propagated from the fit configuration.

    Attributes
    ----------
    shape, param_names, n_spaxels
    fit_info : dict
        Aggregate batch statistics set after fitting completes.

    Additional component accessors
    ------------------------------
    comps : MultiFitComponents
        Batch component accessor based on ``get_components(..., additive=False)``.
        Access by index or name and evaluate a selected component over all
        spaxels, e.g. ``result.comps['nlr'](x)``.
    addcomps : MultiFitComponents
        Additive-only component accessor (multiplicative terms expanded).
    """

    def __init__(self, shape, param_names, template_model=None, x=None,
                 has_cov=False, spectral_axis=-1, fit_indices=None,
                 parameter_metadata=None):
        self.shape = tuple(shape)
        self.param_names = tuple(param_names)
        self._template_model = template_model.copy() if template_model is not None else None
        self._x = None if x is None else np.asarray(x)
        self.spectral_axis = spectral_axis
        self._fit_indices = None if fit_indices is None else np.asarray(fit_indices)
        self._parameter_metadata = parameter_metadata or {}
        self.fit_info = {}
        self._comps = None
        self._addcomps = None
        self._template_components = {}

        self.n_spaxels = int(np.prod(shape))
        self._params = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._errs   = {name: np.full(self.n_spaxels, np.nan) for name in param_names}
        self._stats  = {name: np.full(self.n_spaxels, np.nan) for name in _MULTIFIT_STAT_KEYS}
        self._success  = np.zeros(self.n_spaxels, dtype=bool)
        self._nfev     = np.zeros(self.n_spaxels, dtype=float)
        self._messages = np.full(self.n_spaxels, '', dtype=object)

        self.has_cov = has_cov
        if has_cov:
            self._cov = np.full(
                (self.n_spaxels, len(param_names), len(param_names)), np.nan
            )

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _normalize_spaxel_index(self, index):
        """Convert a flat int or spatial tuple index to a flat integer."""
        if isinstance(index, tuple):
            if len(index) != len(self.shape):
                raise IndexError(
                    f"Expected {len(self.shape)} spatial indices, got {len(index)}."
                )
            return int(np.ravel_multi_index(index, self.shape))
        flat = int(index)
        if flat < 0:
            flat += self.n_spaxels
        if not (0 <= flat < self.n_spaxels):
            raise IndexError(f"Spaxel index {index} is out of bounds for shape {self.shape}.")
        return flat

    def _parameter_vector(self, flat_index):
        return np.array([self._params[n][flat_index] for n in self.param_names], dtype=float)

    def _error_vector(self, flat_index):
        return np.array([self._errs[n][flat_index] for n in self.param_names], dtype=float)

    def _param_array(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._params[name].reshape(self.shape)

    def _std_array(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._errs[name].reshape(self.shape)

    def _parameter_meta(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        return self._parameter_metadata.get(name, {})

    def _template_parameter(self, name):
        if self._template_model is None:
            return None
        return getattr(self._template_model, name)

    def _metadata_value(self, name, key, default=None):
        meta = self._parameter_meta(name)
        if key in meta:
            return meta[key]
        param = self._template_parameter(name)
        return default if param is None else getattr(param, key, default)

    def _full_bounds_array(self, bounds):
        expanded = np.empty(self.shape, dtype=object)
        bound_pair = tuple(bounds)
        for idx in range(self.n_spaxels):
            expanded.flat[idx] = bound_pair
        return expanded

    def _bounds_component_array(self, name, component_index):
        bounds = self.get_bounds(name)
        values = [pair[component_index] for pair in bounds.flat]
        if any(v is None for v in values):
            component = np.empty(self.shape, dtype=object)
            component.flat[:] = values
            return component
        return np.asarray(values, dtype=float).reshape(self.shape)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_param(self, name) -> 'MultiFitParameter':
        """Return an array-like parameter view across all spaxels."""
        return MultiFitParameter(self, name)

    # Alias
    get_parameter = get_param

    def get_std(self, name) -> np.ndarray:
        """Return the standard-deviation array for one parameter."""
        return self._std_array(name)

    def get_bounds(self, name) -> np.ndarray:
        """Return a per-spaxel object array of (lower, upper) bound tuples."""
        bounds = self._metadata_value(name, 'bounds', default=None)
        if isinstance(bounds, np.ndarray):
            return bounds.reshape(self.shape)
        if bounds is not None:
            return self._full_bounds_array(bounds)
        param = self._template_parameter(name)
        return self._full_bounds_array((None, None) if param is None else param.bounds)

    def get_min(self, name) -> np.ndarray:
        """Return the per-spaxel lower bound array for one parameter."""
        return self._bounds_component_array(name, 0)

    def get_max(self, name) -> np.ndarray:
        """Return the per-spaxel upper bound array for one parameter."""
        return self._bounds_component_array(name, 1)

    def get_fixed(self, name) -> bool:
        """Return True if the parameter is fixed (shared across all spaxels)."""
        return bool(self._metadata_value(name, 'fixed', default=False))

    def get_tied(self, name):
        """Return the tied callable, or False if the parameter is free."""
        return self._metadata_value(name, 'tied', default=False)

    def get_unit(self, name):
        """Return the parameter unit, or None."""
        return self._metadata_value(name, 'unit', default=None)

    def get_stat(self, name) -> np.ndarray:
        """Return a shaped array for one fit statistic."""
        if name not in _MULTIFIT_STAT_KEYS:
            raise KeyError(f"Unknown statistic '{name}'. Available: {list(_MULTIFIT_STAT_KEYS)}")
        return self._stats[name].reshape(self.shape)

    def get_model(self, index, copy_model=True):
        """
        Return the fitted astropy model for one spaxel.

        Parameters
        ----------
        index : int or tuple
            Flat or spatial index of the spaxel.
        copy_model : bool
            If True (default), return a fresh copy of the template.

        Returns
        -------
        model : astropy Model
            Fitted model with parameter values, bounds, and stdevs attached.
        """
        if self._template_model is None:
            raise ValueError("Model reconstruction requires a template model.")
        flat_index = self._normalize_spaxel_index(index)
        model = self._template_model.copy() if copy_model else self._template_model
        self._apply_parameter_metadata(model, flat_index)
        model.parameters = self._parameter_vector(flat_index)
        self._attach_model_stds(model, flat_index, self._error_vector(flat_index))
        return model

    def _apply_parameter_metadata(self, model, flat_index):
        for name in self.param_names:
            param = getattr(model, name)
            param.bounds = tuple(self.get_bounds(name).flat[flat_index])
            param.fixed  = self.get_fixed(name)
            param.tied   = self.get_tied(name)
        return model

    def _attach_model_stds(self, model, flat_index, stds):
        if stds is None:
            return model
        for name, std in zip(self.param_names, stds):
            getattr(model, name).std = std
        if self.has_cov:
            model._param_cov = self._cov[flat_index]
        if self._fit_indices is not None:
            model._fit_indices = self._fit_indices
        return model

    def get_fit_info(self, index) -> dict:
        """Return fit diagnostics for a single spaxel."""
        flat_index = self._normalize_spaxel_index(index)
        spatial_index = np.unravel_index(flat_index, self.shape)
        info = {
            'index':   spatial_index,
            'success': bool(self._success[flat_index]),
            'nfev':    self._nfev[flat_index],
            'message': self._messages[flat_index],
        }
        for name in _MULTIFIT_STAT_KEYS:
            info[name] = self._stats[name][flat_index]
        cov = self.covariance
        info['param_cov'] = None if cov is None else cov[spatial_index]
        return info

    def _get_template_components(self, additive=False):
        if self._template_model is None:
            raise ValueError("Component access requires a template model.")
        key = bool(additive)
        cached = self._template_components.get(key)
        if cached is None:
            from ..models.components import get_components
            cached = get_components(self._template_model, additive=additive)
            self._template_components[key] = cached
        return cached

    def get_component_model(self, index, component, additive=False):
        """
        Return one fitted component model for a single spaxel.

        Parameters
        ----------
        index : int or tuple
            Flat or spatial index of the spaxel.
        component : int or str
            Component index or name from ``.comps`` / ``.addcomps``.
        additive : bool
            If True, resolve from additive components.
        """
        from ..models.components import get_components
        model = self.get_model(index)
        return get_components(model, additive=additive)[component]

    def _evaluate_component(self, component, x=None, spectral_axis=None, additive=False):
        """Internal component-evaluation implementation shared by helpers."""
        if self._template_model is None:
            raise ValueError("Component evaluation requires a template model.")

        eval_x = self._x if x is None else np.asarray(x)
        if eval_x is None:
            raise ValueError("Provide x= or construct with x= to enable component evaluation().")

        out = np.empty(self.shape + (len(eval_x),), dtype=float)
        for flat_index in range(self.n_spaxels):
            spatial_index = np.unravel_index(flat_index, self.shape)
            comp_model = self.get_component_model(flat_index, component, additive=additive)
            out[spatial_index] = comp_model(eval_x)

        target_axis = self.spectral_axis if spectral_axis is None else spectral_axis
        if target_axis < 0:
            target_axis += out.ndim
        if target_axis != out.ndim - 1:
            out = np.moveaxis(out, -1, target_axis)
        return out

    def evaluate_component(self, component, x=None, spectral_axis=None):
        """
        Evaluate one (non-additive) component across all spaxels.

        Parameters
        ----------
        component : int or str
            Component index or name from ``result.comps``.
        x : array_like, optional
            Spectral grid. Uses the stored grid when omitted.
        spectral_axis : int, optional
            Desired output spectral-axis position (same semantics as ``evaluate()``).
        """
        return self._evaluate_component(component, x=x, spectral_axis=spectral_axis,
                                        additive=False)

    def evaluate_addcomponent(self, component, x=None, spectral_axis=None):
        """
        Evaluate one additive component across all spaxels.

        Parameters
        ----------
        component : int or str
            Component index or name from ``result.addcomps``.
        x : array_like, optional
            Spectral grid. Uses the stored grid when omitted.
        spectral_axis : int, optional
            Desired output spectral-axis position (same semantics as ``evaluate()``).
        """
        return self._evaluate_component(component, x=x, spectral_axis=spectral_axis,
                                        additive=True)

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x=None, spectral_axis=None) -> np.ndarray:
        """
        Evaluate all fitted models on ``x`` and return a shaped cube.

        Parameters
        ----------
        x : array_like, optional
            Spectral grid. Uses the stored grid when omitted.
        spectral_axis : int, optional
            Desired position of the spectral axis in the output.
            Defaults to the original spectral axis of the input data.

        Returns
        -------
        cube : ndarray, shape ``spatial_shape + (len(x),)`` (before axis move)
        """
        if self._template_model is None:
            raise ValueError("Model evaluation requires a template model.")
        eval_x = self._x if x is None else np.asarray(x)
        if eval_x is None:
            raise ValueError("Provide x= or construct with x= to enable evaluate().")

        out = np.empty(self.shape + (len(eval_x),), dtype=float)
        for flat_index in range(self.n_spaxels):
            out[np.unravel_index(flat_index, self.shape)] = self.get_model(flat_index)(eval_x)

        target_axis = self.spectral_axis if spectral_axis is None else spectral_axis
        if target_axis < 0:
            target_axis += out.ndim
        if target_axis != out.ndim - 1:
            out = np.moveaxis(out, -1, target_axis)
        return out

    def __call__(self, x=None, spectral_axis=None):
        """Alias for ``evaluate()``."""
        return self.evaluate(x=x, spectral_axis=spectral_axis)

    # ------------------------------------------------------------------
    # Update (called per-spaxel during fitting)
    # ------------------------------------------------------------------

    def update(self, idx, model_params, stdevs, fit_info, cov=None):
        """Store the result of one 1-D fit at flattened spaxel index ``idx``."""
        for i, name in enumerate(self.param_names):
            self._params[name][idx] = model_params[i]

        if stdevs is not None:
            if self._fit_indices is None:
                for i, name in enumerate(self.param_names[:len(stdevs)]):
                    self._errs[name][idx] = stdevs[i]
            else:
                for free_idx, sigma in zip(self._fit_indices, stdevs):
                    self._errs[self.param_names[int(free_idx)]][idx] = sigma

        self._success[idx]  = bool(fit_info.get('success', False))
        nfev = fit_info.get('nfev', np.nan)
        self._nfev[idx]     = np.nan if nfev is None else nfev
        self._messages[idx] = fit_info.get('message', '')

        for name in _MULTIFIT_STAT_KEYS:
            v = fit_info.get(name, np.nan)
            self._stats[name][idx] = np.nan if v is None else v

        if self.has_cov and cov is not None:
            if self._fit_indices is None or cov.shape == self._cov[idx].shape:
                self._cov[idx] = cov
            else:
                expanded = np.full_like(self._cov[idx], np.nan)
                fi = self._fit_indices.astype(int)
                expanded[np.ix_(fi, fi)] = cov
                self._cov[idx] = expanded

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        """Return a SpectrumFitResult view for the spaxel at ``index``."""
        return SpectrumFitResult(self, index)

    def __getattr__(self, name):
        """Allow ``result.amplitude_0``, ``result.cost``, etc."""
        try:
            param_names = object.__getattribute__(self, 'param_names')
        except Exception:
            param_names = ()
        if name in param_names:
            return self.get_param(name)
        if name in _MULTIFIT_STAT_KEYS:
            return self.get_stat(name)
        if name in {'message', 'messages'}:
            return self.messages
        raise AttributeError(f"'MultiFitResult' has no attribute '{name}'")

    # ------------------------------------------------------------------
    # Aggregate array properties
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> np.ndarray:
        """Shaped array ``(*spatial, n_params)`` of fitted parameter values."""
        return np.stack([self._param_array(n) for n in self.param_names], axis=-1)

    @property
    def stds(self) -> np.ndarray:
        """Shaped array ``(*spatial, n_params)`` of parameter standard deviations."""
        return np.stack([self._std_array(n) for n in self.param_names], axis=-1)

    @property
    def errors(self):
        return self.stds

    @property
    def stdevs(self):
        return self.stds

    @property
    def messages(self) -> np.ndarray:
        return self._messages.reshape(self.shape)

    @property
    def statistics(self) -> dict:
        return {n: self.get_stat(n) for n in _MULTIFIT_STAT_KEYS}

    @property
    def comps(self):
        """
        Access fitted model components (all leaf components).

        Examples
        --------
        >>> comp = result.comps['line']
        >>> cube = comp(x)                  # full cube for one component
        >>> model_ij = comp.get_model((j, i))
        """
        if self._comps is None:
            self._comps = _MultiFitComponentsAccessor(self, additive=False)
        return self._comps

    @property
    def addcomps(self):
        """
        Access additive-only fitted components.

        Multiplicative branches are expanded before evaluation.
        """
        if self._addcomps is None:
            self._addcomps = _MultiFitComponentsAccessor(self, additive=True)
        return self._addcomps

    @property
    def success(self) -> np.ndarray:
        return self._success.reshape(self.shape)

    @property
    def nfev(self) -> np.ndarray:
        return self._nfev.reshape(self.shape)

    @property
    def covariance(self):
        """Per-spaxel covariance cubes ``(*spatial, n_params, n_params)``, or None."""
        if not self.has_cov:
            return None
        return self._cov.reshape(self.shape + (len(self.param_names), len(self.param_names)))

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_fits(self, filename, wcs=None, overwrite=False,
                parfilter=None, std=True, model=True, **custom_exts):
        """
        Save the fit result to a multi-extension FITS file.

        Parameters
        ----------
        filename : str
            Output file path.
        wcs : astropy.wcs.WCS, optional
            Spatial WCS written into the PRIMARY header. Pass the cube's
            ``.wcs`` object for spatially-aware parameter maps.
        overwrite : bool
            Overwrite ``filename`` if it already exists.
        parfilter : list of str, optional
            Save only the listed parameters.  Excluding parameters raises a
            warning because the MODEL_DEF will then be incomplete for restart.
        std : bool
            Also save standard-deviation maps as ``<PARNAME>_ERR`` extensions.
        model : bool
            Serialize the template model as a ``MODEL_DEF`` YAML extension
            (required for ``load_model()``).
        **custom_exts
            Any additional user arrays to store as extra extensions, e.g.
            ``CHISQ=chi2_map, DOF=dof_map``.
        """
        from ..io import multifit_to_fits
        multifit_to_fits(
            self, filename, wcs=wcs, overwrite=overwrite,
            parfilter=parfilter, std=std, model=model, **custom_exts
        )

    #: Alias for :meth:`to_fits`.
    save = to_fits



# =============================================================================
# SpectrumFitResult
# =============================================================================

class SpectrumFitResult:
    """
    Lightweight view of a single fitted spectrum extracted from a MultiFitResult.

    Access all parameter and diagnostic information for one spaxel without
    copying the underlying arrays.
    """

    def __init__(self, parent: MultiFitResult, index):
        self._parent = parent
        self._flat_index = parent._normalize_spaxel_index(index)
        self.index = np.unravel_index(self._flat_index, parent.shape)
        self._model = None
        self._comps = None
        self._addcomps = None

    @property
    def model(self):
        """The fitted astropy model for this spaxel (lazy, cached)."""
        if self._model is None:
            self._model = self._parent.get_model(self._flat_index)
        return self._model

    @property
    def parameters(self) -> np.ndarray:
        return self._parent._parameter_vector(self._flat_index)

    @property
    def errors(self) -> np.ndarray:
        return self._parent._error_vector(self._flat_index)

    @property
    def stds(self):
        return self.errors

    @property
    def stdevs(self):
        return self.errors

    @property
    def covariance(self):
        cov = self._parent.covariance
        return None if cov is None else cov[self.index]

    @property
    def fit_info(self) -> dict:
        return self._parent.get_fit_info(self._flat_index)

    @property
    def success(self) -> bool:
        return bool(self._parent.success[self.index])

    @property
    def nfev(self):
        return self._parent.nfev[self.index]

    @property
    def message(self) -> str:
        return self._parent.messages[self.index]

    @property
    def comps(self):
        """Model components for this fitted spaxel (all leaf components)."""
        if self._comps is None:
            from ..models.components import get_components
            self._comps = get_components(self.model, additive=False)
        return self._comps

    @property
    def addcomps(self):
        """Additive-only model components for this fitted spaxel."""
        if self._addcomps is None:
            from ..models.components import get_components
            self._addcomps = get_components(self.model, additive=True)
        return self._addcomps

    def __getattr__(self, name):
        try:
            parent = object.__getattribute__(self, '_parent')
            param_names = object.__getattribute__(parent, 'param_names')
        except Exception:
            parent = None
            param_names = ()

        if name in param_names:
            return getattr(self.model, name)
        if name in _MULTIFIT_STAT_KEYS:
            if parent is None:
                raise AttributeError(f"'SpectrumFitResult' has no attribute '{name}'")
            return parent.get_stat(name)[self.index]
        raise AttributeError(f"'SpectrumFitResult' has no attribute '{name}'")

    def __repr__(self):
        return (
            f"SpectrumFitResult(index={self.index}, "
            f"success={self.success}, "
            f"nfev={self.nfev})"
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_fits(self, filename, overwrite=False,
                parfilter=None, std=True, model=True, **custom_exts):
        """
        Save the single-spectrum fit result to a FITS file.

        A human-readable ``PARS`` binary table stores parameter names,
        values, standard deviations, and units.  The template model is
        written to a ``MODEL_DEF`` YAML extension (needed by
        ``load_model()``).

        Parameters
        ----------
        filename : str
        overwrite : bool
        parfilter : list of str, optional
        std : bool
        model : bool
        **custom_exts
            Extra extensions, e.g. ``CHISQ=chi2_scalar``.
        """
        from ..io import spectrumfit_to_fits
        spectrumfit_to_fits(
            self, filename, overwrite=overwrite,
            parfilter=parfilter, std=std, model=model, **custom_exts
        )

    #: Alias for :meth:`to_fits`.
    save = to_fits



# =============================================================================
# MultiFitMixin
# =============================================================================

class MultiFitMixin(abc.ABC):
    """
    Mixin that adds batched spectral fitting to any Fitter subclass.

    Mixing this in gives the fitter three user-facing entry points:

    * ``fitter(model, x, y, ...)``           — auto-dispatches to multifit when y is 2D/3D
    * ``fitter.multifit(model, x, y, ...)``  — explicit batch fitting
    * ``fitter.jointfit(...)``               — future joint fitting (placeholder)

    Parameter initialisation and constraint overrides
    --------------------------------------------------
    initpars : dict, optional   ``{'param_name': scalar_or_spatial_array}``
        Override initial parameter values per spaxel.  A scalar applies to every
        spaxel; an array must have shape ``== spatial_shape``.

    bounds : dict, optional   ``{'param_name': (lower, upper)}``
        Override parameter bounds.  Each component (lower/upper) may be a scalar
        or a spatial array of the same shape as the data.

    fixed : dict, optional   ``{'param_name': True/False}``
        Fix/release parameters (shared across all spaxels).

    tied : dict, optional   ``{'param_name': callable_or_False}``
        Tie parameters (shared callable, applied before every evaluation).

    Error handling
    --------------
    * Unknown parameter names → ``KeyError`` with the list of valid names.
    * Wrong array shapes       → ``ValueError`` with expected vs actual shape.
    * Non-scalar fixed/tied    → ``ValueError`` (per-spaxel not supported).
    * Invalid dict type        → ``TypeError``.
    All checks run **before** any spectrum is fitted, so errors surface instantly.

    Example
    -------
    >>> from prism.modeling import models
    >>> from prism.modeling.fitting import AstroTRF
    >>> import numpy as np
    >>>
    >>> model    = models.GaussianLine(amplitude=10, mean=5000, fwhm=10)
    >>> cube     = ...  # ndarray shape (n_y, n_x, n_wave)
    >>> wave     = ...  # 1-D array, length n_wave
    >>> amp_init = np.random.uniform(5, 15, size=(n_y, n_x))
    >>>
    >>> fitter  = AstroTRF(calc_uncertainties=True)
    >>> results = fitter.multifit(
    ...     model, wave, cube,
    ...     yerr=cube_err,
    ...     initpars={'amplitude': amp_init},
    ...     bounds={'mean': (4950, 5050)},
    ... )
    >>> results.amplitude.value          # shaped (n_y, n_x)
    >>> results.amplitude.std            # shaped (n_y, n_x)
    >>> results[5, 10].model             # fitted model for spaxel (5, 10)
    """

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _resolve_spectral_axis(self, y, wave_len, spectral_axis=None):
        if spectral_axis is not None:
            axis = np.lib.array_utils.normalize_axis_index(spectral_axis, y.ndim)
            if y.shape[axis] != wave_len:
                raise ValueError(f"spectral_axis={spectral_axis} has length {y.shape[axis]}, expected {wave_len}.")
            return axis
        matches = [axis for axis, size in enumerate(y.shape) if size == wave_len]
        if not matches:
            raise ValueError(f"Could not match wave axis length ({wave_len}) to any dimension in y shape {y.shape}.")
        if len(matches) > 1:
            raise ValueError("Ambiguous spectral axis. Pass spectral_axis explicitly.")
        return matches[0]

    def _validate_multifit_param_name(self, model, name, control_name):
        if name not in model.param_names:
            raise KeyError(
                f"Unknown parameter '{name}' in {control_name}. "
                f"Available parameters: {list(model.param_names)}"
            )
        param = getattr(model, name)
        if param.size != 1:
            raise ValueError(
                f"{control_name} currently supports only scalar parameters. "
                f"Parameter '{name}' has size {param.size}."
            )
        return param

    def _normalize_multifit_value(self, value, spatial_shape, control_name, param_name):
        """Return (shared_scalar, array_or_None)."""
        if np.isscalar(value) or value is None:
            return value, None
        array = np.asarray(value)
        if array.shape != tuple(spatial_shape):
            raise ValueError(
                f"{control_name} for '{param_name}' must be a scalar or have shape "
                f"{tuple(spatial_shape)}, got {array.shape}."
            )
        return None, array

    def _normalize_multifit_bounds_pair(self, bounds, spatial_shape, param_name):
        if not isinstance(bounds, (tuple, list, np.ndarray)) or len(bounds) != 2:
            raise ValueError(
                f"bounds for '{param_name}' must be a 2-tuple (lower, upper)."
            )
        lower_shared, lower_map = self._normalize_multifit_value(
            bounds[0], spatial_shape, 'bounds', param_name
        )
        upper_shared, upper_map = self._normalize_multifit_value(
            bounds[1], spatial_shape, 'bounds', param_name
        )
        return (lower_shared, lower_map), (upper_shared, upper_map)

    def _build_multifit_bounds_array(self, spatial_shape,
                                     lower_shared, lower_map,
                                     upper_shared, upper_map):
        n = int(np.prod(spatial_shape))
        result = np.empty(n, dtype=object)
        lower_flat = None if lower_map is None else np.asarray(lower_map, dtype=object).reshape(-1)
        upper_flat = None if upper_map is None else np.asarray(upper_map, dtype=object).reshape(-1)
        for i in range(n):
            lo = lower_shared if lower_flat is None else lower_flat[i]
            hi = upper_shared if upper_flat is None else upper_flat[i]
            result[i] = (lo, hi)
        return result.reshape(spatial_shape)

    def _apply_scalar_multifit_overrides(self, model, initpars=None, bounds=None):
        if initpars:
            for name, value in initpars.items():
                getattr(model, name).value = value
        if bounds:
            for name, pair in bounds.items():
                getattr(model, name).bounds = pair
        return model

    # ------------------------------------------------------------------
    # Configuration build  (runs before any spectrum is fitted)
    # ------------------------------------------------------------------

    def _prepare_multifit_configuration(self, model, spatial_shape,
                                        initpars=None, bounds=None,
                                        fixed=None, tied=None):
        """
        Validate all override dicts, build shared/per-spaxel maps, and return
        a configuration dict consumed by the per-spaxel task loop.
        """
        template_model = model.copy()
        init_shared, init_maps = {}, {}
        bounds_shared, bounds_maps = {}, {}
        parameter_metadata = {}

        if fixed is not None:
            if not isinstance(fixed, dict):
                raise TypeError("fixed must be a dict mapping parameter names to booleans.")
            for name, value in fixed.items():
                self._validate_multifit_param_name(template_model, name, 'fixed')
                if isinstance(value, np.ndarray) or (not np.isscalar(value) and value is not None):
                    raise ValueError(
                        f"fixed['{name}'] must be a shared boolean, not a per-spectrum array."
                    )
                bv = bool(value)
                getattr(template_model, name).fixed = bv
                parameter_metadata.setdefault(name, {})['fixed'] = bv

        if tied is not None:
            if not isinstance(tied, dict):
                raise TypeError("tied must be a dict mapping parameter names to callables or False.")
            for name, value in tied.items():
                self._validate_multifit_param_name(template_model, name, 'tied')
                if isinstance(value, np.ndarray):
                    raise ValueError(
                        f"tied['{name}'] must be a shared callable, not a per-spectrum array."
                    )
                if value not in (False, None) and not callable(value):
                    raise TypeError(
                        f"tied['{name}'] must be a callable, False, or None."
                    )
                tv = False if value in (False, None) else value
                getattr(template_model, name).tied = tv
                parameter_metadata.setdefault(name, {})['tied'] = tv

        if initpars is not None:
            if not isinstance(initpars, dict):
                raise TypeError("initpars must be a dict mapping parameter names to values.")
            for name, value in initpars.items():
                self._validate_multifit_param_name(template_model, name, 'initpars')
                shared, mapped = self._normalize_multifit_value(
                    value, spatial_shape, 'initpars', name
                )
                if mapped is None:
                    init_shared[name] = shared
                else:
                    init_maps[name] = np.asarray(mapped, dtype=float).reshape(-1)

        if bounds is not None:
            if not isinstance(bounds, dict):
                raise TypeError("bounds must be a dict mapping parameter names to (lower, upper).")
            for name, value in bounds.items():
                self._validate_multifit_param_name(template_model, name, 'bounds')
                (lo_s, lo_m), (hi_s, hi_m) = self._normalize_multifit_bounds_pair(
                    value, spatial_shape, name
                )
                if lo_m is None and hi_m is None:
                    pair = (lo_s, hi_s)
                    getattr(template_model, name).bounds = pair
                    bounds_shared[name] = pair
                    parameter_metadata.setdefault(name, {})['bounds'] = \
                        self._build_multifit_bounds_array(spatial_shape, lo_s, None, hi_s, None)
                else:
                    lo_flat = None if lo_m is None else np.asarray(lo_m, dtype=object).reshape(-1)
                    hi_flat = None if hi_m is None else np.asarray(hi_m, dtype=object).reshape(-1)
                    bounds_maps[name] = (lo_s, lo_flat, hi_s, hi_flat)
                    parameter_metadata.setdefault(name, {})['bounds'] = \
                        self._build_multifit_bounds_array(spatial_shape, lo_s, lo_m, hi_s, hi_m)

        for name in template_model.param_names:
            param = getattr(template_model, name)
            meta = parameter_metadata.setdefault(name, {})
            meta.setdefault('fixed', param.fixed)
            meta.setdefault('tied', param.tied)
            meta.setdefault('bounds', param.bounds)
            meta.setdefault('unit', getattr(param, 'unit', None))

        return {
            'template_model':   template_model,
            'init_shared':      init_shared,
            'init_maps':        init_maps,
            'bounds_shared':    bounds_shared,
            'bounds_maps':      bounds_maps,
            'parameter_metadata': parameter_metadata,
        }

    def _multifit_task_overrides(self, idx, config):
        """Return (initpars_1d, bounds_1d) for the spaxel at flat index ``idx``."""
        initpars = {}
        bounds = {}
        for name, v in config['init_shared'].items():
            initpars[name] = v
        for name, arr in config['init_maps'].items():
            initpars[name] = arr[idx]
        for name, pair in config['bounds_shared'].items():
            bounds[name] = pair
        for name, info in config['bounds_maps'].items():
            lo_s, lo_flat, hi_s, hi_flat = info
            lo = lo_s if lo_flat is None else lo_flat[idx]
            hi = hi_s if hi_flat is None else hi_flat[idx]
            lo = None if lo is None or (np.isscalar(lo) and np.isnan(lo)) else lo
            hi = None if hi is None or (np.isscalar(hi) and np.isnan(hi)) else hi
            bounds[name] = (lo, hi)
        return initpars, bounds

    # ------------------------------------------------------------------
    # Per-spaxel worker
    # ------------------------------------------------------------------

    def _fit_single_target(self, args):
        """Worker function for multiprocess mapping (one 1-D fit per call)."""
        idx, model, x, y_1d, yerr_1d, statistic, weights_1d, initpars_1d, bounds_1d, kwargs = args
        try:
            local_model = model.copy()
            self._apply_scalar_multifit_overrides(local_model, initpars=initpars_1d, bounds=bounds_1d)
            fitted = self(
                model=local_model, x=x, y=y_1d, yerr=yerr_1d,
                statistic=statistic, weights=weights_1d,
                inplace=True, **kwargs
            )
            return (idx, fitted.parameters, self.stdevs,
                    self._multifit_entry_from_fit_info(), self.covariance)
        except Exception as e:
            if getattr(self, 'verbose', False):
                print(f"Worker {idx} failed: {e}")
            return (
                idx,
                np.full(len(model.parameters), np.nan),
                None,
                {
                    'success': False, 'nfev': 0, 'message': str(e),
                    **{k: np.nan for k in _MULTIFIT_STAT_KEYS},
                },
                None,
            )

    def _fit_single_direct(self, idx, model, x, y_1d, yerr_1d,
                           statistic, weights_1d, initpars_1d, bounds_1d,
                           kwargs):
        """Direct in-process single-target fit used by the serial multifit path."""
        try:
            local_model = model.copy()
            self._apply_scalar_multifit_overrides(local_model, initpars=initpars_1d, bounds=bounds_1d)
            fitted = self(
                model=local_model, x=x, y=y_1d, yerr=yerr_1d,
                statistic=statistic, weights=weights_1d,
                inplace=True, **kwargs
            )
            return (idx, fitted.parameters, self.stdevs,
                    self._multifit_entry_from_fit_info(), self.covariance)
        except Exception as e:
            if getattr(self, 'verbose', False):
                print(f"Worker {idx} failed: {e}")
            return (
                idx,
                np.full(len(model.parameters), np.nan),
                None,
                {
                    'success': False, 'nfev': 0, 'message': str(e),
                    **{k: np.nan for k in _MULTIFIT_STAT_KEYS},
                },
                None,
            )

    def _multifit_spawn_fitter(self):
        """Build a lightweight fitter clone for spawn workers."""
        worker_fitter = copy.copy(self)
        worker_fitter.fit_info = {}
        if hasattr(worker_fitter, '_template_model'):
            worker_fitter._template_model = None
        return worker_fitter

    # ------------------------------------------------------------------
    # Aggregate fit info
    # ------------------------------------------------------------------

    def _set_batch_fit_info(self, result, total_time):
        n_success = int(result._success.sum())
        n_failed = result.n_spaxels - n_success
        summary = {
            'time': total_time,
            'success': n_failed == 0,
            'nfev': result.nfev,
            'message': f"{n_success}/{result.n_spaxels} spectra converged",
            'messages': result.messages,
            'result': result,
            'param_cov': result.covariance,
            'n_success': n_success,
            'n_failed': n_failed,
            'success_fraction': n_success / result.n_spaxels if result.n_spaxels else np.nan,
            'statistic': 'batch',
        }
        for name in _MULTIFIT_STAT_KEYS:
            summary[name] = result.get_stat(name)
        result.fit_info = summary
        self.fit_info = summary

    def _multifit_entry_from_fit_info(self):
        entry = {
            'success': self.fit_info.get('success', False),
            'nfev':    self.fit_info.get('nfev', np.nan),
            'message': self.fit_info.get('message', ''),
        }
        for name in _MULTIFIT_STAT_KEYS:
            entry[name] = self.fit_info.get(name, np.nan)
        return entry

    def _multifit_chunksize(self, n_spaxels, nproc):
        """Choose a low-overhead chunksize with predictable scheduling."""
        if nproc <= 1:
            return 1
        if n_spaxels <= 32 * nproc:
            return max(1, n_spaxels // nproc)
        if n_spaxels <= 256 * nproc:
            return max(1, n_spaxels // (2 * nproc))
        return max(1, n_spaxels // (4 * nproc))

    # ------------------------------------------------------------------
    # Main batch fitting loop
    # ------------------------------------------------------------------

    def _fit_multi(self, model, x, y, yerr=None, statistic='chi2', weights=None,
                   nproc=1, spectral_axis=None, progress=True, batch=False,
                   binmap=None,
                   initpars=None, bounds=None, fixed=None, tied=None,
                   **kwargs):
        """Internal: run fits over all spaxels and accumulate a MultiFitResult."""
        y = np.asarray(y)
        if y.ndim < 2:
            raise ValueError("Multi-spectrum fitting requires y.ndim >= 2.")
        if yerr is not None and np.asarray(yerr).ndim != y.ndim:
            raise ValueError(
                "Asymmetric yerr is not supported by direct multifit calls. "
                "Use symmetric yerr or bootstrap(..., noise_dist='uniform') for asymmetric resampling."
            )

        wave_len = len(x)
        resolved_axis = self._resolve_spectral_axis(y, wave_len, spectral_axis=spectral_axis)
        y = np.moveaxis(y, resolved_axis, -1)
        mapped_yerr    = np.moveaxis(np.asarray(yerr), resolved_axis, -1)    if yerr    is not None else None
        mapped_weights = np.moveaxis(np.asarray(weights), resolved_axis, -1) if weights is not None else None
        spatial_shape  = y.shape[:-1]

        config = self._prepare_multifit_configuration(
            model, spatial_shape,
            initpars=initpars, bounds=bounds, fixed=fixed, tied=tied,
        )
        template_model = config['template_model']

        n_spaxels    = int(np.prod(spatial_shape))
        y_flat       = y.reshape((n_spaxels, wave_len))
        yerr_flat    = mapped_yerr.reshape((n_spaxels, wave_len))    if mapped_yerr    is not None else [None] * n_spaxels
        weights_flat = mapped_weights.reshape((n_spaxels, wave_len)) if mapped_weights is not None else [None] * n_spaxels

        if binmap is not None:
            validated_binmap = parse_binmap(binmap, spatial_shape=spatial_shape, name='binmap')
            flat_binmap = validated_binmap.reshape(-1)
            fit_flat_indices = []
            fit_groups = {}
            for bin_id in np.unique(flat_binmap):
                if int(bin_id) < 0:
                    continue
                members = np.flatnonzero(flat_binmap == bin_id)
                if members.size == 0:
                    continue
                representative = int(members[0])
                fit_flat_indices.append(representative)
                fit_groups[representative] = members.astype(int)
            if not fit_flat_indices:
                raise ValueError('binmap does not contain any non-negative bin labels.')
        else:
            fit_flat_indices = list(range(n_spaxels))
            fit_groups = {idx: np.array([idx], dtype=int) for idx in fit_flat_indices}

        res = MultiFitResult(
            spatial_shape,
            template_model.param_names,
            template_model=template_model,
            x=x,
            has_cov=self.calc_uncertainties,
            spectral_axis=resolved_axis,
            fit_indices=model_to_fit_params(template_model)[1],
            parameter_metadata=config['parameter_metadata'],
        )

        def task_generator():
            for idx in fit_flat_indices:
                ip, bo = self._multifit_task_overrides(idx, config)
                yield (idx, template_model, x, y_flat[idx],
                       yerr_flat[idx], statistic, weights_flat[idx], ip, bo, kwargs)

        if progress is None:
            progress = True

        show_progress = progress

        if self.verbose:
            if binmap is None:
                print(f"Fitting {n_spaxels} spectra on {nproc} core(s)…")
            else:
                print(f"Fitting {len(fit_flat_indices)} unique bins over {n_spaxels} spectra on {nproc} core(s)…")

        t0 = time.perf_counter()

        if nproc <= 1:
            indices = fit_flat_indices
            if show_progress:
                indices = tqdm(indices, total=len(fit_flat_indices), desc="Fitting bins" if binmap is not None else "Fitting spectra")
            for idx in indices:
                ip, bo = self._multifit_task_overrides(idx, config)
                entry = self._fit_single_direct(
                    idx=idx,
                    model=template_model,
                    x=x,
                    y_1d=y_flat[idx],
                    yerr_1d=yerr_flat[idx],
                    statistic=statistic,
                    weights_1d=weights_flat[idx],
                    initpars_1d=ip,
                    bounds_1d=bo,
                    kwargs=kwargs,
                )
                for target_idx in fit_groups[idx]:
                    res.update(target_idx, entry[1], entry[2], entry[3], entry[4])
        else:
            ctx = mp.get_context('spawn')
            worker_fitter = self._multifit_spawn_fitter()
            with ctx.Pool(
                nproc,
                initializer=_multifit_worker_initializer,
                initargs=(worker_fitter, template_model, x, statistic, kwargs),
            ) as pool:
                def worker_task_generator():
                    for idx in fit_flat_indices:
                        ip, bo = self._multifit_task_overrides(idx, config)
                        yield (idx, y_flat[idx], yerr_flat[idx], weights_flat[idx], ip, bo)

                chunksize = self._multifit_chunksize(len(fit_flat_indices), nproc) if batch else 1
                it = pool.imap_unordered(_multifit_worker_task, worker_task_generator(),
                                         chunksize=chunksize)
                if show_progress:
                    it = tqdm(it, total=len(fit_flat_indices), desc="Fitting bins" if binmap is not None else "Fitting spectra")
                for result in it:
                    rep_idx, model_params, stdevs, fit_info, cov = result
                    for target_idx in fit_groups[rep_idx]:
                        res.update(target_idx, model_params, stdevs, fit_info, cov)

        total_time = time.perf_counter() - t0
        self._set_batch_fit_info(res, total_time)

        if self.verbose:
            n_ok = self.fit_info['n_success']
            n_fail = self.fit_info['n_failed']
            print(f"Done in {total_time:.1f}s — {n_ok}/{n_spaxels} converged"
                  + (f", {n_fail} failed" if n_fail else "") + ".")

        return res

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def multifit(self, model, x, y, z=None, yerr=None, statistic='chi2', weights=None,
                 inplace=False, nproc=1, spectral_axis=None, progress=True, batch=False,
                 binmap=None,
                 initpars=None, bounds=None, fixed=None, tied=None,
                 **kwargs) -> MultiFitResult:
        """
        Fit the same model independently across multiple spectra.

        This is the explicit batch-fitting entry point. For a full description
        of ``initpars``, ``bounds``, ``fixed``, ``tied``, and error handling see
        the class-level docstring of ``MultiFitMixin``.

        Parameters
        ----------
        model : astropy Model
            Template model (copied per spaxel).
        x : array_like
            1-D spectral grid shared by every spectrum.
        y : array_like
            Data cube.  The axis matching ``len(x)`` is detected automatically
            unless ``spectral_axis`` is given.
        z : ignored
            Placeholder for future 2-D model support.
        yerr : array_like, optional
            Uncertainty cube (same shape as ``y``).
        statistic : {'chi2', 'poisson'}
            Statistic used to convert ``yerr`` to weights.
        weights : array_like, optional
            Explicit weight cube (overrides ``yerr``).
        inplace : bool
            Kept for API symmetry, but ignored in multifit mode. Each spectrum
            is always fit using its own model copy. Passing ``inplace=True``
            emits a warning.
        nproc : int
            Number of worker processes.  ``1`` disables multiprocessing.
        spectral_axis : int, optional
            Axis of ``y`` that is the spectral dimension.
        progress : bool, optional
            Show tqdm progress bar.  Defaults to ``fitter.verbose``.
        batch : bool, optional
            When ``True`` and ``nproc > 1``, results are delivered to the
            progress bar in chunks (smarter scheduling, slightly lower IPC
            overhead).  When ``False`` (default) the bar updates once per
            completed fit.  The difference is negligible for real spectral
            fits (>1 ms each); use ``batch=True`` only for very fast synthetic
            fits where IPC dominates.
        initpars : dict, optional
            Per-parameter initial value overrides.
        bounds : dict, optional
            Per-parameter bound overrides.
        fixed : dict, optional
            Per-parameter fixed-state overrides.
        tied : dict, optional
            Per-parameter tie-function overrides.
        **kwargs
            Forwarded to the underlying 1-D fitter implementation.

        Returns
        -------
        MultiFitResult
        """
        if z is not None:
            raise NotImplementedError("2-D fitting is not yet implemented.")
        if inplace:
            warnings.warn(
                "'inplace=True' has no effect in multifit mode: each spectrum is "
                "fit on an independent model copy.",
                UserWarning,
                stacklevel=2,
            )
        return self._fit_multi(
            model, x, y, yerr=yerr, statistic=statistic, weights=weights,
            nproc=nproc, spectral_axis=spectral_axis, progress=progress, batch=batch,
            binmap=binmap,
            initpars=initpars, bounds=bounds, fixed=fixed, tied=tied,
            **kwargs
        )

    def jointfit(self, *args, **kwargs):
        """Placeholder for future joint optimisation across datasets with shared parameters."""
        raise NotImplementedError(
            "Joint fitting is not implemented yet. "
            "Use multifit() for independent fits over multiple spectra."
        )
