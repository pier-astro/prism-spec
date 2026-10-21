"""Batch result arrays, component views and individual-spectrum access."""
import numpy as np
import pandas as pd
from types import MappingProxyType

_MULTIFIT_STAT_KEYS = ('cost', 'optimality', 'ndata', 'nparam', 'nfree', 'dof', 'status')

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

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype) if copy is None else np.array(self.value, dtype=dtype, copy=copy)

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

    def evaluate(self, x=None, axis=None):
        return self._parent._result._evaluate_component(
            self._key,
            x=x,
            axis=axis,
            additive=self._additive,
        )

    def __call__(self, x=None, axis=None):
        return self.evaluate(x=x, axis=axis)

    def get_model(self, index):
        return self._parent._result.get_component_model(
            index,
            self._key,
            additive=self._additive,
        )

    @property
    def flux(self):
        """Return a Metric array with flux value, uncertainty, and confidence limits."""
        return self._parent._result._component_flux(self._key, additive=self._additive)

    def ew(self, continuum=None, method='constant-continuum', x=None, window=None, num=4096):
        """Compute the equivalent width of the component."""
        return self._parent._result._component_ew(
            self._key, additive=self._additive, continuum=continuum, method=method, x=x, window=window, num=num)

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
        Coordinate grid; stored for ``evaluate()``.
    has_cov : bool
        Whether to allocate storage for per-spaxel covariance matrices.
    axis : int
        Original sample axis index before it was moved to the last position.
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
                 has_cov=False, axis=-1, fit_indices=None,
                 parameter_metadata=None):
        self.shape = tuple(shape)
        self.param_names = tuple(param_names)
        self._template_model = template_model.copy() if template_model is not None else None
        self._x = None if x is None else np.asarray(x)
        self.axis = axis
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
        values = self._params[name].reshape(self.shape).view()
        values.setflags(write=False)
        return values

    def _std_array(self, name):
        if name not in self.param_names:
            raise KeyError(f"Unknown parameter '{name}'. Available: {list(self.param_names)}")
        values = self._errs[name].reshape(self.shape).view()
        values.setflags(write=False)
        return values

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

    def _bounds_at(self, name, flat_index):
        """Read one pair without allocating a full map for model reconstruction."""
        bounds = self._metadata_value(name, 'bounds', default=(None, None))
        if isinstance(bounds, np.ndarray):
            return tuple(bounds.flat[flat_index])
        pair = tuple(value.flat[flat_index] if isinstance(value, np.ndarray) else value
                     for value in bounds)
        return tuple(None if value is None or np.isnan(value) else value for value in pair)

    def _bounds_component_array(self, name, component_index):
        bounds = self._metadata_value(name, 'bounds', default=(None, None))
        if isinstance(bounds, np.ndarray):
            values = [pair[component_index] for pair in bounds.flat]
            return np.asarray(values).reshape(self.shape)
        # Scalar constraints broadcast with zero strides; mapped sides share
        # their input arrays. Materialize tuple objects only when bounds is asked.
        value = np.asarray(bounds[component_index])
        if np.issubdtype(value.dtype, np.number) and np.any(np.isnan(value)):
            value = np.where(np.isnan(value), None, value)
        return np.broadcast_to(value, self.shape)

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
        bounds = self._metadata_value(name, 'bounds', default=(None, None))
        if isinstance(bounds, np.ndarray):
            return bounds.reshape(self.shape)
        result = np.empty(self.shape, dtype=object)
        if all(np.ndim(value) == 0 for value in bounds):
            result.fill(tuple(bounds))
        else:
            lower, upper = self.get_min(name), self.get_max(name)
            for index, pair in enumerate(zip(lower.flat, upper.flat)):
                result.flat[index] = pair
        return result

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
        if not np.all(np.isfinite(self._parameter_vector(flat_index))):
            raise ValueError('No finite fitted model is available for this target.')
        model = self._template_model.copy() if copy_model else self._template_model
        self._apply_parameter_metadata(model, flat_index)
        model.parameters = self._parameter_vector(flat_index)
        self._attach_model_stds(model, flat_index, self._error_vector(flat_index))
        return model

    def _apply_parameter_metadata(self, model, flat_index):
        for name in self.param_names:
            param = getattr(model, name)
            param.bounds = self._bounds_at(name, flat_index)
            # The copied template already carries fixed state and links.
            # Reassigning ties from metadata would reintroduce old graph nodes.
        return model

    def _attach_model_stds(self, model, flat_index, stds):
        from ..extension import _clear_fit_uncertainties
        _clear_fit_uncertainties(model)
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
            from ...models.components import get_components
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
        from ...models.components import get_components
        model = self.get_model(index)
        return get_components(model, additive=additive)[component]

    def _evaluate_component(self, component, x=None, axis=None, additive=False):
        """Internal component-evaluation implementation shared by helpers."""
        if self._template_model is None:
            raise ValueError("Component evaluation requires a template model.")

        eval_x = self._x if x is None else np.asarray(x)
        if eval_x is None:
            raise ValueError("Provide x= or construct with x= to enable component evaluation().")

        out = np.full(self.shape + (len(eval_x),), np.nan, dtype=float)
        for flat_index in range(self.n_spaxels):
            if not np.all(np.isfinite(self._parameter_vector(flat_index))):
                continue
            spatial_index = np.unravel_index(flat_index, self.shape)
            comp_model = self.get_component_model(flat_index, component, additive=additive)
            out[spatial_index] = comp_model(eval_x)

        target_axis = self.axis if axis is None else axis
        if target_axis < 0:
            target_axis += out.ndim
        if target_axis != out.ndim - 1:
            out = np.moveaxis(out, -1, target_axis)
        return out

    def evaluate_component(self, component, x=None, axis=None):
        """
        Evaluate one (non-additive) component across all spaxels.

        Parameters
        ----------
        component : int or str
            Component index or name from ``result.comps``.
        x : array_like, optional
            Coordinate grid. Uses the stored grid when omitted.
        axis : int, optional
            Desired output spectral-axis position (same semantics as ``evaluate()``).
        """
        return self._evaluate_component(component, x=x, axis=axis,
                                        additive=False)

    def evaluate_addcomponent(self, component, x=None, axis=None):
        """
        Evaluate one additive component across all spaxels.

        Parameters
        ----------
        component : int or str
            Component index or name from ``result.addcomps``.
        x : array_like, optional
            Coordinate grid. Uses the stored grid when omitted.
        axis : int, optional
            Desired output spectral-axis position (same semantics as ``evaluate()``).
        """
        return self._evaluate_component(component, x=x, axis=axis,
                                        additive=True)

    def _component_flux(self, component, additive=False):
        """Return array-valued Metric for the flux of a component."""
        vals, stds, los, his = [], [], [], []
        for flat_index in range(self.n_spaxels):
            comp_model = self.get_component_model(flat_index, component, additive=additive)
            if hasattr(comp_model, 'flux'):
                f = comp_model.flux
                # If f is a DataFrame (from LineGroups), we can extract its columns
                if isinstance(f, pd.DataFrame):
                    vals.append(np.asarray(f['value']))
                    stds.append(np.asarray(f['std']))
                    los.append(np.asarray(f['lolim']))
                    his.append(np.asarray(f['uplim']))
                else:
                    vals.append(f.value)
                    stds.append(f.std)
                    los.append(f.lolim)
                    his.append(f.uplim)
            else:
                vals.append(np.nan)
                stds.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
        from ...metrics import Metric
        vals_arr = np.array(vals)
        target_shape = self.shape + vals_arr.shape[1:]
        
        # If the original returned a DataFrame, return a DataFrame of Metric objects
        # Or just a single Metric holding arrays. The Metric object supports arrays!
        return Metric(
            value=vals_arr.reshape(target_shape),
            std=np.array(stds).reshape(target_shape),
            lolim=np.array(los).reshape(target_shape),
            uplim=np.array(his).reshape(target_shape)
        )

    def _component_ew(self, component, additive=False, continuum=None, method='constant-continuum', x=None, window=None, num=4096):
        """Return array-valued Metric for the equivalent width of a component."""
        vals, stds, los, his = [], [], [], []
        is_cont_array = isinstance(continuum, np.ndarray) and continuum.shape == self.shape
        for flat_index in range(self.n_spaxels):
            comp_model = self.get_component_model(flat_index, component, additive=additive)
            if hasattr(comp_model, 'eqw'):
                idx_cont = continuum[np.unravel_index(flat_index, self.shape)] if is_cont_array else continuum
                f = comp_model.eqw(continuum=idx_cont, method=method, x=x, window=window, num=num)
                if isinstance(f, pd.DataFrame):
                    vals.append(np.asarray(f['value']))
                    stds.append(np.asarray(f['std']))
                    los.append(np.asarray(f['lolim']))
                    his.append(np.asarray(f['uplim']))
                else:
                    vals.append(f.value)
                    stds.append(f.std)
                    los.append(f.lolim)
                    his.append(f.uplim)
            else:
                vals.append(np.nan)
                stds.append(np.nan)
                los.append(np.nan)
                his.append(np.nan)
        from ...metrics import Metric
        vals_arr = np.array(vals)
        target_shape = self.shape + vals_arr.shape[1:]
        return Metric(
            value=vals_arr.reshape(target_shape),
            std=np.array(stds).reshape(target_shape),
            lolim=np.array(los).reshape(target_shape),
            uplim=np.array(his).reshape(target_shape)
        )

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x=None, axis=None) -> np.ndarray:
        """
        Evaluate all fitted models on ``x`` and return a shaped cube.

        Parameters
        ----------
        x : array_like, optional
            Coordinate grid. Uses the stored grid when omitted.
        axis : int, optional
            Desired position of the sample axis in the output.
            Defaults to the original sample axis of the input data.

        Returns
        -------
        cube : ndarray, shape ``spatial_shape + (len(x),)`` (before axis move)
        """
        if self._template_model is None:
            raise ValueError("Model evaluation requires a template model.")
        eval_x = self._x if x is None else np.asarray(x)
        if eval_x is None:
            raise ValueError("Provide x= or construct with x= to enable evaluate().")

        out = np.full(self.shape + (len(eval_x),), np.nan, dtype=float)
        for flat_index in range(self.n_spaxels):
            if not np.all(np.isfinite(self._parameter_vector(flat_index))):
                continue
            out[np.unravel_index(flat_index, self.shape)] = self.get_model(flat_index)(eval_x)

        target_axis = self.axis if axis is None else axis
        if target_axis < 0:
            target_axis += out.ndim
        if target_axis != out.ndim - 1:
            out = np.moveaxis(out, -1, target_axis)
        return out

    def __call__(self, x=None, axis=None):
        """Alias for ``evaluate()``."""
        return self.evaluate(x=x, axis=axis)

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
    def params(self):
        """Read-only parameter views with the nonsample input shape.

        Values and standard deviations share stored numeric maps. Accessing one
        parameter never constructs per-target models or a stacked parameter cube.
        """
        return MappingProxyType({name: self.get_param(name) for name in self.param_names})

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
        from ...io import multifit_to_fits
        multifit_to_fits(
            self, filename, wcs=wcs, overwrite=overwrite,
            parfilter=parfilter, std=std, model=model, **custom_exts
        )

    #: Alias for :meth:`to_fits`.
    save = to_fits

    def to_image(self):
        raise NotImplementedError("MultiFitResult.to_image() is not implemented. Use to_fits() instead.")



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
            from ...models.components import get_components
            self._comps = get_components(self.model, additive=False)
        return self._comps

    @property
    def addcomps(self):
        """Additive-only model components for this fitted spaxel."""
        if self._addcomps is None:
            from ...models.components import get_components
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
        from ...io import spectrumfit_to_fits
        spectrumfit_to_fits(
            self, filename, overwrite=overwrite,
            parfilter=parfilter, std=std, model=model, **custom_exts
        )

    #: Alias for :meth:`to_fits`.
    save = to_fits



# =============================================================================
# MultiFitMixin
# =============================================================================

