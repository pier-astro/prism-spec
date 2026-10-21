"""Prepare shared and spatially mapped parameter controls for independent fits."""
from collections.abc import Mapping
import numpy as np
from ..utils import _get_tied_info
from ..constraints import _box_limits

class _MultiFitConfiguration:
    def _resolve_sample_axis(self, y, n_samples, axis=None):
        if axis is not None:
            axis = np.lib.array_utils.normalize_axis_index(axis, y.ndim)
            if y.shape[axis] != n_samples:
                raise ValueError(f"axis={axis} has length {y.shape[axis]}, expected {n_samples}.")
            return axis
        matches = [axis for axis, size in enumerate(y.shape) if size == n_samples]
        if not matches:
            raise ValueError(f"Could not match coordinate length ({n_samples}) to any dimension in y shape {y.shape}.")
        if len(matches) > 1:
            raise ValueError("Ambiguous sample axis. Pass axis explicitly.")
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

    def _apply_scalar_multifit_overrides(self, model, initpars=None, bounds=None):
        if initpars:
            for name, value in initpars.items():
                if value is None:
                    continue
                if np.isscalar(value) and not np.isfinite(value):
                    continue
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
                                        fixed=None, tied=None, *, _copy_model=True):
        """
        Validate all override dicts, build shared/per-spaxel maps, and return
        a configuration dict consumed by the per-spaxel task loop.
        """
        template_model = model.copy() if _copy_model else model
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
                    parameter_metadata.setdefault(name, {})['bounds'] = pair
                else:
                    lo_flat = None if lo_m is None else np.asarray(lo_m, dtype=float).reshape(-1)
                    hi_flat = None if hi_m is None else np.asarray(hi_m, dtype=float).reshape(-1)
                    bounds_maps[name] = (lo_s, lo_flat, hi_s, hi_flat)
                    parameter_metadata.setdefault(name, {})['bounds'] = (
                        lo_s if lo_m is None else np.array(lo_m, dtype=float, copy=True).reshape(spatial_shape),
                        hi_s if hi_m is None else np.array(hi_m, dtype=float, copy=True).reshape(spatial_shape))

        if _copy_model:
            _get_tied_info(template_model)

        for name in template_model.param_names:
            param = getattr(template_model, name)
            meta = parameter_metadata.setdefault(name, {})
            meta.setdefault('fixed', param.fixed)
            meta.setdefault('tied', param.tied)
            meta.setdefault('bounds', param.bounds)
            meta.setdefault('unit', getattr(param, 'unit', None))

        config = {
            'template_model':   template_model,
            'init_shared':      init_shared,
            'init_maps':        init_maps,
            'bounds_shared':    bounds_shared,
            'bounds_maps':      bounds_maps,
            'parameter_metadata': parameter_metadata,
        }
        if _copy_model:
            self._validate_multifit_bounds(template_model, config)
        return config

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

    def _named_multifit_controls(self, values, labels, *, bounds=False):
        """Resolve explicitly named per-target controls to insertion-order arrays."""
        if values is None:
            return None
        if not isinstance(values, dict):
            raise TypeError('Parameter controls must be dictionaries.')

        def resolve(value):
            if isinstance(value, Mapping):
                if set(value) != set(labels):
                    raise ValueError('Per-target controls must name every Dataset exactly once.')
                return np.array([value[label] for label in labels], dtype=object)
            if value is not None and not np.isscalar(value):
                raise TypeError('Named Dataset controls require scalars or mappings keyed by Dataset label.')
            return value

        result = {}
        for name, value in values.items():
            if bounds:
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError(f"bounds for '{name}' must be a 2-tuple (lower, upper).")
                result[name] = (resolve(value[0]), resolve(value[1]))
            else:
                result[name] = resolve(value)
        return result

    def _multifit_bound_controls(self, config):
        """Assemble array controls once for exact, vectorized preflight checks."""
        initial = dict(config['init_shared'], **config['init_maps'])
        bounds = config['bounds_shared'].copy()
        for name, (lo, lower, hi, upper) in config['bounds_maps'].items():
            lower = np.asarray(lo if lower is None else lower, dtype=float)
            upper = np.asarray(hi if upper is None else upper, dtype=float)
            bounds[name] = (np.where(np.isnan(lower), -np.inf, lower),
                            np.where(np.isnan(upper), np.inf, upper))
        return initial, bounds

    def _validate_multifit_bounds(self, model, config):
        """Validate all mapped starts and linked bounds before any worker starts."""
        initial, bounds = self._multifit_bound_controls(config)
        self._validate_fit_constraints(model, initial=initial, bounds=bounds)

    def _validate_fit_constraints(self, model, *, initial=None, bounds=None):
        """Batch preflight capability hook, shared by array and Dataset routes.

        Current adapters request boxes. A specialized fitter can override this
        hook together with its solver, without replacing batch orchestration or
        changing the generic residual/parameter problem.
        """
        _box_limits(model, initial=initial, bounds=bounds)
