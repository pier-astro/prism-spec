"""Independent fit execution, process startup and streamed result collection."""
import time
import warnings
import numpy as np
import abc
import os
import copy
import threading
from contextlib import contextmanager
from collections.abc import Mapping
from tempfile import TemporaryDirectory
import dill
import multiprocess as mp
from tqdm.auto import tqdm
from ..utils import _free_parameter_indices, _ParameterMap
from ..data import Dataset, _validate_fit_shapes
from ..results import FitResult, _clear_model_estimates
from ..extension import _validate_spectral_inputs
from ...composition import Model
from ....data.core import parse_binmap
from .results import MultiFitResult, _MULTIFIT_STAT_KEYS
from .configuration import _MultiFitConfiguration

_MULTIFIT_WORKER_STATE = {}
_MULTIFIT_SPAWN_LOCK = threading.Lock()

def _combine_validity_masks(base, extra):
    """Combine Prism-style validity masks where True means usable."""
    if extra is None:
        return base
    extra = np.asarray(extra, dtype=bool)
    if base is None:
        return extra
    return np.asarray(base, dtype=bool) & extra


def _extract_multifit_inputs(y, yerr=None, weights=None, mask=None):
    """Normalize multidimensional inputs and infer optional validity masks."""
    inferred_mask = None

    if hasattr(y, 'values') and hasattr(y, 'mask'):
        if mask is None:
            mask = y.mask
        if yerr is None and hasattr(y, 'err'):
            yerr = y.err
        y = y.values

    if np.ma.isMaskedArray(y):
        inferred_mask = _combine_validity_masks(inferred_mask, ~np.ma.getmaskarray(y))
        y = np.ma.getdata(y)
    if np.ma.isMaskedArray(yerr):
        inferred_mask = _combine_validity_masks(inferred_mask, ~np.ma.getmaskarray(yerr))
        yerr = np.ma.getdata(yerr)
    if np.ma.isMaskedArray(weights):
        inferred_mask = _combine_validity_masks(inferred_mask, ~np.ma.getmaskarray(weights))
        weights = np.ma.getdata(weights)

    mask = _combine_validity_masks(inferred_mask, mask)
    return y, yerr, weights, mask


def _nan_multifit_payload(model, message):
    """Return the standard NaN-filled multifit payload for one failed target."""
    return (
        np.full(len(model.parameters), np.nan),
        None,
        {
            'success': False,
            'nfev': 0,
            'message': message,
            **{k: np.nan for k in _MULTIFIT_STAT_KEYS},
        },
        None,
    )


def _multifit_worker_initializer(snapshot_file):
    """Load one private worker configuration from the parent-owned snapshot."""
    from ..extension import set_max_evaluations
    with open(snapshot_file, 'rb') as stream:
        state = dill.load(stream)
    set_max_evaluations(state.pop('max_evaluations'))
    if state.pop('astropy_pipe'):
        from ...interop.astropy import install_pipe_patch
        install_pipe_patch()
    global _MULTIFIT_WORKER_STATE
    _MULTIFIT_WORKER_STATE = state


@contextmanager
def _multifit_pool(nproc, **kwargs):
    """Set child thread defaults before imports, without changing the parent."""
    # Spawn copies the environment when each child starts. Setting defaults in
    # the initializer is too late for NumPy/BLAS imported during module loading.
    with _MULTIFIT_SPAWN_LOCK:
        names = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.setdefault(name, '1')
            pool = mp.get_context('spawn').Pool(nproc, **kwargs)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    with pool:
        yield pool


def _multifit_worker_task(args):
    """Run one multifit target using preloaded worker state."""
    idx, y_1d, yerr_1d, weights_1d, mask_1d, initpars_1d, bounds_1d = args
    state = _MULTIFIT_WORKER_STATE
    return state['fitter']._fit_single_direct(
        idx, state['template_model'], state['x'], y_1d, yerr_1d,
        state['statistic'], weights_1d, mask_1d, state['skip_invalid'],
        initpars_1d, bounds_1d, state['kwargs'],
    )


def _multifit_dataset_worker_task(args):
    """Keep each source and response in the same serialized local graph."""
    state = _MULTIFIT_WORKER_STATE
    return state['fitter']._fit_dataset_direct(
        *args, statistic=state['statistic'], skip_invalid=state['skip_invalid'],
        kwargs=state['kwargs'])


# =============================================================================
# MultiFitParameter
# =============================================================================

class MultiFitMixin(_MultiFitConfiguration, abc.ABC):
    """Fit a model recipe independently over regular arrays or named Datasets.

    Model states, optimizations, convergence and covariance are local to each
    target. Internal links are copied with the model; links across independent
    targets require jointfit instead. Serial and process-parallel execution use
    the same local optimization path.
    """

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _fit_single_direct(self, idx, model, x, y_1d, yerr_1d,
                           statistic, weights_1d, mask_1d, skip_invalid, initpars_1d, bounds_1d,
                           kwargs):
        """Fit one independent target in either the parent or a worker process."""
        try:
            local_model = model.copy()
            self._apply_scalar_multifit_overrides(local_model, initpars=initpars_1d, bounds=bounds_1d)
            # Let the single-spectrum fitter prepare data exactly once. Keep
            # coordinates intact for responses and profile-wide transforms.
            fit = getattr(self, '_fit_single', self)
            fitted = fit(
                model=local_model, x=x, y=y_1d, yerr=yerr_1d,
                statistic=statistic, weights=weights_1d, mask=mask_1d,
                inplace=True, **kwargs
            )
            return (idx, fitted.parameters, self.stdevs,
                    self._multifit_entry_from_fit_info(), self.covariance)
        except Exception as e:
            empty = isinstance(e, ValueError) and str(e) == 'No valid spectral samples remain after masking.'
            if empty and not skip_invalid:
                raise
            if getattr(self, 'verbose', False):
                print(f"Worker {idx} failed: {e}")
            message = 'skipped: no valid spectral samples' if empty else str(e)
            payload = _nan_multifit_payload(model, message)
            return (idx, payload[0], payload[1], payload[2], payload[3])

    def _fit_dataset_direct(self, label, model, data, initpars, bounds, *,
                            statistic, skip_invalid, kwargs):
        """Fit one independent Dataset, retaining its observed/source separation."""
        source, response = copy.deepcopy((model, data.response))
        local = copy.copy(data)
        local.response = response
        self._apply_scalar_multifit_overrides(source, initpars, bounds)
        try:
            result = self(source, local, statistic=statistic, inplace=True, **kwargs)
        except Exception as error:
            empty = isinstance(error, ValueError) and str(error) == 'No valid spectral samples remain after masking.'
            if empty and not skip_invalid:
                raise
            # A missing fit is a diagnostic snapshot, not a model at its initial
            # values presented as fitted. Preserve the recipe for an explicit retry.
            source, response = copy.deepcopy((model, data.response))
            local = copy.copy(data)
            local.response = response
            self._apply_scalar_multifit_overrides(source, initpars, bounds)
            _clear_model_estimates(source)
            if local.response is not None:
                _clear_model_estimates(local.response)
            graph = local._observed_model(source)
            parameters = _ParameterMap(graph)
            info = dict(success=False, nfev=0, message=str(error),
                        statistic=statistic, stat=np.nan, ndata=np.nan,
                        nfree=len(parameters.free), dof=np.nan)
            result = FitResult({'data': (source, local)}, graph, info,
                {'data': {'stat': np.nan, 'ndata': np.nan}}, _single=True, _failed=True)
        return label, result

    @contextmanager
    def _multifit_workers(self, nproc, *, statistic, skip_invalid, kwargs, **state):
        """Prepare the common spawn snapshot and keep it alive until collection."""
        from ..extension import get_max_evaluations
        from astropy.modeling import CompoundModel
        state.update(fitter=self._multifit_spawn_fitter(), statistic=statistic,
                     skip_invalid=bool(skip_invalid), kwargs=kwargs,
                     max_evaluations=get_max_evaluations(),
                     astropy_pipe=bool(getattr(CompoundModel, '_prism_linear_operator_patch', False)))
        with TemporaryDirectory(prefix='prism-workers-') as directory:
            snapshot_file = os.path.join(directory, 'worker.pkl')
            with open(snapshot_file, 'wb') as stream:
                dill.dump(state, stream)
            with _multifit_pool(nproc, initializer=_multifit_worker_initializer,
                               initargs=(snapshot_file,)) as pool:
                yield pool

    def _fit_datasets(self, model, datasets, *, statistic, skip_invalid, nproc,
                      progress, initpars, bounds, fixed, tied, **kwargs):
        """Prepare independent named observations without homogenizing their grids."""
        if not datasets:
            raise ValueError('multifit requires at least one Dataset.')
        labels = tuple(datasets)
        for label, data in datasets.items():
            if not isinstance(label, str) or not label or not isinstance(data, Dataset):
                raise TypeError('multifit requires nonempty string labels mapped to Datasets.')
            _validate_fit_shapes(data.x, data.y, data.yerr, data.weights, data.mask)
        source = Model.from_astropy(model)
        source._spectrum()
        # Clone every root together before configuring it, so a source/response
        # link remains intact. Each task will then copy its complete local graph.
        source, responses = copy.deepcopy((source, [data.response for data in datasets.values()]))
        config = self._prepare_multifit_configuration(
            source, (len(labels),),
            initpars=self._named_multifit_controls(initpars, labels),
            bounds=self._named_multifit_controls(bounds, labels, bounds=True),
            fixed=fixed, tied=tied, _copy_model=False)
        prepared = []
        for index, (label, data) in enumerate(datasets.items()):
            local = copy.copy(data)
            local.response = responses[index]
            # Declared dependencies must belong to THIS fit, never another task.
            observed = local._observed_model(source)
            _validate_spectral_inputs(observed, local.x, local.y, yerr=local.yerr,
                                      weights=local.weights, statistic=statistic)
            _ParameterMap(observed)  # includes dependencies on this response
            initial, limits = self._multifit_task_overrides(index, config)
            names = {id(getattr(observed, name)): name for name in observed.param_names}
            local_names = {name: names[id(getattr(source, name))] for name in source.param_names}
            self._validate_fit_constraints(observed,
                initial={local_names[name]: value for name, value in initial.items()},
                bounds={local_names[name]: value for name, value in limits.items()})
            prepared.append((label, source, local, initial, limits))

        results = {}
        start = time.perf_counter()
        nproc = min(nproc, len(labels))
        def collect(entries):
            if progress:
                entries = tqdm(entries, total=len(labels), desc='Fitting Datasets')
            for label, result in entries:
                results[label] = result
        if nproc == 1:
            collect(self._fit_dataset_direct(*task, statistic=statistic,
                    skip_invalid=skip_invalid, kwargs=kwargs) for task in prepared)
        else:
            with self._multifit_workers(nproc, statistic=statistic,
                    skip_invalid=skip_invalid, kwargs=kwargs) as pool:
                collect(pool.imap_unordered(_multifit_dataset_worker_task, prepared, chunksize=1))
        results = {label: results[label] for label in labels}
        succeeded = sum(bool(result.fit_info['success']) for result in results.values())
        self.fit_info = dict(time=time.perf_counter()-start, statistic='batch',
            success=succeeded == len(labels), n_success=succeeded,
            n_failed=len(labels)-succeeded, success_fraction=succeeded/len(labels),
            message=f'{succeeded}/{len(labels)} independent fits converged',
            result=results)
        return results

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

    # ------------------------------------------------------------------
    # Main batch fitting loop
    # ------------------------------------------------------------------

    def _fit_multi(self, model, x, y, yerr=None, statistic='chi2', weights=None,
                   mask=None, skip_invalid=True,
                   nproc=1, axis=None, progress=True,
                   binmap=None,
                   initpars=None, bounds=None, fixed=None, tied=None,
                   **kwargs):
        """Internal: run fits over all spaxels and accumulate a MultiFitResult."""
        if not isinstance(nproc, (int, np.integer)) or nproc < 1:
            raise ValueError('nproc must be a positive integer.')
        if 'batch' in kwargs:
            raise TypeError('multifit no longer accepts batch; completed fits are streamed individually.')
        y, yerr, weights, mask = _extract_multifit_inputs(y, yerr=yerr, weights=weights, mask=mask)
        y = np.asarray(y)
        if y.ndim < 2:
            raise ValueError("Multi-spectrum fitting requires y.ndim >= 2.")
        if yerr is not None and np.asarray(yerr).ndim != y.ndim:
            raise ValueError(
                "Asymmetric yerr is not supported by direct multifit calls. "
                "Use symmetric yerr or bootstrap(..., noise_dist='uniform') for asymmetric resampling."
            )
        for name, array in [('yerr', yerr), ('weights', weights), ('mask', mask)]:
            if array is not None and np.shape(array) != y.shape:
                raise ValueError(f'{name} must have shape {y.shape}; got {np.shape(array)}.')

        n_samples = len(x)
        resolved_axis = self._resolve_sample_axis(y, n_samples, axis=axis)
        y = np.moveaxis(y, resolved_axis, -1)
        mapped_yerr    = np.moveaxis(np.asarray(yerr), resolved_axis, -1)    if yerr    is not None else None
        mapped_weights = np.moveaxis(np.asarray(weights), resolved_axis, -1) if weights is not None else None
        mapped_mask    = np.moveaxis(np.asarray(mask, dtype=bool), resolved_axis, -1) if mask is not None else None
        spatial_shape  = y.shape[:-1]

        config = self._prepare_multifit_configuration(
            model, spatial_shape,
            initpars=initpars, bounds=bounds, fixed=fixed, tied=tied,
        )
        template_model = config['template_model']

        n_spaxels    = int(np.prod(spatial_shape))
        y_flat       = y.reshape((n_spaxels, n_samples))
        yerr_flat    = mapped_yerr.reshape((n_spaxels, n_samples))    if mapped_yerr    is not None else [None] * n_spaxels
        weights_flat = mapped_weights.reshape((n_spaxels, n_samples)) if mapped_weights is not None else [None] * n_spaxels
        mask_flat    = mapped_mask.reshape((n_spaxels, n_samples))    if mapped_mask    is not None else [None] * n_spaxels

        if binmap is not None:
            validated_binmap = parse_binmap(binmap, spatial_shape=spatial_shape, name='binmap')
            flat_binmap = validated_binmap.reshape(-1)
            # A stable sort groups equal labels and keeps their first spatial
            # member as representative, without rescanning the entire map.
            order = np.argsort(flat_binmap, kind='stable')
            order = order[flat_binmap[order] >= 0]
            if order.size == 0:
                raise ValueError('binmap does not contain any non-negative bin labels.')
            boundaries = np.flatnonzero(np.diff(flat_binmap[order])) + 1
            groups = np.split(order, boundaries)
            fit_flat_indices = [int(group[0]) for group in groups]
            fit_groups = dict(zip(fit_flat_indices, groups))
        else:
            fit_flat_indices = range(n_spaxels)
            fit_groups = None

        res = MultiFitResult(
            spatial_shape,
            template_model.param_names,
            template_model=template_model,
            x=x,
            has_cov=getattr(self, "_estimate_covariance", getattr(self, "calc_uncertainties", False)),
            axis=resolved_axis,
            fit_indices=_free_parameter_indices(template_model),
            parameter_metadata=config['parameter_metadata'],
        )

        if progress is None:
            progress = True

        show_progress = progress
        nproc = min(nproc, max(1, len(fit_flat_indices)))

        if self.verbose:
            if binmap is None:
                print(f"Fitting {n_spaxels} spectra on {nproc} core(s)…")
            else:
                print(f"Fitting {len(fit_flat_indices)} unique bins over {n_spaxels} spectra on {nproc} core(s)…")

        t0 = time.perf_counter()

        def tasks():
            for idx in fit_flat_indices:
                ip, bo = self._multifit_task_overrides(idx, config)
                yield (idx, y_flat[idx], yerr_flat[idx], weights_flat[idx], mask_flat[idx], ip, bo)

        def collect(entries):
            if show_progress:
                entries = tqdm(entries, total=len(fit_flat_indices),
                               desc="Fitting bins" if binmap is not None else "Fitting spectra")
            for idx, params, stds, info, covariance in entries:
                targets = (idx,) if fit_groups is None else fit_groups[idx]
                for target in targets:
                    res.update(target, params, stds, info, covariance)

        if nproc == 1:
            entries = (
                self._fit_single_direct(
                    idx, template_model, x, spectrum, error, statistic,
                    weight, mask, skip_invalid, ip, bo, kwargs)
                for idx, spectrum, error, weight, mask, ip, bo in tasks()
            )
            collect(entries)
        else:
            # Large initializer arguments can block spawn's startup pipe.
            # Each worker restores its own invariant configuration from one file.
            with self._multifit_workers(nproc, template_model=template_model, x=x,
                    statistic=statistic, skip_invalid=skip_invalid, kwargs=kwargs) as pool:
                collect(pool.imap_unordered(_multifit_worker_task, tasks(), chunksize=1))

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

    def multifit(self, model, x, y=None, z=None, yerr=None, statistic='chi2', weights=None,
                 mask=None, skip_invalid=True,
                 nproc=1, axis=None, progress=True,
                 binmap=None,
                 initpars=None, bounds=None, fixed=None, tied=None,
                 **kwargs):
        """Fit one recipe independently to many observations.

        Parameters
        ----------
        model : Model or astropy.modeling.Model
            Source recipe copied for each fit. Existing source pipes are retained.
        x : array-like or mapping of str to Dataset
            Common one-dimensional coordinates, or named observations with their
            own grids, errors, masks and optional responses. No interpolation is
            performed to homogenize Dataset sampling.
        y : array-like or Cube, optional
            Regular data with at least two dimensions; omit for named Datasets.
        z : optional
            Reserved; native spatial multifit is not implemented.
        yerr, weights, mask : array-like, optional
            Same shape as y. Errors are standard deviations; weights multiply
            residuals and override errors. True mask values include samples.
            Named observations store these controls on each Dataset instead.
        statistic : {'chi2', 'leastsq'}, optional
            Weighted or unweighted least squares. Poisson is not supported.
        skip_invalid : bool, optional
            Store empty selections as failures by default; False raises on them.
            Other local numerical failures retain diagnostics and missing values.
            Collection configuration errors raise before any fit is started.
        nproc : int, optional
            Positive worker count. Default 1 runs serially. Excess workers are
            capped at the number of independent targets.
        axis : int, optional
            Sample dimension of regular y. Inferred only if exactly one dimension
            matches len(x); the result retains its original position.
        progress : bool, optional
            Show completion progress; default True.
        binmap : array-like of int, optional
            Regular-array target labels. Fit the first member of each nonnegative
            bin and replicate its result; negative labels remain excluded.
        initpars, bounds : dict, optional
            Source parameter controls. Regular data accept scalars or arrays with
            the nonsample shape. Named data accept scalars or mappings with every
            Dataset label. Each bound is a (lower, upper) pair; either side can be
            mapped. Free starting values must lie within their bounds. None or
            nonfinite initial overrides retain the recipe's initial value;
            mapped None/NaN bounds are open. Configure response parameters on
            each Dataset response, rather than with these source controls.
        fixed, tied : dict, optional
            Source parameter booleans or tie functions applied to each copy;
            per-target arrays are not accepted for these controls.
        **kwargs
            Optimizer controls forwarded to the one-dimensional fitter. inplace
            is not a batch option; the supplied source and responses are unchanged.

        Returns
        -------
        MultiFitResult or dict of str to FitResult
            Regular arrays give compact parameter/statistic/covariance maps.
            Named Datasets give an insertion-ordered dictionary of independent
            single-fit snapshots, each with model, dataset, predict and save.
            A fit that raises stores NaN estimates and its diagnostic; model,
            predict and save then raise rather than presenting the initial model
            as a fitted result. associations retains the recipe for retrying.
        """
        if z is not None:
            raise NotImplementedError("2-D fitting is not yet implemented.")
        if 'inplace' in kwargs:
            raise TypeError('multifit copies each model; inplace is not a batch-fit option.')
        if not isinstance(nproc, (int, np.integer)) or nproc < 1:
            raise ValueError('nproc must be a positive integer.')
        if 'batch' in kwargs:
            raise TypeError('multifit no longer accepts batch; completed fits are streamed individually.')
        if 'spectral_axis' in kwargs:
            raise TypeError('Use axis for the sample dimension.')
        if progress is None:
            progress = True
        if isinstance(x, Mapping):
            if any(value is not None for value in (y, yerr, weights, mask, axis, binmap)):
                raise TypeError('Named Datasets carry their own coordinates, errors, weights and masks; axis and binmap apply to arrays.')
            return self._fit_datasets(model, x, statistic=statistic,
                skip_invalid=skip_invalid, nproc=nproc, progress=progress,
                initpars=initpars, bounds=bounds, fixed=fixed, tied=tied, **kwargs)
        return self._fit_multi(
            model, x, y, yerr=yerr, statistic=statistic, weights=weights,
            mask=mask, skip_invalid=skip_invalid,
            nproc=nproc, axis=axis, progress=progress,
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
