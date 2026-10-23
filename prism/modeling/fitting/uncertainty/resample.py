"""Resampling-based parameter uncertainty estimation."""

import warnings
import numpy as np
from ..utils import _free_parameter_indices

__all__ = [
    'Bootstrap',
    'bootstrap',
    'BootstrapResult',
    'ResampleError',
]


class ResampleError(Exception):
    """Raised when resampling fails or inputs are invalid."""


def is_batched_result_like(model) -> bool:
    """Return True for MultiFit-like containers (batched spectra/cubes)."""
    return (
        hasattr(model, 'n_spaxels')
        and hasattr(model, 'get_model')
        and hasattr(model, 'param_names')
    )


def prepare_noise_and_weights(y, yerr, weights, statistic):
    """Prepare noise widths and residual weights, consistently with the fitters."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ResampleError('Resampling expects one-dimensional observations.')
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != y.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ResampleError('weights must match y and be finite and nonnegative.')
    valid = np.ones(y.shape, dtype=bool) if weights is None else weights > 0
    if not np.any(valid):
        raise ResampleError('No valid samples remain for resampling.')
    if np.any(~np.isfinite(y[valid])):
        raise ResampleError('Included observations must be finite.')
    if statistic.lower() == 'poisson' and np.any(y[valid] < 0):
        raise ResampleError('Poisson resampling requires nonnegative counts.')

    if yerr is None:
        if weights is not None:
            yerr = np.zeros_like(y)
            np.divide(1.0, weights, out=yerr, where=valid)
        elif statistic.lower() == 'poisson':
            yerr = np.sqrt(np.maximum(y, 1.0))
        else:
            raise ResampleError('yerr or residual weights are required for Gaussian resampling.')
    yerr = np.asarray(yerr, dtype=float)
    if yerr.shape not in (y.shape, (2, len(y))):
        raise ResampleError('yerr must have shape (N,) or (2, N), matching y.')
    included = yerr[..., valid]
    if np.any(~np.isfinite(included)) or np.any(included <= 0):
        raise ResampleError('Included uncertainties must be finite and strictly positive.')
    # Excluded samples carry zero weight and require no random noise draw.
    yerr = np.where(valid, yerr, 0.0)
    effective_yerr = yerr.mean(axis=0) if yerr.ndim == 2 else yerr
    if weights is None:
        weights = 1.0 / effective_yerr
    return yerr, effective_yerr, weights


def extract_limits(param, param_samples, lower_percentile, upper_percentile,
                   boundary_epsilon=1e-10, boundary_tolerance=0.5):
    """Extract confidence limits with simple boundary truncation detection."""
    valid_samples = param_samples[~np.isnan(param_samples)]
    if len(valid_samples) == 0:
        return np.nan, np.nan

    lolim = np.percentile(valid_samples, lower_percentile)
    uplim = np.percentile(valid_samples, upper_percentile)

    has_lower_bound = hasattr(param, 'min') and param.min is not None
    has_upper_bound = hasattr(param, 'max') and param.max is not None

    at_lower = 0
    at_upper = 0
    if has_lower_bound:
        at_lower = np.sum(valid_samples <= param.min + boundary_epsilon)
    if has_upper_bound:
        at_upper = np.sum(valid_samples >= param.max - boundary_epsilon)

    if at_lower == 0 and at_upper == 0:
        return lolim, uplim

    iqr = np.percentile(valid_samples, 75) - np.percentile(valid_samples, 25)
    tolerance = max(boundary_tolerance * iqr, 1e-10)

    if has_lower_bound and at_lower > 0:
        if (lolim - param.min) < tolerance:
            lolim = np.nan

    if has_upper_bound and at_upper > 0:
        if (param.max - uplim) < tolerance:
            uplim = np.nan

    return lolim, uplim


class BootstrapResult:
    """Detached bootstrap estimates and correlated parameter samples.

    limits() summarizes existing refits at a requested confidence level.
    params is a convenience summary at the default 68% level. samples contains read-only arrays of free-parameter refits; failed
    realizations remain NaN. The original model is never updated. Selecting a
    line from this result propagates these same refits into its analytic flux.
    """
    def __init__(self, model, samples, fit_info=None):
        import copy
        from types import MappingProxyType
        from ..results import _clear_model_estimates
        self._model = copy.deepcopy(model)
        _clear_model_estimates(self._model)
        arrays = {name: np.array(values, dtype=float, copy=True) for name, values in samples.items()}
        expected = Bootstrap._fit_param_names(self._model)
        if list(arrays) != expected:
            raise ValueError('Samples must match the free-parameter names and ordering.')
        if not arrays or any(a.ndim != 1 for a in arrays.values()) or len({len(a) for a in arrays.values()}) != 1:
            raise ValueError('Samples must be equally sized one-dimensional arrays.')
        for values in arrays.values():
            values.setflags(write=False)
        self.samples = MappingProxyType(arrays)
        self.fit_info = MappingProxyType(dict(fit_info or {}))
        # Default for derived line measurements; generation itself has no level.
        self._confidence = 68

    def limits(self, confidence=68, boundary_tolerance=0.05):
        """Summarize existing refits without running new fits.

        Parameters
        ----------
        confidence : float, optional
            Central percentile coverage in percent, strictly between 0 and 100.
        boundary_tolerance : float, optional
            Nonnegative width of the near-bound band as a fraction of the IQR.
            An endpoint is unavailable when at least its excluded-tail fraction
            of valid draws occupies that band. Zero uses numerical tolerance only.

        Returns
        -------
        astropy.table.Table
            Nominal values, sample standard deviations and interval endpoints,
            with units per row and provenance in meta. Bound-dominated endpoints
            are NaN; the opposite endpoint keeps its original percentile.
        """
        import copy
        from astropy.table import Table
        from ..utils import apply_tied
        from .limits import _sample_interval, _parameter_state
        if not np.isfinite(confidence) or not 0 < confidence < 100:
            raise ValueError('confidence must lie strictly between 0 and 100.')
        if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
            raise ValueError('boundary_tolerance must be finite and nonnegative.')
        working = copy.deepcopy(self._model)
        values = {name: [] for name in working.param_names}
        for draw in zip(*self.samples.values()):
            if not np.isfinite(draw).all():
                continue
            self._set_values(working, draw)
            apply_tied(working)
            if not np.isfinite(working.parameters).all():
                continue
            for name in values:
                values[name].append(np.array(getattr(working, name).value, copy=True))
        rows = []
        boundary = {}
        for name, draws in values.items():
            parameter = getattr(self._model, name)
            shape = np.shape(parameter.value)
            std = np.full(shape, np.nan)
            lower, upper = np.full(shape, np.nan), np.full(shape, np.nan)
            if len(draws) >= 2 and not parameter.fixed:
                draws = np.asarray(draws)
                std = np.std(draws, axis=0, ddof=1)
                lower, upper, sides = _sample_interval(draws, parameter.bounds, confidence, boundary_tolerance)
                if sides: boundary[name] = sides
            rows.append((name, np.array(parameter.value, copy=True), std, lower, upper,
                         '' if parameter.unit is None else str(parameter.unit)))
        names = ('name', 'value', 'std', 'lolim', 'uplim', 'unit')
        if any(np.ndim(row[1]) for row in rows):
            table = Table()
            for column, name in enumerate(names):
                column_values = np.empty(len(rows), dtype=object)
                column_values[:] = [row[column] for row in rows]
                table[name] = column_values
        else:
            table = Table(rows=rows, names=names)
        table.meta.update(method='bootstrap', confidence=float(confidence), criterion='central_percentile',
                          boundary_tolerance=float(boundary_tolerance), boundary=boundary,
                          n_success=max(map(len, values.values()), default=0),
                          sampling=dict(self.fit_info), state=_parameter_state(self._model))
        return table

    @property
    def params(self):
        """Convenience default summary (68%); call limits() for explicit choices."""
        from types import MappingProxyType
        from ...metrics import Metric
        return MappingProxyType({row['name']: Metric(row['value'], row['std'], row['lolim'],
                                                     row['uplim'], row['unit'] or None)
                                 for row in self.limits()})

    def save(self, filename, overwrite=False):
        """Save samples and association metadata to NPZ, without duplicating the model."""
        import json
        from .limits import _parameter_state
        metadata = dict(format='prism-bootstrap', version=1, state=_parameter_state(self._model),
                        fit_info=dict(self.fit_info))
        with open(filename, 'wb' if overwrite else 'xb') as stream:
            np.savez_compressed(stream, __prism_bootstrap__=json.dumps(metadata, default=lambda value: value.item()), **self.samples)

    @classmethod
    def load(cls, filename, result):
        """Load samples associated with a FitResult; reject a different parameter state."""
        import json
        from ..results import FitResult
        from .limits import _parameter_state
        if not isinstance(result, FitResult):
            raise TypeError('Supply the associated FitResult when loading bootstrap samples.')
        model = result.model
        with np.load(filename, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['__prism_bootstrap__']))
            if metadata.get('format') != 'prism-bootstrap' or metadata.get('version') != 1:
                raise ValueError('Unsupported bootstrap archive.')
            if metadata['state'] != _parameter_state(model):
                raise ValueError('Bootstrap samples do not match this fitted parameter state.')
            samples = {name: archive[name] for name in archive.files if name != '__prism_bootstrap__'}
        return cls(model, samples, metadata['fit_info'])

    @property
    def model(self):
        """Detached nominal model, without statistical estimates attached."""
        import copy
        return copy.deepcopy(self._model)

    def _set_values(self, model, values):
        for name, value in zip(self.samples, values):
            if '[' in name:
                base, index = name.split('[')
                parameter = getattr(model, base)
                updated = np.array(parameter.value, copy=True)
                updated[int(index.rstrip(']'))] = value
                parameter.value = updated
            else:
                getattr(model, name).value = value

    def show(self, mode='auto'):
        """Display nominal values and bootstrap uncertainties, without mutation."""
        from ...display import show_result
        return show_result(self, mode=mode)


class Bootstrap:
    """Flux Randomization (Empirical Monte Carlo) uncertainty estimator.

    Synthetic datasets are generated by perturbing the observed data with
    the measured uncertainties, not from the fitted model. This preserves
    systematic residuals and is appropriate when the model is phenomenological
    or incomplete (e.g., multi-Gaussian emission-line fitting).

    For each iteration, pixel i is drawn as:

    - ``gauss``:   y_new[i] ~ Normal(y_obs[i], sigma[i])
    - ``uniform``: y_new[i] ~ Uniform(y_obs[i] ± √3·sigma[i])
                   (bounds scaled by √3 so that Var = sigma²,
                    matching the Gaussian case)
    - ``poisson``: y_new[i] ~ Poisson(lam = y_obs[i])
                   (requires raw non-negative integer counts)

    The same effective errors and residual weights are used in every fit.
    Symmetric errors give weights 1/yerr; asymmetric errors use their mean for
    fitting and retain the supplied lower/upper widths for noise generation.

    References
    ----------
    Peterson et al. (1998), PASP 110, 660 — flux randomization for
    emission-line uncertainty estimation.
    """

    def __init__(self, model, fitter=None, x=None, y=None, yerr=None, weights=None,
                 n_samples=1000, statistic='gauss', fitter_kwargs=None,
                 seed=None, noise_dist='uniform',
                 verbose=True, nproc=1,
                 xerr=None, perturb_x=False):
        # Frozen copy: all bootstrap iterations start from this state.
        # The original model is never mutated during the run.
        self._initial_model = model.copy()
        self.model = model
        if fitter is None:
            from ..scipyfit import TRF
            fitter = TRF()
        self.fitter = fitter
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.xerr = None if xerr is None else np.asarray(xerr, dtype=float)
        self.yerr = yerr
        self.weights = weights
        self.n_samples = int(n_samples)
        self.statistic = statistic
        self.fitter_kwargs = {} if fitter_kwargs is None else dict(fitter_kwargs)
        self.seed = seed
        self.verbose = bool(verbose)
        self.nproc = int(nproc)
        self.noise_dist = noise_dist.lower()
        self.perturb_x = bool(perturb_x)

    @staticmethod
    def _fit_param_names(model):
        fit_indices = _free_parameter_indices(model)
        names = []
        cumulative_idx = 0
        for pname in model.param_names:
            param = getattr(model, pname)
            for offset in range(param.size):
                if cumulative_idx in fit_indices:
                    names.append(pname if param.size == 1 else f"{pname}[{offset}]")
                cumulative_idx += 1
        return names

    def run(self):
        """Run bootstrap and return a detached BootstrapResult."""
        if is_batched_result_like(self.model):
            raise NotImplementedError(
                "Resampling for batched spectra/cubes is not implemented yet. "
                "Use ad-hoc single-spectrum fitting for bootstrap uncertainties."
            )

        stat = self.statistic.lower()
        if stat not in {'gauss', 'poisson'}:
            raise ResampleError(f"Unknown statistic '{self.statistic}'. Use 'gauss' or 'poisson'.")

        if self.noise_dist not in {'gauss', 'uniform'}:
            raise ResampleError(f"Unknown noise distribution '{self.noise_dist}'. Use 'gauss' or 'uniform'.")

        if self.n_samples < 1:
            raise ResampleError("n_samples must be >= 1.")

        raw_yerr, fit_yerr, weights = prepare_noise_and_weights(self.y, self.yerr, self.weights, stat)
        # Flux Randomization: perturb observed data with measured uncertainties.
        # The observed spectrum carries the signal; only the noise realization
        # changes across iterations (Peterson et al. 1998, PASP 110, 660).
        y_obs = np.where(weights > 0, self.y, 0.0)
        n_pts = y_obs.size
        _sqrt3 = np.sqrt(3.0)

        if raw_yerr.ndim == 2 and self.noise_dist == 'gauss':
            raise ResampleError(
                "Asymmetric yerr is ambiguous for Gaussian resampling. "
                "Use noise_dist='uniform' or provide symmetric yerr."
            )

        if self.perturb_x:
            if self.xerr is None:
                raise ResampleError('perturb_x=True requires explicit xerr.')
            if self.xerr.ndim != 1:
                raise ResampleError('Only symmetric xerr is supported when perturb_x=True.')
            if self.xerr.shape != self.x.shape:
                raise ResampleError('xerr must match x shape when perturb_x=True.')
            if np.any(~np.isfinite(self.xerr)) or np.any(self.xerr < 0):
                raise ResampleError('xerr must be finite and non-negative.')

        rng = np.random.default_rng(self.seed)
        if stat == 'gauss':
            if self.noise_dist == 'uniform':
                # ±√3·σ gives Var = σ², matching the Gaussian case.
                if raw_yerr.ndim == 2:
                    y_synth = rng.uniform(
                        y_obs - raw_yerr[0],
                        y_obs + raw_yerr[1],
                        size=(self.n_samples, n_pts),
                    )
                else:
                    y_synth = rng.uniform(
                        y_obs - _sqrt3 * raw_yerr,
                        y_obs + _sqrt3 * raw_yerr,
                        size=(self.n_samples, n_pts),
                    )
            else:
                y_synth = rng.normal(y_obs, raw_yerr, size=(self.n_samples, n_pts))
            fit_stat = 'chi2'
        else:
            # Poisson: λ = observed counts. y_obs must be raw non-negative counts.
            lam = np.maximum(y_obs, 0.0)
            y_synth = rng.poisson(lam, size=(self.n_samples, n_pts)).astype(float)
            # Poisson draws are refitted with the same fixed-error chi-square
            # approximation; this does not implement a Poisson likelihood.
            fit_stat = 'chi2'

        if self.perturb_x:
            if self.noise_dist == 'uniform':
                x_synth = rng.uniform(
                    self.x - _sqrt3 * self.xerr,
                    self.x + _sqrt3 * self.xerr,
                    size=(self.n_samples, n_pts),
                )
            else:
                x_synth = rng.normal(self.x, self.xerr, size=(self.n_samples, n_pts))
        else:
            x_synth = None

        yerr_batch = np.broadcast_to(fit_yerr, (self.n_samples, n_pts))
        weights_batch = np.broadcast_to(weights, (self.n_samples, n_pts))

        param_names = self._fit_param_names(self._initial_model)
        samples = {name: np.full(self.n_samples, np.nan, dtype=float) for name in param_names}

        n_success = 0
        max_warn = 5
        warned = 0
        if x_synth is None:
            result = self.fitter.multifit(
                model=self._initial_model.copy(),
                x=self.x,
                y=y_synth,
                yerr=yerr_batch,
                weights=weights_batch,
                statistic=fit_stat,
                nproc=self.nproc,
                progress=True,
                **self.fitter_kwargs,
            )

            for i in range(self.n_samples):
                if not bool(result.success.flat[i]):
                    if self.verbose and warned < max_warn:
                        warnings.warn(
                            f"Bootstrap iteration {i} failed: {result.messages.flat[i]}",
                            RuntimeWarning,
                        )
                        warned += 1
                    continue

                fitted_model = result.get_model(i)
                fitted_params = fitted_model.parameters[_free_parameter_indices(fitted_model)]
                for name, value in zip(param_names, fitted_params):
                    samples[name][i] = value
                n_success += 1

        else:
            for i in range(self.n_samples):
                try:
                    fit = getattr(self.fitter, '_fit_single', self.fitter)
                    fitted_model = fit(
                        model=self._initial_model.copy(),
                        x=x_synth[i],
                        y=y_synth[i],
                        yerr=fit_yerr,
                        weights=weights,
                        statistic=fit_stat,
                        inplace=False,
                        **self.fitter_kwargs,
                    )
                    if not self.fitter.fit_info.get('success', False):
                        raise RuntimeError(self.fitter.fit_info.get('message', 'Fit did not converge.'))
                except Exception as exc:
                    if self.verbose and warned < max_warn:
                        warnings.warn(f'Bootstrap iteration {i} failed: {exc}', RuntimeWarning)
                        warned += 1
                    continue

                fitted_params = fitted_model.parameters[_free_parameter_indices(fitted_model)]
                for name, value in zip(param_names, fitted_params):
                    samples[name][i] = value
                n_success += 1

        # Post-run diagnostic: warn if all samples are identical (degenerate)
        if n_success > 1:
            vals = np.column_stack(
                [samples[name] for name in param_names])
            valid_mask = np.all(np.isfinite(vals), axis=1)
            valid_vals = vals[valid_mask]
            if valid_vals.shape[0] > 1:
                spread = np.ptp(valid_vals, axis=0)
                if np.all(spread == 0.0):
                    warnings.warn(
                        "All bootstrap parameter samples are identical. "
                        "The fitter converged to the same solution for every "
                        "noise realization. Uncertainties are unreliable. "
                        "Possible causes: the fitter is stuck at a hard "
                        "minimum, tolerance is too loose, or parameter "
                        "bounds are too tight.",
                        RuntimeWarning,
                    )

        success_rate = n_success / self.n_samples
        if success_rate < 0.9:
            warnings.warn(
                f"Only {success_rate * 100:.1f}% of bootstrap iterations succeeded. "
                "Results may be unreliable.",
                RuntimeWarning,
            )

        if self.verbose:
            print(f"✓ Bootstrap complete: {n_success}/{self.n_samples} successful")

        return BootstrapResult(self._initial_model, samples,
            dict(n_samples=self.n_samples, n_success=n_success, seed=self.seed,
                 statistic=self.statistic, noise_dist=self.noise_dist, perturb_x=self.perturb_x))


def bootstrap(model, fitter=None, x=None, y=None, yerr=None, weights=None,
              n_samples=1000, statistic='gauss', fitter_kwargs=None, seed=None,
              verbose=False, nproc=1,
              noise_dist='gauss',
              xerr=None, perturb_x=False):
    """Estimate parameter uncertainties via Flux Randomization (Empirical Monte Carlo).

    Generates n_samples synthetic spectra by perturbing the observed flux
    with the measured uncertainties (flux randomization, Peterson et al. 1998,
    PASP 110, 660). The fitted model is refitted to each realization; the
    resulting parameter distributions give the uncertainties.

    Parameters
    ----------
    model : native or adapted model
        Fitted model whose parameters carry the starting values.
    fitter : Fitter, optional
        Fitter for each iteration. Defaults to ``TRF()``, which uses analytic
        derivatives when available and otherwise finite differences. Every
        realization starts from the same supplied model state.
    x, y, yerr : array-like
        Observed coordinates, values, and 1-sigma uncertainties.
    n_samples : int
        Number of Monte Carlo iterations.
    statistic : str
        ``'gauss'`` (default) for chi-squared fitting with Gaussian/uniform
        noise; ``'poisson'`` for raw-count data.
    noise_dist : str
        ``'gauss'`` (default) — perturb each pixel as Normal(y_obs, sigma).
        ``'uniform'`` — perturb as Uniform(y_obs ± √3·sigma), which has the
        same variance as the Gaussian case.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Nominal model, parameter estimates and correlated refit samples.
    """
    return Bootstrap(
        model=model,
        fitter=fitter,
        x=x,
        y=y,
        yerr=yerr,
        weights=weights,
        n_samples=n_samples,
        statistic=statistic,
        fitter_kwargs=fitter_kwargs,
        seed=seed,
        verbose=verbose,
        nproc=nproc,
        noise_dist=noise_dist,
        xerr=xerr,
        perturb_x=perturb_x,
    ).run()
