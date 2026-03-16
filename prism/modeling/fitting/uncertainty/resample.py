"""Resampling-based parameter uncertainty estimation."""

import warnings
import numpy as np

__all__ = [
    'Bootstrap',
    'bootstrap',
    'attach',
    'detach',
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
    """
    Normalize yerr/weights inputs and return both arrays.
    - If only weights are provided, derive yerr for noise generation.
    - If only yerr is provided, compute chi-squared style weights.
    - If neither is provided and statistic is Poisson, synthesize yerr/weights.
    """
    stat = statistic.lower()

    if yerr is None:
        if weights is not None:
            weights = np.asarray(weights)
            if np.any(~np.isfinite(weights)):
                raise ResampleError("weights contains non-finite values.")
            safe_weights = np.maximum(weights, 1e-20)
            yerr = np.sqrt(1.0 / safe_weights)
        elif stat == 'poisson':
            base = np.maximum(np.abs(y), 1)
            yerr = np.sqrt(base)
        else:
            raise ResampleError(
                "yerr or weights required for Gaussian bootstrap. "
                "Provide data uncertainties or weights for noise generation."
            )

    yerr = np.asarray(yerr)
    yerr = np.maximum(yerr, 1e-10)
    if np.any(~np.isfinite(yerr)):
        raise ResampleError("yerr contains non-finite values.")

    if weights is None:
        if stat == 'poisson':
            weights = 1.0 / np.maximum(np.abs(y), 1e-10)
        else:
            weights = 1.0 / (yerr ** 2)
    else:
        weights = np.asarray(weights)
        if np.any(~np.isfinite(weights)):
            raise ResampleError("weights contains non-finite values.")
        weights = np.maximum(weights, 1e-20)

    return yerr, weights


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


class Bootstrap:
    """Parametric bootstrap uncertainty estimator using MultiFit batch refits."""

    def __init__(self, model, fitter, x, y, yerr=None, weights=None,
                 n_samples=1000, statistic='gauss', fitter_kwargs=None,
                 seed=None, verbose=True, nproc=1, batch=False):
        self.model = model
        self.fitter = fitter
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.yerr = yerr
        self.weights = weights
        self.n_samples = int(n_samples)
        self.statistic = statistic
        self.fitter_kwargs = {} if fitter_kwargs is None else dict(fitter_kwargs)
        self.seed = seed
        self.verbose = bool(verbose)
        self.nproc = int(nproc)
        self.batch = bool(batch)

    @staticmethod
    def _fit_param_names(model):
        from astropy.modeling.fitting import model_to_fit_params

        _, fit_indices, _ = model_to_fit_params(model)
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
        """Run bootstrap and return parameter sample arrays."""
        from astropy.modeling.fitting import model_to_fit_params

        if is_batched_result_like(self.model):
            raise NotImplementedError(
                "Resampling for batched spectra/cubes is not implemented yet. "
                "Use ad-hoc single-spectrum fitting for bootstrap uncertainties."
            )

        stat = self.statistic.lower()
        if stat not in {'gauss', 'poisson'}:
            raise ResampleError(f"Unknown statistic '{self.statistic}'. Use 'gauss' or 'poisson'.")

        if self.n_samples < 1:
            raise ResampleError("n_samples must be >= 1.")

        yerr, weights = prepare_noise_and_weights(self.y, self.yerr, self.weights, stat)
        y_model = np.asarray(self.model(self.x), dtype=float)

        rng = np.random.default_rng(self.seed)
        if stat == 'gauss':
            noise = rng.normal(0.0, yerr, size=(self.n_samples, y_model.size))
            y_synth = y_model[np.newaxis, :] + noise
            fit_stat = 'chi2'
        else:
            lam = np.maximum(y_model, 0.0)
            y_synth = rng.poisson(lam, size=(self.n_samples, y_model.size)).astype(float)
            fit_stat = 'poisson'

        yerr_batch = np.broadcast_to(yerr, (self.n_samples, y_model.size))
        weights_batch = np.broadcast_to(weights, (self.n_samples, y_model.size))

        result = self.fitter.multifit(
            model=self.model.copy(),
            x=self.x,
            y=y_synth,
            yerr=yerr_batch,
            weights=weights_batch,
            statistic=fit_stat,
            nproc=self.nproc,
            progress=self.verbose,
            batch=self.batch,
            **self.fitter_kwargs,
        )

        param_names = self._fit_param_names(self.model)
        samples = {name: np.full(self.n_samples, np.nan, dtype=float) for name in param_names}

        n_success = 0
        max_warn = 5
        warned = 0
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
            fitted_params, _, _ = model_to_fit_params(fitted_model)
            for name, value in zip(param_names, fitted_params):
                samples[name][i] = value
            n_success += 1

        success_rate = n_success / self.n_samples
        if success_rate < 0.5:
            warnings.warn(
                f"Only {success_rate * 100:.1f}% of bootstrap iterations succeeded. "
                "Results may be unreliable.",
                RuntimeWarning,
            )

        if self.verbose:
            print(f"✓ Bootstrap complete: {n_success}/{self.n_samples} successful")

        return samples

    @staticmethod
    def attach(model, samples, confidence=68, percentiles=None, verbose=False, set_values=True):
        """Attach bootstrap uncertainties to model parameters in-place."""
        if is_batched_result_like(model):
            raise NotImplementedError(
                "Attaching resampling limits to batched spectra/cubes is not implemented yet. "
                "Use ad-hoc single-spectrum fitting for uncertainty attachment."
            )

        if percentiles is None:
            percentiles = [16, 50, 84]
        if 50 not in percentiles:
            percentiles = list(percentiles) + [50]

        alpha = (100 - confidence) / 2
        lower_percentile = alpha
        upper_percentile = 100 - alpha

        for param_name, param_samples in samples.items():
            if '[' in param_name:
                base_name = param_name.split('[')[0]
                idx = int(param_name.split('[')[1].rstrip(']'))
                param = getattr(model, base_name)

                if not hasattr(param, 'std') or param.std is None or np.isscalar(param.std):
                    param.std = np.full(param.size, np.nan)
                    param.median = np.full(param.size, np.nan)
                    param.lolim = np.full(param.size, np.nan)
                    param.uplim = np.full(param.size, np.nan)
                    param.percentiles = {}

                valid_samples = param_samples[~np.isnan(param_samples)]
                if len(valid_samples) == 0:
                    continue

                param.std[idx] = np.std(valid_samples)
                param.median[idx] = np.percentile(valid_samples, 50)
                lolim, uplim = extract_limits(param, param_samples, lower_percentile, upper_percentile)
                param.lolim[idx] = lolim
                param.uplim[idx] = uplim

                for p in percentiles:
                    if p not in param.percentiles:
                        param.percentiles[p] = np.full(param.size, np.nan)
                    param.percentiles[p][idx] = np.percentile(valid_samples, p)

                if set_values and np.isfinite(param.median[idx]):
                    updated = np.array(param.value, copy=True)
                    updated.flat[idx] = param.median[idx]
                    param.value = updated
                continue

            param = getattr(model, param_name)
            valid_samples = param_samples[~np.isnan(param_samples)]
            if len(valid_samples) == 0:
                continue

            param.std = np.std(valid_samples)
            param.median = np.percentile(valid_samples, 50)
            lolim, uplim = extract_limits(param, param_samples, lower_percentile, upper_percentile)
            param.lolim = lolim
            param.uplim = uplim
            param.percentiles = {p: np.percentile(valid_samples, p) for p in percentiles}

            if set_values and np.isfinite(param.median):
                param.value = param.median

            if verbose:
                print(
                    f"  {param_name}: {param.value:.6g} ± {param.std:.6g} "
                    f"(median: {param.median:.6g}) [{param.lolim:.6g}, {param.uplim:.6g}]"
                )

        if verbose:
            print(f"✓ Attached uncertainties with {confidence}% confidence intervals")

    @staticmethod
    def detach(model):
        """Remove bootstrap uncertainty attributes from model parameters."""
        for param_name in model.param_names:
            param = getattr(model, param_name)
            for attr in ['std', 'median', 'lolim', 'uplim', 'percentiles', 'samples']:
                if not hasattr(param, attr):
                    continue
                try:
                    delattr(param, attr)
                except (AttributeError, TypeError):
                    setattr(param, attr, None)


def bootstrap(model, fitter, x, y, yerr=None, weights=None, n_samples=1000,
             statistic='gauss', fitter_kwargs=None, seed=None,
             verbose=True, nproc=1, batch=False):
    """Compatibility wrapper around Bootstrap(...).run()."""
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
        batch=batch,
    ).run()


def attach(model, samples, confidence=68, percentiles=None, verbose=False, set_values=True):
    """Compatibility wrapper around Bootstrap.attach(...)."""
    Bootstrap.attach(
        model=model,
        samples=samples,
        confidence=confidence,
        percentiles=percentiles,
        verbose=verbose,
        set_values=set_values,
    )


def detach(model):
    """Compatibility wrapper around Bootstrap.detach(...)."""
    Bootstrap.detach(model)
