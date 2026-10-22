"""Fit snapshots, with separate source and observed-space predictions."""
import copy
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

from .extension import _clear_fit_uncertainties
from .utils import _free_parameter_indices


def _estimate_value(value):
    """Freeze scalar or array estimates without discarding parameter shape."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class _FitParameter:
    name: str
    value: object
    std: object
    bounds: tuple
    fixed: bool
    tied: bool
    unit: object


def _clear_model_estimates(model):
    """Remove fit metadata from a materialized model and its physical leaves."""
    graph = getattr(model, '_evaluation_parent', None)
    if graph is not None:
        _clear_fit_uncertainties(graph)
    for node in getattr(model, '_expressions', lambda: (model,))():
        _clear_fit_uncertainties(node)
    for component in getattr(model, '_components', ()):
        _clear_fit_uncertainties(component)


class _DatasetFitView:
    """One observation in an immutable, globally fitted parameter graph."""
    def __init__(self, parent, label):
        self._parent = parent
        self._label = label

    @property
    def model(self):
        """Independent source model for evaluation or a new fit.

        The copy retains the complete parameter context for evaluating external
        links. Its uncertainty attributes are cleared: the joint covariance and
        fitted parameter estimates belong to FitResult, not a standalone model.
        """
        self._parent._require_fitted()
        model = copy.deepcopy(self._parent._associations[self._label][0])
        _clear_model_estimates(model)
        return model

    @property
    def dataset(self):
        """Independent copy of this observation and its fitted response."""
        data = copy.deepcopy(self._parent._associations[self._label][1])
        if data.response is not None:
            _clear_model_estimates(data.response)
        return data

    @property
    def params(self):
        """Read-only fitted source parameter estimates in local model order."""
        model = self._parent._associations[self._label][0]
        return MappingProxyType({name: self._parent._by_identity[id(getattr(model, name))]
                                 for name in model.param_names})

    @property
    def fit_info(self):
        """Local statistic and sample count; degrees of freedom are global."""
        return self._parent._contributions[self._label]

    def predict(self, x=None):
        """Predict on x, or the original grid, through the fitted response."""
        self._parent._require_fitted()
        model, data = self._parent._associations[self._label]
        return data.predict(model, x=x)


class FitResult:
    """Snapshot of one single or joint fit, with source and observation views.

    Access observations by their labels: ``result['blue'].model`` evaluates the
    source, ``.dataset`` supplies data and its response, and ``.predict()`` gives
    the observed-space prediction. Source models and datasets are detached copies
    for inspection or editing; changing them cannot invalidate this fit snapshot.

    ``params`` contains unique global parameter estimates. Per-observation params
    use local source names and refer to the same estimate for shared nodes.
    ``param_names`` orders all nodes; ``fit_indices`` selects the free nodes that
    label rows and columns of ``covariance``. Linked-target standard deviations
    are not inferred from marginal errors. Derived measurements require the full
    global covariance, including correlations between observations.

    ``fit_info`` contains global termination and fit diagnostics. Per-observation
    views expose their statistic contributions and sample counts, without local
    degrees of freedom that would double-count shared parameters.

    Notes
    -----
    A single fit exposes model, dataset and predict(x=None) directly. A joint
    fit requires an observation label, even when it has only one observation.
    Explicit source pipes are retained; a Dataset response remains separate.
    Named multifit failures retain missing estimates and a retry recipe in
    associations. model, predict and save raise when no fit was obtained;
    a completed optimization with success=False retains its last finite model.
    """

    def __init__(self, associations, graph, fit_info, contributions, *, _single=False, _failed=False):
        self._single = _single
        self._failed = _failed
        self._graph, self._associations = copy.deepcopy((graph, associations))
        self.param_names = self._graph.param_names
        self._by_identity = {}
        estimates = {}
        for name in self.param_names:
            parameter = getattr(self._graph, name)
            std = getattr(parameter, 'std', None)
            estimate = _FitParameter(name, _estimate_value(np.nan if _failed else parameter.value),
                None if _failed or std is None else _estimate_value(std), tuple(parameter.bounds),
                bool(parameter.fixed), bool(parameter.tied), parameter.unit)
            estimates[name] = estimate
            self._by_identity[id(parameter)] = estimate
        self.params = MappingProxyType(estimates)
        self.parameters = np.full_like(self._graph.parameters, np.nan) if _failed else self._graph.parameters.copy()
        self.parameters.setflags(write=False)
        self.fit_indices = _free_parameter_indices(self._graph)
        self.fit_indices.setflags(write=False)
        covariance = None if _failed else getattr(self._graph, '_param_cov', None)
        self.covariance = None if covariance is None else np.array(covariance, copy=True)
        if self.covariance is not None:
            self.covariance.setflags(write=False)
        info = copy.deepcopy(fit_info)
        info['param_cov'] = self.covariance
        info['nparam'] = len(self.param_names)
        self.fit_info = MappingProxyType(info)
        self._contributions = {label: MappingProxyType(dict(info))
                               for label, info in contributions.items()}

    def _require_fitted(self):
        if self._failed:
            raise ValueError('No fitted model is available; inspect fit_info or retry from associations.')

    def __setstate__(self, state):
        # Object identities change on a worker transfer. Rebuild the index used
        # by per-observation params and retain the snapshot's read-only arrays.
        self.__dict__.update(state)
        self._by_identity = {id(getattr(self._graph, name)): self.params[name]
                             for name in self.param_names}
        self.parameters.setflags(write=False)
        self.fit_indices.setflags(write=False)
        if self.covariance is not None:
            self.covariance.setflags(write=False)
        for parameter in self.params.values():
            for value in (parameter.value, parameter.std):
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)

    @property
    def associations(self):
        """Editable observation mapping copied as one complete parameter graph.

        Shared components, response parameters and cross-observation links retain
        their identities within this copy. Single fits use the label 'data'.
        Estimates are cleared; editing this mapping never changes the snapshot.
        Refit a joint mapping with fitter.jointfit(associations), or unpack the
        single pair and use fitter(model, dataset).
        """
        associations = copy.deepcopy(self._associations)
        for model, data in associations.values():
            _clear_model_estimates(model)
            if data.response is not None:
                _clear_model_estimates(data.response)
        return associations

    def show(self, mode='auto'):
        """Display fitted parameter estimates and global fit diagnostics."""
        from ..display import show_result
        return show_result(self, mode=mode)

    def _repr_html_(self):
        return self.show(mode='html')

    def save(self, filename, overwrite=False):
        """Save models, data, response resources, links and fit estimates to FITS.

        Custom function definitions are supplied explicitly when loading. Opaque
        callable links and operators without reconstruction rules are rejected;
        no Python code or pickle is stored. Fixed matrices use verified relative
        sidecar files, as in model-only sessions. Fitted-session persistence
        currently requires scalar parameters.
        """
        self._require_fitted()
        from .session import _save_result
        _save_result(self, filename, overwrite=overwrite)

    @classmethod
    def load(cls, filename, *, components=None):
        """Load a fitted snapshot, supplying named custom Component definitions.

        The FITS session and any referenced response sidecars must be available.
        Definitions supply functions/derivatives; fitted values, constraints and
        declarative links are restored from the session. Optimizer runtime
        objects are not persisted; portable fit diagnostics are retained.
        """
        from .session import _load_result
        return _load_result(filename, components=components)

    def _single_view(self):
        if not self._single:
            raise AttributeError('Select a joint observation by label before accessing model, dataset or predict().')
        return self['data']

    @property
    def model(self):
        """Detached fitted source; explicit pipes remain part of the model."""
        return self._single_view().model

    @property
    def dataset(self):
        """Detached observation and fitted response for a single fit."""
        return self._single_view().dataset

    def predict(self, x=None):
        """Predict a single observation on x, or its original grid."""
        return self._single_view().predict(x=x)

    def __setattr__(self, name, value):
        if not name.startswith('_') and name in self.__dict__:
            raise AttributeError('FitResult attributes are read-only snapshots.')
        object.__setattr__(self, name, value)

    def __getitem__(self, label):
        if label not in self._associations:
            raise KeyError(label)
        return _DatasetFitView(self, label)

    def __iter__(self):
        return iter(self._associations)

    def __len__(self):
        return len(self._associations)

    def __repr__(self):
        observations = '' if self._single else f'observations={tuple(self)}, '
        return (f'FitResult({observations}'
                f'nfree={self.fit_info["nfree"]}, success={self.fit_info["success"]})')
