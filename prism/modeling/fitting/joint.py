"""Joint least squares: observation blocks over one unique parameter graph."""
import copy
from collections.abc import Mapping
import numpy as np

from ..composition import Model
from ..parameters import _Expression
from .data import Dataset, _prepare_fit_data
from .constraints import _check_fitted_bounds
from .extension import _validate_spectral_inputs, _clear_fit_uncertainties
from .problem import _FitBlock
from .utils import (_ParameterMap, _apply_tied_fast, _get_tied_info,
                    _analytic_jacobian_parameter_major, _reduce_tied_analytic_jacobian)


class _ParameterGraph:
    """Unique scalar nodes and callable contexts, without a source expression."""
    col_fit_deriv = True

    def __init__(self, associations):
        self._parameters = {}
        self._contexts = {}
        seen = set()
        for label, (source, data) in associations.items():
            owners = [(source, '')]
            if data.response is not None:
                owners.append((data.response, 'response.'))
            for owner, prefix in owners:
                for local in owner.param_names:
                    parameter = getattr(owner, local)
                    if id(parameter) in seen:
                        continue
                    name = f'{label}:{prefix}{local}'
                    seen.add(id(parameter))
                    self._parameters[name] = parameter
                    self._contexts[name] = owner
        self.param_names = tuple(self._parameters)

    def __getattr__(self, name):
        parameters = self.__dict__.get('_parameters', {})
        if name in parameters:
            return parameters[name]
        raise AttributeError(name)

    @property
    def indices(self):
        return {id(parameter): index for index, parameter in enumerate(self._parameters.values())}

    @property
    def parameters(self):
        return np.array([p.value for p in self._parameters.values()], dtype=float)

    @parameters.setter
    def parameters(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.param_names),):
            raise ValueError('Parameter vector does not match the joint graph.')
        for parameter, value in zip(self._parameters.values(), values):
            parameter.value = value

    @property
    def tied(self):
        # Opaque callables keep the model context in which they were declared.
        # Declarative expressions instead expose the complete global dependency.
        ties = {}
        for name, parameter in self._parameters.items():
            function = parameter.tied
            if function and not isinstance(function, _Expression):
                context = self._contexts[name]
                function = lambda graph, function=function, context=context: function(context)
            ties[name] = function
        return ties

    @property
    def fixed(self):
        return {name: parameter.fixed for name, parameter in self._parameters.items()}

    @property
    def bounds(self):
        return {name: parameter.bounds for name, parameter in self._parameters.items()}

    @property
    def has_tied(self):
        return any(p.tied for p in self._parameters.values())


class _JointProblem:
    """Sum selected residuals while preserving each observation's own grid."""

    def __init__(self, associations, *, statistic='chi2', filter_non_finite=False):
        if not isinstance(associations, Mapping) or not associations:
            raise TypeError('jointfit requires a nonempty named mapping of (model, Dataset) pairs.')
        sources, responses, inputs = {}, {}, {}
        adapted = {}
        dataset_ids = set()
        for label, pair in associations.items():
            if not isinstance(label, str) or not label:
                raise TypeError('Joint observation labels must be nonempty strings.')
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not isinstance(pair[1], Dataset):
                raise TypeError(f'{label!r} must contain a (model, Dataset) pair.')
            source, data = pair
            if id(data) in dataset_ids:
                raise ValueError('The same Dataset appears more than once; combine its source components instead.')
            dataset_ids.add(id(data))
            if id(source) not in adapted:
                adapted[id(source)] = Model.from_astropy(source)
                adapted[id(source)]._spectrum()
            sources[label] = adapted[id(source)]
            responses[label] = data.response
            inputs[label] = data

        # Copy all model/response roots together: sharing and cross-model links
        # survive, while user arrays remain referenced during optimization.
        sources, responses = copy.deepcopy((sources, responses))
        self.associations = {}
        for label, data in inputs.items():
            local = copy.copy(data)
            local.response = responses[label]
            self.associations[label] = (sources[label], local)

        self.model = _ParameterGraph(self.associations)
        self.parameters = _ParameterMap(self.model)
        self.parameters.tied = _get_tied_info(self.model)
        self.cache = self.parameters.initial.copy()
        self.blocks, self.sections = [], {}
        self.ndata = 0
        self.weighted = False
        global_indices = self.model.indices
        for label, (source, data) in self.associations.items():
            observed = data._observed_model(source)
            _validate_spectral_inputs(observed, data.x, data.y,
                                     yerr=data.yerr, weights=data.weights, statistic=statistic)
            x, y, errors, weights, valid = _prepare_fit_data(
                data.x, data.y, data.yerr, data.weights, data.mask,
                filter_non_finite=filter_non_finite)
            data.mask = valid.copy()
            if weights is None and errors is not None:
                weights = np.zeros_like(errors, dtype=float)
                np.divide(1., errors, out=weights, where=valid)
            block = _FitBlock(observed, x, y, weights, valid)
            indices = np.array([global_indices[id(getattr(observed, name))]
                                for name in observed.param_names], dtype=int)
            self.sections[label] = slice(self.ndata, self.ndata+block.ndata)
            self.ndata += block.ndata
            self.weighted |= weights is not None
            self.blocks.append((label, block, indices))
            for node in observed._expressions():
                _clear_fit_uncertainties(node)
                node._evaluation_parent = self.model

        self.evaluation = self

    def update(self, values):
        self.cache[self.parameters.free] = values
        self.model.parameters = self.cache
        if self.parameters.tied:
            _apply_tied_fast(self.model, self.parameters.tied, self.cache)
        return self.cache

    def residual(self, values):
        parameters = self.update(values)
        result = np.empty(self.ndata)
        for label, block, indices in self.blocks:
            residual = block.evaluation(parameters[indices])[block.select]-block.y
            if block.weights is not None:
                residual = residual*block.weights
            result[self.sections[label]] = residual
        return result

    def fit_deriv(self, parameters):
        """Scatter raw local derivatives into unique global parameter rows."""
        full = np.zeros((len(self.model.param_names), self.ndata))
        for label, block, indices in self.blocks:
            local = _analytic_jacobian_parameter_major(
                block.model, block.evaluation.fit_deriv(parameters[indices]))
            full[indices, self.sections[label]] = local[:, block.select]
        return full

    def jacobian(self, values):
        parameters = self.update(values)
        full = _reduce_tied_analytic_jacobian(self.model, self.fit_deriv(parameters))
        result = full[self.parameters.free].T
        for label, block, _ in self.blocks:
            if block.weights is not None:
                result[self.sections[label]] *= block.weights[:, None]
        if not np.isfinite(result).all():
            raise ValueError('Invalid derivatives encountered')
        return result

    def finish(self, values):
        self.update(values)
        _check_fitted_bounds(self.model)
        return self.model


class _JointFitMixin:
    def jointfit(self, associations, *, statistic='chi2', filter_non_finite=False, **kwargs):
        """Fit named (model, Dataset) pairs in a single least-squares problem.

        Reusing a model or component shares its parameter nodes. Distinct
        models remain independent unless linked. Grids, masks and responses
        belong to each Dataset. The complete model graph is copied once before
        optimization; the caller's models and responses are not fitted in place.

        Returns a FitResult with a global covariance and per-observation views.
        Residuals are assumed statistically independent between datasets. Linked
        bounds are compiled exactly to free-node intervals when affine in one
        free scalar. Unsupported bounded links raise before optimization; raw
        expressions and bounds remain available to future constrained solvers.
        """
        from .results import FitResult
        if 'inplace' in kwargs:
            raise TypeError('jointfit copies the complete graph; inplace is not a joint-fit option.')
        problem = _JointProblem(associations, statistic=statistic,
                                filter_non_finite=filter_non_finite)
        self._fit_problem(problem, statistic=statistic, weighted=problem.weighted, **kwargs)
        residual = problem.residual(problem.model.parameters[problem.parameters.free])
        contributions = {label: {'stat': float(residual[section]@residual[section]),
                                 'ndata': section.stop-section.start}
                         for label, section in problem.sections.items()}
        return FitResult(problem.associations, problem.model, self.fit_info, contributions)
