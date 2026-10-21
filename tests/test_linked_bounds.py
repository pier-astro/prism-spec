"""Exact interval propagation preserves declared links and the data statistic."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from astropy import units as u
from astropy.modeling import Fittable1DModel, Parameter
from scipy.sparse import eye

from prism.modeling import Component, LinearOperatorModel
from prism.modeling.parameters import _Parameter
from prism.modeling.fitting import Dataset, FitResult, TRF, SherpaLM, HAS_SHERPA
from prism.modeling.fitting.constraints import _box_limits, _box_bounds
from prism.modeling.fitting.utils import _ParameterMap
from prism.modeling.fitting.problem import _FitProblem

ENGINES = [TRF, pytest.param(SherpaLM, marks=pytest.mark.skipif(
    not HAS_SHERPA, reason='Sherpa unavailable'))]


def trend(x, a, b):
    return a+b*x


def derivative(x, a, b):
    return np.array([np.ones_like(x), x])


def source():
    model = Component(trend, dict(a=2., b=4.), derivative=derivative, name='trend')
    model.b.link = 2*model.a
    model.a.bounds = (1., 8.)
    model.b.bounds = (0., 10.)
    return model


@pytest.mark.parametrize('engine', ENGINES)
def test_linked_upper_bound_changes_driver_minimum_without_changing_statistic(engine, tmp_path):
    model = source()
    x = np.linspace(0., 2., 51)
    profile = 1+2*x
    y, error = 7*profile, np.full(x.size, .2)
    result = engine(covariance=True)(model, x, y, yerr=error)
    assert result.fit_info['success']
    assert result.model.a.value == pytest.approx(5., abs=2e-6)
    assert result.model.b.value == pytest.approx(10., abs=4e-6)
    assert result.model.b.value == 2*result.model.a.value
    assert result.fit_info['stat'] == pytest.approx(np.sum(((5*profile-y)/error)**2), rel=2e-6)
    assert result.fit_info['nfree'] == 1
    assert result.fit_info['dof'] == x.size-1
    assert result.covariance.shape == (1, 1)
    assert_array_equal(model.parameters, [2., 4.])
    # Declared bounds are retained; the solver's intersection is not substituted
    # into the user's physical model or serialized as a different model recipe.
    assert result.params['a'].bounds == (1., 8.)
    assert result.params['b'].bounds == (0., 10.)
    path = tmp_path/'linked.fits'
    result.save(path)
    loaded = FitResult.load(path, components={'trend': model})
    assert_allclose(loaded.parameters, result.parameters, rtol=0., atol=0.)
    recipe, data = loaded.associations['data']
    repeated = engine()(recipe, data)
    assert_allclose(repeated.parameters, result.parameters, atol=2e-6)


@pytest.mark.parametrize('expression, bounds, expected', [
    (lambda m: m.a, (2., 4.), (2., 4.)),
    (lambda m: 2*m.a+1, (3., 9.), (1., 4.)),
    (lambda m: -3*m.a+9, (0., 6.), (1., 3.)),
    (lambda m: m.a/2, (None, 2.), (1., 4.)),
    (lambda m: -m.a, (-4., None), (1., 4.)),
    (lambda m: m.a**1, (1., 5.), (1., 5.)),
    (lambda m: m.a+m.a-m.a, (1., 5.), (1., 5.)),
    (lambda m: m.a*(2*u.dimensionless_unscaled), (2., 8.), (1., 4.)),
])
def test_exact_affine_reduction(expression, bounds, expected):
    model = source()
    model.b.link = expression(model)
    model.b.bounds = bounds
    parameters = _ParameterMap(model)
    declared = parameters.bounds.copy()
    assert_allclose(_box_bounds(model, parameters), [expected])
    assert_allclose(parameters.bounds, declared)


def three():
    return Component(lambda x, a, b, c: a+b*x+c*x*x,
                     dict(a=2., b=6., c=4.))


def test_multiple_targets_and_link_chains_intersect_at_the_free_root():
    model = three()
    model.a.bounds = (0., 10.)
    model.b.link = 10-2*model.a
    model.b.bounds = (1., 9.)
    model.c.link = .5*model.b+1
    model.c.bounds = (2., 4.)
    assert_allclose(_box_bounds(model, _ParameterMap(model)), [[2., 4.]])


@pytest.mark.parametrize('constant', [3., 0.])
def test_constant_links_are_validated_without_adding_free_nodes(constant):
    model = source()
    model.b.link = constant
    model.b.bounds = (0., 4.)
    assert_allclose(_box_bounds(model, _ParameterMap(model)), [[1., 8.]])
    model.b.bounds = (5., 6.)
    with pytest.raises(ValueError, match='Constant linked parameter'):
        _box_bounds(model, _ParameterMap(model))


def test_fixed_parameters_in_affine_coefficients_and_array_preflight():
    model = three()
    model.b.fixed = True
    model.c.link = model.a*model.b+1
    model.a.bounds = (-10., 10.)
    model.c.bounds = (-1., 5.)
    limits = _box_limits(model, initial={'a': np.array([-1., 2., 1.]),
                                       'b': np.array([-2., 0., 2.])})
    assert_allclose(limits['a'][0], [-2., -10., -1.])
    assert_allclose(limits['a'][1], [1., 10., 2.])
    assert_array_equal(model.parameters, [2., 6., 13.])
    with pytest.raises(ValueError, match='Constant linked parameter'):
        _box_limits(model, initial={'b': np.array([0., 0.])}, bounds={'c': (2., 5.)})


@pytest.mark.parametrize('kind', ['incompatible', 'initial', 'fixed', 'singleton'])
def test_invalid_boxes_raise_before_optimization(kind):
    model = source()
    if kind == 'incompatible':
        model.b.bounds = (20., 30.)
    elif kind == 'initial':
        model.b.bounds = (0., 3.)
    elif kind == 'fixed':
        model.a.fixed = True
        model.a.bounds = (3., 4.)
    else:
        model.b.bounds = (16., 20.)  # a == 8 is feasible but must be fixed explicitly
    with pytest.raises(ValueError):
        _box_bounds(model, _ParameterMap(model))


@pytest.mark.parametrize('kind', ['square', 'product', 'division', 'callable'])
def test_unsupported_bounded_links_are_solver_capabilities_not_problem_restrictions(kind):
    model = three()
    model.a.bounds = (1., 5.)
    if kind == 'square':
        model.c.link = model.a**2
    elif kind == 'product':
        model.c.link = model.a*model.b
    elif kind == 'division':
        model.c.link = model.a/model.b
    else:
        model.c.tied = lambda m: 2*m.a.value
    model.c.bounds = (0., 100.)
    x = np.linspace(0., 1., 17)
    problem = _FitProblem(model, x, np.ones_like(x))
    # A future constrained solver can use this same problem, including its raw
    # links and bounds. Only the current adapter's box request is unsupported.
    assert np.isfinite(problem.residual(problem.parameters.values)).all()
    with pytest.raises(NotImplementedError, match='Reparameterize'):
        _box_bounds(model, problem.parameters)
    model.c.bounds = (None, None)
    assert _box_bounds(model, _ParameterMap(model)).shape[1] == 2


def test_multivariate_affine_bound_is_explicitly_rejected():
    model = three()
    model.c.link = model.a+model.b
    model.c.bounds = (0., 10.)
    with pytest.raises(NotImplementedError, match='several free parameters'):
        _box_bounds(model, _ParameterMap(model))


@pytest.mark.parametrize('engine', ENGINES)
def test_general_unbounded_link_still_fits(engine):
    model = source()
    model.b.link = model.a**2
    model.b.bounds = (None, None)
    x = np.linspace(0., 1., 51)
    result = engine()(model, x, 3.+9*x)
    assert_allclose(result.parameters, [3., 9.], atol=1e-6)


@pytest.mark.parametrize('engine', ENGINES)
def test_joint_bounds_restrict_a_cross_dataset_driver(engine):
    first, second = source(), source()
    first.b.link = None
    first.b.value = 0.
    first.b.fixed = True
    first.a.bounds = (0., 10.)
    second.b.link = None
    second.b.value = 0.
    second.b.fixed = True
    second.a.link = 10-2*first.a
    second.a.bounds = (2., 6.)
    x = np.linspace(0., 1., 31)
    result = engine(covariance=True).jointfit({
        'first': (first, Dataset(x, np.full(x.size, 7.))),
        'second': (second, Dataset(x, np.full(x.size, -4.)))})
    assert result['first'].model.a.value == pytest.approx(4., abs=2e-6)
    assert result['second'].model.a.value == pytest.approx(2., abs=4e-6)
    assert result.fit_info['nfree'] == 1
    assert result.fit_info['stat'] == pytest.approx(x.size*(3.**2+6.**2), rel=2e-6)


@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('nproc', [1, 2])
def test_regular_and_named_multifit_propagate_per_target_bounds(engine, nproc):
    model = source()
    model.a.bounds = (0., 10.)
    x = np.linspace(0., 1., 31)
    values = np.array([7., 8.])[:, None]*(1+2*x)
    fitter = engine()
    regular = fitter.multifit(model, x, values, axis=1, nproc=nproc, progress=False,
        initpars={'a': np.array([1., 2.])}, bounds={'b': (0., np.array([4., 6.]))})
    assert_allclose(regular.params['a'].value, [2., 3.], atol=2e-6)
    assert_allclose(regular.params['b'].value, [4., 6.], atol=4e-6)
    named = fitter.multifit(model, {'early': Dataset(x, values[0]), 'late': Dataset(x, values[1])},
        nproc=nproc, progress=False, initpars={'a': {'early': 1., 'late': 2.}},
        bounds={'b': (0., {'early': 4., 'late': 6.})})
    assert_allclose([named[label].model.a.value for label in named], [2., 3.], atol=2e-6)


@pytest.mark.parametrize('kind', ['initial', 'incompatible', 'unsupported'])
def test_regular_invalid_mapped_constraint_precedes_workers(kind, monkeypatch):
    model = source()
    options = {'bounds': {'b': (0., np.array([3., 8.]))}}
    if kind == 'incompatible':
        options['bounds']['b'] = (np.array([20., 0.]), np.array([30., 8.]))
    if kind == 'unsupported':
        model.b.link = model.a**2
    fitter = TRF()
    monkeypatch.setattr(fitter, '_fit_single_direct', lambda *args, **kwargs: pytest.fail('worker started'))
    with pytest.raises((ValueError, NotImplementedError)):
        fitter.multifit(model, np.arange(17.), np.ones((2, 17)), progress=False, **options)


class Gain(LinearOperatorModel):
    gain = Parameter(default=.7, bounds=(.1, 3.))
    def get_matrix(self, x, gain):
        return gain*self.matrix


@pytest.mark.parametrize('engine', ENGINES)
def test_linked_source_bound_can_restrict_a_dataset_response_parameter(engine):
    x = np.linspace(0., 1., 41)
    model = Component(trend, dict(a=2., b=.7), derivative=derivative)
    model.a.fixed = True
    response = Gain(matrix=eye(x.size), x=x)
    model.b.link = response.gain
    model.b.bounds = (0., 1.)
    data = Dataset(x, 1.5*(2.+1.5*x), response=response)
    result = engine()(model, data)
    assert result.dataset.response.gain.value == pytest.approx(1., abs=1e-6)
    assert result.model.b.value == pytest.approx(1., abs=1e-6)
    named = engine().multifit(model, {'one': data}, progress=False)['one']
    assert_allclose(named.parameters, result.parameters, atol=1e-6)


class Velocities(Fittable1DModel):
    first = _Parameter(default=2000., unit=u.m/u.s)
    second = _Parameter(default=2., unit=u.km/u.s)
    @staticmethod
    def evaluate(x, first, second):
        return first+x*second


def test_unit_conversion_is_preserved_in_the_linked_interval():
    model = Velocities()
    model.second.link = model.first
    model.first.bounds = (0., 4000.)
    model.second.bounds = (1., 3.)
    assert_allclose(_box_bounds(model, _ParameterMap(model)), [[1000., 3000.]])


def test_covariance_draws_obey_the_linked_target_interval():
    from line_helpers import make_line
    from prism.modeling import select_line
    from prism.modeling.lines.measurements.sampling import _draw_selection_samples
    first = make_line(amplitude=2., name='first')
    second = make_line(amplitude=4., name='second')
    for component in (first, second):
        for name in component.param_names:
            getattr(component, name).fixed = True
    first.amplitude.fixed = second.amplitude.fixed = False
    first.amplitude.bounds = (0., 100.)
    second.amplitude.bounds = (0., 6.)
    second.amplitude.link = 2*first.amplitude
    model = first+second
    model._param_cov = np.array([[25.]])
    selected = select_line(model, 'line', components='second')
    specs, draws = _draw_selection_samples(selected, 256, method='covariance', random_state=17)
    assert len(specs) == 1
    assert_allclose(specs[0].bounds, (0., 3.))
    assert np.all((draws[:, 0] >= 0.) & (2*draws[:, 0] <= 6.))
    assert draws[:, 0].max() == 3.


def test_declared_bounds_are_checked_after_committing_a_fit():
    from prism.modeling.fitting.constraints import _check_fitted_bounds
    model = source()
    model.a.value = 6.
    with pytest.raises(ValueError, match='violates its declared bounds'):
        _check_fitted_bounds(model)


def test_roundoff_at_a_cancelled_link_boundary_is_not_a_physical_violation():
    from prism.modeling.fitting.constraints import _check_fitted_bounds
    model = source()
    model.b.link = 1.-.3*model.a
    model.b.bounds = (0., None)
    model.a.value = np.nextafter(1./.3, np.inf)
    _check_fitted_bounds(model)
    model.a.value = 1./.3+1e-5
    with pytest.raises(ValueError, match='violates its declared bounds'):
        _check_fitted_bounds(model)


def test_future_batch_preflight_can_receive_raw_constraints_without_box_compilation(monkeypatch):
    model = three()
    model.c.link = model.a+model.b
    model.c.bounds = (0., 10.)
    received = []
    def inspect_constraints(prepared_model, *, initial, bounds):
        received.append((prepared_model, initial, bounds))
    fitter = TRF()
    monkeypatch.setattr(fitter, '_validate_fit_constraints', inspect_constraints)
    fitter._prepare_multifit_configuration(model, (2,),
        initpars={'a': np.array([1., 2.])}, bounds={'c': (0., np.array([9., 10.]))})
    prepared, initial, bounds = received[0]
    assert prepared is not model
    assert_array_equal(initial['a'], [1., 2.])
    assert_array_equal(bounds['c'][1], [9., 10.])
    assert len(set(map(id, prepared.c.link.references()))) == 2
    assert prepared.c.bounds == (0., 10.)
    assert model.c.bounds == (0., 10.)
