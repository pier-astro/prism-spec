"""Public physical component views retain identity, algebra and link context."""
import numpy as np
import pytest
from astropy.modeling import Parameter, FittableModel
from astropy.modeling.models import Gaussian1D
from numpy.testing import assert_allclose
from scipy.sparse import eye

from prism.modeling import Component, LinearOperatorModel, Model
from prism.modeling.lines import GaussianLines


def constant(x, amplitude):
    return np.full_like(x, amplitude)


def test_named_views_are_live_models_with_read_only_structure():
    first = Component(constant, {'amplitude': 2.}, name='first')
    second = Component(constant, {'amplitude': 3.}, name='second')
    model = first + second
    components = model.components
    assert tuple(components) == ('first', 'second')
    assert isinstance(components['first'], Model)
    assert components['first'].params['amplitude'] is first.amplitude
    components['first'].params['amplitude'].value = 4.
    assert first.amplitude.value == 4.
    assert_allclose(model(np.arange(5.)), 7.)
    with pytest.raises(TypeError):
        components['first'] = second
    with pytest.raises(KeyError):
        components['missing']


def test_component_evaluation_resolves_parent_links_without_mutating_source_scope():
    first = Component(constant, {'amplitude': 2.}, name='first')
    second = Component(constant, {'amplitude': 1.}, name='second')
    second.amplitude.link = 3*first.amplitude
    model = first + second
    view = model.components['second']
    first.amplitude.value = 4.
    assert_allclose(view(np.arange(5.)), 12.)
    assert not hasattr(second, '_evaluation_parent')
    with pytest.raises(ValueError, match='outside'):
        second(np.arange(5.))
    copied = model.copy()
    copied.components['first'].params['amplitude'].value = 5.
    assert_allclose(copied.components['second'](np.arange(5.)), 15.)
    assert first.amplitude.value == 4.


def test_components_preserve_physical_leaves_without_distributing_products():
    first = Component(constant, {'amplitude': 2.}, name='first')
    second = Component(constant, {'amplitude': 3.}, name='second')
    factor = Component(constant, {'amplitude': .5}, name='factor')
    model = (first + second)*factor
    x = np.arange(5.)
    assert tuple(model.components) == ('first', 'second', 'factor')
    assert_allclose(model.components['first'](x), 2.)
    assert_allclose(model.components['second'](x), 3.)
    assert_allclose(model.components['factor'](x), .5)
    assert_allclose(model(x), 2.5)


def test_unnamed_and_duplicate_named_components_have_explicit_position_keys():
    first = Model.from_astropy(Gaussian1D())
    second = Model.from_astropy(Gaussian1D())
    for names, keys in [((None, None), (0, 1)), (('line', 'line'), (0, 1)),
                        (('line', None), ('line', 1))]:
        first.name, second.name = names
        components = (first+second).components
        assert tuple(components) == keys
        assert components[keys[0]].params['amplitude'] is first.amplitude
        assert components[keys[1]].params['amplitude'] is second.amplitude


class Gain(LinearOperatorModel):
    gain = Parameter(default=1.)

    def get_matrix(self, x, gain):
        return self.matrix*float(np.asarray(gain).item())


def test_pipe_access_retains_the_public_operator_and_its_flux_action():
    x = np.linspace(-2., 2., 21)
    source = Component(constant, {'amplitude': 2.}, name='source')
    response = Gain(matrix=eye(x.size), x=x, name='response')
    model = source | response
    parts = model.components
    assert tuple(parts) == ('source', 'response')
    assert parts['response'] is response
    assert parts['response'].params['gain'] is response.gain
    parts['response'].params['gain'].value = 3.
    assert_allclose(parts['response'](source(x)), 6.)
    assert_allclose(parts['source'](x), 2.)
    assert_allclose(model(x), 6.)


def test_lineset_and_astropy_leaves_share_the_same_public_parameter_access():
    from astropy import units as u
    lines = GaussianLines({'name': ['A'], 'position': [5000.]*u.AA},
                           amplitude=2., fwhm=300., name='lines', normalization='amplitude')
    continuum = Model.from_astropy(Gaussian1D(1., 5000., 100., name='continuum'))
    model = lines + continuum
    x = np.linspace(4990., 5010., 101)
    assert_allclose(model.components['lines'](x), lines(x))
    assert_allclose(model.components['continuum'](x), continuum(x))
    assert model.components['lines'].params['fwhm'] is lines.fwhm
    assert model.components['continuum'].params['mean'] is continuum.mean


class Transform(FittableModel):
    n_inputs = 2
    n_outputs = 1
    norm = Parameter(default=1.)

    @staticmethod
    def evaluate(x, flux, norm):
        return norm*flux


def test_transform_component_view_resolves_ties_in_its_complete_parent():
    source = Component(constant, {'amplitude': 2.}, name='source')
    transform = Model.from_astropy(Transform(name='transform'))
    model = source | transform
    model.norm_1.tied = lambda parent: 3*parent['source'].amplitude.value
    view = model.components['transform']
    source.amplitude.value = 4.
    x = np.arange(5.)
    assert_allclose(view(x, np.ones_like(x)), 12.)
