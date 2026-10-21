from prism.modeling import Model
"""Explicit component/member selection and retained parent-model contracts."""
from line_helpers import make_line as single_line, make_group
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.models import Const1D
from prism.modeling.lines.measurements import select_line
from prism.modeling.lines import profiles


def make_line(name='narrow', amplitude=2., fwhm=200.):
    line = single_line(position=5000., amplitude=amplitude, fwhm=fwhm, name=name)
    line.offset.fixed = True
    line.fwhm.fixed = True
    return line


def test_explicit_independent_component_list_matches_analytic_sum():
    narrow, broad = make_line(), make_line('broad',1.,600.)
    selected = select_line([narrow,broad], 'Hb')
    x = np.linspace(4970.,5030.,1001)
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    expected = 2.*np.exp(-.5*((x-5000.)/sigma)**2)
    expected += np.exp(-.5*((x-5000.)/(3.*sigma))**2)
    np.testing.assert_allclose(selected(x),expected,rtol=1e-12,atol=1e-15)
    narrow.amplitude = 3.
    np.testing.assert_allclose(selected(5000.),4.)
    parts = selected.evaluate_components(x=x[[3,1]])
    np.testing.assert_allclose(parts['narrow']+parts['broad'],selected(x[[3,1]]))


def test_parent_named_object_and_index_aliases_count_component_once():
    model = Model.from_astropy(Const1D(3.,name='continuum'))+make_line()+make_line('broad')
    selected = select_line(model, 'Hb', components=['narrow',model['narrow'],1])
    assert selected.source_model is model
    assert selected.component_names == ['narrow']
    assert selected.entries[0].component_key == 1
    assert selected(5000.) == 2.
    assert select_line(model,'Hb',components=model['narrow'])(5000.) == 2.
    foreign = make_line()  # Same name does not establish parent membership.
    with pytest.raises(ValueError,match='supplied parent'):
        select_line(model,'Hb',components=[foreign])


def test_explicit_parent_selection_preserves_joint_covariance_after_continuum():
    model = Model.from_astropy(Const1D(3.,name='continuum'))+make_line()+make_line('broad',1.,600.)
    model._param_cov = np.array([[.09,0.,0.],[0.,.04,.015],[0.,.015,.01]])
    selected = select_line(model,'Hb',components=['narrow','broad'])
    _, samples = selected.sample(method='covariance',n_samples=64,random_state=71,return_samples=True)
    draws = np.random.default_rng(71).multivariate_normal([2.,1.],model._param_cov[1:,1:],size=64)
    sigma = 5000.*200./profiles.C_KMS/profiles.SIGMA2FWHM
    expected = (draws[:,0]+3.*draws[:,1])*sigma*np.sqrt(2*np.pi)
    np.testing.assert_allclose(samples['flux'],expected,rtol=1e-12)


def test_quantity_member_position_and_live_group_parameters():
    group = make_group(names=['pair','pair'],pos=[5000.,5010.]*u.AA,
                                      weights=[1.,.5],amplitude=4.,fwhm=200.)
    selected = select_line(group,'pair',position=501.*u.nm)
    assert selected.position == 5010.
    assert selected(5010.) == 2.
    group.amp_pair = 6.
    assert selected(5010.) == 3.
    group.offset.value = 100.
    center = 5010.*np.exp(100./profiles.C_KMS)
    np.testing.assert_allclose(selected(center),3.)
    with pytest.raises(ValueError,match='unique matching member'):
        select_line(group,'pair',position=503.*u.nm)


def test_native_coordinate_mismatch_is_rejected():
    a = make_line()
    b = single_line(position=500.*u.nm,name='other')
    with pytest.raises(ValueError,match='same native spectral unit'):
        select_line([a,b],'Hb')




def test_non_additive_model_requires_explicit_intrinsic_choice():
    model = Model.from_astropy(Const1D(3.,name='scale'))*make_line()
    with pytest.raises(ValueError,match='non-additive'):
        select_line(model,'narrow')
    selected = select_line(model,'Hb',components='narrow')
    assert selected(5000.) == 2.
    with pytest.raises(TypeError,match='not a line model'):
        select_line(model,'Hb',components='scale')


def test_linked_component_list_requires_original_parent():
    line = make_line()
    line.amplitude.tied = lambda model: 2.*model.amplitude_1
    with pytest.raises(ValueError,match='original parent'):
        select_line([line,make_line('driver')],'Hb')


@pytest.mark.parametrize('weights', [[1.,1.],[.3,.5]])
def test_ambiguous_group_requires_explicit_rest_position(weights):
    group = make_group(names=['pair','pair'],pos=[5000.,5010.]*u.AA,
                                      weights=weights,amplitude=4.,fwhm=200.)
    with pytest.raises(ValueError,match='unique matching member'):
        select_line(group,'pair')
    selected = select_line(group,'pair',position=5010.)
    np.testing.assert_allclose(selected(5010.),4.*weights[1])


def test_sole_template_member_needs_no_unit_weight_anchor():
    group = make_group(names=['one'],pos=[5000.]*u.AA,
                                      weights=[.3],amplitude=4.,fwhm=200.)
    np.testing.assert_allclose(select_line(group,'one')(5000.),1.2)


def test_cube_choice_uses_portable_names_or_indices():
    from prism.modeling.fitting.multifit import MultiFitResult
    line = make_line()
    x = np.linspace(4990.,5010.,101)
    result = MultiFitResult((2,),line.param_names,template_model=line,x=x)
    for index, amplitude in enumerate([2.,4.]):
        values = line.parameters.copy()
        values[0] = amplitude
        result.update(index, values, None, {'success':True})
    selected = select_line(result,'Hb',components='narrow')
    np.testing.assert_allclose(selected(x)[:,50],[2.,4.])
    assert select_line(result,'Hb',components=0,index=1)(5000.) == 4.
    with pytest.raises(TypeError,match='name or index'):
        select_line(result,'Hb',components=line)
