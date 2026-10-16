"""Independent checks for analytic line adapters and finite-window measures."""
import numpy as np
import pytest
from astropy import units as u
from scipy.special import voigt_profile

from prism.modeling.models.lines import (
    GaussianLine, LorentzianLine, VoigtLine,
    GaussianLines, LorentzianLines, VoigtLines,
)
from prism.modeling.models.lines import profiles, _measurements
from prism.modeling.line_analysis import select_line


@pytest.mark.parametrize('sigma,gamma', [(0.,1.),(1.,0.),(1.,.5)])
def test_voigt_jacobian_matches_independent_profile_limits(sigma,gamma):
    x = np.linspace(-5.,5.,401)
    derivatives = profiles.voigt_deriv(x,2.,0.,sigma,gamma)
    def reference(s,g):
        return 2.*voigt_profile(x,s,g)/voigt_profile(0.,s,g)
    if sigma > 0:
        step=1e-5
        numeric=(reference(sigma+step,gamma)-reference(sigma-step,gamma))/(2*step)
        np.testing.assert_allclose(derivatives[3],numeric,rtol=2e-5,atol=1e-8)
    step=1e-6
    if gamma == 0:
        numeric=(reference(sigma,step)-reference(sigma,0.))/step
    else:
        numeric=(reference(sigma,gamma+step)-reference(sigma,gamma-step))/(2*step)
    np.testing.assert_allclose(derivatives[4],numeric,rtol=3e-5,atol=2e-6)


@pytest.mark.parametrize('single,group', [(GaussianLine,GaussianLines),
                                       (LorentzianLine,LorentzianLines),(VoigtLine,VoigtLines)])
@pytest.mark.parametrize('velocity_unit', [u.km/u.s,u.m/u.s])
def test_single_and_group_share_profile_and_flux(single,group,velocity_unit):
    widths={'fwhm_G':180*u.km/u.s,'fwhm_L':110*u.km/u.s} if single is VoigtLine else {'fwhm':180*u.km/u.s}
    kwargs=dict(amplitude=2.*u.Jy,offset=35.*u.km/u.s,redshift=.025,instfwhm=80.,velocity_unit=velocity_unit,**widths)
    line=single(position=5000.*u.AA,**kwargs)
    template=group.from_arrays(names=['one'],pos=[5000.]*u.AA,weights=[1.],**kwargs)
    x=np.linspace(5090.,5160.,701)
    np.testing.assert_allclose(line(x*u.AA),template(x*u.AA),rtol=3e-13,atol=1e-14*u.Jy)
    np.testing.assert_allclose(line.flux.value,template.flux.loc['one','value'],rtol=1e-13)
    assert line.flux.unit == u.Jy*u.AA
    assert template.flux.loc['one','unit'] == line.flux.unit
    selected=select_line(template,'one')
    np.testing.assert_allclose(selected(x*u.AA),line(x*u.AA),rtol=3e-13,atol=1e-14*u.Jy)
    center=5000.*1.025*np.exp(35./profiles.C_KMS)
    sigma=center/profiles.C_KMS/profiles.SIGMA2FWHM
    sigma*=np.hypot(180.,80.) if single is not LorentzianLine else 80.
    gamma=center/profiles.C_KMS/2*(110. if single is VoigtLine else 180. if single is LorentzianLine else 0.)
    expected=2./1.025/voigt_profile(0.,sigma,gamma)
    np.testing.assert_allclose(line.flux.value,expected,rtol=2e-13)
    np.testing.assert_allclose(line.eqw(continuum=2000.*u.mJy).value,expected/2.,rtol=1e-13)
    assert line.eqw(continuum=2.*u.Jy).unit == u.AA


def test_template_integral_contains_distant_members_and_quantity_windows():
    group=GaussianLines.from_arrays(names=['pair','pair'],pos=[4900.,5100.]*u.AA,
                                    weights=[1.,.33],amplitude=2.*u.Jy,fwhm=180.,instfwhm=80.)
    analytic=group.flux.loc['pair','value']/2.
    for window in [None,(480.*u.nm,520.*u.nm)]:
        measured=group.eqw(continuum=2000.*u.mJy,method='integration',window=window,num=20001)
        np.testing.assert_allclose(measured.loc['pair','value'],analytic,rtol=1e-9)
        assert measured.loc['pair','unit'] == u.AA
    x=np.linspace(4800.,5200.,20001)*u.AA
    a=group.eqw(continuum=2.*u.Jy,method='integration',x=x)
    b=group.eqw(continuum=2.*u.Jy,method='integration',x=x[::-1])
    np.testing.assert_allclose(a['value'],b['value'],rtol=1e-13)


@pytest.mark.parametrize('kind',[GaussianLine,GaussianLines])
def test_flux_zero_error_and_parameter_restoration_on_failure(kind):
    model=kind() if kind is GaussianLine else kind.from_arrays(names=['one'],pos=[5000.]*u.AA,weights=[1.])
    amplitude=model.param_names[0]
    getattr(model,amplitude).std=0.
    flux=model.flux
    assert (flux.std if kind is GaussianLine else flux['std'].iloc[0]) == 0.
    before=model.parameters.copy()
    def fails_when_perturbed():
        if getattr(model,amplitude).value != before[0]:
            raise RuntimeError('measurement failed')
        return 1.
    getattr(model,amplitude).std=.1
    with pytest.raises(RuntimeError,match='measurement failed'):
        _measurements._flux(model,fails_when_perturbed,[amplitude],[amplitude])
    np.testing.assert_array_equal(model.parameters,before)
    with pytest.raises(RuntimeError,match='measurement failed'):
        _measurements._evaluate_at(model,fails_when_perturbed,{amplitude:2.})
    np.testing.assert_array_equal(model.parameters,before)


def test_invalid_measurement_inputs_do_not_return_false_zero():
    line=GaussianLine()
    with pytest.raises(ValueError,match='at least two'):
        line.eqw(continuum=1.,method='integration',x=[])
    with pytest.raises(ValueError,match='strictly monotonic'):
        line.eqw(continuum=1.,method='integration',x=[5000.,5000.])
    with pytest.raises(ValueError,match='method'):
        line.eqw(method='unknown')
    with pytest.raises(ValueError,match='zero-width'):
        GaussianLine(fwhm=0.).eqw(continuum=1.,method='integration')


@pytest.mark.parametrize('kind',[GaussianLine,GaussianLines])
def test_automatic_ew_grid_is_independent_of_native_length_unit(kind):
    values=[]
    for unit in [u.AA,u.nm,u.um]:
        position=(5000.*u.AA).to(unit)
        if kind is GaussianLine:
            line=kind(position=position,amplitude=2.*u.Jy,fwhm=180.)
            measured=line.eqw(continuum=2.*u.Jy,method='integration')
        else:
            line=kind.from_arrays(names=['one'],pos=[position.value]*unit,weights=[1.],
                                  amplitude=2.*u.Jy,fwhm=180.)
            row=line.eqw(continuum=2.*u.Jy,method='integration').iloc[0]
            from prism.modeling.metrics import Metric
            measured=Metric(**row.to_dict())
        values.append(measured.quantity.to_value(u.AA))
    np.testing.assert_allclose(values,values[0],rtol=2e-12)


@pytest.mark.parametrize('instfwhm',[np.nan,np.inf,-1.,[[5000.,np.nan]],[[5000.,-1.]],np.empty((0,2))])
def test_invalid_instrumental_input_is_rejected_before_evaluation(instfwhm):
    with pytest.raises(ValueError,match='finite'):
        GaussianLine(instfwhm=instfwhm)


@pytest.mark.parametrize('pos,weight',[([np.nan],[1.]),([5000.],[np.nan])])
def test_invalid_line_table_does_not_silently_drop_members(pos,weight):
    with pytest.raises(ValueError,match='finite'):
        GaussianLines.from_arrays(names=['one'],pos=pos,weights=weight,position_unit=u.AA)
