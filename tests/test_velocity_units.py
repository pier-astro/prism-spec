"""Physical equivalence and derivatives across user-selected velocity units."""
import numpy as np
import pytest
from astropy import units as u
from astropy.io.misc import yaml

from prism.modeling.models.lines import (
    GaussianLine, LorentzianLine, VoigtLine,
    GaussianLines, LorentzianLines, VoigtLines,
)
from prism.modeling import fitting
from prism.modeling.io import load_model
from prism.modeling.fitting.uncertainty import covar_attach


KINDS = (GaussianLine, LorentzianLine, VoigtLine,
         GaussianLines, LorentzianLines, VoigtLines)


def test_incompatible_or_ambiguous_velocity_units_are_rejected():
    with pytest.raises(u.UnitConversionError):
        GaussianLine(fwhm=3*u.AA)
    with pytest.raises(u.UnitConversionError):
        GaussianLine(instfwhm=3*u.AA)
    with pytest.raises(ValueError, match='scalar velocity'):
        GaussianLine(instfwhm=np.array([[5000.,80.],[5100.,85.]])*u.km/u.s)


def _model(kind, velocity_unit=u.km/u.s, quantities=False, spectral_unit=u.AA,
           instfwhm=80., **kwargs):
    values = {'offset': 35., 'fwhm': 180.}
    if kind in (VoigtLine, VoigtLines):
        values = {'offset': 35., 'fwhm_G': 180., 'fwhm_L': 110.}
    values.update(kwargs)
    for name in values:
        if name in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
            values[name] = values[name] * u.km/u.s
            if quantities:
                # Deliberately differs from the selected parameter unit.
                other = u.m/u.s if velocity_unit == u.km/u.s else u.km/u.s
                values[name] = values[name].to(other)
            else:
                values[name] = values[name].to_value(velocity_unit)
    if kind.__name__.endswith('Lines'):
        return kind.from_arrays(names=['pair', 'pair'], pos=[5000., 5010.]*spectral_unit,
                                weights=[1., .33], amplitude=2., velocity_unit=velocity_unit,
                                instfwhm=instfwhm, **values)
    return kind(position=5000.*spectral_unit, amplitude=2., velocity_unit=velocity_unit,
                instfwhm=instfwhm, **values)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('velocity_unit', [u.km/u.s, u.m/u.s])
@pytest.mark.parametrize('quantities', [False, True])
def test_physical_profiles_and_parameter_labels(kind, velocity_unit, quantities):
    reference = _model(kind)
    model = _model(kind, velocity_unit, quantities)
    x = np.linspace(4970., 5040., 701)
    np.testing.assert_allclose(model(x), reference(x), rtol=2e-12, atol=2e-14)
    for name in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
        if name in model.param_names:
            assert getattr(model, name).unit == velocity_unit
            np.testing.assert_allclose(getattr(model, name).quantity.to_value(u.km/u.s),
                                       getattr(reference, name).value, rtol=1e-14)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('spectral_unit', [u.AA, u.eV])
@pytest.mark.parametrize('instfwhm', [80., np.array([[4800., 65.], [5000., 80.], [5200., 95.]])])
@pytest.mark.parametrize('offset', [0.,35.])
def test_velocity_jacobian_matches_independent_differences(kind, spectral_unit, instfwhm, offset):
    model = _model(kind, u.m/u.s, spectral_unit=spectral_unit, instfwhm=instfwhm, offset=offset)
    x = np.linspace(4970., 5040., 701)
    p = model.parameters.copy()
    analytic = np.asarray(model.fit_deriv(x, *p))
    for i, name in enumerate(model.param_names):
        if name not in ('offset', 'fwhm', 'fwhm_G', 'fwhm_L'):
            continue
        # A physical 0.01 km/s floor avoids cancellation around zero offset.
        step = max(abs(p[i])*1e-6, .01/model._to_kms)
        upper, lower = model.copy(), model.copy()
        a, b = p.copy(), p.copy()
        a[i] += step
        b[i] -= step
        upper.parameters, lower.parameters = a, b
        numeric = (upper(x)-lower(x))/(2*step)
        # Current Lorentzian derivatives are first-order forward differences.
        tolerance = 1e-4 if kind in (LorentzianLine, LorentzianLines) else 1e-6
        np.testing.assert_allclose(analytic[i], numeric, rtol=2e-3,
                                   atol=tolerance*max(abs(numeric).max(), 1e-12))


@pytest.mark.parametrize('kind', KINDS)
def test_defaults_bounds_instrument_and_serialization(kind, tmp_path):
    kwargs = {'bounds': {'offset': (-100*u.km/u.s, 100*u.km/u.s)}}
    if kind.__name__.endswith('Lines'):
        plain = kind.from_arrays(names=['one'], pos=[5000.]*u.AA)
        selected = kind.from_arrays(names=['one'], pos=[5000.]*u.AA, velocity_unit=u.m/u.s, **kwargs)
    else:
        plain = kind(position=5000.*u.AA)
        selected = kind(position=5000.*u.AA, velocity_unit=u.m/u.s, **kwargs)
    x = np.linspace(4950.,5050.,401)
    np.testing.assert_array_equal(selected(x),plain(x))
    assert selected.offset.bounds == (-100000.,100000.)
    model = _model(kind, u.m/u.s, instfwhm=80000*u.m/u.s)
    np.testing.assert_allclose(model(x),_model(kind)(x),rtol=2e-12,atol=2e-14)
    model.offset.std = 1500.
    model.offset.lolim, model.offset.uplim = 30000.,40000.
    model.offset.bounds = (-100000.,100000.)
    path = tmp_path/'units.fits'
    model.save(path, overwrite=True)
    for restored in (yaml.load(yaml.dump(model)), load_model(path)):
        assert restored.velocity_unit == u.m/u.s
        assert restored.offset.unit == u.m/u.s
        assert restored.offset.bounds == model.offset.bounds
        assert restored.offset.std == model.offset.std
        assert restored.offset.lolim == model.offset.lolim
        np.testing.assert_array_equal(restored(x),model(x))
    legacy = yaml.dump(_model(kind))
    legacy = '\n'.join(line for line in legacy.splitlines() if '_velocity_unit:' not in line)
    restored = yaml.load(legacy)
    assert restored.velocity_unit == u.km/u.s
    np.testing.assert_array_equal(restored(x),_model(kind)(x))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('offset', [0.,35.])
def test_flux_uncertainty_is_independent_of_velocity_unit(kind, offset):
    models = [_model(kind,offset=offset), _model(kind,u.m/u.s,offset=offset)]
    for model in models:
        scale = (u.km/u.s).to(model.velocity_unit)
        amp = model.param_names[0]
        getattr(model, amp).std = .1
        model.offset.std = 10.*scale
        for name in ('fwhm', 'fwhm_G', 'fwhm_L'):
            if name in model.param_names:
                getattr(model, name).std = 5.*scale
        # Independent-error propagation exercises _template_flux for groups.
    def unpack(model):
        flux = model.flux
        return np.array([flux['value'].iloc[0],flux['std'].iloc[0]]) if kind.__name__.endswith('Lines') else np.array([flux.value,flux.std])
    np.testing.assert_allclose(unpack(models[1]),unpack(models[0]),rtol=2e-5)
    # Include correlations and the correct row/column parameter scaling.
    from astropy.modeling.fitting import model_to_fit_params
    _, indices, _ = model_to_fit_params(models[0])
    scale = np.array([1000. if models[0].param_names[i] in ('offset','fwhm','fwhm_G','fwhm_L') else 1.
                      for i in indices])
    cov = np.eye(len(indices))*.01
    cov[0,-1] = cov[-1,0] = .002
    models[0]._param_cov = cov
    models[1]._param_cov = cov*scale[:,None]*scale[None,:]
    np.testing.assert_allclose(unpack(models[1]),unpack(models[0]),rtol=2e-5)


@pytest.mark.parametrize('kind', [GaussianLine, GaussianLines])
@pytest.mark.parametrize('method', ['TRFLSQFitter','ScipyTRF','SherpaLM'])
def test_cold_fit_and_covariance_scale(kind, method, tmp_path):
    if method == 'SherpaLM':
        pytest.importorskip('sherpa')
    x = np.linspace(4970.,5040.,501)
    truth = _model(kind)
    # Deterministic small deviations; the fit starts far from the known inputs.
    y = truth(x)+.002*np.sin(np.arange(x.size)*.7)
    yerr = np.full_like(x,.02)
    results = []
    covariances = []
    for unit in (u.km/u.s,u.m/u.s):
        scale = (u.km/u.s).to(unit)
        initial = _model(kind,unit,offset=-60.,fwhm=350.,
                         bounds={'offset':(-200.*scale,200.*scale),'fwhm':(50.*scale,500.*scale)})
        fitter = getattr(fitting,method)(calc_uncertainties=True)
        fitted = fitter(initial,x,y,yerr=yerr)
        assert fitter.fit_info['success']
        assert fitted.fwhm.unit == unit
        np.testing.assert_allclose(fitted.offset.quantity.to_value(u.km/u.s),35.,atol=.1)
        np.testing.assert_allclose(fitted.fwhm.quantity.to_value(u.km/u.s),180.,atol=.1)
        covar_attach(fitted,fitter)
        results.append(fitted(x))
        # Unscale covariance to km/s before comparing inferred uncertainties.
        factors = np.array([scale if name in ('offset','fwhm') else 1.
                            for name in fitted.param_names if not fitted.fixed[name]])
        covariances.append(fitter.fit_info['param_cov']/factors[:,None]/factors[None,:])
        path = tmp_path/(unit.to_string().replace('/','_')+'.fits')
        fitted.save(path,overwrite=True)
        restored = load_model(path)
        assert restored.fwhm.unit == unit
        np.testing.assert_allclose(restored.covariance,fitted.covariance)
    np.testing.assert_allclose(results[1],results[0],rtol=2e-5,atol=1e-6)
    np.testing.assert_allclose(covariances[1],covariances[0],rtol=2e-3,atol=1e-9)


@pytest.mark.parametrize('unit', [u.km/u.s, u.m/u.s])
def test_quantity_data_and_tied_response_fit(unit):
    from prism.modeling.operators.instrument import InstrumentResponse, SpectralResponse
    x = np.linspace(4970.,5040.,501)*u.AA
    response = SpectralResponse(instrument=InstrumentResponse.from_fixed_fwhm(x.value, .8),
                                wave=x.value,flexible=False)
    def construct(offset, fwhm):
        first = GaussianLine(name='first',amplitude=2.*u.Jy,position=5000.*u.AA,
                             offset=offset*u.km/u.s,fwhm=fwhm*u.km/u.s,velocity_unit=unit)
        second = GaussianLine(name='second',amplitude=1.*u.Jy,position=5010.*u.AA,
                              offset=offset*u.km/u.s,fwhm=fwhm*u.km/u.s,velocity_unit=unit)
        model = (first+second)|response
        model['second'].offset.tied=fitting.tie('first',lambda m:m.offset)
        model['second'].fwhm.tied=fitting.tie('first',lambda m:m.fwhm)
        return model
    truth = construct(35.,180.)
    initial = construct(-60.,350.)
    fitted = fitting.TRFLSQFitter()(initial,x,truth(x))
    assert fitted['first'].fwhm.unit == unit
    assert fitted['first'].amplitude.unit == u.Jy
    np.testing.assert_allclose(fitted(x),truth(x),rtol=2e-5,atol=1e-6*u.Jy)
    np.testing.assert_allclose(fitted['first'].offset.quantity.to_value(u.km/u.s),35.,atol=.01)
    np.testing.assert_allclose(fitted['first'].fwhm.quantity.to_value(u.km/u.s),180.,atol=.01)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('spectral_unit', [u.AA, u.eV])
def test_selected_profiles_and_measurements_preserve_velocity_units(kind, spectral_unit):
    from prism.modeling.line_analysis import select_line
    pair = [_model(kind, unit, spectral_unit=spectral_unit) for unit in (u.km/u.s,u.m/u.s)]
    selector = 'pair' if kind.__name__.endswith('Lines') else kind.__name__
    selections = [select_line(model,selector) for model in pair]
    x = np.linspace(4970.,5040.,701)
    single_kind = {GaussianLines:GaussianLine,LorentzianLines:LorentzianLine,VoigtLines:VoigtLine}.get(kind,kind)
    reference = _model(single_kind,spectral_unit=spectral_unit)
    for selected in selections:
        np.testing.assert_allclose(selected(x),reference(x),rtol=2e-12,atol=2e-14)
    np.testing.assert_allclose(selections[1].infer_window(),selections[0].infer_window(),rtol=1e-13)
    # Symmetric windows must be centered on the kinematic shift in the native domain.
    center = 5000.*np.exp(35./299792.458) if spectral_unit == u.AA else 5000./np.exp(35./299792.458)
    np.testing.assert_allclose(np.mean(selections[0].infer_window()),center,rtol=1e-13)
    measured = [selected.measure() for selected in selections]
    for key in ('flux','fwhm','peak_position'):
        np.testing.assert_allclose(getattr(measured[1],key).value,getattr(measured[0],key).value,rtol=1e-11)
