"""Native scientific fixtures; no Astropy line subclasses or production shims."""
import numpy as np
from astropy import units as u
from prism.modeling import Component, LineSet
from prism.modeling.lines import GaussianProfile
from prism.modeling.lines.tables import normalize_linetable


def make_line(amplitude=1., position=5000., offset=0., fwhm=1000., redshift=0.,
         instfwhm=0., name="line", bounds=None, fixed=None, tied=None):
    unit = getattr(position, 'unit', u.AA)
    profile = GaussianProfile(instfwhm=instfwhm, position_unit=unit)
    values = dict(amplitude=float(getattr(amplitude, 'value', amplitude)),
                  position=float(getattr(position, 'value', position)),
                  offset=offset.to_value(u.km/u.s) if isinstance(offset, u.Quantity) else offset,
                  fwhm=fwhm.to_value(u.km/u.s) if isinstance(fwhm, u.Quantity) else fwhm,
                  redshift=redshift)
    model = Component(profile, values, derivative=profile.derivative, name=name)
    model.position.fixed = model.redshift.fixed = True
    model.component._separable_linear_params = ('amplitude',)
    for flag, settings in [('bounds', bounds), ('fixed', fixed), ('tied', tied)]:
        for key, value in (settings or {}).items():setattr(getattr(model, key), flag, value)
    return model


def make_group(names=None, pos=None, weights=None, *, table=None, position_unit=None,
          amplitude=1., offset=0., fwhm=1000., redshift=0., instfwhm=0.,
          name=None, bounds=None, fixed=None, tied=None, **amplitudes):
    if table is None:
        table = dict(name=names, position=pos)
        if weights is not None:table['weight']=weights
    table = normalize_linetable(table, position_unit=position_unit)
    profile = GaussianProfile(instfwhm=instfwhm, position_unit=table['position'].unit)
    model = LineSet(profile, table, shared=dict(offset=offset.to_value(u.km/u.s) if isinstance(offset,u.Quantity) else offset,
                                      fwhm=fwhm.to_value(u.km/u.s) if isinstance(fwhm,u.Quantity) else fwhm, redshift=redshift),
                          parameters={str(tag): {'amplitude': amplitude} for tag in dict.fromkeys(table['name'])},
                          position='position', derivative=profile.derivative, name=name)
    model.redshift.fixed=True
    for key,value in amplitudes.items():setattr(model,key,value)
    for flag,settings in [('bounds',bounds),('fixed',fixed),('tied',tied)]:
        for key,value in (settings or {}).items():
            targets=[n for n in model.param_names if n.startswith('amp_')] if key=='amplitude' else [key]
            for target in targets:setattr(getattr(model,target),flag,value)
    return model


def reference_profile(x, amplitude, position, offset, widths, redshift, instrumental, unit):
    """Independent peak-normalized reference using scipy.special.voigt_profile."""
    from scipy.special import voigt_profile
    factor = (1.+redshift) if unit.is_equivalent(u.m) else 1./(1.+redshift)
    center = position*factor*np.exp((offset if unit.is_equivalent(u.m) else -offset)/299792.458)
    gaussian, lorentzian = widths
    sigma = np.hypot(gaussian, instrumental)*center/299792.458/np.sqrt(8*np.log(2))
    gamma = lorentzian*center/299792.458/2.
    return amplitude/(1.+redshift)*voigt_profile(x-center,sigma,gamma)/voigt_profile(0.,sigma,gamma)
