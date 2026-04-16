import numpy as np
from prism.data import Spectrum
from astropy import units as u
import prism.modeling.models as models
import prism.modeling.fitting as fitting
from prism.modeling.fitting import tie

filename = 'examples/data/agnspec.txt'
dec, ra = -25.12345, 13.12345
redshift = 0.0622
spec = Spectrum.from_txt(filename=filename, ra=ra, dec=dec, z=redshift, wave_unit=u.AA, unit=1e-15 * u.erg / (u.s * u.cm**2 * u.AA), name='AGN Spectrum')
spec.deredden()
spec.zcorrect()
models.setup_local_lines(wmin=np.min(spec.wave), wmax=np.max(spec.wave), overwrite=True)

instfwhm = 120. # km/s
bknpower = models.BrokenPowerlaw(name='bknpower', x_ref=5450, amplitude=np.median(spec.flux), index1=-2.0, index2=0.0, bounds={'x_ref': (5100, 5710), 'amplitude': (1.5, 5.0), 'index1': (-3.0, 0.0), 'index2': (-1.0, 1.5)})
blr = models.agn.blr(name='blr', amplitude=10, offset=-360, fwhm=3727, instfwhm=instfwhm, bounds={'amplitude': (0, 1000), 'offset': (-1000, 1000), 'fwhm': (500, 8000)})
nlr = models.agn.nlr(name='nlr', amplitude=1, offset=0, fwhm=500, instfwhm=instfwhm, bounds={'amplitude': (0, 100), 'offset': (-300, 300), 'fwhm': (0, 800)})
hhe_nlr = models.GaussianLines.from_csv(name='hhe_nlr', csv_files=['hydrogen.csv', 'helium.csv'], amplitude=1, offset=0, fwhm=500, instfwhm=instfwhm, bounds={'amplitude': (0, 100), 'offset': (-300, 300), 'fwhm': (0, 800)})
hhe_nlr.offset.tied = tie("nlr", lambda m: m.offset)
hhe_nlr.fwhm.tied = tie("nlr", lambda m: m.fwhm)
oiii_out = models.GaussianLines.from_csv(name='oiii', csv_files=['oiii.csv'], amplitude=1, offset=-560, fwhm=850, instfwhm=instfwhm, bounds={'amplitude': (0, 100), 'offset': (-1000, 750), 'fwhm': (50, 1500)})
fe = models.agn.fe(name='fe', offset=0, fwhm=1200, instfwhm=instfwhm, bounds={'amplitude': (0, 100), 'offset': (-3000, 3000), 'fwhm': (800, 3000)})
model = bknpower + blr + nlr + hhe_nlr + oiii_out + fe

fitter_scipy = fitting.ScipyTRF()
print("Fitting with ScipyTRF defaults")
fitted_scipy = fitter_scipy(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print("Cost:", fitter_scipy.fit_info['cost'])
print("Info:", fitter_scipy.fit_info['message'])

print("\nFitting with ScipyTRF tightened (loss=soft_l1)")
fitted_scipy_tight = fitter_scipy(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err, ftol=1e-12, xtol=1e-12, gtol=1e-12, loss='soft_l1')
print("Cost:", fitter_scipy.fit_info['cost'])
print("Info:", fitter_scipy.fit_info['message'])

from astropy.modeling.fitting import TRFLSQFitter
fitter_trf = TRFLSQFitter()
fitted_trf = fitter_trf(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print("\nFitting with TRFLSQFitter")
print("Info:", fitter_trf.fit_info.get('message', 'NA'))

