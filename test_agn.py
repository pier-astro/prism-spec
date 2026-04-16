import numpy as np
from astropy import units as u
from prism.data import Spectrum
import prism.modeling.models as models
import prism.modeling.fitting as fitting
from prism.modeling.fitting import tie
from astropy.modeling.fitting import TRFLSQFitter

# 1. Load Data
filename = 'examples/data/agnspec.txt'
dec, ra = -25.12345, 13.12345
redshift = 0.0622

spec = Spectrum.from_txt(filename=filename, ra=ra, dec=dec, z=redshift,
                         wave_unit=u.AA, unit=1e-15 * u.erg / (u.s * u.cm**2 * u.AA),
                         name='AGN Spectrum')
spec.deredden()
spec.zcorrect()

models.setup_local_lines(wmin=np.min(spec.wave), wmax=np.max(spec.wave), overwrite=True)

instfwhm = 120. # km/s

bknpower = models.BrokenPowerlaw(
    name='bknpower', x_ref=5450, amplitude=np.median(spec.flux), index1=-2.0, index2=0.0,
    bounds={'x_ref': (5100, 5710), 'amplitude': (1.5, 5.0), 'index1': (-3.0, 0.0), 'index2': (-1.0, 1.5)}
)
blr = models.agn.blr(
    name='blr', amplitude=10, offset=-360, fwhm=3727, instfwhm=instfwhm,
    bounds={'amplitude': (0, 1000), 'offset': (-1000, 1000), 'fwhm': (500, 8000)}
)
nlr = models.agn.nlr(
    name='nlr', amplitude=1, offset=0, fwhm=500, instfwhm=instfwhm,
    bounds={'amplitude': (0, 100), 'offset': (-300, 300), 'fwhm': (0, 800)}
)
hhe_nlr = models.GaussianLines.from_csv(
    name='hhe_nlr', csv_files=['hydrogen.csv', 'helium.csv'], amplitude=1, offset=0, fwhm=500, instfwhm=instfwhm,
    bounds={'amplitude': (0, 100), 'offset': (-300, 300), 'fwhm': (0, 800)}
)
hhe_nlr.offset.tied = tie("nlr", lambda m: m.offset)
hhe_nlr.fwhm.tied = tie("nlr", lambda m: m.fwhm)
oiii_out = models.GaussianLines.from_csv(
    name='oiii', csv_files=['oiii.csv'], amplitude=1, offset=-560, fwhm=850, instfwhm=instfwhm,
    bounds={'amplitude': (0, 100), 'offset': (-1000, 750), 'fwhm': (50, 1500)}
)
fe = models.agn.fe(
    name='fe', offset=0, fwhm=1200, instfwhm=instfwhm,
    bounds={'amplitude': (0, 100), 'offset': (-3000, 3000), 'fwhm': (800, 3000)}
)

model = bknpower + blr + nlr + hhe_nlr + oiii_out + fe

# Use ScipyTRF to get a good starting best fit
print("Initial fit with ScipyTRF...")
fitter_scipy = fitting.ScipyTRF()
fitted_model = fitter_scipy(model=model, x=spec.wave, y=spec.flux, weights=1.0/spec.err)

print("\nSimulating Bootstrap Iteration...")
np.random.seed(42)
y_synth = np.random.normal(spec.flux, spec.err)

print("\n--- TRFLSQFitter (estimate_jacobian=False) ---")
trf_fitter = TRFLSQFitter()
fitted_trf_analytic = trf_fitter(fitted_model.copy(), spec.wave, y_synth, weights=1.0/spec.err, estimate_jacobian=False)
print("Changes?", not np.allclose(fitted_trf_analytic.parameters, fitted_model.parameters))
print("Info:", trf_fitter.fit_info.get('message', 'NA'))

print("\n--- TRFLSQFitter (estimate_jacobian=True) ---")
fitted_trf_numeric = trf_fitter(fitted_model.copy(), spec.wave, y_synth, weights=1.0/spec.err, estimate_jacobian=True)
print("Changes?", not np.allclose(fitted_trf_numeric.parameters, fitted_model.parameters))
print("Info:", trf_fitter.fit_info.get('message', 'NA'))

print("\n--- TRFLSQFitter (acc=1e-12) ---")
fitted_trf_acc = trf_fitter(fitted_model.copy(), spec.wave, y_synth, weights=1.0/spec.err, estimate_jacobian=False, acc=1e-12)
print("Changes?", not np.allclose(fitted_trf_acc.parameters, fitted_model.parameters))
print("Info:", trf_fitter.fit_info.get('message', 'NA'))
fitted_scipy_boot = fitter_scipy(fitted_model.copy(), spec.wave, y_synth, weights=1.0/spec.err)
print("Changes?", not np.allclose(fitted_scipy_boot.parameters, fitted_model.parameters))
print("Info nfev:", fitter_scipy.fit_info['nfev'])

