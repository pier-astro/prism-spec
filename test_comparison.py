import numpy as np
import time
from prism.data import Spectrum
from astropy import units as u
from astropy.modeling.fitting import TRFLSQFitter
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
init_params = model.parameters.copy()

print("0. Initial Model Evaluation")
initial_chi2 = np.sum((model(spec.wave) - spec.flux)**2 / spec.err**2)
print("   Initial True Chi2:", initial_chi2)

print("\n1. Testing SherpaLM")
f_sherpa = fitting.SherpaLM()
t0 = time.time()
fit_sherpa = f_sherpa(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print(f"   SherpaLM Time: {time.time()-t0:.2f}s | Cost: {f_sherpa.fit_info.get('cost', 'N/A')} | stat: {f_sherpa.fit_info.get('stat', 'NA')}")
chi2_sherpa = np.sum((fit_sherpa(spec.wave) - spec.flux)**2 / spec.err**2)
print("   SherpaLM True Chi2:", chi2_sherpa)

print("\n2. Testing ScipyTRF default")
f_scipy = fitting.ScipyTRF()
t0 = time.time()
fit_scipy1 = f_scipy(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print(f"   ScipyTRF Time: {time.time()-t0:.2f}s | Cost: {f_scipy.fit_info.get('cost', 'N/A')} | stat: {f_scipy.fit_info.get('stat', 'NA')}")
chi2_scipy1 = np.sum((fit_scipy1(spec.wave) - spec.flux)**2 / spec.err**2)
print("   ScipyTRF True Chi2:", chi2_scipy1)

print("\n3. Testing ScipyTRF with 2-point jacobian")
f_scipy3 = fitting.ScipyTRF(jac='2-point')
t0 = time.time()
fit_scipy3 = f_scipy3(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print(f"   ScipyTRF 2-point Time: {time.time()-t0:.2f}s")
chi2_scipy3 = np.sum((fit_scipy3(spec.wave) - spec.flux)**2 / spec.err**2)
print("   ScipyTRF 2-point True Chi2:", chi2_scipy3)

print("\n5. Testing ScipyTRF with x_scale=1.0")
f_scipy_xs = fitting.ScipyTRF(x_scale=1.0)
t0 = time.time()
fit_scipy_xs = f_scipy_xs(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print(f"   ScipyTRF x_scale=1.0 Time: {time.time()-t0:.2f}s")
chi2_scipy_xs = np.sum((fit_scipy_xs(spec.wave) - spec.flux)**2 / spec.err**2)
print("   ScipyTRF x_scale=1.0 True Chi2:", chi2_scipy_xs)

print("\n7. Testing ScipyTRF with large diff_step")
f_scipy_diff = fitting.ScipyTRF(jac='2-point', diff_step=1e-2)
t0 = time.time()
fit_scipy_diff = f_scipy_diff(model.copy(), spec.wave, spec.flux, weights=1.0/spec.err)
print(f"   ScipyTRF BigDiff Time: {time.time()-t0:.2f}s")
chi2_scipy_diff = np.sum((fit_scipy_diff(spec.wave) - spec.flux)**2 / spec.err**2)
print("   ScipyTRF BigDiff True Chi2:", chi2_scipy_diff)


