import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from prism.modeling.fitting.scipyfit import ScipyTRF
from prism.modeling.models.agn import nlr
from astropy.modeling import models

np.random.seed(42)
x = np.linspace(4000, 7000, 1000)

base_model = nlr(amplitude=1.5, offset=0, fwhm=300)
y_true = base_model(x)
yerr = np.ones_like(x) * 0.05
y = y_true + np.random.normal(0, yerr, size=x.size)

print("Fitting original data with ScipyTRF...")
fitter = ScipyTRF()
fitted_model = fitter(base_model, x, y, weights=1./yerr)
print("Best fit:", fitted_model.parameters)

print("\n--- Simulating 1 Bootstrap Iteration ---")
# Bootstrap perturb
y_synth = np.random.normal(y, yerr)

trf_fitter = TRFLSQFitter()

print("Using TRFLSQFitter on perturbed data:")
fitted_trf = trf_fitter(fitted_model.copy(), x, y_synth, weights=1./yerr)
print("New params:", fitted_trf.parameters)
print("Is it identical to init?", np.allclose(fitted_trf.parameters, fitted_model.parameters))
print("Info:", trf_fitter.fit_info['message'])

print("\nUsing ScipyTRF on perturbed data:")
fitted_scipy = fitter(fitted_model.copy(), x, y_synth, weights=1./yerr)
print("New params:", fitted_scipy.parameters)
print("Is it identical to init?", np.allclose(fitted_scipy.parameters, fitted_model.parameters))
print("Info eval:", fitter.fit_info['nfev'])

