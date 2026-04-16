import numpy as np
from astropy.modeling import models, fitting
import warnings

# Create a noisy Gaussian
x = np.linspace(-10, 10, 100)
baseline = 10.0
amp = 50.0
mean = 1.2
stddev = 2.5
y_true = baseline + amp * np.exp(-0.5 * ((x - mean) / stddev)**2)

np.random.seed(42)
yerr = np.ones_like(x) * 2.0
y = y_true + np.random.normal(0, yerr, size=x.size)

# Create a compound model
model = models.Const1D(amplitude=5.0) + models.Gaussian1D(amplitude=10.0, mean=0.0, stddev=1.0)
model.amplitude_0.bounds = (0, 20)
model.amplitude_1.bounds = (0, 100)
model.mean_1.bounds = (-5, 5)
model.stddev_1.bounds = (0.1, 5.0)

fitter = fitting.TRFLSQFitter()

print("--- Default TRFLSQFitter ---")
try:
    fitted_1 = fitter(model, x, y, weights=1./yerr)
    print(fitted_1)
    print(f"Info: {fitter.fit_info['message']}")
except Exception as e:
    print(e)


# Try without analytic derivatives? Astropy fitters allow passing estimate_jacobian=False but maybe TRFLSQFitter doesn't use analytic by default, or it does if the model has derivatives.
from scipy.optimize import least_squares

print("\n--- TRFLSQFitter with better tolerances ---")
try:
    fitted_2 = fitter(model, x, y, weights=1./yerr, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    print(fitted_2)
    print(f"Info: {fitter.fit_info['message']}")
except Exception as e:
    print(e)

print("\n--- ScipyTRF ---")
try:
    from prism.modeling.scipyfit import ScipyTRF
    scipy_fitter = ScipyTRF()
    fitted_3 = scipy_fitter(model, x, y, weights=1./yerr)
    print(fitted_3)
except Exception as e:
    print("ScipyTRF failed or not found:", e)

