import numpy as np
from prism.data import Spectrum
from astropy import units as u
import prism.modeling.models as models
from prism.modeling.models.lines import GaussianLines
from scipy.optimize import check_grad

filename = 'examples/data/agnspec.txt'
dec, ra = -25.12345, 13.12345
redshift = 0.0622
spec = Spectrum.from_txt(filename=filename, ra=ra, dec=dec, z=redshift, wave_unit=u.AA, unit=1e-15 * u.erg / (u.s * u.cm**2 * u.AA), name='AGN Spectrum')

models.setup_local_lines(wmin=np.min(spec.wave), wmax=np.max(spec.wave), overwrite=True)

# Test simple nlr line
model = models.agn.nlr(name='nlr', amplitude=1, offset=0, fwhm=500, instfwhm=120)

x = spec.wave[:100]  # Just take 100 points
p = model.parameters

def func(p_arr):
    model.parameters = p_arr
    return model(x)

def jac(p_arr):
    model.parameters = p_arr
    return np.array(model.fit_deriv(x, *p_arr))

# Let's check gradient for a specific point, e.g., the sum
def scalar_func(p_arr):
    return np.sum(func(p_arr))

def scalar_jac(p_arr):
    return np.sum(jac(p_arr), axis=1)

err = check_grad(scalar_func, scalar_jac, p)
print("Gradient check error for AGN NLR:", err)

# Let's evaluate exactly for point 50
model.parameters = p
np_jac = np.zeros((len(p), len(x)))
epsilon = 1e-6
for i in range(len(p)):
    p_up = p.copy()
    p_up[i] += epsilon
    model.parameters = p_up
    y_up = model(x)
    
    p_dn = p.copy()
    p_dn[i] -= epsilon
    model.parameters = p_dn
    y_dn = model(x)
    
    np_jac[i] = (y_up - y_dn) / (2 * epsilon)

an_jac = jac(p)

print("\nMax difference between analytic and numeric jacobian:")
print(np.max(np.abs(an_jac - np_jac)))

for i, name in enumerate(model.param_names):
    diff = np.max(np.abs(an_jac[i] - np_jac[i]))
    print(f"Param {name}: max diff = {diff:.2e}")
