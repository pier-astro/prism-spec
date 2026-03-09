#!/usr/bin/env python3
"""
Test script to verify current fitting architecture works before refactoring.
Tests all available fitters with a simple Gaussian model.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
import prism.modeling.fitting as fitting

# Generate test data
np.random.seed(42)
x = np.linspace(-5, 5, 100)
true_gaussian = models.Gaussian1D(amplitude=10, mean=1, stddev=2)
y_true = true_gaussian(x)
y_noise = 0.5 * np.random.randn(len(x))
y = y_true + y_noise
yerr = np.full_like(y, 0.5)

# Test model
model = models.Gaussian1D(amplitude=5, mean=0, stddev=1)
model.amplitude.bounds = (0, 20)
model.mean.bounds = (-3, 3)  
model.stddev.bounds = (0.1, 5)

print("Testing Current Fitting Architecture")
print("=" * 50)

# Test each fitter type
fitters_to_test = [
    ('AstropyTRF', fitting.AstropyTRF(calc_uncertainties=True, verbose=False)),
    ('AstropyDogBox', fitting.AstropyDogBox(calc_uncertainties=True, verbose=False)),
    ('ScipyTRF', fitting.ScipyTRF(calc_uncertainties=True, verbose=False)),
    ('ScipyDogBox', fitting.ScipyDogBox(calc_uncertainties=True, verbose=False)),
    ('LMFitter', fitting.LMFitter(calc_uncertainties=True, verbose=False)),
]

# Add Sherpa fitters if available
if fitting.HAS_SHERPA:
    fitters_to_test.append(
        ('SherpaLevMar', fitting.SherpaLevMar(calc_uncertainties=True, verbose=False))
    )

results = {}
for name, fitter in fitters_to_test:
    try:
        print(f"\n{name}:")
        fitted_model = fitter(model.copy(), x, y, yerr=yerr, maxiter=1000)
        
        # Check results
        fit_info = fitter.fit_info
        success = fit_info.get('success', False)
        nfev = fit_info.get('nfev', '?')
        time_ms = fit_info.get('time', 0) * 1000
        
        # Parameter values
        amp = fitted_model.amplitude.value
        mean = fitted_model.mean.value  
        std = fitted_model.stddev.value
        
        # Check covariance
        has_cov = fitter.covariance is not None
        has_stds = fitter.stdevs is not None
        
        print(f"  ✓ Success: {success}, nfev: {nfev}, time: {time_ms:.1f}ms")
        print(f"  ✓ Fitted: amp={amp:.2f}, mean={mean:.2f}, std={std:.2f}")
        print(f"  ✓ Covariance: {has_cov}, Stdevs: {has_stds}")
        
        results[name] = {
            'success': success,
            'params': [amp, mean, std],
            'has_uncertainties': has_cov and has_stds,
            'time': time_ms
        }
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results[name] = {'success': False, 'error': str(e)}

# Summary
print("\n" + "=" * 50)
print("SUMMARY:")
successful = [name for name, res in results.items() if res.get('success', False)]
failed = [name for name, res in results.items() if not res.get('success', False)]

print(f"✓ Successful fitters ({len(successful)}): {', '.join(successful)}")
if failed:
    print(f"✗ Failed fitters ({len(failed)}): {', '.join(failed)}")

# Check parameter consistency
if len(successful) > 1:
    print(f"\nParameter Consistency Check:")
    ref_params = results[successful[0]]['params']
    print(f"  Reference ({successful[0]}): amp={ref_params[0]:.3f}, mean={ref_params[1]:.3f}, std={ref_params[2]:.3f}")
    
    for name in successful[1:]:
        params = results[name]['params']
        amp_diff = abs(params[0] - ref_params[0])
        mean_diff = abs(params[1] - ref_params[1])  
        std_diff = abs(params[2] - ref_params[2])
        
        consistent = amp_diff < 0.1 and mean_diff < 0.1 and std_diff < 0.1
        status = "✓" if consistent else "✗"
        print(f"  {status} {name}: amp={params[0]:.3f} (Δ{amp_diff:.3f}), mean={params[1]:.3f} (Δ{mean_diff:.3f}), std={params[2]:.3f} (Δ{std_diff:.3f})")

print(f"\nTrue values: amp={true_gaussian.amplitude.value}, mean={true_gaussian.mean.value}, std={true_gaussian.stddev.value}")
print("Current architecture test completed.")