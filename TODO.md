# Deferred improvements

- Evaluate the model on an independent, finer grid and integrate the response-folded prediction over observed bins; consider analytic profile integrals and selective component refinement where useful.
- Optional initialization and parameter-scaling strategies for difficult fits, benchmarked with both SciPy and Sherpa.
- Rework LMFitter for reliable bounded and linked fits.
- Optimize SeparableTRF for line models; validate reduced derivatives, active bounds and covariance.
- Evaluate an optimized XSPEC-inspired Levenberg–Marquardt implementation against the existing engines.
