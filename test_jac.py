import numpy as np
from test_agn import fitted_model, model, spec

J = np.array(model.fit_deriv(spec.wave, *fitted_model.parameters))
print("Jacobian shape:", J.shape)
print("Jacobian contains NaN or Inf?", np.any(~np.isfinite(J)))

valid_cols = ~np.any(~np.isfinite(J), axis=1)
print("Invalid cols:", np.where(~valid_cols)[0])
for idx in np.where(~valid_cols)[0]:
    print("Param:", fitted_model.param_names[idx])
