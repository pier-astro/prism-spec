# Operators

`prism.modeling.operators` extends Astropy's native pipe operator so a right-hand Prism operator can see both:

- the left-hand model output `f(x, θ)`
- the original input grid `x`

This makes it possible to write native Astropy expressions such as:

```python
model = source | rsp
```

and evaluate them as

```text
h(x, θ) = M(x, φ) @ f(x, θ)
```

where `M` is a matrix operator and `φ` are optional operator parameters.

## Why A Monkey-Patch Exists Here

This module intentionally monkey-patches `astropy.modeling.CompoundModel` for a narrow, opt-in case.

That patch is kept because removing it would break the cleanest workflow Prism users rely on:

```python
model = native_astropy_source | prism_operator
```

Without the patch, Astropy's native pipe machinery does not expose enough information to the right-hand Prism operator, especially the original input grid needed by wavelength-aware operators.

Prism keeps this patch because it preserves Astropy transparency better than introducing a separate opaque compound-model system.

The result remains an Astropy `CompoundModel`, not a hidden Prism-only replacement.

## How The Patch Works

Astropy's native `|` operator only passes the left-hand output into the right-hand model. That is not enough for wavelength-aware operators such as instrumental responses, because they also need the original grid `x`.

Prism patches `astropy.modeling.CompoundModel` only for an opt-in case:

- the compound node uses `op == '|'`
- the right-hand model defines `_prism_pipe_evaluate(...)`

When that is true, Prism intercepts:

- `CompoundModel._evaluate` for `model(x)`
- `CompoundModel.evaluate` for fitter/objective evaluation
- `CompoundModel.fit_deriv` for analytic Jacobian propagation

Everything else remains ordinary Astropy behavior.

This is the main compatibility-critical monkey-patch Prism still intends to keep unless Astropy eventually provides a small public hook for this evaluation path.

## Core Type

The generic right-hand operator is `LinearOperatorModel` from `matop.py`.

The base class stores:

- `matrix`: the fixed matrix
- `x`: optional input-grid metadata
- `recipe`: reconstruction metadata used by Prism serialization

The base hook behavior is:

- `_prism_pipe_evaluate(leftval, left_inputs, right_params, **kwargs)`
- `_prism_pipe_fit_deriv(left_deriv, left_inputs, left_params, right_params, left_model)`

The default Jacobian rule is the linear chain rule:

```text
J_h = M @ J_f
```

## Accepted Expression Shapes

These are supported analytically:

- `source | rsp`
- `(source1 + source2) | rsp`
- `(source | rsp) + plain_model`
- `(source | rsp) * plain_model`
- `(source | rsp1) | rsp2`
- `(source1 | rsp1) + (source2 | rsp2)`

These evaluate correctly but fall back to numeric Jacobians because native Astropy `|` nodes do not provide `fit_deriv`:

- `(source | native_pipe) | rsp`
- `(source | rsp) | native_pipe`
- `(source | rsp) + (source2 | native_pipe)`

Use `has_native_pipe(model)` if you want to detect that case early.

## Instrumental Response Example

```python
import numpy as np
from astropy.modeling.models import Gaussian1D
from prism.modeling.operators import InstrumentResponse, SpectralResponse

wave = np.arange(4700.0, 5100.0, 1.25)
source = Gaussian1D(amplitude=1.0, mean=5007.0, stddev=3.0)
response = InstrumentResponse.from_fixed_fwhm(wave, fwhm=2.5)
rsp = SpectralResponse(instrument=response, wave=wave)
model = source | rsp
flux = model(wave)
```

## Custom Operator Example

If your operator is still a matrix transform but depends on the input grid, subclass `LinearOperatorModel` and override `get_matrix`:

```python
import numpy as np
from prism.modeling.operators import LinearOperatorModel

class DiagonalAttenuator(LinearOperatorModel):
    def get_matrix(self, x, *right_params):
        tau = 0.001 * np.asarray(x)
        return np.diag(np.exp(-tau))
```

Then use it with the native pipe:

```python
op = DiagonalAttenuator(np.eye(len(wave)), x=wave, name='atten')
model = source | op
```

## Custom Operator With Analytic Derivatives

If your operator has its own parameters and the matrix depends on them, override `_prism_pipe_fit_deriv`.

The hook receives:

- `left_deriv`: analytic derivative of the left-hand source model
- `left_inputs`: original source inputs
- `left_params`: current source parameters
- `right_params`: current operator parameters
- `left_model`: the left-hand source model object

A common pattern is:

1. transform the source derivatives through your matrix
2. append additional derivatives with respect to operator parameters

Sketch:

```python
def _prism_pipe_fit_deriv(self, left_deriv, left_inputs, left_params, right_params, left_model):
    base = super()._prism_pipe_fit_deriv(
        left_deriv, left_inputs, left_params, right_params, left_model
    )
    x = np.asarray(left_inputs[0])
    source_flux = np.asarray(left_model.evaluate(*left_inputs, *left_params)).ravel()

    # Example: derivative of M(theta) @ f with respect to theta
    dmatrix_dtheta = ...
    dtheta = np.asarray(dmatrix_dtheta.dot(source_flux)).ravel()
    return np.vstack([base, dtheta])
```

That keeps the left-hand analytic derivatives and extends them with operator-parameter derivatives.

## Practical Notes

- `model.left` is the intrinsic source model.
- `model.right` is the right-hand operator model.
- `is_linear_operator_pipe(model)` detects Prism-enhanced `|` expressions.
- `has_native_pipe(model)` detects a native `|` subtree that will force numeric Jacobians upstream.

## Astropy Transparency Summary

- expressions still use Astropy's native operator syntax
- the resulting object is still an Astropy `CompoundModel`
- native Astropy fitters can still operate on the resulting model
- Prism only intercepts the specific evaluation and Jacobian path needed to keep wavelength-aware operators correct
