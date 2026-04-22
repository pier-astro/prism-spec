# prism.modeling

`prism.modeling` is the Astropy-facing layer of Prism: models, operators, fitting helpers, line analysis, display, and model I/O live here.

## Compatibility Policy

Prism follows a hybrid compatibility strategy:

- prefer Prism-owned subclasses and wrappers where the workflow can stay clean without mutating Astropy globally
- keep a very small number of Astropy monkey-patches only where native Astropy currently blocks transparent interoperability
- keep the resulting objects Astropy-compatible whenever possible

In practice this means Prism aims to let users combine native Astropy models and Prism models naturally, while avoiding a large parallel abstraction layer.

## Import-Time Behavior

Two convenience hooks are intentionally enabled at import time:

- `enable_model_display()`
- `install_model_io_methods()`

These are kept because Prism users typically want richer display and direct FITS save/load methods immediately.

They are convenience patches, not compatibility-critical patches.

Astropy transparency is preserved in the sense that:

- Prism models remain ordinary Astropy-style models
- native Astropy models can still be passed through Prism utilities
- free-function APIs still exist for display and I/O when explicit control is preferred

## Remaining Compatibility-Critical Patches

The only patches Prism currently treats as structurally necessary are the ones that preserve smooth mixed-model behavior for:

- operator-aware `CompoundModel` pipe expressions
- tied analytic Jacobians in those enhanced fitting paths

Those behaviors are documented in the module READMEs under `operators/` and `fitting/`.