# Interoperability tools

`interop.astropy` contains optional helpers that extend Astropy rather than the
native Prism model. Importing it does not change Astropy display methods.

```python
from prism.modeling.interop.astropy import tie, show_model
model.stddev_1.tied = tie('core', lambda component: 2 * component.stddev)
show_model(model, mode='text')
```

For automatic notebook display of Astropy models, explicitly call
`enable_model_display()`; `disable_model_display()` restores their methods.
Native Prism models provide their own display and use declarative `.link`
expressions. No display activation is needed for them.

Use `Model.from_astropy(model)` to bring a supported Astropy model/expression
into Prism. Numerical fitting adapters remain in `modeling.fitting`; they are
part of Prism's fitting functionality rather than generic Astropy extensions.

Migration: import `tie` here instead of `fitting` or `fitting.utils`. Shared
formatting functions live in `modeling.display`; importing that module no longer
activates extensions. The former `modeling.models.display` module is removed.
For Astropy-root expressions only, explicitly call `install_model_io_methods()`
to add `.save()` / `.to_fits()` and `.covariance` to Astropy models, or
`install_pipe_patch()` to evaluate `astropy_source | prism_operator`. Both live
here and are idempotent; importing Prism, fitting or I/O no longer installs them.
They modify Astropy classes for the process and should be chosen deliberately.
Without I/O installation, `modeling.io.model_to_fits(model, filename)` remains
available for supported Astropy models. For the native path, convert with
`Model.from_astropy(source)` before composing an operator; native `.save()` and
`.covariance` require no class extensions. This namespace is an extraction boundary,
not yet a standalone distribution independent of Prism.
