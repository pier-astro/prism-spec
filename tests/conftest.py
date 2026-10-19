"""Explicit activation for tests of the optional Astropy extension path."""
import pytest


@pytest.fixture
def astropy_extensions():
    from astropy.modeling import Model, CompoundModel
    from prism.modeling.interop.astropy import install_model_io_methods, install_pipe_patch
    missing = object()
    attributes = {
        Model: ('save', 'to_fits', 'covariance', '_prism_io_installed'),
        CompoundModel: ('_evaluate', 'evaluate', 'fit_deriv', '_prism_linear_operator_patch'),
    }
    original = {(cls, name): cls.__dict__.get(name, missing)
                for cls, names in attributes.items() for name in names}
    install_model_io_methods()
    install_pipe_patch()
    try:
        yield
    finally:
        for (cls, name), value in original.items():
            if value is missing:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)
