"""
Phase 3 refactor verification tests.

Covers:
  Test 1 — dynamic domain_family inference from position unit physical_type
  Test 2 — Astropy input_units property for automatic unit conversion
  Test 3 — velocity_unit scaling factor for optimizer step-size control
"""
import numpy as np
import pytest
import astropy.units as u

from prism.modeling.models.lines import GaussianLine, GaussianLines


# ---------------------------------------------------------------------------
# Test 1: Dynamic domain inference
# ---------------------------------------------------------------------------

def test_domain_family_wavelength_from_angstrom():
    """AA is length -> 'wavelength' kinematics."""
    line = GaussianLine(position=5000 * u.AA)
    assert line.domain_family == 'wavelength'


def test_domain_family_wavelength_from_nm():
    """nm is also length -> 'wavelength' kinematics."""
    line = GaussianLine(position=500 * u.nm)
    assert line.domain_family == 'wavelength'


def test_domain_family_linear_from_ev():
    """eV is energy -> 'linear' kinematics."""
    line = GaussianLine(position=2.5 * u.eV)
    assert line.domain_family == 'linear'


def test_domain_family_linear_from_hz():
    """Hz is frequency -> 'linear' kinematics."""
    line = GaussianLine(position=6e14 * u.Hz)
    assert line.domain_family == 'linear'


def test_domain_family_is_always_set_because_class_declares_unit():
    """
    GaussianLine.position declares unit=u.AA at the class level. _coerce_init_parameter
    wraps any bare float in that unit, so domain_family is always resolvable even when
    the user passes a plain number.
    """
    line = GaussianLine(position=5000.0)   # coerced to 5000 AA by the class default
    assert line.domain_family == 'wavelength'  # AA -> length -> 'wavelength'


# ---------------------------------------------------------------------------
# Test 2: Astropy quantity evaluation (input_units wrapper)
# ---------------------------------------------------------------------------

def test_input_units_maps_to_position_unit():
    """input_units property must expose the position unit under the first input name."""
    line = GaussianLine(position=5000 * u.AA, fwhm=300 * u.km / u.s)
    iu = line.input_units
    assert iu is not None
    assert line.inputs[0] in iu
    assert iu[line.inputs[0]] == u.AA


def test_input_units_always_set_because_class_declares_unit():
    """
    Because GaussianLine.position carries unit=u.AA, even a bare-float construction
    yields a position with a unit, so input_units always returns a dict.
    """
    line = GaussianLine(position=5000.0)   # coerced to AA
    assert line.input_units == {line.inputs[0]: u.AA}


def test_astropy_auto_converts_nm_to_angstrom():
    """
    Astropy intercepts the nm Quantity, converts it to AA (position unit),
    and passes bare floats to evaluate -> result must equal bare-float call.
    """
    line = GaussianLine(position=5000 * u.AA, fwhm=300 * u.km / u.s)

    val_bare = line(5000.0)
    val_quant = line(500.0 * u.nm)   # 500 nm == 5000 AA

    assert np.isclose(val_bare, val_quant)


def test_astropy_auto_converts_micron_to_angstrom():
    """Same check with microns."""
    line = GaussianLine(position=5000 * u.AA, fwhm=300 * u.km / u.s)
    val_bare = line(5000.0)
    val_um = line(0.5 * u.micron)    # 0.5 µm == 5000 AA
    assert np.isclose(val_bare, val_um)


# ---------------------------------------------------------------------------
# Test 3: velocity_unit scaling (optimizer step-size control)
# ---------------------------------------------------------------------------

def test_velocity_unit_ms_equals_kms():
    """
    A model expecting floats in m/s and one expecting km/s must evaluate
    identically when given numerically equivalent velocities.
    """
    # 300 000 m/s == 300 km/s
    line_ms = GaussianLine(
        position=5000 * u.AA,
        fwhm=300_000.0,           # bare float, interpreted as m/s
        velocity_unit=u.m / u.s,
    )
    line_kms = GaussianLine(
        position=5000 * u.AA,
        fwhm=300.0,               # bare float, interpreted as km/s (default)
    )

    assert np.isclose(line_ms(5000.0), line_kms(5000.0))


def test_velocity_unit_scaling_factor():
    """_to_kms must hold the correct multiplier."""
    line_ms = GaussianLine(position=5000 * u.AA, velocity_unit=u.m / u.s)
    # 1 m/s -> km/s multiplier is 0.001
    assert np.isclose(line_ms._to_kms, 1e-3)

    line_kms = GaussianLine(position=5000 * u.AA)
    assert np.isclose(line_kms._to_kms, 1.0)


def test_velocity_unit_rejects_non_velocity():
    """Passing a non-velocity unit must raise immediately."""
    with pytest.raises(ValueError, match="velocity"):
        GaussianLine(position=5000 * u.AA, velocity_unit=u.AA)


# ---------------------------------------------------------------------------
# Regression: groups still infer domain from position unit
# ---------------------------------------------------------------------------

def test_gaussian_lines_domain_family_inferred_from_position():
    """GaussianLines built with eV positions must report 'linear'."""
    model = GaussianLines.from_arrays(
        names=['Ha', 'NII'],
        pos=(np.array([6563.0, 6583.0]) * u.AA).to(u.eV, equivalencies=u.spectral()),
        amplitude=1.0,
        fwhm=300 * u.km / u.s,
    )
    assert model.domain_family == 'linear'


def test_gaussian_lines_domain_family_wavelength():
    """GaussianLines built with AA positions must report 'wavelength'."""
    model = GaussianLines.from_arrays(
        names=['Ha', 'NII'],
        pos=[6563.0, 6583.0],
        position_unit=u.AA,      # explicit, no silent default
        amplitude=1.0,
        fwhm=300 * u.km / u.s,
    )
    assert model.domain_family == 'wavelength'
