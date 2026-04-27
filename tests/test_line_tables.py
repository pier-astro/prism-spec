import numpy as np
import pytest
import astropy.units as u
from astropy.table import QTable

from prism.modeling.models.lines import GaussianLine, GaussianLines, profiles, set_medium, setup_local_lines, trim_line_lists
from prism.modeling.models.lines import base as lines_base
from prism.utils.tools import air_to_vac


def _write_ecsv(path, positions, *, unit=u.AA, medium='air'):
    table = QTable()
    table['name'] = ['A', 'B', 'C']
    table['position'] = np.asarray(positions, dtype=float) * unit
    table['weight'] = np.ones(3, dtype=float)
    table.meta['medium'] = medium
    table.write(path, format='ascii.ecsv', overwrite=True)


def _write_csv(path, rows, header='name,position,weight\n'):
    body = ''.join(f"{name},{pos},{weight}\n" for name, pos, weight in rows)
    path.write_text(header + body)


def test_setup_local_lines_accepts_quantity_bounds_and_writes_ecsv_in_angstrom(tmp_path, monkeypatch):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'local'
    _write_ecsv(source_dir / 'demo.ecsv', [5000.0, 6000.0, 7000.0])

    monkeypatch.setattr(lines_base, 'resource_path', str(source_dir))

    min_nu = (6500.0 * u.AA).to(u.Hz, equivalencies=u.spectral())
    max_nu = (5500.0 * u.AA).to(u.Hz, equivalencies=u.spectral())
    setup_local_lines(min=min_nu, max=max_nu, dirpath=str(output_dir), overwrite=True)

    trimmed = QTable.read(output_dir / 'demo.ecsv', format='ascii.ecsv')
    np.testing.assert_allclose(trimmed['position'].to_value(u.AA), [6000.0])
    assert trimmed['position'].unit == u.AA
    assert trimmed.meta['medium'] == 'air'
    text = (output_dir / 'demo.ecsv').read_text()
    assert '# meta:' in text
    assert '__serialized_columns__' not in text
    assert lines_base._wmin == 5500.0
    assert lines_base._wmax == 6500.0


def test_setup_local_lines_accepts_scalar_bounds_with_explicit_unit(tmp_path, monkeypatch):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'local'
    _write_ecsv(source_dir / 'demo.ecsv', [5000.0, 6000.0, 7000.0])

    monkeypatch.setattr(lines_base, 'resource_path', str(source_dir))

    setup_local_lines(min=550.0, max=650.0, unit=u.nm, dirpath=str(output_dir), overwrite=True)

    trimmed = QTable.read(output_dir / 'demo.ecsv', format='ascii.ecsv')
    np.testing.assert_allclose(trimmed['position'].to_value(u.AA), [6000.0])


def test_setup_local_lines_rejects_mixed_quantity_and_scalar_bounds(tmp_path):
    with pytest.raises(ValueError, match='must both be spectral Quantities or both be plain scalars'):
        setup_local_lines(min=5500.0 * u.AA, max=6500.0, dirpath=str(tmp_path), overwrite=False)


def test_trim_line_lists_converts_plain_csv_to_ecsv(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'trimmed'
    _write_csv(source_dir / 'demo.csv', [
        ('A', 1.0, 1.0),
        ('B', 2.0, 1.0),
        ('C', 3.0, 1.0),
    ])

    trim_line_lists(1.5, 2.5, str(source_dir), str(output_dir), unit=u.eV, overwrite=True)

    trimmed = QTable.read(output_dir / 'demo.ecsv', format='ascii.ecsv')
    assert list(trimmed['name']) == ['B']
    np.testing.assert_allclose(trimmed['position'].to_value(u.AA), [(2.0 * u.eV).to_value(u.AA, equivalencies=u.spectral())])
    assert trimmed.meta['medium'] == 'vacuum'


def test_plain_csv_requires_explicit_position_unit(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'trimmed'
    _write_csv(source_dir / 'demo.csv', [
        ('A', 1.0, 1.0),
        ('B', 2.0, 1.0),
        ('C', 3.0, 1.0),
    ])

    with pytest.raises(ValueError, match='Plain CSV line tables require position_unit'):
        trim_line_lists(1.5, 2.5, str(source_dir), str(output_dir), overwrite=True)


def test_from_arrays_defaults_plain_numeric_positions_to_angstrom():
    lines_base.set_wavelength_range(wmin=0.0, wmax=1.0e9)
    model = GaussianLines.from_arrays(
        names=['HeII1640'],
        pos=[1640.42],
        amplitude=1.0,
        fwhm=300.0,
    )

    np.testing.assert_allclose(model.lines['position'].to_value(u.AA), [1640.42])
    assert model.lines['position'].unit == u.AA
    assert model.lines.meta['medium'] == 'air'


def test_from_csv_explicit_medium_overrides_header_and_converts_positions(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_path = source_dir / 'demo.ecsv'
    _write_ecsv(output_path, [5000.0, 6000.0, 7000.0], medium='air')

    model = GaussianLines.from_csv('demo.ecsv', dirpath=str(source_dir), amplitude=1.0, medium='vacuum')

    expected = air_to_vac(np.array([5000.0, 6000.0, 7000.0]))
    np.testing.assert_allclose(model.lines['position'].to_value(u.AA), expected)
    assert model.lines.meta['medium'] == 'vacuum'
    assert model.medium == 'vacuum'


def test_header_medium_wins_over_session_default_for_file_inputs(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_path = source_dir / 'demo.ecsv'
    _write_ecsv(output_path, [5000.0, 6000.0, 7000.0], medium='air')

    original_medium = lines_base._default_medium
    try:
        set_medium('vacuum')
        model = GaussianLines.from_csv('demo.ecsv', dirpath=str(source_dir), amplitude=1.0)
    finally:
        set_medium(original_medium)

    np.testing.assert_allclose(model.lines['position'].to_value(u.AA), [5000.0, 6000.0, 7000.0])
    assert model.lines.meta['medium'] == 'air'
    assert model.medium == 'air'


def test_set_medium_changes_headerless_defaults_for_arrays_and_single_lines():
    original_medium = lines_base._default_medium
    try:
        set_medium('vacuum')
        array_model = GaussianLines.from_arrays(
            names=['HeII1640'],
            pos=[1640.42],
            amplitude=1.0,
            fwhm=300.0,
        )
        single_model = GaussianLine(amplitude=1.0, position=5007.0, fwhm=300.0)
    finally:
        set_medium(original_medium)

    np.testing.assert_allclose(array_model.lines['position'].to_value(u.AA), [1640.42])
    assert array_model.lines.meta['medium'] == 'vacuum'
    assert array_model.medium == 'vacuum'
    assert single_model.medium == 'vacuum'


def test_redshifted_frequency_domain_matches_wavelength_domain_jacobian():
    wavelength_model = GaussianLine(
        amplitude=3.0,
        position=5007.0,
        fwhm=400.0,
        redshift=0.25,
    )
    frequency_model = GaussianLine(
        amplitude=3.0,
        position=5007.0,
        fwhm=400.0,
        redshift=0.25,
        domain='frequency',
    )

    lam = np.linspace(6000.0, 6550.0, 512)
    nu = profiles.from_wavelength_values(lam, 'frequency')

    y_lam = np.asarray(wavelength_model(lam), dtype=float)
    y_nu = np.asarray(frequency_model(nu), dtype=float)
    expected = y_lam * profiles.domain_jacobian(lam, 'frequency')

    np.testing.assert_allclose(y_nu, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(y_lam, lam), abs(np.trapezoid(y_nu, nu)), rtol=5e-6)