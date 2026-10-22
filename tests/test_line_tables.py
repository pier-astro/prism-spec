import numpy as np
from line_helpers import make_line, make_group
import pytest
import astropy.units as u
from astropy.table import QTable

from prism.modeling.lines import profiles
from prism.modeling.lines import export, set_directory, trim_line_lists
from prism.modeling.lines import tables as line_tables
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


def test_export_accepts_quantity_window_and_preserves_ecsv_units(tmp_path, monkeypatch):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'local'
    _write_ecsv(source_dir / 'demo.ecsv', [5000.0, 6000.0, 7000.0])

    monkeypatch.setattr(line_tables, 'resource_path', str(source_dir))

    min_nu = (6500.0 * u.AA).to(u.Hz, equivalencies=u.spectral())
    max_nu = (5500.0 * u.AA).to(u.Hz, equivalencies=u.spectral())
    export('demo', window=(550*u.nm, 650*u.nm), directory=output_dir)

    trimmed = QTable.read(output_dir / 'demo.ecsv', format='ascii.ecsv')
    np.testing.assert_allclose(trimmed['position'].to_value(u.AA), [6000.0])
    assert trimmed['position'].unit == u.AA
    assert trimmed.meta['medium'] == 'air'
    text = (output_dir / 'demo.ecsv').read_text()


def test_export_accepts_scalar_window_in_native_unit(tmp_path, monkeypatch):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_dir = tmp_path / 'local'
    _write_ecsv(source_dir / 'demo.ecsv', [5000.0, 6000.0, 7000.0])

    monkeypatch.setattr(line_tables, 'resource_path', str(source_dir))

    export('demo', window=(5500.,6500.), directory=output_dir)

    trimmed = QTable.read(output_dir / 'demo.ecsv', format='ascii.ecsv')
    np.testing.assert_allclose(trimmed['position'].to_value(u.AA), [6000.0])


def test_export_rejects_reversed_window(tmp_path):
    with pytest.raises(ValueError, match='increasing'):
        export('oiii', window=(6500.,5500.), directory=tmp_path)


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
    np.testing.assert_allclose(
        trimmed['position'].to(u.AA, equivalencies=u.spectral()).value,
        [(2.0 * u.eV).to_value(u.AA, equivalencies=u.spectral())]
    )


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


def test_from_arrays_requires_explicit_position_unit_for_plain_arrays():
    """Bare float arrays must always come with an explicit position_unit."""
    model = make_group(
        names=['HeII1640'],
        pos=[1640.42],
        amplitude=1.0,
        fwhm=300.0,
        position_unit=u.AA,
    )

    np.testing.assert_allclose(np.asarray(model.component._setup['lines']['position']), [1640.42])
    assert model.component._setup['unit'] == u.AA.to_string()


def test_from_arrays_raises_without_position_unit():
    """Passing bare floats without position_unit must raise a clear error."""
    with pytest.raises(ValueError, match='position_unit'):
        make_group(
            names=['HeII1640'],
            pos=[1640.42],
            amplitude=1.0,
            fwhm=300.0,
        )


def test_convert_medium_csv(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    output_path = source_dir / 'demo.ecsv'
    _write_ecsv(output_path, [5000.0, 6000.0, 7000.0], medium='air')
    
    # 1. Convert to vacuum
    line_tables.convert_medium_csv(str(output_path), 'vacuum', overwrite=True)
    table_vac = QTable.read(output_path, format='ascii.ecsv')
    expected_vac = air_to_vac(np.array([5000.0, 6000.0, 7000.0]))
    np.testing.assert_allclose(table_vac['position'].to_value(u.AA), expected_vac)
    assert table_vac.meta['medium'] == 'vacuum'
    
    # 2. Convert back to air using directory processing
    line_tables.convert_medium_csv(str(source_dir), 'air', overwrite=True)
    table_air = QTable.read(output_path, format='ascii.ecsv')
    np.testing.assert_allclose(table_air['position'].to_value(u.AA), [5000.0, 6000.0, 7000.0])
    assert table_air.meta['medium'] == 'air'
    
    # 3. Blind conversion (no medium in meta) - CSV requires explicit input_unit
    csv_path = source_dir / 'blind.csv'
    _write_csv(csv_path, [('A', 5000.0, 1.0)])
    line_tables.convert_medium_csv(str(csv_path), 'vacuum', input_unit=u.AA, overwrite=False)
    
    blind_ecsv_path = source_dir / 'blind.ecsv'
    table_blind = QTable.read(blind_ecsv_path, format='ascii.ecsv')
    expected_blind = air_to_vac(np.array([5000.0]))
    np.testing.assert_allclose(table_blind['position'].to_value(u.AA), expected_blind)
    assert table_blind.meta['medium'] == 'vacuum'
    
    # 4. Verify that CSV without input_unit raises ValueError
    csv_no_unit = source_dir / 'no_unit.csv'
    _write_csv(csv_no_unit, [('B', 6000.0, 1.0)])
    with pytest.raises(ValueError, match="input_unit"):
        line_tables.convert_medium_csv(str(csv_no_unit), 'vacuum', overwrite=True)


def test_native_kinematics_linear_vs_wavelength_domain():
    z = 0.25
    pos_aa = 5007.0
    pos_ev = (pos_aa * u.AA).to_value(u.eV, equivalencies=u.spectral())
    fwhm_kms = 400.0

    lam = np.linspace(6000.0, 6550.0, 512)
    ev_grid = (lam * u.AA).to_value(u.eV, equivalencies=u.spectral())

    # domain_family is inferred from the unit on position
    wavelength_model = make_line(
        amplitude=3.0,
        position=pos_aa * u.AA,      # length unit -> 'wavelength' kinematics
        fwhm=fwhm_kms * u.km / u.s,
        redshift=z,
    )
    linear_model = make_line(
        amplitude=3.0,
        position=pos_ev * u.eV,      # energy unit -> 'linear' kinematics
        fwhm=fwhm_kms * u.km / u.s,
        redshift=z,
    )

    y_lam = np.asarray(wavelength_model(lam), dtype=float)
    y_ev = np.asarray(linear_model(ev_grid), dtype=float)

    # Peak positions must map to the same physical wavelength.
    peak_lam = lam[np.argmax(y_lam)]
    peak_ev = ev_grid[np.argmax(y_ev)]

    peak_ev_as_lam = (peak_ev * u.eV).to_value(u.AA, equivalencies=u.spectral())
    np.testing.assert_allclose(peak_lam, peak_ev_as_lam, rtol=1e-3)


def test_agn_factory_uses_current_local_line_table_path(tmp_path,monkeypatch):
    from prism.modeling.models import agn
    monkeypatch.setattr(line_tables, '_directory', None)
    set_directory(tmp_path/'lines')
    export('feII_model', window=(4700.,7000.), directory=tmp_path/'lines')
    iron=agn.fe(name='iron',fwhm=1200.,window=(4800.,6900.))
    positions=iron.component._setup['lines']['position']
    assert iron.name=='iron' and len(positions)>0
    assert min(positions)>=4800 and max(positions)<=6900


@pytest.mark.parametrize("conversion", ["domain", "medium"])
@pytest.mark.parametrize("selection", ["file", "directory", "sequence"])
def test_table_converters_accept_path_objects(tmp_path, conversion, selection):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "lines.ecsv"
    _write_ecsv(path, [4958.911, 5006.843, 6562.8])
    output = tmp_path / "converted"
    paths = {"file": path, "directory": source, "sequence": [path]}[selection]
    if conversion == "domain":
        line_tables.convert_domain_csv(paths, u.eV, output_dir=output)
    else:
        line_tables.convert_medium_csv(paths, "vacuum", output_dir=output)
    result = QTable.read(output / "lines.ecsv", format="ascii.ecsv")
    if conversion == "domain":
        expected = (np.array([4958.911, 5006.843, 6562.8])*u.AA).to_value(
            u.eV, equivalencies=u.spectral())
        np.testing.assert_allclose(result["position"].to_value(u.eV), expected)
    else:
        assert result.meta["medium"] == "vacuum"
        assert np.all(result["position"].to_value(u.AA) > [4958.911, 5006.843, 6562.8])
