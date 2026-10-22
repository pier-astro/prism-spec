"""Editable presets and authoritative Spectrum arrays, independent of hidden copies."""
from pathlib import Path
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from prism.modeling import lines
from prism.data import Spectrum


@pytest.fixture(autouse=True)
def isolated_directory(monkeypatch):
    monkeypatch.setattr(lines.tables, '_directory', None)


def test_presets_export_override_fallback_and_explicit_path(tmp_path,monkeypatch):
    directory=tmp_path/'local'
    lines.set_directory(directory)
    baseline=lines.load('oiii')
    assert not directory.exists()
    paths=lines.export(['oiii'],directory=directory,window=(4900.,5100.))
    assert paths==[directory/'oiii.ecsv']
    assert '__serialized_columns__' not in paths[0].read_text()
    local=lines.load('oiii');local['weight']*=2
    local.write(paths[0],overwrite=True)
    np.testing.assert_allclose(lines.load('oiii')['weight'],baseline['weight']*2)
    model=lines.GaussianLines('oiii',fwhm=250.,name='narrow')
    lines.set_directory(None)
    np.testing.assert_allclose(lines.load('oiii')['weight'],baseline['weight'])
    np.testing.assert_allclose(model.component._setup['lines']['weight'],baseline['weight']*2)
    # Export always reads packaged defaults even with local overrides configured.
    lines.set_directory(directory)
    lines.export('oiii',directory=directory,overwrite=True)
    np.testing.assert_allclose(lines.load('oiii')['weight'],baseline['weight'])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):lines.load(Path('oiii.ecsv'))
    with pytest.raises(FileNotFoundError):lines.load('oiii.ecsv')
    np.testing.assert_allclose(lines.load('hydrogen')['weight'],lines.load(lines.tables._preset_path('hydrogen'))['weight'])


def test_export_all_and_overwrite_preflight(tmp_path):
    original=lines.load('oiii')
    original.write(tmp_path/'oiii.ecsv')
    before=(tmp_path/'oiii.ecsv').read_bytes()
    with pytest.raises(FileExistsError):lines.export(['hydrogen','oiii'],directory=tmp_path)
    assert not (tmp_path/'hydrogen.ecsv').exists()
    assert (tmp_path/'oiii.ecsv').read_bytes()==before
    with pytest.raises(ValueError):lines.export('oiii',directory=tmp_path,all=True)
    with pytest.raises(ValueError):lines.export([],directory=tmp_path)
    paths=lines.export(directory=tmp_path,all=True,window=(4900.,5100.),overwrite=True)
    assert len(paths)==len(list(Path(lines.tables.resource_path).glob('*.ecsv')))
    for path in paths:
        table=lines.load(path)
        assert np.all((table['position'].value>=4900.) & (table['position'].value<=5100.))
    assert lines.tables._directory is None


def test_bad_local_override_is_not_silently_replaced(tmp_path):
    lines.set_directory(tmp_path)
    QTable({'name':['A']}).write(tmp_path/'oiii.ecsv')
    with pytest.raises(ValueError, match='position'):lines.load('oiii')
    with pytest.raises(FileNotFoundError,match='Unknown line preset'):lines.load('not_a_preset')


def spectrum():
    return Spectrum(np.linspace(5000.,5100.,11),np.linspace(1.,2.,11),
                    yerr=np.full(11,.1),xerr=np.full(11,.5),
                    xunit='AA',yunit='erg / (s cm2 Angstrom)',redshift=.1,
                    frame='observed',dtype=float,meta={'source':'known'})


def transform(spec):
    spec.x /= 1+spec.redshift
    spec.xerr /= 1+spec.redshift
    spec.y *= 1+spec.redshift
    spec.yerr *= 1+spec.redshift
    spec.frame='rest'


def test_direct_edits_survive_copy_cutout_with_independent_metadata():
    spec=spectrum().cutout(min=5010.,max=5090.)
    original=spec.copy()
    transform(spec)
    spec.yunit=u.Unit('W / (m2 Angstrom)');spec.y*=.001;spec.yerr*=.001
    spec.meta['ebv']=.03
    copied=spec.copy()
    for name in ('x','y','xerr','yerr'):
        np.testing.assert_array_equal(getattr(copied,name),getattr(spec,name))
        assert not np.shares_memory(getattr(copied,name),getattr(spec,name))
    selected=spec.cutout(mask=np.arange(spec.x.size)%2==0)
    np.testing.assert_array_equal(selected.y,spec.y[::2])
    np.testing.assert_array_equal(selected.x,spec.x[::2])
    assert selected.frame=='rest'
    np.testing.assert_allclose(spec.fwhm,spectrum().cutout(min=5010.,max=5090.).fwhm/1.1)
    assert copied.frame=='rest' and copied.yunit==spec.yunit
    np.testing.assert_array_equal(copied.x,spec.x)
    assert copied.meta['ebv']==.03
    copied.meta['ebv']=.5
    assert spec.meta['ebv']==.03
    np.testing.assert_array_equal(original.x,spectrum().x[1:-1])
    assert original.frame=='observed'
    assert not any(hasattr(spec,name) for name in ('mask','reset','orig_x','orig_y'))


def test_slice_edits_and_rebin_keep_current_data_and_uncertainties(tmp_path):
    spec=spectrum();spec.y[:]*=3.;spec.yerr[:]*=3.;transform(spec)
    current=spec.copy()
    np.testing.assert_array_equal(current.y,spec.y)
    rebinned=spec.copy();rebinned.rebin(new_x=spec.x.copy())
    np.testing.assert_allclose(rebinned.y[1:-1],spec.y[1:-1],rtol=1e-12)
    assert rebinned.frame=='rest'
    filename=tmp_path/'spectrum.fits';spec.save(filename);loaded=Spectrum.load(filename)
    np.testing.assert_array_equal(loaded.x,spec.x)
    np.testing.assert_array_equal(loaded.yerr,spec.yerr)
    assert loaded.frame=='rest' and loaded.meta==spec.meta
    selected=rebinned.cutout(mask=np.arange(rebinned.x.size)%2==0)
    np.testing.assert_array_equal(selected.x,rebinned.x[::2])
    assert selected.frame=='rest'
    with pytest.raises(ValueError):spec.frame='velocity'
    spec.frame=None;assert spec.frame is None


def test_reassigned_array_shapes_are_checked_before_data_operations(tmp_path):
    spec=spectrum();spec.x=spec.x[:-1]
    for operation in (spec.copy, lambda: spec.cutout(min=5000.),
                      lambda: spec.rebin(factor=2), lambda: spec.save(tmp_path/'bad.fits')):
        with pytest.raises(ValueError,match='same shape'):operation()
    assert not (tmp_path/'bad.fits').exists()
