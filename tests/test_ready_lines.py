"""One-line native constructors preserve fitting, graph and persistence contracts."""
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import Model as AstropyModel
from astropy.io.misc import yaml
from prism.modeling import LineSet
from prism.modeling.lines import GaussianLines, LorentzianLines, VoigtLines, tables
from prism.modeling.fitting import TRF
from prism.modeling.io import load_model


@pytest.mark.parametrize('factory,widths',[(GaussianLines,dict(fwhm=250.)),
    (LorentzianLines,dict(fwhm=250.)),(VoigtLines,dict(fwhm_G=180.,fwhm_L=200.))])
def test_ready_line_fit_links_and_profile_roundtrip(factory,widths,tmp_path):
    table=dict(name=['A','A','B'],position=[4959.,5007.,5100.]*u.AA,weight=[1/3,1.,1.])
    curve=np.array([[4900.,80.],[5200.,140.]])
    model=factory(table,amplitude=1.,instfwhm=curve,bounds={'amplitude':(0.,10.)},name='lines',**widths)
    assert isinstance(model,LineSet) and not isinstance(model,AstropyModel)
    assert model.redshift.fixed
    assert model.amp_a.bounds==(0.,10.) and model.amp_b.bounds==(0.,10.)
    for name in model.param_names:
        getattr(model,name).fixed=not name.startswith('amp_')
    model.amp_b.link=1.75*model.amp_a
    truth=model.copy();truth.amp_a=4.
    x=np.linspace(4900.,5150.,2000)
    fitted=TRF()(model,x,truth(x),inplace=False)
    assert fitted.model.amp_a.value==pytest.approx(4.,rel=1e-7)
    assert fitted.model.amp_b.value==pytest.approx(7.,rel=1e-7)
    assert model.amp_a.value==1.
    fitted.model.save(tmp_path/'ready.fits')
    restored=load_model(tmp_path/'ready.fits')
    np.testing.assert_allclose(restored(x),fitted.predict(x),rtol=1e-13)
    np.testing.assert_allclose(restored.fit_deriv(x,*restored.parameters),fitted.model.fit_deriv(x,*fitted.parameters))


def test_ready_lines_file_and_omitted_weights(tmp_path,monkeypatch):
    from astropy.table import QTable
    table=QTable(dict(name=['A','A'],position=[4959.,5007.]*u.AA,weight=[1/3,1.]))
    table.write(tmp_path/'pair.ecsv')
    monkeypatch.setattr(tables,'linetable_path',str(tmp_path))
    model=GaussianLines('pair.ecsv',fwhm=300.,name='file')
    assert model.param_names==('offset','fwhm','redshift','amp_a')
    omitted=GaussianLines(dict(name=['A','A'],position=[4959.,5007.]*u.AA),name='unweighted')
    assert omitted.param_names==model.param_names
    table.remove_column('weight')
    table.write(tmp_path/'unweighted.ecsv')
    assert GaussianLines('unweighted.ecsv').param_names == model.param_names
    with pytest.raises(ValueError,match='Instrumental'):
        GaussianLines(table,instfwhm=-1.)


def test_read_previous_native_lineset_tag():
    model=GaussianLines(dict(name=['A'],position=[5000.]*u.AA),name='native')
    text=yaml.dump(model)
    assert '!prism.LineSet' in text
    restored=yaml.load(text.replace('!prism.LineSet','!prism.LineComponent'))
    np.testing.assert_allclose(restored(np.array([5000.])),model(np.array([5000.])))
