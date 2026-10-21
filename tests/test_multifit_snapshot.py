"""Private worker snapshots preserve runtime context and have bounded lifetime."""
from pathlib import Path
import numpy as np
import pytest
from astropy.modeling.models import Const1D
from prism.modeling import Model
from prism.modeling.fitting import TRF
from prism.modeling.fitting.extension import get_max_evaluations, set_max_evaluations
import prism.modeling.fitting.multifit.execution as module


def track_directories(monkeypatch, tmp_path):
    original = module.TemporaryDirectory
    paths = []
    def temporary(*args, **kwargs):
        kwargs['dir'] = tmp_path
        directory = original(*args, **kwargs)
        paths.append(Path(directory.name))
        return directory
    monkeypatch.setattr(module, 'TemporaryDirectory', temporary)
    return paths


@pytest.mark.parametrize('invalid', [False, True])
def test_one_snapshot_cleaned_after_success_or_fit_failure(tmp_path, monkeypatch, invalid):
    paths = track_directories(monkeypatch, tmp_path)
    original = module.dill.dump
    dumps = []
    def dump(value, stream, *args, **kwargs):
        dumps.append(stream.name)
        return original(value, stream, *args, **kwargs)
    monkeypatch.setattr(module.dill, 'dump', dump)
    x = np.arange(5.)
    data = np.ones((2, 5))*3.
    mask = np.ones_like(data, dtype=bool)
    if invalid:
        mask[1] = False
    fitter = TRF()
    model = Model.from_astropy(Const1D(1.))
    if invalid:
        with pytest.raises(ValueError, match='No valid spectral samples'):
            fitter.multifit(model, x, data, mask=mask, skip_invalid=False, nproc=2, progress=False)
    else:
        result = fitter.multifit(model, x, data, nproc=2, progress=False)
        np.testing.assert_allclose(result.amplitude.value, 3.)
    assert len(dumps) == 1
    assert len(paths) == 1
    assert not paths[0].exists()
    assert model.amplitude.value == 1.


def test_global_budget_is_preserved_across_spawn():
    previous = get_max_evaluations()
    try:
        set_max_evaluations(1)
        x = np.arange(5.)
        y = np.ones((2, 5))*3.
        model = Model.from_astropy(Const1D(1.))
        serial = TRF().multifit(model, x, y, progress=False)
        parallel = TRF().multifit(model, x, y, nproc=2, progress=False)
        np.testing.assert_array_equal(parallel.nfev, [1, 1])
        np.testing.assert_array_equal(parallel._success, serial._success)
        assert not parallel._success.any()
        np.testing.assert_array_equal(parallel.amplitude.value, serial.amplitude.value)
    finally:
        set_max_evaluations(previous)


def test_serialization_failure_is_cleaned_before_starting_workers(tmp_path, monkeypatch):
    paths = track_directories(monkeypatch, tmp_path)
    class Unserializable:
        def __reduce__(self):
            raise TypeError('cannot serialize config')
    def unexpected_pool(*args, **kwargs):
        raise AssertionError('Pool must not start after failed serialization')
    monkeypatch.setattr(module, '_multifit_pool', unexpected_pool)
    fitter = TRF()
    fitter.extra = Unserializable()
    with pytest.raises(TypeError, match='cannot serialize config'):
        fitter.multifit(Const1D(), np.arange(3.), np.ones((2, 3)), nproc=2, progress=False)
    assert len(paths) == 1
    assert not paths[0].exists()


@pytest.mark.usefixtures('astropy_extensions')
def test_explicit_astropy_pipe_extension_is_preserved_in_workers():
    from astropy.modeling.models import Gaussian1D
    from scipy.sparse import eye
    from prism.modeling import LinearOperatorModel
    from prism.modeling.interop.astropy import TRFLSQFitter
    native = np.linspace(-3., 3., 81)
    x = native[::2]
    source = Gaussian1D(1., 0., .8, fixed={'mean': True, 'stddev': True})
    model = source | LinearOperatorModel(eye(len(native)), x=native)
    data = np.broadcast_to(2.*np.exp(-.5*(x/.8)**2), (2, len(x)))
    serial = TRFLSQFitter().multifit(model, x, data, progress=False)
    parallel = TRFLSQFitter().multifit(model, x, data, nproc=2, progress=False)
    assert parallel._success.all()
    np.testing.assert_allclose(parallel._params['amplitude_0'], 2., atol=1e-10)
    np.testing.assert_array_equal(parallel._params['amplitude_0'], serial._params['amplitude_0'])
