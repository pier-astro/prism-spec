"""Completion streaming, spawn defaults and bounded worker counts."""
import json
import os
import time
from pathlib import Path
import numpy as np
import pytest
from astropy.modeling.models import Const1D
from prism.modeling import Model
from prism.modeling.fitting import ScipyTRF
from prism.modeling.fitting.multifit import MultiFitMixin
import prism.modeling.fitting.multifit as module

NAMES = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')


class CompletionProbe(MultiFitMixin):
    calc_uncertainties = False
    verbose = False

    def __init__(self, marker):
        self.marker = marker

    def _fit_single_direct(self, idx, *args, **kwargs):
        if idx == 0:
            deadline = time.monotonic()+15
            while not Path(self.marker).exists() and time.monotonic() < deadline:
                time.sleep(.01)
            completed = Path(self.marker).exists()
        else:
            completed = True
        return idx, np.array([float(idx)]), None, {'success': completed, 'nfev': 1}, None


class EnvironmentFitter(ScipyTRF):
    def __call__(self, *args, **kwargs):
        model = super().__call__(*args, **kwargs)
        self.fit_info['message'] = json.dumps({name: os.environ.get(name) for name in NAMES})
        return model


def test_parent_receives_short_fit_before_blocked_fit_and_keeps_indices(tmp_path, monkeypatch):
    marker = tmp_path/'received'
    completions = []
    def progress(entries, **kwargs):
        assert kwargs['total'] == 32
        for entry in entries:
            completions.append(entry[0])
            if entry[0] == 1:
                marker.touch()
            yield entry
    monkeypatch.setattr(module, 'tqdm', progress)
    result = CompletionProbe(str(marker)).multifit(
        Model.from_astropy(Const1D()), np.arange(3.), np.ones((32, 3)), nproc=2)
    assert completions.index(1) < completions.index(0)
    assert sorted(completions) == list(range(32))
    assert result._success.all()
    np.testing.assert_array_equal(result.amplitude.value, np.arange(32.))


def test_spawn_thread_defaults_and_parent_restoration(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    expected_parent = {name: os.environ.get(name) for name in NAMES}
    x = np.linspace(0., 1., 11)
    result = EnvironmentFitter().multifit(Model.from_astropy(Const1D()), x,
                                        np.ones((2, len(x))), nproc=2, progress=False)
    for message in result.messages.flat:
        assert json.loads(message) == {'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '3'}
    assert {name: os.environ.get(name) for name in NAMES} == expected_parent


def test_parent_environment_restored_when_pool_creation_fails(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    class BrokenContext:
        def Pool(self, *args, **kwargs):
            assert all(os.environ[name] == '1' for name in NAMES)
            raise RuntimeError('spawn failed')
    monkeypatch.setattr(module.mp, 'get_context', lambda method: BrokenContext())
    with pytest.raises(RuntimeError, match='spawn failed'):
        with module._multifit_pool(2):
            pass
    assert all(name not in os.environ for name in NAMES)


def test_single_unique_bin_does_not_start_idle_workers(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('No worker pool needed for one target')
    monkeypatch.setattr(module, '_multifit_pool', unexpected)
    x = np.arange(5.)
    result = ScipyTRF().multifit(Model.from_astropy(Const1D()), x, np.ones((4, 5))*2.,
                               binmap=np.zeros(4, dtype=int), nproc=8, progress=False)
    np.testing.assert_allclose(result.amplitude.value, 2.)


@pytest.mark.parametrize('nproc', [0, -1, 1.5])
def test_invalid_worker_count_is_rejected_before_fitting(nproc):
    with pytest.raises(ValueError, match='positive integer'):
        ScipyTRF().multifit(Const1D(), np.arange(3.), np.ones((2, 3)), nproc=nproc)


def test_removed_batch_option_is_rejected_before_fitting():
    with pytest.raises(TypeError, match='no longer accepts batch'):
        ScipyTRF().multifit(Const1D(), np.arange(3.), np.ones((2, 3)), batch=True)
