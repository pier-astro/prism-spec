"""
PyYAML / Astropy-IO serialization hooks for prism-spec custom models.

Calling ``register()`` (done automatically on import) ensures that
``astropy.io.misc.yaml.dump / load`` can round-trip every prism model,
including dynamically-built ``LineGroupBase`` subclasses and compound models.

Strategy
--------
* ``LineModelBase`` subclasses (``GaussianLine``, ``VoigtLine``, …) are
  ordinary Astropy ``Fittable1DModel`` subclasses → representer encodes
  class name + per-parameter state; constructor inverts this.

* ``LineGroupBase`` subclasses (``GaussianLines``, …) are dynamically created
  via ``type()`` inside ``from_templates``.  The representer stores:
  - the *base* class name (e.g. ``GaussianLines``)
  - the ``_df`` as a list of records
  - ``instfwhm``
  - current parameter values / bounds / fixed flags
  The constructor calls ``cls.from_templates(df, ...)`` to reconstruct the
  identical dynamic class, then restores the parameter state.
"""

import warnings
import numpy as np
import pandas as pd
import re

from astropy.modeling.core import CompoundModel

from astropy.io.misc.yaml import AstropyDumper, AstropyLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_state(param):
    """Compact serialisable dict for one Astropy Parameter."""
    state = {
        'value':  float(param.value),
        'fixed':  bool(param.fixed),
        'bounds': list(param.bounds) if param.bounds != (None, None) else [None, None],
    }
    # Persist prism uncertainty attributes when available; scalars only.
    for attr in ('std', 'lolim', 'uplim', 'median'):
        raw = getattr(param, attr, None)
        if raw is None:
            continue
        try:
            val = float(raw)
            if np.isfinite(val):
                state[attr] = val
        except (TypeError, ValueError):
            pass  # skip array-valued or non-serialisable extras
    return state

def _restore_param(model, name, state):
    param = getattr(model, name)
    param.value = state['value']
    param.fixed = state['fixed']
    param.bounds = tuple(state['bounds'])
    # Restore prism uncertainty attributes if present.
    for attr in ('std', 'lolim', 'uplim', 'median'):
        if attr in state:
            try:
                setattr(param, attr, float(state[attr]))
            except (TypeError, ValueError):
                pass

# ---------------------------------------------------------------------------
# LineModelBase  (GaussianLine, VoigtLine, LorentzianLine)
# ---------------------------------------------------------------------------

def _linemodel_representer(dumper, obj):
    state = {pn: _param_state(getattr(obj, pn)) for pn in obj.param_names}
    instfwhm_raw = getattr(obj, '_instfwhm_raw', getattr(obj, 'instfwhm', 0.0))
    if isinstance(instfwhm_raw, np.ndarray):
        instfwhm_raw = instfwhm_raw.tolist()
    state['_instfwhm'] = instfwhm_raw
    if obj.name:
        state['_name'] = obj.name
    return dumper.represent_mapping(f'!prism.{obj.__class__.__name__}', state)

def _linemodel_constructor(loader, node, cls):
    mapping = loader.construct_mapping(node, deep=True)
    instfwhm = mapping.pop('_instfwhm', 0.0)
    mdl_name = mapping.pop('_name', None)
    obj = cls(instfwhm=instfwhm, name=mdl_name)
    for pn, state in mapping.items():
        if hasattr(obj, pn):
            _restore_param(obj, pn, state)
    return obj

# ---------------------------------------------------------------------------
# LineGroupBase  (GaussianLines, LorentzianLines, VoigtLines)
# ---------------------------------------------------------------------------

def _linegroup_representer(dumper, obj):
    # Base class name for reconstruction
    for base in type(obj).__mro__:
        if base.__name__ in ('GaussianLines', 'LorentzianLines', 'VoigtLines'):
            base_name = base.__name__
            break
    else:
        base_name = type(obj).__name__

    state = {
        '_base': base_name,
        '_df':   obj._df.to_dict(orient='records'),
    }
    instfwhm_raw = getattr(obj, '_instfwhm_raw', getattr(obj, 'instfwhm', 0.0))
    if isinstance(instfwhm_raw, np.ndarray):
        instfwhm_raw = instfwhm_raw.tolist()
    state['_instfwhm'] = instfwhm_raw
    if obj.name:
        state['_name'] = obj.name

    for pn in obj.param_names:
        state[pn] = _param_state(getattr(obj, pn))

    return dumper.represent_mapping('!prism.LineGroup', state)

def _linegroup_constructor(loader, node):
    from . import lines as _lines_mod
    mapping = loader.construct_mapping(node, deep=True)

    base_name = mapping.pop('_base')
    df_records = mapping.pop('_df')
    instfwhm = mapping.pop('_instfwhm', 0.0)
    mdl_name = mapping.pop('_name', None)

    base_cls = getattr(_lines_mod, base_name)
    df = pd.DataFrame(df_records)
    obj = base_cls.from_templates(df, instfwhm=instfwhm, name=mdl_name)

    for pn, state in mapping.items():
        if hasattr(obj, pn):
            _restore_param(obj, pn, state)
    return obj

# ---------------------------------------------------------------------------
# Continuum models  (Powerlaw, BrokenPowerlaw — plain Fittable1DModel)
# ---------------------------------------------------------------------------

def _continuum_representer(dumper, obj):
    state = {pn: _param_state(getattr(obj, pn)) for pn in obj.param_names}
    if obj.name:
        state['_name'] = obj.name
    return dumper.represent_mapping(f'!prism.{obj.__class__.__name__}', state)

def _continuum_constructor(loader, node, cls):
    mapping = loader.construct_mapping(node, deep=True)
    mdl_name = mapping.pop('_name', None)
    obj = cls(name=mdl_name)
    for pn, state in mapping.items():
        if hasattr(obj, pn):
            _restore_param(obj, pn, state)
    return obj


# ---------------------------------------------------------------------------
# Compound models
# ---------------------------------------------------------------------------

def _compound_representer(dumper, obj):
    state = {
        '_expr': obj._format_expression(),
        '_components': list(obj._leaflist),
    }
    if obj.name:
        state['_name'] = obj.name
    return dumper.represent_mapping('!prism.CompoundModel', state)


def _compound_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    expr = mapping['_expr']
    components = mapping['_components']
    mdl_name = mapping.get('_name')

    namespace = {f'c{i}': comp for i, comp in enumerate(components)}
    py_expr = re.sub(r'\[(\d+)\]', lambda m: f"c{m.group(1)}", expr)
    obj = eval(py_expr, {'__builtins__': {}}, namespace)
    if mdl_name is not None:
        obj.name = mdl_name
    return obj

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register():
    """Register all prism models with the Astropy YAML loader/dumper."""
    from .lines import GaussianLine, VoigtLine, LorentzianLine
    from .lines import GaussianLines, LorentzianLines, VoigtLines, LineGroupBase
    from .continuum import Powerlaw, BrokenPowerlaw, BalmerContinuum

    line_models = [GaussianLine, VoigtLine, LorentzianLine]
    continuum_models = [Powerlaw, BrokenPowerlaw, BalmerContinuum]

    for cls in line_models:
        tag = f'!prism.{cls.__name__}'
        AstropyDumper.add_representer(cls, _linemodel_representer)
        AstropyLoader.add_constructor(
            tag, lambda loader, node, cls=cls: _linemodel_constructor(loader, node, cls)
        )

    for cls in continuum_models:
        tag = f'!prism.{cls.__name__}'
        AstropyDumper.add_representer(cls, _continuum_representer)
        AstropyLoader.add_constructor(
            tag, lambda loader, node, cls=cls: _continuum_constructor(loader, node, cls)
        )

    AstropyDumper.add_representer(CompoundModel, _compound_representer)
    AstropyLoader.add_constructor('!prism.CompoundModel', _compound_constructor)

    # LineGroup - all dynamic subclasses share a single representer/constructor
    for cls in [GaussianLines, LorentzianLines, VoigtLines]:
        AstropyDumper.add_representer(cls, _linegroup_representer)

    # Dynamic subclasses created by LineGroupBase.from_templates inherit from LineGroupBase.
    AstropyDumper.add_multi_representer(LineGroupBase, _linegroup_representer)
    AstropyLoader.add_constructor('!prism.LineGroup', _linegroup_constructor)


register()
