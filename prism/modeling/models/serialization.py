"""
YAML serialization hooks for prism-spec models.

Calling ``register()`` (done automatically on import) registers all
prism model types with ``astropy.io.misc.yaml``, enabling round-trip
``dump``/``load`` for fitted models saved to FITS via ``prism.modeling.io``.

Supported types
---------------
* ``LineModelBase`` (``GaussianLine``, ``VoigtLine``, ``LorentzianLine``) —
  stored as class name + per-parameter state.
* ``LineGroupBase`` (``GaussianLines``, …) — dynamically-created classes
  rebuilt via ``from_templates`` from the embedded ``_df`` table.
* ``CompoundModel`` — tree of sub-models with Astropy operators.
* ``LinearOperatorCompoundModel`` — stores the source model plus a
  reconstruction *recipe* (instrument name, redshift, wavelength grid).
  The matrix is never serialized; it is rebuilt on load.
* Generic ``Fittable1DModel`` — fallback for any Astropy model.

Response serialization
----------------------
The ``LinearOperatorCompoundModel`` (LOCM) is saved with a *recipe*
that records how the operator matrix was constructed:

* **Instrument recipe** (``type='instrument'``): the instrument name,
  redshift, and wavelength grid are stored.  On load, the matrix is
  rebuilt from the archived instrument file.  This is fully
  round-trippable as long as the instrument is registered::

      from prism.modeling.operators.instrument import SpectralResponse
      rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave, z=z)
      model = rsp(source)
      model.save('fit.fits')               # recipe = {type: 'instrument', ...}
      loaded = load_model('fit.fits')       # LOCM reconstructed

* **Direct recipe** (``type='direct'``): the operator was built from an
  ``InstrumentResponse`` object directly, without an instrument name.
  The matrix cannot be reconstructed from the recipe alone.  On load
  the source model is returned *without* convolution and a warning is
  issued.  Re-apply the operator manually::

      loaded = load_model('fit.fits')       # source model only (warning)
      loaded = rsp(loaded)                  # re-apply manually

If the ``serialization`` module is not imported before loading (e.g. in
a fresh session), call ``prism.modeling.models.serialization.register()``
or simply ``import prism.modeling.models`` to activate the hooks.
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
# Generic Astropy Fittable1DModel  (fallback for unregistered models)
# ---------------------------------------------------------------------------

def _astropy_model_representer(dumper, obj):
    """Generic representer for any Astropy Fittable1DModel without a prism hook."""
    cls = type(obj)
    state = {
        '_class': f'{cls.__module__}.{cls.__qualname__}',
    }
    if obj.name:
        state['_name'] = obj.name
    for pn in obj.param_names:
        state[pn] = _param_state(getattr(obj, pn))
    return dumper.represent_mapping('!prism.AstropyModel', state)


def _astropy_model_constructor(loader, node):
    """Reconstruct an Astropy model from its module path + parameter state."""
    import importlib
    mapping = loader.construct_mapping(node, deep=True)
    class_path = mapping.pop('_class')
    mdl_name = mapping.pop('_name', None)

    if not class_path.startswith('astropy.modeling.'):
        raise ValueError(
            f"Cannot deserialize model class '{class_path}': "
            "only astropy.modeling classes are supported."
        )

    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
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
# LinearOperatorCompoundModel  (instrumental response wrapper)
# ---------------------------------------------------------------------------

def _locm_representer(dumper, obj):
    """Serialize a LinearOperatorCompoundModel.

    Stores the source model (serialized recursively) and a reconstruction
    recipe for the operator.  The full matrix is *not* stored — it is
    rebuilt from the recipe on load.
    """
    state = {
        '_source': obj.left,  # serialized by its own representer
    }
    operator = obj.right_operator
    recipe = getattr(operator, '_recipe', {})
    if recipe:
        state['_recipe'] = dict(recipe)

    wave = getattr(operator, '_wave', None)
    if wave is not None:
        state['_wave'] = wave.tolist()

    if obj.name:
        state['_name'] = obj.name

    if not recipe or recipe.get('type') == 'direct':
        warnings.warn(
            "LinearOperatorCompoundModel has no reconstruction recipe. "
            "The model will be saved without the response operator. "
            "Re-apply the response after loading.",
            UserWarning, stacklevel=4,
        )

    return dumper.represent_mapping('!prism.ConvolvedModel', state)


def _locm_constructor(loader, node):
    """Reconstruct a LinearOperatorCompoundModel from its recipe."""
    mapping = loader.construct_mapping(node, deep=True)
    source = mapping['_source']
    recipe = mapping.get('_recipe', {})
    wave_list = mapping.get('_wave')
    name = mapping.get('_name')

    wave = np.asarray(wave_list) if wave_list else None
    rtype = recipe.get('type', 'unknown')

    if rtype == 'instrument' and wave is not None:
        from ..operators.instrument import SpectralResponse
        instrument_name = recipe['instrument']
        z = recipe.get('z', 0)
        try:
            rsp = SpectralResponse(instrument=instrument_name, wave=wave, z=z)
            return rsp(source, name=name)
        except Exception as exc:
            warnings.warn(
                f"Could not reconstruct response from recipe "
                f"(instrument='{instrument_name}', z={z}): {exc}. "
                "Returning source model without convolution.",
                UserWarning, stacklevel=2,
            )
            return source

    if rtype == 'direct':
        warnings.warn(
            "Model was saved with a direct response matrix that cannot be "
            "reconstructed from a recipe.  Re-apply the response manually.",
            UserWarning, stacklevel=2,
        )
        return source

    warnings.warn(
        f"Unknown response recipe type '{rtype}'. "
        "Returning source model without convolution.",
        UserWarning, stacklevel=2,
    )
    return source


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

    # LinearOperatorCompoundModel — must be registered BEFORE CompoundModel
    # multi-representer, since LOCM is a subclass of CompoundModel.
    from ..operators.convolved import LinearOperatorCompoundModel
    AstropyDumper.add_representer(LinearOperatorCompoundModel, _locm_representer)
    AstropyLoader.add_constructor('!prism.ConvolvedModel', _locm_constructor)

    # LineGroup - all dynamic subclasses share a single representer/constructor
    for cls in [GaussianLines, LorentzianLines, VoigtLines]:
        AstropyDumper.add_representer(cls, _linegroup_representer)

    # Dynamic subclasses created by LineGroupBase.from_templates inherit from LineGroupBase.
    AstropyDumper.add_multi_representer(LineGroupBase, _linegroup_representer)
    AstropyLoader.add_constructor('!prism.LineGroup', _linegroup_constructor)

    # Generic fallback for vanilla Astropy Fittable1DModel subclasses that
    # have no prism-specific representer.  Uses add_multi_representer so
    # that exact-match representers (prism models) take priority.
    from astropy.modeling import Fittable1DModel
    AstropyDumper.add_multi_representer(Fittable1DModel, _astropy_model_representer)
    AstropyLoader.add_constructor('!prism.AstropyModel', _astropy_model_constructor)


register()
