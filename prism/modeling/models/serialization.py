"""
YAML serialization hooks for Prism models and explicit composition trees.

Registration supports line models/groups, compound models, generic Astropy
one-dimensional models and fixed linear-operator pipes. FITS sessions reference
external sparse operator resources through per-call loader/dumper context;
standalone YAML embeds fixed matrices. Registered instrument recipes rebuild
responses and verify their content fingerprint. Missing resources or unsupported
operator behavior raise errors instead of dropping the operator.

Declarative links are stored by the FITS session writer as an explicit graph.
Callable parameter links are marked but must be restored explicitly by analysis
code. Import ``prism.modeling.models`` to register these hooks.
"""

import warnings
import numpy as np
import pandas as pd
import astropy.units as u

from astropy.table import QTable

from astropy.modeling.core import CompoundModel

from astropy.io.misc.yaml import AstropyDumper, AstropyLoader

from ..operators.matop import is_linear_operator_pipe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_state(param, dumper):
    """Compact serialisable dict for one Astropy Parameter."""
    from ..parameters import _Expression
    declarative = isinstance(param.tied, _Expression)
    if declarative and not getattr(dumper, 'declarative_links', False):
        raise ValueError('Save declarative parameter links through model.save(), which records the complete model graph.')
    state = {
        'value':  float(param.value),
        'unit': param.unit.to_string() if param.unit is not None else None,
        'fixed':  bool(param.fixed),
        'tied':   bool(param.tied) and not declarative,
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
    if state.get('unit') is not None:
        setattr(model, name, state['value'] * u.Unit(state['unit']))
    else:
        param.value = state['value']
    param.fixed = state['fixed']
    param.bounds = tuple(state['bounds'])
    if state.get('tied', False):
        warnings.warn(
            f"Restore the analysis's callable link for {model.name or type(model).__name__}.{name} "
            "before refitting or covariance propagation; Python links are not serialized.",
            UserWarning, stacklevel=3)
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
    state = {pn: _param_state(getattr(obj, pn), dumper) for pn in obj.param_names}
    state['_velocity_unit'] = obj.velocity_unit.to_string()
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
    velocity_unit = mapping.pop('_velocity_unit', 'km/s')
    parameters = {name: state['value'] * u.Unit(state['unit'])
                  for name, state in mapping.items() if state.get('unit') is not None}
    obj = cls(instfwhm=instfwhm, name=mdl_name, velocity_unit=velocity_unit, **parameters)
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

    # Serialise positions in the model's native spectral unit (not forced to AA).
    pos_unit = getattr(type(obj), '_position_unit', u.AA)
    pos_unit_str = pos_unit.to_string()

    state = {
        '_base': base_name,
        '_velocity_unit': obj.velocity_unit.to_string(),
        '_position_unit': pos_unit_str,
        '_linetable': [
            {
                'name': str(row['name']),
                'position': float(row['position'].to(pos_unit, equivalencies=u.spectral()).value),
                'weight': float(row['weight']),
            }
            for row in obj._linetable
        ],
    }
    instfwhm_raw = getattr(obj, '_instfwhm_raw', getattr(obj, 'instfwhm', 0.0))
    if isinstance(instfwhm_raw, np.ndarray):
        instfwhm_raw = instfwhm_raw.tolist()
    state['_instfwhm'] = instfwhm_raw
    if obj.name:
        state['_name'] = obj.name

    for pn in obj.param_names:
        state[pn] = _param_state(getattr(obj, pn), dumper)

    return dumper.represent_mapping('!prism.LineGroup', state)

def _linegroup_constructor(loader, node):
    from . import lines as _lines_mod
    mapping = loader.construct_mapping(node, deep=True)

    base_name = mapping.pop('_base')
    records = mapping.pop('_linetable')
    instfwhm = mapping.pop('_instfwhm', 0.0)
    mdl_name = mapping.pop('_name', None)
    velocity_unit = mapping.pop('_velocity_unit', 'km/s')

    # Restore the native spectral unit (defaults to AA for old files).
    pos_unit_str = mapping.pop('_position_unit', 'Angstrom')
    pos_unit = u.Unit(pos_unit_str)

    base_cls = getattr(_lines_mod, base_name)
    linetable = QTable()
    linetable['name'] = [record['name'] for record in records]
    linetable['position'] = [record['position'] for record in records] * pos_unit
    linetable['weight'] = [record['weight'] for record in records]
    obj = base_cls.from_templates(
        linetable, instfwhm=instfwhm, name=mdl_name, velocity_unit=velocity_unit)

    for pn, state in mapping.items():
        if hasattr(obj, pn):
            _restore_param(obj, pn, state)
    return obj

# ---------------------------------------------------------------------------
# Continuum models  (BalmerContinuum — plain Fittable1DModel)
# ---------------------------------------------------------------------------

def _continuum_representer(dumper, obj):
    state = {pn: _param_state(getattr(obj, pn), dumper) for pn in obj.param_names}
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
    if not cls.__module__.startswith('astropy.modeling.'):
        raise ValueError(f'No reconstruction rule for model class {cls.__module__}.{cls.__qualname__}.')
    state = {
        '_class': f'{cls.__module__}.{cls.__qualname__}',
    }
    if obj.name:
        state['_name'] = obj.name
    for pn in obj.param_names:
        state[pn] = _param_state(getattr(obj, pn), dumper)
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

def _native_representer(dumper, obj):
    if is_linear_operator_pipe(obj):
        state = _linear_operator_pipe_representer(dumper, obj)
        return state
    state = {'name': obj.name}
    if obj.op is None:
        state['component'] = obj.component
    else:
        state.update(op=obj.op, left=obj.left, right=obj.right)
    return dumper.represent_mapping('!prism.Model', state)


def _native_constructor(loader, node):
    from ..composition import Model
    state = loader.construct_mapping(node, deep=True)
    if 'component' in state:
        return Model.from_astropy(state['component'])
    result = Model(op=state['op'], left=state['left'], right=state['right'], name=state['name'])
    if result.op not in ('+', '-', '*', '/', '**', '|'):
        raise ValueError('Unsupported native model operation.')
    return result


def _compound_representer(dumper, obj):
    if is_linear_operator_pipe(obj):
        return _linear_operator_pipe_representer(dumper, obj)
    return dumper.represent_mapping('!prism.CompoundModel',
        {'_op': obj.op, '_left': obj.left, '_right': obj.right, '_name': obj.name})


def _compound_constructor(loader, node):
    import operator
    from astropy.modeling import fix_inputs
    mapping = loader.construct_mapping(node, deep=True)
    operations = {'+': operator.add, '-': operator.sub, '*': operator.mul,
                  '/': operator.truediv, '**': operator.pow, '|': operator.or_,
                  '&': operator.and_, 'fix_inputs': fix_inputs}
    op = mapping['_op']
    if op not in operations:
        raise ValueError(f'Unsupported model composition: {op!r}.')
    result = operations[op](mapping['_left'], mapping['_right'])
    result.name = mapping.get('_name')
    return result


# ---------------------------------------------------------------------------
# Linear-operator pipe models
# ---------------------------------------------------------------------------

def _linear_operator_pipe_representer(dumper, obj):
    """Persist source and a fixed operator, using a session resource if available."""
    from ..operators.storage import _operator_state, _operator_digest
    from scipy.sparse import csr_matrix
    operator = obj.right
    metadata = _operator_state(operator)
    state = {'_source': obj.left, '_name': operator.name, '_model_name': obj.name}
    resource = getattr(dumper, 'operator_files', {}).get(id(operator))
    if resource is not None:
        state['_operator_file'] = {'path': resource['path'], 'fingerprint': resource['fingerprint']}
    elif metadata['kind'] == 'spectral' and operator.recipe.get('type') == 'instrument':
        requested = getattr(operator, '_requested_wave', None)
        state.update(_recipe=operator.recipe, _name=operator.name,
                     _wave=operator.x if requested is None else requested,
                     _fingerprint=_operator_digest(operator))
    else:
        matrix = csr_matrix(operator.matrix)
        state['_operator'] = dict(state=metadata, x=operator.x, xout=operator.xout,
            matrix=dict(data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape))
    return dumper.represent_mapping('!prism.LinearOperatorPipe', state)


def _linear_operator_pipe_constructor(loader, node):
    """Rebuild the fixed operator and verify external resources/calibrations."""
    from pathlib import Path
    from scipy.sparse import csr_matrix
    from ..operators.matop import LinearOperatorModel
    from ..operators.storage import _operator_digest, _restore_operator
    mapping = loader.construct_mapping(node, deep=True)
    source = mapping['_source']
    if '_operator_file' in mapping:
        resource = mapping['_operator_file']
        base = getattr(loader, 'base_path', None)
        path = Path(resource['path'])
        if not path.is_absolute():
            if base is None:
                raise ValueError('Relative operator files require loading through load_model(filename).')
            path = base/path
        operator = LinearOperatorModel.from_fits(path)
        if _operator_digest(operator) != resource['fingerprint']:
            raise ValueError(f'Operator resource differs from the saved session: {path}')
    elif '_operator' in mapping:
        payload = mapping['_operator']
        matrix = payload['matrix']
        operator = _restore_operator(csr_matrix((matrix['data'], matrix['indices'], matrix['indptr']),
                                               shape=tuple(matrix['shape'])),
                                     payload['x'], payload['xout'], payload['state'])
    elif mapping.get('_recipe', {}).get('type') == 'instrument' and '_wave' in mapping:
        from ..operators.instrument import SpectralResponse
        recipe = mapping['_recipe']
        operator = SpectralResponse(instrument=recipe['instrument'], wave=mapping['_wave'],
            z=recipe.get('z', 0), flexible=recipe.get('flexible', True),
            renormalize=recipe.get('renormalize', False), name=mapping.get('_name', 'rsp'))
        if '_fingerprint' not in mapping or _operator_digest(operator) != mapping['_fingerprint']:
            raise ValueError('Instrument calibration or response configuration differs from the saved session.')
    else:
        raise ValueError('Cannot reconstruct an operator without a saved matrix, resource, or verified recipe.')
    operator.name = mapping.get('_name', operator.name)
    result = source | operator
    result.name = mapping.get('_model_name')
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register():
    """Register all prism models with the Astropy YAML loader/dumper."""
    from .lines import GaussianLine, VoigtLine, LorentzianLine
    from .lines import GaussianLines, LorentzianLines, VoigtLines, LineGroupBase
    from .continuum import BalmerContinuum

    line_models = [GaussianLine, VoigtLine, LorentzianLine]
    continuum_models = [BalmerContinuum]

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

    from ..composition import Model
    AstropyDumper.add_representer(Model, _native_representer)
    AstropyLoader.add_constructor('!prism.Model', _native_constructor)
    AstropyDumper.add_representer(CompoundModel, _compound_representer)
    AstropyLoader.add_constructor('!prism.CompoundModel', _compound_constructor)
    AstropyLoader.add_constructor('!prism.LinearOperatorPipe', _linear_operator_pipe_constructor)

    # LineGroup - all dynamic subclasses share a single representer/constructor
    for cls in [GaussianLines, LorentzianLines, VoigtLines]:
        AstropyDumper.add_representer(cls, _linegroup_representer)

    # Dynamic subclasses created by LineGroupBase.from_templates inherit from LineGroupBase.
    AstropyDumper.add_multi_representer(LineGroupBase, _linegroup_representer)
    AstropyLoader.add_constructor('!prism.LineGroup', _linegroup_constructor)

    # Generic fallback for vanilla Astropy Fittable1DModel subclasses that
    # have no prism-specific representer.  Uses add_multi_representer so
    # that exact-match representers (prism models) take priority.
    from astropy.modeling import Model as AstropyModel, Fittable1DModel, Fittable2DModel
    AstropyDumper.add_multi_representer(AstropyModel, _astropy_model_representer)
    AstropyDumper.add_multi_representer(Fittable1DModel, _astropy_model_representer)
    AstropyDumper.add_multi_representer(Fittable2DModel, _astropy_model_representer)
    AstropyLoader.add_constructor('!prism.AstropyModel', _astropy_model_constructor)


register()
