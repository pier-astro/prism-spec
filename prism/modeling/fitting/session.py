"""Portable fitted graphs: explicit definitions, FITS arrays and fixed responses."""
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits
from astropy.io.misc import yaml as astropy_yaml

from ..composition import Model, _is_compound
from ..parameters import _Expression, _encode_links, _restore_links
from ..operators.matop import LinearOperatorModel
from ..operators.storage import _operator_digest, _operator_state
from ..io import _operator_files, _write_session, _model_hdu
from .data import Dataset
from .extension import _attach_covariance
from .joint import _ParameterGraph
from .results import FitResult
from .utils import _free_parameter_indices, _get_tied_info


def _portable(value):
    """Keep numerical diagnostics and metadata, never optimizer/Python objects."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _operators(associations):
    """Walk source and response roots, visiting shared objects only once."""
    seen = set()
    def visit(node):
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, LinearOperatorModel):
            _operator_state(node)
            yield node
        elif _is_compound(node):
            yield from visit(node.left)
            yield from visit(node.right)
    for source, data in associations.values():
        yield from visit(source)
        yield from visit(data.response)


def _save_array(hdus, value, name):
    if value is None:
        return None
    unit = getattr(value, 'unit', None)
    array = np.asanyarray(np.ma.getdata(value))
    boolean = array.dtype.kind == 'b'
    if array.ndim != 1 or array.dtype.kind not in 'biuf':
        raise ValueError('Result datasets require one-dimensional real numeric arrays.')
    hdus.append(fits.ImageHDU(array.astype(np.uint8) if boolean else array, name=name))
    state = dict(hdu=name, unit=None if unit is None else unit.to_string(), boolean=boolean)
    if np.ma.isMaskedArray(value):
        state['mask'] = name+'_MASK'
        hdus.append(fits.ImageHDU(np.ma.getmaskarray(value).astype(np.uint8), name=state['mask']))
    return state


def _load_array(hdus, state):
    if state is None:
        return None
    raw = np.asarray(hdus[state['hdu']].data)
    array = np.array(raw, dtype=bool if state['boolean'] else raw.dtype.newbyteorder('='), copy=True)
    if state.get('unit') is not None:
        array = array*u.Unit(state['unit'])
    if 'mask' in state:
        array = np.ma.array(array, mask=np.asarray(hdus[state['mask']].data, dtype=bool))
    return array


def _session_dumper(resources):
    # Register only on a per-call subclass; model-only serialization is unchanged.
    from ..models import serialization  # noqa: F401
    class Dumper(astropy_yaml.AstropyDumper):
        declarative_links = True
        result_graph = True
    Dumper.operator_files = resources
    def operator(dumper, obj):
        resource = resources.get(id(obj))
        if resource is not None:
            state = dict(path=resource['path'], fingerprint=resource['fingerprint'], name=obj.name)
        else:
            # Registered responses retain verified archive recipes.
            state = dict(recipe=obj.recipe, wave=getattr(obj, '_requested_wave', None),
                         name=obj.name, fingerprint=_operator_digest(obj))
            if state['wave'] is None:
                state['wave'] = obj.x
        return dumper.represent_mapping('!prism.ResultOperator', state)
    from ..operators.instrument import SpectralResponse
    for cls in (LinearOperatorModel, SpectralResponse):
        Dumper.add_representer(cls, operator)
    return Dumper


def _validate_definitions(associations):
    from ..component import _Function
    from ..models.serialization import _profile_recipe
    definitions = {}
    for source, data in associations.values():
        for owner in (source, data.response):
            for leaf in getattr(owner, '_components', ()):
                if not isinstance(leaf, _Function):
                    continue
                function = getattr(leaf, '_profile', getattr(leaf, '_function', None))
                derivative = leaf._derivative
                if _profile_recipe(function, derivative) is not None:
                    continue
                if not leaf.name:
                    raise ValueError('Name custom components before saving a fitted session.')
                definition = (function, derivative)
                if leaf.name in definitions and definitions[leaf.name] != definition:
                    raise ValueError('Different custom definitions require distinct component names: '+leaf.name)
                definitions[leaf.name] = definition


def _save_result(result, filename, overwrite=False):
    """Write one graph document so YAML aliases preserve shared components."""
    graph = result._graph
    _get_tied_info(graph)
    for name in graph.param_names:
        parameter = getattr(graph, name)
        if parameter.size != 1:
            raise ValueError('Fitted-session persistence currently requires scalar parameters.')
        if parameter.tied and not isinstance(parameter.tied, _Expression):
            raise ValueError('Callable parameter links cannot be persisted; use declarative links for a fitted session.')
    _validate_definitions(result._associations)
    resources = {}
    for operator in _operators(result._associations):
        resources.update(_operator_files(operator, filename))
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['FORMAT'] = 'PRISMFIT'
    hdus[0].header['VERSION'] = 1
    observations = []
    for index, (label, (source, data)) in enumerate(result._associations.items()):
        arrays = {key: _save_array(hdus, getattr(data, key), f'D{index}_{key.upper()}')
                  for key in ('x', 'y', 'yerr', 'weights', 'mask')}
        observations.append(dict(label=label, model=source, response=data.response,
            arrays=arrays, meta=_portable(data.meta), xunit=data.xunit, yunit=data.yunit))
    info = {key: _portable(value) for key, value in result.fit_info.items()
            if key not in ('param_cov', 'native_result')}
    payload = dict(version=1, single=result._single, observations=observations,
        param_names=list(graph.param_names), fit_indices=result.fit_indices.tolist(),
        links=_encode_links(graph),
        fit_info=info, contributions={key: dict(value) for key, value in result._contributions.items()})
    # A small parameter table supports initialization without rebuilding the graph.
    names = list(result.param_names)
    width = max(map(len, names), default=1)
    hdus.append(fits.BinTableHDU.from_columns([
        fits.Column(name='NAME', format=f'{width}A', array=names),
        fits.Column(name='VALUE', format='D', array=result.parameters),
    ], name='PARS'))
    if result.covariance is not None:
        hdus.append(fits.ImageHDU(result.covariance, name='PARAM_COV'))
    document = yaml.dump(payload, Dumper=_session_dumper(resources), default_flow_style=None)
    hdus.append(_model_hdu(document))
    _write_session(hdus, filename, resources, overwrite)


def _session_loader(filename, components):
    from ..models import serialization  # noqa: F401
    class Loader(astropy_yaml.AstropyLoader):
        pass
    Loader.base_path = Path(filename).resolve().parent
    Loader.components = {} if components is None else components
    def operator(loader, node):
        state = loader.construct_mapping(node, deep=True)
        if 'path' in state:
            path = Path(state['path'])
            restored = LinearOperatorModel.from_fits(path if path.is_absolute() else loader.base_path/path)
        else:
            from ..operators.instrument import SpectralResponse
            recipe = state['recipe']
            if recipe.get('type') != 'instrument':
                raise ValueError('Unsupported saved response recipe.')
            restored = SpectralResponse(instrument=recipe['instrument'], wave=state['wave'],
                redshift=recipe.get('redshift', recipe.get('z', 0.)), flexible=recipe.get('flexible', True),
                renormalize=recipe.get('renormalize', False), name=state['name'])
        if _operator_digest(restored) != state['fingerprint']:
            raise ValueError('Response content differs from the fitted session.')
        restored.name = state['name']
        return restored
    Loader.add_constructor('!prism.ResultOperator', operator)
    return Loader


def _load_result(filename, components=None):
    """Rebuild associations first, then restore links in the global node order."""
    with fits.open(filename, memmap=False) as hdus:
        if hdus[0].header.get('FORMAT') != 'PRISMFIT' or hdus[0].header.get('VERSION') != 1:
            raise ValueError('Unsupported fitted-session format or version.')
        for hdu in hdus:
            if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                raise ValueError('Fitted-session checksum is missing or invalid.')
        payload = yaml.load(str(hdus['MODEL_DEF'].data['YAML'][0]),
                            Loader=_session_loader(filename, components))
        if payload.get('version') != 1:
            raise ValueError('Unsupported fitted-session definition version.')
        associations = {}
        for state in payload['observations']:
            label = state['label']
            if not isinstance(label, str) or not label or label in associations:
                raise ValueError('Observation labels must be unique nonempty strings.')
            arrays = {key: _load_array(hdus, value) for key, value in state['arrays'].items()}
            data = Dataset(**arrays, response=state['response'])
            data.meta, data.xunit, data.yunit = state['meta'], state['xunit'], state['yunit']
            associations[label] = (state['model'], data)
        if not associations or payload['single'] and tuple(associations) != ('data',):
            raise ValueError('Invalid observation mapping for a fitted session.')
        if payload['single']:
            source, data = associations['data']
            graph = data._observed_model(source)
        else:
            graph = _ParameterGraph(associations)
        if tuple(payload['param_names']) != graph.param_names:
            raise ValueError('Saved parameter order does not match the restored graph.')
        _restore_links(graph, payload['links'])
        _get_tied_info(graph)
        if not np.array_equal(payload['fit_indices'], _free_parameter_indices(graph)):
            raise ValueError('Saved covariance order does not match the restored free parameters.')
        covariance = None
        if 'PARAM_COV' in hdus:
            covariance = np.array(hdus['PARAM_COV'].data, dtype=float, copy=True)
            free = _free_parameter_indices(graph)
            if covariance.shape != (len(free), len(free)) or not np.all(np.isfinite(covariance)):
                raise ValueError('Covariance does not match the restored free parameters.')
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
                raise ValueError('Saved covariance must be symmetric.')
        _attach_covariance(graph, covariance)
        for source, data in associations.values():
            for owner in (source, data.response):
                if isinstance(owner, Model):
                    for expression in owner._expressions():
                        expression._evaluation_parent = graph
        info = payload['fit_info']
        info['param_cov'] = covariance
        if info['nfree'] != len(_free_parameter_indices(graph)):
            raise ValueError('Saved diagnostics disagree with the free parameter order.')
        return FitResult(associations, graph, info, payload['contributions'], _single=payload['single'])
