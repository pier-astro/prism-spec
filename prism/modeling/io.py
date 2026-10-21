"""
prism.modeling.io
=================

Save and load best-fit parameters and model definitions to / from FITS files.

Public API
----------
``model_to_fits(model, filename)`` for single fitted Astropy models
``MultiFitResult.to_fits()`` / ``MultiFitResult.save()``
``SpectrumFitResult.to_fits()`` / ``SpectrumFitResult.save()``
``load_params(filename)``
``load_model(filename)``

FITS structure
--------------
Batch / cube results (``MultiFitResult``)
    PRIMARY            Metadata header; optional WCS keywords.
    <PARNAME>          ImageHDU — parameter value map.
    <PARNAME>_ERR      ImageHDU — parameter std map  (when ``std=True``).
    MODEL_DEF          BinTableHDU — single-column YAML string.
    <CUSTOM>           ImageHDU — any extra user-supplied array.

Single spectrum (``SpectrumFitResult``)
    PRIMARY            Metadata header.
    PARS               BinTableHDU — columns NAME, VALUE, STD, UNIT.
    MODEL_DEF          BinTableHDU — single-column YAML string.
    <CUSTOM>           ImageHDU — any extra user-supplied scalar/array.
"""

from __future__ import annotations

import warnings
import os
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime, timezone
from astropy.io import fits
from astropy.io.misc import yaml as astropy_yaml
import astropy.units as u

# Extension name constants
EXT_PARS       = 'PARS'
EXT_MODEL      = 'MODEL_DEF'
EXT_ERR_SUFFIX = '_ERR'
EXT_COV        = 'PARAM_COV'

_PARNAME_MAXLEN = 18  # FITS extension-name maximum usable length


def _ext_name(param_name: str) -> str:
    """Truncate & upper-case a parameter name for use as a FITS extension name."""
    return param_name.upper()[:_PARNAME_MAXLEN]


def _operator_files(model, filename, operator_file=None):
    """Plan fixed operator resources; archived recipes need no sidecar by default."""
    from astropy.modeling import CompoundModel
    from .composition import Model, _is_compound
    from .operators.matop import LinearOperatorModel
    from .operators.storage import _operator_digest, _operator_state
    operators = {}
    def collect(node):
        if isinstance(node, LinearOperatorModel):
            _operator_state(node)
            if operator_file is not None or node.recipe.get('type') != 'instrument':
                operators[id(node)] = node
        elif _is_compound(node):
            collect(node.left)
            collect(node.right)
    if model is not None:
        collect(model)
    if operator_file is not None and len(operators) != 1:
        raise ValueError('operator_file requires exactly one fixed operator in the saved model.')
    session = Path(filename).resolve()
    resources = {}
    for index, (identity, operator) in enumerate(operators.items(), 1):
        fingerprint = _operator_digest(operator)
        if operator_file is not None:
            target = Path(operator_file)
            if not target.is_absolute():
                target = session.parent/target
        else:
            suffix = '' if len(operators) == 1 else str(index)
            target = session.with_name(f'{session.stem}_operator{suffix}_{fingerprint[:12]}.fits')
        target = target.resolve()
        if target == session:
            raise ValueError('The operator resource and session must use different filenames.')
        resources[identity] = dict(path=os.path.relpath(target, session.parent),
            fingerprint=fingerprint, target=target, operator=operator)
    return resources


def _serialize_model(model, operator_files=None):
    """Serialize a definition with per-call resources, never shared global state."""
    if model is None:
        return None
    from .models import serialization  # noqa: F401 (registers YAML hooks)
    class SessionDumper(astropy_yaml.AstropyDumper):
        pass
    from .parameters import _encode_links
    from .fitting.utils import _get_tied_info
    _get_tied_info(model)  # validate dependencies before saving
    from .component import _Function
    leaves = getattr(model, '_components', ())
    names = [leaf.name for leaf in leaves if isinstance(leaf, _Function)]
    if len(names) != len(set(names)):
        raise ValueError('Custom components require unique names for explicit restoration.')
    links = _encode_links(model)
    SessionDumper.operator_files = {} if operator_files is None else operator_files
    SessionDumper.declarative_links = True
    payload = {'_prism_links': 1, 'model': model, 'links': links} if links else model
    return yaml.dump(payload, Dumper=SessionDumper, default_flow_style=None)


def _write_session(hdus, filename, resources, overwrite):
    """Reuse verified resources; publish the session only after all writes succeed."""
    from .operators.matop import LinearOperatorModel
    from .operators.storage import _operator_digest, _atomic_write
    session = Path(filename).resolve()
    if session.exists() and not overwrite:
        raise FileExistsError(f'File exists: {session}')
    created = []
    try:
        for resource in resources.values():
            target = resource['target']
            if target.exists():
                existing = LinearOperatorModel.from_fits(target)
                if _operator_digest(existing) != resource['fingerprint']:
                    raise FileExistsError(f'Operator resource has different content; choose another operator_file: {target}')
            else:
                resource['operator'].save(target)
                created.append(target)
        _atomic_write(hdus, session, overwrite)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _model_hdu(yaml_str: str) -> fits.BinTableHDU:
    """Pack a YAML string into a single-cell BinTableHDU."""
    from astropy.io.fits import Column, BinTableHDU
    col = Column(name='YAML', format=f'{len(yaml_str) + 1}A', array=[yaml_str])
    hdu = BinTableHDU.from_columns([col], name=EXT_MODEL)
    hdu.header['COMMENT'] = 'prism-spec model definition (Astropy YAML)'
    return hdu


def _primary_hdu() -> fits.PrimaryHDU:
    hdr = fits.Header()
    hdr['AUTHOR']  = 'prism-spec'
    hdr['DATE']    = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    hdr['HISTORY'] = 'Created by prism-spec MultiFit I/O'
    return fits.PrimaryHDU(header=hdr)


def _image_hdu(name: str, data: np.ndarray, unit=None, wcs=None, cd_matrix=False) -> fits.ImageHDU:
    from prism.data.cube import wcs_to_cd_matrix
    hdr = fits.Header()
    if unit is not None:
        try:
            hdr['BUNIT'] = u.Unit(unit).to_string('fits')
        except Exception:
            hdr['BUNIT'] = str(unit)

    if wcs is not None:
        wcs_header = wcs.to_header()
        if cd_matrix:
            wcs_to_cd_matrix(wcs_header)
        hdr.update(wcs_header)

    return fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=hdr, name=name)


# ---------------------------------------------------------------------------
# Single fitted Astropy model  to_fits / save
# ---------------------------------------------------------------------------

def model_to_fits(model, filename: str, overwrite: bool = False,
                  parfilter=None, std: bool = True, model_def: bool = True, operator_file=None,
                  **custom_exts):
    """
    Save a fitted single-spectrum Astropy model directly to FITS.

    This is the minimal path for standard (non-multifit) fit outputs where
    the result is already an Astropy model instance.

    Parameters
    ----------
    model : astropy.modeling.Model
    filename : str
    overwrite : bool
    parfilter : list of str, optional
        Export only these parameters.
    std : bool
        Include standard deviations in the PARS table when available.
    model_def : bool
        Serialize the model into the MODEL_DEF extension.
    operator_file : str or pathlib.Path, optional
        External fixed-operator FITS file; relative paths resolve beside the
        session. Requires exactly one operator. By default, direct matrices use
        content-named resources and registered responses retain verified recipes.
    **custom_exts
        Extra scalar or array extensions.
    """
    param_names = list(model.param_names)

    if parfilter is not None:
        missing = [p for p in parfilter if p not in param_names]
        if missing:
            raise KeyError(f"parfilter contains unknown parameters: {missing}")
        param_names = [p for p in param_names if p in parfilter]

    hdus = [_primary_hdu()]

    names = np.array(param_names, dtype=object)
    values = np.array([float(getattr(model, p).value) for p in param_names], dtype=float)
    def _to_float_or_nan(value):
        return np.nan if value is None else float(value)

    if std:
        errors = np.array([
            _to_float_or_nan(getattr(getattr(model, p), 'std', np.nan))
            for p in param_names
        ], dtype=float)
    else:
        errors = np.full(len(param_names), np.nan, dtype=float)

    units = []
    for p in param_names:
        unit = getattr(getattr(model, p), 'unit', None)
        if unit is None:
            units.append('')
        else:
            try:
                units.append(u.Unit(unit).to_string('fits'))
            except Exception:
                units.append(str(unit))

    has_lolim = any(hasattr(getattr(model, p), 'lolim') for p in param_names)
    has_uplim = any(hasattr(getattr(model, p), 'uplim') for p in param_names)
    if has_lolim:
        lolims = [_to_float_or_nan(getattr(getattr(model, p), 'lolim', np.nan)) for p in param_names]
    if has_uplim:
        uplims = [_to_float_or_nan(getattr(getattr(model, p), 'uplim', np.nan)) for p in param_names]

    name_fmt = f'{max(len(p) for p in param_names) + 1}A' if param_names else '1A'
    unit_fmt = '20A'

    from astropy.io.fits import Column, BinTableHDU
    cols = [
        Column(name='NAME',  format=name_fmt, array=names),
        Column(name='VALUE', format='D',      array=values),
        Column(name='STD',   format='D',      array=errors),
    ]
    if has_lolim:
        cols.append(Column(name='LOLIM', format='D', array=lolims))
    if has_uplim:
        cols.append(Column(name='UPLIM', format='D', array=uplims))
    cols.append(Column(name='UNIT', format=unit_fmt, array=units))

    hdus.append(BinTableHDU.from_columns(cols, name=EXT_PARS))

    for ext_name, data in custom_exts.items():
        hdus.append(_image_hdu(ext_name.upper(), np.atleast_1d(np.asarray(data, dtype=np.float32))))

    # Save covariance matrix if available
    cov = getattr(model, '_param_cov', None)
    if cov is not None:
        cov_arr = np.asarray(cov, dtype=np.float64)
        hdu = fits.ImageHDU(data=cov_arr, name=EXT_COV)
        from .fitting.utils import _free_parameter_indices
        indices = getattr(model, '_fit_indices', _free_parameter_indices(model))
        hdu.header['FITIND'] = ','.join(str(int(i)) for i in indices)
        hdus.append(hdu)

    resources = _operator_files(model, filename, operator_file) if model_def else {}
    if model_def:
        yaml_str = _serialize_model(model, resources)
        if yaml_str is not None:
            hdus.append(_model_hdu(yaml_str))

    _write_session(hdus, filename, resources, overwrite)


# ---------------------------------------------------------------------------
# MultiFitResult  to_fits / save
# ---------------------------------------------------------------------------

def multifit_to_fits(result, filename: str, wcs=None, overwrite: bool = False,
                     parfilter=None, std: bool = False, model: bool = True,
                     cd_matrix: bool = False, operator_file=None, **custom_exts):
    """
    Save a ``MultiFitResult`` to a multi-extension FITS file.

    Parameters
    ----------
    result : MultiFitResult
    filename : str
    wcs : astropy.wcs.WCS, optional
        Spatial WCS for cube results loaded into each ImageHDU header.
    overwrite : bool
    parfilter : list of str, optional
        Export only these parameters. Omitting parameters raises a warning
        because the saved file will not fully reconstruct the model.
    std : bool
        Also export parameter standard-deviation maps as ``<PARNAME>_ERR``.
    model : bool
        Serialize the template model into a ``MODEL_DEF`` extension.
    cd_matrix : bool
        If True, convert WCS PC matrix + CDELT to the legacy CD matrix format.
    operator_file : str or pathlib.Path, optional
        External fixed-operator FITS file; relative paths resolve beside the
        session. Requires exactly one operator. By default, direct matrices use
        content-named resources and registered responses retain verified recipes.
    **custom_exts
        Arbitrary extra extensions, e.g. ``CHISQ=chi2_array, DOF=dof_array``.
        Each value must be array-like and broadcastable to result.shape.
    """
    from .fitting.multifit import MultiFitResult  # avoid circular at module level

    param_names = list(result.param_names)

    if parfilter is not None:
        missing = [p for p in parfilter if p not in param_names]
        if missing:
            raise KeyError(f"parfilter contains unknown parameters: {missing}")
        excluded = [p for p in param_names if p not in parfilter]
        if excluded:
            warnings.warn(
                f"Parameters {excluded} are excluded. The saved file cannot be "
                "used to fully reconstruct the template model via load_model().",
                UserWarning, stacklevel=2
            )
        param_names = [p for p in param_names if p in parfilter]

    hdus = [_primary_hdu()]

    for pname in param_names:
        val  = result._param_array(pname)
        unit = result.get_unit(pname)
        hdus.append(_image_hdu(_ext_name(pname), val, unit, wcs=wcs, cd_matrix=cd_matrix))
        if std:
            err = result._std_array(pname)
            hdus.append(_image_hdu(_ext_name(pname) + EXT_ERR_SUFFIX, err, unit, wcs=wcs, cd_matrix=cd_matrix))

    for ext_name, data in custom_exts.items():
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape((1,))
        hdus.append(_image_hdu(ext_name.upper(), arr, wcs=wcs, cd_matrix=cd_matrix))

    resources = _operator_files(result._template_model, filename, operator_file) if model else {}
    if model:
        yaml_str = _serialize_model(result._template_model, resources)
        if yaml_str is not None:
            hdus.append(_model_hdu(yaml_str))

    _write_session(hdus, filename, resources, overwrite)


# ---------------------------------------------------------------------------
# SpectrumFitResult  to_fits / save
# ---------------------------------------------------------------------------

def spectrumfit_to_fits(result, filename: str, overwrite: bool = False,
                        parfilter=None, std: bool = True, model: bool = True, operator_file=None,
                        **custom_exts):
    """
    Save a single ``SpectrumFitResult`` to a FITS file with a human-readable
    binary table (``PARS``) listing names, values, errors, and units.

    Parameters
    ----------
    result : SpectrumFitResult
    filename : str
    overwrite : bool
    parfilter : list of str, optional
        Export only these parameters.
    std : bool
        Include standard deviations in the PARS table.
    model : bool
        Serialize the fitted model into a ``MODEL_DEF`` extension.
    operator_file : str or pathlib.Path, optional
        External fixed-operator FITS file; relative paths resolve beside the
        session. Requires exactly one operator. By default, direct matrices use
        content-named resources and registered responses retain verified recipes.
    **custom_exts
        Extra scalar or array extensions.
    """
    return model_to_fits(result.model, filename, overwrite=overwrite,
                         parfilter=parfilter, std=std, model_def=model,
                         operator_file=operator_file, **custom_exts)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_params(filename: str) -> dict:
    """
    Load best-fit parameter maps/values from a prism FITS file.

    Returns a dict suitable for passing directly to ``fitter.multifit(initpars=...)``.

    * ``PARS`` BinTableHDU (single spectrum) → dict of scalars.
    * ``ImageHDU`` extensions (batch / cube) → dict of numpy arrays.
    """
    params = {}
    with fits.open(filename) as hdul:
        if hdul[0].header.get('FORMAT') == 'PRISMFIT':
            raise ValueError('This file stores a FitResult; use FitResult.load(...).')
        ext_names = [hdu.name.upper() for hdu in hdul]

        if EXT_PARS in ext_names:
            data = hdul[EXT_PARS].data
            for row in data:
                params[str(row['NAME']).strip().lower()] = float(row['VALUE'])
            return params

        skip = {'PRIMARY', EXT_MODEL}
        for hdu in hdul:
            name = hdu.name.upper()
            if name in skip or name.endswith(EXT_ERR_SUFFIX):
                continue
            if isinstance(hdu, fits.ImageHDU) and hdu.data is not None:
                params[name.lower()] = hdu.data
    return params


def load_model(filename: str, *, components=None):
    """
    Deserialize the model and declarative links stored in a Prism FITS file.

    Custom functions are supplied explicitly as named Component definitions through
    ``components={name: component}``. Definitions are not mutated.

    Requires the file to contain a ``MODEL_DEF`` extension written by
    ``to_fits(model=True)`` (the default).
    """
    # Ensure prism YAML hooks are active
    try:
        import prism.modeling.models.serialization  # noqa: F401
    except ImportError:
        pass
    with fits.open(filename) as hdul:
        if hdul[0].header.get('FORMAT') == 'PRISMFIT':
            raise ValueError('This file stores a FitResult; use FitResult.load(...).')
        if EXT_MODEL not in [h.name.upper() for h in hdul]:
            raise ValueError(
                f"'{filename}' does not contain a '{EXT_MODEL}' extension. "
                "Was the file saved with model=True (default)?"
            )
        yaml_str = str(hdul[EXT_MODEL].data['YAML'][0])
        class SessionLoader(astropy_yaml.AstropyLoader):
            pass
        SessionLoader.base_path = Path(filename).resolve().parent
        SessionLoader.components = {} if components is None else components
        model = yaml.load(yaml_str, Loader=SessionLoader)
        if isinstance(model, dict) and '_prism_links' in model:
            if model['_prism_links'] != 1 or set(model) != {'_prism_links', 'model', 'links'}:
                raise ValueError('Unsupported parameter-link session format.')
            from .parameters import _restore_links
            payload = model
            model = payload['model']
            _restore_links(model, payload['links'])

        # Restore covariance matrix if saved
        if EXT_COV in [h.name.upper() for h in hdul]:
            model._param_cov = np.asarray(hdul[EXT_COV].data, dtype=np.float64)
            order = hdul[EXT_COV].header.get('FITIND')
            if order is not None:
                model._fit_indices = np.array([int(i) for i in order.split(',') if i], dtype=int)

        # Restore .std from PARS table if available
        if EXT_PARS in [h.name.upper() for h in hdul]:
            data = hdul[EXT_PARS].data
            if 'STD' in data.columns.names:
                for row in data:
                    pname = str(row['NAME']).strip()
                    std_val = float(row['STD'])
                    if np.isfinite(std_val) and hasattr(model, pname):
                        getattr(model, pname).std = std_val
            if 'LOLIM' in data.columns.names:
                for row in data:
                    pname = str(row['NAME']).strip()
                    val = float(row['LOLIM'])
                    if np.isfinite(val) and hasattr(model, pname):
                        getattr(model, pname).lolim = val
            if 'UPLIM' in data.columns.names:
                for row in data:
                    pname = str(row['NAME']).strip()
                    val = float(row['UPLIM'])
                    if np.isfinite(val) and hasattr(model, pname):
                        getattr(model, pname).uplim = val

        return model
