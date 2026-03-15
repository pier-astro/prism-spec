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
import numpy as np
from datetime import datetime
from astropy.io import fits
from astropy.io.misc import yaml as astropy_yaml
import astropy.units as u

# Extension name constants
EXT_PARS       = 'PARS'
EXT_MODEL      = 'MODEL_DEF'
EXT_ERR_SUFFIX = '_ERR'

_PARNAME_MAXLEN = 18  # FITS extension-name maximum usable length


def _ext_name(param_name: str) -> str:
    """Truncate & upper-case a parameter name for use as a FITS extension name."""
    return param_name.upper()[:_PARNAME_MAXLEN]


def _serialize_model(model) -> str | None:
    """Serialize an Astropy model to a YAML string; returns None on failure."""
    if model is None:
        return None
    # Ensure prism YAML hooks are active
    try:
        import prism.modeling.models.serialization  # noqa: F401 (triggers register())
    except ImportError:
        pass
    try:
        return astropy_yaml.dump(model)
    except Exception as exc:
        warnings.warn(
            f"Could not serialize model to YAML: {exc}. "
            "MODEL_DEF extension will be omitted.",
            UserWarning, stacklevel=4
        )
        return None


def _model_hdu(yaml_str: str) -> fits.BinTableHDU:
    """Pack a YAML string into a single-cell BinTableHDU."""
    from astropy.io.fits import Column, BinTableHDU
    col = Column(name='YAML', format=f'{len(yaml_str) + 1}A', array=[yaml_str])
    hdu = BinTableHDU.from_columns([col], name=EXT_MODEL)
    hdu.header['COMMENT'] = 'prism-spec model definition (Astropy YAML)'
    return hdu


def _primary_hdu(wcs=None) -> fits.PrimaryHDU:
    hdr = fits.Header()
    hdr['AUTHOR']  = 'prism-spec'
    hdr['DATE']    = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    hdr['HISTORY'] = 'Created by prism-spec MultiFit I/O'
    if wcs is not None:
        hdr.update(wcs.to_header())
    return fits.PrimaryHDU(header=hdr)


def _image_hdu(name: str, data: np.ndarray, unit=None) -> fits.ImageHDU:
    hdr = fits.Header()
    if unit is not None:
        try:
            hdr['BUNIT'] = u.Unit(unit).to_string('fits')
        except Exception:
            hdr['BUNIT'] = str(unit)
    return fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=hdr, name=name)


# ---------------------------------------------------------------------------
# Single fitted Astropy model  to_fits / save
# ---------------------------------------------------------------------------

def model_to_fits(model, filename: str, overwrite: bool = False,
                  parfilter=None, std: bool = True, model_def: bool = True,
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

    if model_def:
        yaml_str = _serialize_model(model)
        if yaml_str is not None:
            hdus.append(_model_hdu(yaml_str))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


def install_model_io_methods():
    """Install class-level ``to_fits`` / ``save`` on Astropy models (idempotent)."""
    from astropy.modeling.core import Model as AstropyModel

    if getattr(AstropyModel, '_prism_io_installed', False):
        return

    def _to_fits(self, filename, overwrite=False,
                 parfilter=None, std=True, model=True, **custom_exts):
        return model_to_fits(
            self,
            filename,
            overwrite=overwrite,
            parfilter=parfilter,
            std=std,
            model_def=model,
            **custom_exts,
        )

    AstropyModel.to_fits = _to_fits
    AstropyModel.save = _to_fits
    AstropyModel._prism_io_installed = True


# Ensure direct model.save()/to_fits() is available whenever this module is imported.
install_model_io_methods()


# ---------------------------------------------------------------------------
# MultiFitResult  to_fits / save
# ---------------------------------------------------------------------------

def multifit_to_fits(result, filename: str, wcs=None, overwrite: bool = False,
                     parfilter=None, std: bool = True, model: bool = True,
                     **custom_exts):
    """
    Save a ``MultiFitResult`` to a multi-extension FITS file.

    Parameters
    ----------
    result : MultiFitResult
    filename : str
    wcs : astropy.wcs.WCS, optional
        Spatial WCS for cube results loaded into the PRIMARY header.
    overwrite : bool
    parfilter : list of str, optional
        Export only these parameters. Omitting parameters raises a warning
        because the saved file will not fully reconstruct the model.
    std : bool
        Also export parameter standard-deviation maps as ``<PARNAME>_ERR``.
    model : bool
        Serialize the template model into a ``MODEL_DEF`` extension.
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

    hdus = [_primary_hdu(wcs)]

    for pname in param_names:
        val  = result._param_array(pname)
        unit = result.get_unit(pname)
        hdus.append(_image_hdu(_ext_name(pname), val, unit))
        if std:
            err = result._std_array(pname)
            hdus.append(_image_hdu(_ext_name(pname) + EXT_ERR_SUFFIX, err, unit))

    for ext_name, data in custom_exts.items():
        hdus.append(_image_hdu(ext_name.upper(), np.asarray(data)))

    if model:
        yaml_str = _serialize_model(result._template_model)
        if yaml_str is not None:
            hdus.append(_model_hdu(yaml_str))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


# ---------------------------------------------------------------------------
# SpectrumFitResult  to_fits / save
# ---------------------------------------------------------------------------

def spectrumfit_to_fits(result, filename: str, overwrite: bool = False,
                        parfilter=None, std: bool = True, model: bool = True,
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
        Serialize the template model into a ``MODEL_DEF`` extension.
    **custom_exts
        Extra scalar or array extensions.
    """
    parent = result._parent
    param_names = list(parent.param_names)

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

    # PARS binary table
    names  = np.array(param_names, dtype=object)
    values = result.parameters[:len(param_names)]
    errors = result.errors[:len(param_names)] if std else np.full(len(param_names), np.nan)
    units  = [
        (parent.get_unit(p).to_string('fits') if parent.get_unit(p) else '') for p in param_names
    ]
    
    has_lolim = any(hasattr(getattr(result.model, p), 'lolim') for p in param_names)
    has_uplim = any(hasattr(getattr(result.model, p), 'uplim') for p in param_names)

    if has_lolim:
        lolims = [getattr(getattr(result.model, p), 'lolim', np.nan) for p in param_names]
    if has_uplim:
        uplims = [getattr(getattr(result.model, p), 'uplim', np.nan) for p in param_names]

    n = len(param_names)
    name_fmt = f'{max(len(p) for p in param_names) + 1}A'
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
        
    cols.append(Column(name='UNIT',  format=unit_fmt, array=units))
    
    hdus.append(BinTableHDU.from_columns(cols, name=EXT_PARS))

    for ext_name, data in custom_exts.items():
        hdus.append(_image_hdu(ext_name.upper(), np.atleast_1d(np.asarray(data, dtype=np.float32))))

    if model:
        yaml_str = _serialize_model(parent._template_model)
        if yaml_str is not None:
            hdus.append(_model_hdu(yaml_str))

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


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


def load_model(filename: str):
    """
    Deserialize the template Astropy model stored in a prism FITS file.

    Requires the file to contain a ``MODEL_DEF`` extension written by
    ``to_fits(model=True)`` (the default).
    """
    # Ensure prism YAML hooks are active
    try:
        import prism.modeling.models.serialization  # noqa: F401
    except ImportError:
        pass
    with fits.open(filename) as hdul:
        if EXT_MODEL not in [h.name.upper() for h in hdul]:
            raise ValueError(
                f"'{filename}' does not contain a '{EXT_MODEL}' extension. "
                "Was the file saved with model=True (default)?"
            )
        yaml_str = str(hdul[EXT_MODEL].data['YAML'][0])
        return astropy_yaml.load(yaml_str)
