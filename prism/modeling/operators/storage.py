"""Lossless FITS storage for fixed discretized linear operators.

This Prism format stores a CSR matrix and explicit coordinate grids. Matrix
entries act on sampled values; no probability normalization is implied.
"""
from pathlib import Path
import hashlib
import os
import tempfile
import json
import numpy as np
from scipy.sparse import csr_matrix
from astropy import units as u
from astropy.io import fits


def _atomic_write(hdus, filename, overwrite=False):
    target = Path(filename)
    if target.exists() and not overwrite:
        raise FileExistsError(f'File exists: {target}')
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=''.join(target.suffixes) or '.fits', delete=False) as stream:
        stage = Path(stream.name)
    try:
        fits.HDUList(hdus).writeto(stage, overwrite=True, checksum=True)
        os.replace(stage, target)
    finally:
        stage.unlink(missing_ok=True)


def _operator_state(operator):
    from .matop import LinearOperatorModel
    from .instrument import SpectralResponse
    if type(operator) not in (LinearOperatorModel, SpectralResponse):
        raise ValueError('Only fixed built-in operators can be stored as matrices; parameter-dependent/custom classes require an explicit reconstruction rule.')
    state = dict(kind='spectral' if type(operator) is SpectralResponse else 'linear',
                 name=operator.name, flexible=bool(operator.flexible))
    if state['kind'] == 'spectral':
        # Keep the on-disk state key stable so saved response fingerprints remain valid.
        state.update(z=float(operator.redshift), renormalize=bool(operator.renormalize))
    return state


def _restore_operator(matrix, x, xout, state):
    from .matop import LinearOperatorModel
    from .instrument import SpectralResponse
    if state['kind'] == 'linear':
        return LinearOperatorModel(matrix, x=x, xout=xout, name=state['name'], flexible=state['flexible'])
    if state['kind'] == 'spectral':
        if xout is not None and (x is None or not np.array_equal(xout, x)):
            raise ValueError('A spectral response requires the same native input/output grid.')
        return SpectralResponse(response_matrix=matrix, wave=x, redshift=state['z'],
                                renormalize=state['renormalize'], flexible=state['flexible'], name=state['name'])
    raise ValueError(f"Unknown fixed operator type: {state['kind']!r}.")


def _operator_digest(operator):
    """Fingerprint coefficients, grids, units and evaluation options."""
    state = _operator_state(operator)
    state.pop('name', None)
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode())
    matrix = csr_matrix(operator.matrix)
    digest.update(np.asarray(matrix.shape, dtype='<i8').tobytes())
    for array in [matrix.data, matrix.indices, matrix.indptr]:
        dtype = '<f8' if array.dtype.kind == 'f' else '<i8'
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    for grid in [operator.x, operator.xout]:
        if grid is None:
            digest.update(b'none')
        else:
            digest.update(str(getattr(grid, 'unit', '')).encode())
            digest.update(np.asarray(grid, dtype='<f8').tobytes())
    return digest.hexdigest()


def _write_operator(operator, filename, overwrite=False):
    state = _operator_state(operator)
    matrix = csr_matrix(operator.matrix)
    if matrix.dtype.kind not in 'fiu' or not np.all(np.isfinite(matrix.data)):
        raise ValueError('Operator coefficients must be finite real numbers.')
    header = fits.Header()
    header['FORMAT'] = 'PRISMLIN'
    header['VERSION'] = 1
    header['NINPUT'] = matrix.shape[1]
    header['NOUTPUT'] = matrix.shape[0]
    header['ACTION'] = 'SAMPLES'
    header['STATE'] = json.dumps(state, sort_keys=True)
    hdus = [fits.PrimaryHDU(header=header),
            fits.ImageHDU(matrix.data, name='DATA'),
            fits.ImageHDU(matrix.indices.astype(np.int64), name='INDICES'),
            fits.ImageHDU(matrix.indptr.astype(np.int64), name='INDPTR')]
    same = operator.x is not None and operator.xout is not None and np.array_equal(operator.x, operator.xout)
    same = same and getattr(operator.x, 'unit', None) == getattr(operator.xout, 'unit', None)
    hdus[0].header['SAMEGRID'] = same
    for name, grid in [('X', operator.x), ('XOUT', None if same else operator.xout)]:
        if grid is not None:
            hdu = fits.ImageHDU(np.asarray(grid), name=name)
            if hasattr(grid, 'unit'):
                hdu.header['BUNIT'] = grid.unit.to_string('fits')
            hdus.append(hdu)
    _atomic_write(hdus, filename, overwrite)


def _read_operator(filename):
    with fits.open(filename) as hdus:
        header = hdus[0].header
        if header.get('FORMAT') != 'PRISMLIN' or header.get('VERSION') != 1 or header.get('ACTION') != 'SAMPLES':
            raise ValueError('Unsupported discretized-operator FITS format or version.')
        for hdu in hdus:
            if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                raise ValueError('Operator FITS checksum is missing or invalid.')
        shape = (header['NOUTPUT'], header['NINPUT'])
        data = np.asarray(hdus['DATA'].data)
        data = data.astype(data.dtype.newbyteorder('='), copy=True)
        indices = np.array(hdus['INDICES'].data, dtype=np.int64)
        indptr = np.array(hdus['INDPTR'].data, dtype=np.int64)
        matrix = csr_matrix((data, indices, indptr), shape=shape)
        matrix.check_format(full_check=True)
        def grid(name):
            if name not in hdus:
                return None
            values = np.array(hdus[name].data, dtype=float)
            unit = hdus[name].header.get('BUNIT')
            return values if unit is None else values*u.Unit(unit)
        x = grid('X')
        xout = x if header['SAMEGRID'] else grid('XOUT')
        state = json.loads(header['STATE'])
    return _restore_operator(matrix, x, xout, state)
