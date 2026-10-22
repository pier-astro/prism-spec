"""
Instrumental response (line-spread-function) convolution.

Build, load, and apply spectral response matrices to Astropy models
via ``SpectralResponse`` — a Prism ``LinearOperatorModel`` used on the
right-hand side of Astropy's native pipe operator.

The observed spectrum is modelled as ``h(x) = R @ f(x, θ)`` where *R*
is the instrument response matrix and *f* is the intrinsic source.
Prism patches Astropy's ``CompoundModel`` pipe evaluation only for
opt-in right-hand operator models, preserving the chain rule
``J_h = R @ J_f``.

Classes
-------
InstrumentResponse
    Build / load / save / crop response matrices.
SpectralResponse
    Linear operator model used as ``source_model | rsp``.

Wavelength conventions
~~~~~~~~~~~~~~~~~~~~~~
When sources are redshifted, the data is typically redshift-corrected to rest
frame.  The instrument LSF is defined in observer frame.
``SpectralResponse`` handles the mapping via the *redshift* parameter:
``λ_obs = λ_rest × (1 + redshift)``.

Instrument storage
~~~~~~~~~~~~~~~~~~
* Package defaults are in ``responses/`` (shipped with prism-spec).
* Custom instruments go to ``~/.prism/instruments/``.
* User instruments with the same name override package defaults.

To register a new instrument for recipe-based serialization::

    resp = InstrumentResponse.from_fixed_fwhm(wave_obs, fwhm=2.5)
    resp.save('MY_INSTRUMENT')

Example
-------
>>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave_rest, redshift=0.1)
>>> convolved = source_model | rsp      # CompoundModel pipe expression
>>> convolved.left                      # unwrapped source
>>> convolved(wave_rest)                # evaluates R @ f(wave_rest, θ)
"""
import numpy as np
import os
import warnings
import yaml

from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, issparse
from scipy.special import ndtr
from astropy.io import fits
from .matop import LinearOperatorModel, is_linear_operator_pipe, _apply_matrix, _sample_response


# --- Path utilities ---

def _get_package_response_dir() -> str:
    """Return the path to the package responses directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources", "responses")


def _get_user_response_dir() -> str:
    """Return the path to the user's local response directory (~/.prism/instruments/)."""
    return os.path.join(os.path.expanduser("~"), ".prism", "instruments")


def _ensure_user_response_dir() -> str:
    """Ensure user response directory exists and return its path."""
    user_dir = _get_user_response_dir()
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def _get_user_yaml_path() -> str:
    """Return the path to the user's responses.yaml file."""
    return os.path.join(_get_user_response_dir(), "rsps.yaml")


def _get_package_yaml_path() -> str:
    """Return the path to the package responses.yaml file."""
    return os.path.join(_get_package_response_dir(), "rsps.yaml")


def load_responses_mapping() -> dict:
    """
    Load the combined instrument-to-FITS mapping.
    
    User instruments (~/.prism/instruments/) override package defaults.
    """
    # Load package defaults
    pkg_yaml = _get_package_yaml_path()
    if os.path.exists(pkg_yaml):
        with open(pkg_yaml, "r") as f:
            pkg_mapping = yaml.safe_load(f) or {}
    else:
        pkg_mapping = {}
    
    # Load user instruments (override package)
    user_yaml = _get_user_yaml_path()
    if os.path.exists(user_yaml):
        with open(user_yaml, "r") as f:
            user_mapping = yaml.safe_load(f) or {}
    else:
        user_mapping = {}
    
    # Merge: user overrides package
    combined = {**pkg_mapping, **user_mapping}
    return combined


def _load_user_mapping() -> dict:
    """Load only the user's instrument mapping."""
    user_yaml = _get_user_yaml_path()
    if os.path.exists(user_yaml):
        with open(user_yaml, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_user_mapping(mapping: dict) -> None:
    """Save the user's instrument mapping."""
    _ensure_user_response_dir()
    user_yaml = _get_user_yaml_path()
    with open(user_yaml, "w") as f:
        yaml.safe_dump(mapping, f)


def add_response_to_archive(inst_name: str, fits_filename: str, clobber: bool = False, user: bool = True) -> None:
    """
    Add or update an instrument-FITS mapping in the archive.
    
    Parameters
    ----------
    inst_name : str
        Name of the instrument
    fits_filename : str
        FITS filename (basename only)
    clobber : bool
        Overwrite existing entry
    user : bool
        If True, save to user directory; if False, save to package directory
    """
    if user:
        mapping = _load_user_mapping()
        if inst_name in mapping and not clobber:
            raise ValueError(f"Instrument '{inst_name}' already in user archive. Use clobber=True to overwrite.")
        mapping[inst_name] = fits_filename
        _save_user_mapping(mapping)
    else:
        # Package directory (for development)
        pkg_yaml = _get_package_yaml_path()
        if os.path.exists(pkg_yaml):
            with open(pkg_yaml, "r") as f:
                mapping = yaml.safe_load(f) or {}
        else:
            mapping = {}
        if inst_name in mapping and not clobber:
            raise ValueError(f"Instrument '{inst_name}' already in package archive. Use clobber=True to overwrite.")
        mapping[inst_name] = fits_filename
        with open(pkg_yaml, "w") as f:
            yaml.safe_dump(mapping, f)


def list_instruments() -> dict:
    """
    List all available instruments with their source (package or user).
    
    Returns
    -------
    dict
        Dictionary with instrument names as keys and 'package' or 'user' as values
    """
    # Load both mappings
    pkg_yaml = _get_package_yaml_path()
    if os.path.exists(pkg_yaml):
        with open(pkg_yaml, "r") as f:
            pkg_mapping = yaml.safe_load(f) or {}
    else:
        pkg_mapping = {}
    
    user_mapping = _load_user_mapping()
    
    result = {}
    for name in pkg_mapping:
        result[name] = 'package'
    for name in user_mapping:
        result[name] = 'user' if name not in pkg_mapping else 'user (override)'
    
    return result


def _response_grid(wavelength_grid):
    """Own a finite, strictly increasing one-dimensional response grid."""
    wave = np.array(wavelength_grid, dtype=float, copy=True)
    if wave.ndim != 1 or wave.size == 0 or not np.all(np.isfinite(wave)):
        raise ValueError('Response wavelengths must be a non-empty finite 1-D array.')
    if np.any(np.diff(wave) <= 0):
        raise ValueError('Response wavelengths must be strictly increasing.')
    return wave


def _bin_edges(wave):
    """Infer contiguous bin edges from centers, extending half a step at each end."""
    if len(wave) < 2:
        raise ValueError('Gaussian responses require at least two wavelength centers.')
    return np.r_[wave[0] - (wave[1] - wave[0])/2,
                 (wave[:-1] + wave[1:])/2,
                 wave[-1] + (wave[-1] - wave[-2])/2]


def _response_matrix(matrix, size=None):
    matrix = csr_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1] or (size is not None and matrix.shape[0] != size):
        raise ValueError('Response matrix must be square and match the wavelength grid.')
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError('Response matrix must contain only finite values.')
    return matrix


# --- InstrumentResponse class ---

class InstrumentResponse:
    """
    Class for building, loading, saving, and cropping instrumental response matrices.
    
    The density operator acts as ``output = matrix @ input``: rows index output
    samples and columns index input samples. It is also directly callable on
    spectra or sample-major Jacobian columns.

    Gaussian builders integrate the LSF over destination bins at each input
    center, including the input/output bin-width ratio for density arrays.
    Flux outside the supplied band is lost, without artificial renormalization.
    Model values are evaluated at native centers; sampling convergence matters.
    Matrices loaded from files are used as stored, without guessing a convention.
    
    Use classmethods to construct:
    - from_variable_gaussian_resolution(): R(λ) varies with wavelength
    - from_fixed_fwhm(): constant FWHM in Angstroms
    - from_fixed_resolution(): constant R = λ/Δλ
    - from_fixed_sigma(): constant σ in Angstroms
    - from_fits(): load from FITS file
    - from_instrument(): load from instrument archive
    - from_array(): wrap existing matrix
    """
    
    def __init__(self, wavelength_grid: np.ndarray, response_matrix: csr_matrix):
        self.wavelength_grid = _response_grid(wavelength_grid)
        self.response_matrix = _response_matrix(response_matrix, len(self.wavelength_grid))
        self._bin_widths = np.diff(_bin_edges(self.wavelength_grid)) if len(self.wavelength_grid) > 1 else None

    def __call__(self, values):
        """Apply the response to a spectrum or sample-major Jacobian columns.

        The first axis must match the native grid. Quantity units are retained;
        normalization and throughput are those encoded in the stored matrix.
        """
        return _apply_matrix(self.response_matrix, values)

    # ---- BUILDERS ----
    
    @classmethod
    def from_variable_gaussian_resolution(
        cls, 
        wavelength_grid: np.ndarray, 
        lambda_R: np.ndarray, 
        R_values: np.ndarray, 
        interp_kind: str = 'linear'
    ) -> "InstrumentResponse":
        """Build a response matrix using a variable spectral resolution R(λ).

        The resolving power R = λ / Δλ is interpolated onto the target grid to
        compute the corresponding Gaussian FWHM and sigma at each wavelength.

        Parameters
        ----------
        wavelength_grid : numpy.ndarray
            The target 1-D wavelength grid.
        lambda_R : numpy.ndarray
            Wavelengths at which the resolving power is defined.
        R_values : numpy.ndarray
            Resolving power R values corresponding to ``lambda_R``.
        interp_kind : str, optional
            Interpolation method used by `scipy.interpolate.interp1d`. Default is ``'linear'``.

        Returns
        -------
        InstrumentResponse
            The generated response matrix.
        """
        sigmas = cls._compute_sigmas_variable(wavelength_grid, lambda_R, R_values, interp_kind)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_fwhm(cls, wavelength_grid: np.ndarray, fwhm: float) -> "InstrumentResponse":
        """Build a response matrix from a fixed Gaussian FWHM.

        Parameters
        ----------
        wavelength_grid : numpy.ndarray
            The target 1-D wavelength grid.
        fwhm : float
            Full-width at half-maximum in the same units as the wavelength grid.

        Returns
        -------
        InstrumentResponse
            The generated response matrix.
        """
        if fwhm <= 0:
            raise ValueError("FWHM must be positive.")
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_resolution(cls, wavelength_grid: np.ndarray, R: float) -> "InstrumentResponse":
        """Build a response matrix from a fixed spectral resolution R.

        Parameters
        ----------
        wavelength_grid : numpy.ndarray
            The target 1-D wavelength grid.
        R : float
            Resolving power R = λ / Δλ.

        Returns
        -------
        InstrumentResponse
            The generated response matrix.
        """
        if R <= 0:
            raise ValueError("Resolution R must be positive.")
        delta_lam = np.asarray(wavelength_grid) / R
        sigmas = delta_lam / (2 * np.sqrt(2 * np.log(2)))
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_sigma(cls, wavelength_grid: np.ndarray, sigma: float) -> "InstrumentResponse":
        """Build a response matrix from a fixed Gaussian sigma.

        Parameters
        ----------
        wavelength_grid : numpy.ndarray
            The target 1-D wavelength grid.
        sigma : float
            Standard deviation of the Gaussian LSF in the same units as the grid.

        Returns
        -------
        InstrumentResponse
            The generated response matrix.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_array(cls, wavelength_grid: np.ndarray, matrix: np.ndarray) -> "InstrumentResponse":
        """Create an InstrumentResponse from an existing matrix.

        Parameters
        ----------
        wavelength_grid : numpy.ndarray
            The 1-D wavelength grid matching the matrix dimensions.
        matrix : numpy.ndarray or scipy.sparse.spmatrix
            The square response matrix (dense or sparse).

        Returns
        -------
        InstrumentResponse
            The wrapped response matrix.
        """
        return cls(wavelength_grid, matrix)

    # ---- LOADERS ----
    
    @classmethod
    def from_fits(cls, filename: str) -> "InstrumentResponse":
        """Load an instrument response from a standard FITS file format.

        Parameters
        ----------
        filename : str
            Path to the FITS file containing the wavelength grid (ext 1) 
            and response matrix (ext 2).

        Returns
        -------
        InstrumentResponse
            The loaded response matrix.
        """
        with fits.open(filename) as hdul:
            wavelength_grid = hdul[1].data
            dense_matrix = hdul[2].data.astype(np.float64)
            matrix = csr_matrix(dense_matrix)
            response = cls(wavelength_grid, matrix)
            if 'BINWIDTH' in hdul:
                widths = np.asarray(hdul['BINWIDTH'].data, dtype=float)
                if widths.shape != response.wavelength_grid.shape or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
                    raise ValueError('Stored bin widths must be finite, positive and match the wavelength grid.')
                response._bin_widths = widths.copy()
        return response

    @classmethod
    def from_instrument(cls, instrument: str) -> "InstrumentResponse":
        """Load an instrument response by its registered archive name.

        Searches the user directory (``~/.prism/instruments/``) first,
        then falls back to package defaults.

        Parameters
        ----------
        instrument : str
            The registered name of the instrument (e.g., 'MUSE-WFM').

        Returns
        -------
        InstrumentResponse
            The loaded response matrix.

        Raises
        ------
        ValueError
            If the instrument name is not found in either the user or package archive.
        """
        # Check user directory first
        user_mapping = _load_user_mapping()
        if instrument in user_mapping:
            filename = os.path.join(_get_user_response_dir(), user_mapping[instrument])
            return cls.from_fits(filename)
        
        # Fall back to package directory
        pkg_yaml = _get_package_yaml_path()
        if os.path.exists(pkg_yaml):
            with open(pkg_yaml, "r") as f:
                pkg_mapping = yaml.safe_load(f) or {}
        else:
            pkg_mapping = {}
        
        if instrument in pkg_mapping:
            filename = os.path.join(_get_package_response_dir(), pkg_mapping[instrument])
            return cls.from_fits(filename)
        
        # Not found
        all_instruments = list_instruments()
        raise ValueError(f"Unknown instrument: '{instrument}'. Available: {list(all_instruments.keys())}")

    # ---- INTERNALS ----
    
    @staticmethod
    def _compute_sigmas_variable(wavelength_grid, lambda_R, R_values, interp_kind):
        """Compute sigma array from variable R(λ)."""
        wavelength_grid = _response_grid(wavelength_grid)
        lambda_R = _response_grid(lambda_R)
        R_values = np.asarray(R_values, dtype=float)
        if R_values.shape != lambda_R.shape or not np.all(np.isfinite(R_values)) or np.any(R_values <= 0):
            raise ValueError('Resolving powers must be finite, positive and match lambda_R.')
        R = interp1d(lambda_R, R_values, kind=interp_kind, bounds_error=False,
                     fill_value=(R_values[0], R_values[-1]))(wavelength_grid)
        return wavelength_grid / R / (2*np.sqrt(2*np.log(2)))

    @staticmethod
    def _build_sparse_gaussian_matrix(wavelength_grid, sigmas):
        """Redistribute bin-integrated flux; do not return escaped flux to the band."""
        wave = _response_grid(wavelength_grid)
        edges = _bin_edges(wave)
        widths = np.diff(edges)
        sigmas = np.asarray(sigmas, dtype=float)
        if sigmas.shape != wave.shape or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError('Gaussian sigmas must be finite, positive and match the wavelength grid.')

        data, rows, columns = [], [], []
        for j, (center, sigma) in enumerate(zip(wave, sigmas)):
            # Only visit bins overlapping +/-8 sigma: omitted Gaussian mass <1.3e-15.
            start = max(0, np.searchsorted(edges, center - 8*sigma, side='right') - 1)
            stop = min(len(wave), np.searchsorted(edges, center + 8*sigma, side='left'))
            i = np.arange(start, stop)
            probability = ndtr((edges[i+1] - center)/sigma) - ndtr((edges[i] - center)/sigma)
            # Input density -> integrated flux -> redistribution -> output density.
            density = probability * widths[j] / widths[i]
            nonzero = density > 0
            data.extend(density[nonzero])
            rows.extend(i[nonzero])
            columns.extend([j]*np.count_nonzero(nonzero))
        return csr_matrix((data, (rows, columns)), shape=(len(wave), len(wave)))

    # ---- SAVE & REGISTER ----
    
    def save(self, filename: str, compress: bool = True) -> None:
        """Save the wavelength grid and response matrix to a FITS file.

        Parameters
        ----------
        filename : str
            The output FITS file path.
        compress : bool, optional
            If True, uses lossless FITS GZIP tile compression. Default is ``True``.
        """
        if self.response_matrix is None:
            raise ValueError("No response matrix to save.")
        primary_hdu = fits.PrimaryHDU()
        wavelength_hdu = fits.ImageHDU(self.wavelength_grid, name='WAVELENGTH')
        dense_matrix = self.response_matrix.toarray()
        if compress:
            matrix_hdu = fits.CompImageHDU(
                dense_matrix, name='RESPONSE', compression_type='GZIP_1', quantize_level=0.0)
        else:
            matrix_hdu = fits.ImageHDU(dense_matrix, name='RESPONSE')
        hdus = [primary_hdu, wavelength_hdu, matrix_hdu]
        if self._bin_widths is not None:
            hdus.append(fits.ImageHDU(self._bin_widths, name='BINWIDTH'))
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=True)

    def save_and_register(
        self, instrument_name: str, 
        compress: bool = True, clobber: bool = False
    ) -> None:
        """Save the response matrix and register it in the user archive.
        
        The file is saved to ``~/.prism/instruments/<instrument_name>.fits``
        and registered in the user's instrument archive YAML file.
        
        Parameters
        ----------
        instrument_name : str
            Name to register the instrument under.
        compress : bool, optional
            Use FITS compression. Default is ``True``.
        clobber : bool, optional
            Overwrite an existing file or registry entry. Default is ``False``.
        """
        user_dir = _ensure_user_response_dir()
        filename = f"{instrument_name}.fits"
        archive_filename = os.path.join(user_dir, filename)
        
        if os.path.exists(archive_filename) and not clobber:
            raise ValueError(f"File '{archive_filename}' exists. Use clobber=True to overwrite.")
        
        self.save(archive_filename, compress=compress)
        add_response_to_archive(instrument_name, filename, clobber=clobber, user=True)
        print(f"Saved and registered '{instrument_name}' to {archive_filename}")

    # ---- CROP ----
    
    def crop(self, new_wavelengths: np.ndarray, renormalize: bool = False) -> "InstrumentResponse":
        """Crop the response matrix to match a new subset wavelength grid.

        Parameters
        ----------
        new_wavelengths : numpy.ndarray
            The target wavelength grid to crop to. Must be a subset of the original grid.
        renormalize : bool, optional
            Default False preserves the stored matrix coefficients and boundary
            losses. True retains the legacy row-normalized smoothing behavior;
            it does not conserve integrated flux in general.

        Returns
        -------
        InstrumentResponse
            A new cropped response matrix.
        """
        cropped_matrix = _crop_response_matrix(
            self.response_matrix, self.wavelength_grid, new_wavelengths, renormalize=renormalize
        )
        cropped = InstrumentResponse(new_wavelengths, cropped_matrix)
        if self._bin_widths is not None:
            cropped._bin_widths = self._bin_widths[_response_indices(self.wavelength_grid, new_wavelengths)]
        return cropped

    def __repr__(self) -> str:
        shape = self.wavelength_grid.shape if self.wavelength_grid is not None else None
        wmin = f"{self.wavelength_grid.min():.1f}" if self.wavelength_grid is not None else "?"
        wmax = f"{self.wavelength_grid.max():.1f}" if self.wavelength_grid is not None else "?"
        return f"<InstrumentResponse({shape}, λ=[{wmin}, {wmax}])>"


def delete_instrument(instrument_name: str, user_only: bool = True) -> None:
    """
    Remove an instrument from the archive and delete its FITS file.
    
    Parameters
    ----------
    instrument_name : str
        Name of the instrument to delete
    user_only : bool
        If True (default), only search/delete from user directory.
        If False, also check package directory (requires write permissions).
    """
    mapping = _load_user_mapping()
    
    if instrument_name in mapping:
        filename = mapping.pop(instrument_name)
        _save_user_mapping(mapping)
        
        user_dir = _get_user_response_dir()
        fits_path = os.path.join(user_dir, filename)
        if os.path.exists(fits_path):
            os.remove(fits_path)
            print(f"Deleted instrument '{instrument_name}' and file {fits_path}")
        else:
            print(f"Removed instrument '{instrument_name}' from archive (file not found)")
        return

    if not user_only:
        pkg_yaml = _get_package_yaml_path()
        if os.path.exists(pkg_yaml):
            with open(pkg_yaml, "r") as f:
                pkg_mapping = yaml.safe_load(f) or {}
            
            if instrument_name in pkg_mapping:
                filename = pkg_mapping.pop(instrument_name)
                with open(pkg_yaml, "w") as f:
                    yaml.safe_dump(pkg_mapping, f)
                
                pkg_dir = _get_package_response_dir()
                fits_path = os.path.join(pkg_dir, filename)
                if os.path.exists(fits_path):
                    os.remove(fits_path)
                    print(f"Deleted package instrument '{instrument_name}' and file {fits_path}")
                else:
                    print(f"Removed package instrument '{instrument_name}' from archive (file not found)")
                return

    raise ValueError(f"Instrument '{instrument_name}' not found in {'user' if user_only else 'any'} archive.")


def _response_indices(matrix_wave, target_wave):
    """Match requested samples to distinct ordered native bins within half a step."""
    matrix_wave = np.asarray(matrix_wave, dtype=float)
    target_wave = np.asarray(target_wave, dtype=float)

    if matrix_wave.ndim != 1 or target_wave.ndim != 1:
        raise ValueError("matrix_wave and target_wave must be 1-D arrays.")
    if len(matrix_wave) == 0 or len(target_wave) == 0:
        raise ValueError("matrix_wave and target_wave must be non-empty.")

    # Match each target bin to the nearest response bin center.
    # Allow offsets up to half a response bin so cubes with the same
    # sampling but slightly different absolute zero-points still crop
    # cleanly while larger grid mismatches remain an error.
    if len(matrix_wave) > 1:
        matrix_step = np.abs(np.diff(matrix_wave))
        atol = 0.5 * matrix_step.min() + np.finfo(float).eps * max(1.0, np.abs(matrix_wave).max())
    else:
        atol = 0.5

    indices = np.searchsorted(matrix_wave, target_wave)
    indices = np.clip(indices, 1, len(matrix_wave) - 1)
    left = indices - 1
    use_left = np.abs(target_wave - matrix_wave[left]) <= np.abs(matrix_wave[indices] - target_wave)
    indices = np.where(use_left, left, indices)

    deltas = np.abs(matrix_wave[indices] - target_wave)
    bad = np.where(deltas > atol)[0]
    if len(bad):
        j = bad[0]
        raise ValueError(
            f"Wavelength {target_wave[j]:.3f} not found within {atol:.3f} Angstrom "
            f"of the matrix grid [{matrix_wave.min():.1f}, {matrix_wave.max():.1f}]. "
            f"Nearest grid value is {matrix_wave[indices[j]]:.3f} "
            f"(offset {matrix_wave[indices[j]] - target_wave[j]:+.3f} Angstrom)."
        )
    if np.any(np.diff(indices) <= 0):
        raise ValueError(
            "target_wave does not map to a strictly increasing subset of the matrix grid. "
            "This usually means the target grid is oversampled, unsorted, or otherwise "
            "incompatible with simple matrix cropping."
        )

    return indices


def _crop_response_matrix(
    matrix: np.ndarray,
    matrix_wave: np.ndarray,
    target_wave: np.ndarray,
    renormalize: bool = False
) -> np.ndarray:
    """
    Crop a response matrix to match the target wavelengths.

    Target wavelengths must be a subset of the matrix wavelengths (within tolerance).
    Gaps in the target grid are allowed (e.g., masked spectral regions).

    Args:
        matrix: 2D sparse or dense matrix (N x N)
        matrix_wave: wavelengths corresponding to the matrix
        target_wave: desired output wavelengths (can have gaps)
        renormalize: legacy row normalization after cropping (default False)

    Returns:
        Cropped (and optionally renormalized) sparse matrix
    """
    indices = _response_indices(matrix_wave, target_wave)

    cropped = matrix[np.ix_(indices, indices)]

    if renormalize:
        row_sums = cropped.sum(axis=1).A1 if issparse(cropped) else cropped.sum(axis=1)
        # Avoid division by zero
        row_sums = np.where(row_sums > 1e-10, row_sums, 1.0)
        if issparse(cropped):
            cropped = cropped.multiply(1 / row_sums[:, np.newaxis])
        else:
            cropped = cropped / row_sums[:, np.newaxis]

    return cropped if issparse(cropped) else csr_matrix(cropped)


# --- SpectralResponse class ---

class SpectralResponse(LinearOperatorModel):
    """
    Instrument-response linear operator for Astropy pipe expressions.

    Use an instance directly on the right-hand side of a native Astropy pipe
    expression, as ``source | rsp``, to convolve a source model while
    preserving analytic Jacobians.

    Parameters
    ----------
    instrument : str or InstrumentResponse, optional
        Name of an archived instrument (e.g. ``'MUSE-WFM'``) **or** an
        ``InstrumentResponse`` instance built manually.
    wave : array-like
        Requested rest-frame samples, matched at lambda_obs = wave*(1+redshift).
        With flexible=True and renormalize=False, the native model grid also
        retains available neighboring source bins from the supplied response.
        Consequently wavelength_grid can be larger than wave.
    redshift : float, default 0
        Redshift used to map rest→observer wavelengths.
    response_matrix : sparse matrix, optional
        Supply a pre-built matrix directly (no archive lookup, no cropping).
    renormalize : bool, default False
        Preserve stored coefficients and physical boundary losses by default.
        True requests the legacy cropped, row-normalized smoothing behavior.
    flexible : bool, default True
        If ``True``, evaluation on a different wavelength grid is handled by
        evaluating the source model on the native response grid and
        interpolating the convolved spectrum back to the requested grid. This
        preserves native-grid contributions while allowing subset or plotting
        grids. False restricts evaluation to the supplied wave grid and disables
        automatic retention of neighboring source bins.
    name : str, default ``'rsp'``
        Name of the response operator model.

    Examples
    --------
    >>> # From archive (serializable — full round-trip supported)
    >>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave, redshift=0.1)
    >>> model = source | rsp

    >>> # Custom name at construction
    >>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave, redshift=0.1, name='muse')
    >>> model = source | rsp

    >>> # From InstrumentResponse instance (matrix serialized directly)
    >>> ir = InstrumentResponse.from_fixed_fwhm(wave, fwhm=2.5)
    >>> rsp = SpectralResponse(instrument=ir, wave=wave)
    >>> model = source | rsp
    """
    
    def __init__(
        self,
        instrument: str | InstrumentResponse | None = None,
        wave: np.ndarray | None = None,
        redshift: float = 0,
        response_matrix=None,
        renormalize: bool = False,
        flexible: bool = True,
        name: str = 'rsp',
    ):
        if not np.isfinite(redshift) or redshift <= -1:
            raise ValueError('Response redshift must be finite and greater than -1.')
        self.redshift = redshift
        self.flexible = flexible
        self.renormalize = renormalize
        self._instrument_name = None
        self._interp_cache = None
        self._requested_wave = None
        
        if response_matrix is not None:
            # Direct matrix mode - use as-is
            self.wavelength_grid = _response_grid(wave) if wave is not None else None
            self.response_matrix = _response_matrix(
                response_matrix, len(self.wavelength_grid) if wave is not None else None)
            self._mode = "direct"
            
        elif instrument is not None and wave is not None:
            wave = _response_grid(wave)
            wave_obs = wave * (1 + redshift) if redshift != 0 else wave
            
            # Check if instrument is a string (name) or InstrumentResponse instance
            if isinstance(instrument, str):
                # Load from archive by name
                response = InstrumentResponse.from_instrument(instrument)
                self._instrument_name = instrument
            elif isinstance(instrument, InstrumentResponse):
                # Use the provided instance directly
                response = instrument
                self._instrument_name = None
            else:
                raise TypeError(
                    f"instrument must be str or InstrumentResponse, got {type(instrument).__name__}"
                )
            
            self._requested_wave = wave.copy()
            if flexible and not renormalize:
                # Keep all available source columns that can illuminate requested outputs.
                _response_indices(response.wavelength_grid, wave_obs)
                # Sampling between centers also needs the two bracketing output rows.
                first = max(0, np.searchsorted(response.wavelength_grid, wave_obs[0], side='right') - 1)
                last = min(len(response.wavelength_grid) - 1,
                           np.searchsorted(response.wavelength_grid, wave_obs[-1], side='left'))
                support = response.response_matrix[first:last+1].indices
                start = min(first, support.min()) if support.size else first
                stop = max(last, support.max()) + 1 if support.size else last + 1
                self.response_matrix = response.response_matrix[start:stop, start:stop]
                self.wavelength_grid = response.wavelength_grid[start:stop] / (1 + redshift)
            else:
                cropped = response.crop(wave_obs, renormalize=renormalize)
                self.response_matrix = cropped.response_matrix
                self.wavelength_grid = wave
            self._mode = "instrument"
            
        else:
            raise ValueError(
                "Provide either (instrument, wave) or (response_matrix, wave)."
            )

        super().__init__(
            self.response_matrix,
            x=self.wavelength_grid,
            flexible=self.flexible,
            recipe=self._build_recipe(),
            name=name,
        )

    @property
    def response_matrix(self):
        """The operator matrix; alias of ``matrix`` with no duplicate storage."""
        return self._matrix

    @response_matrix.setter
    def response_matrix(self, value):
        self._matrix = _response_matrix(value)

    @property
    def wavelength_grid(self):
        """The native operator grid; alias of ``x`` with no duplicate storage."""
        return self._x

    @wavelength_grid.setter
    def wavelength_grid(self, value):
        self._x = None if value is None else _response_grid(value)
        self._interp_cache = None

    @property
    def recipe(self):
        return self._build_recipe()

    def _build_recipe(self):
        """Build a reconstruction recipe for serialization.

        The recipe stores enough metadata to reconstruct the operator
        from the instrument archive without saving the full matrix.
        """
        if self._mode == 'instrument' and self._instrument_name:
            return {
                'type': 'instrument',
                'instrument': self._instrument_name,
                'redshift': float(self.redshift),
                'renormalize': bool(self.renormalize),
                'flexible': bool(self.flexible),
            }
        return {'type': 'direct', 'flexible': bool(self.flexible),
                'renormalize': bool(self.renormalize), 'redshift': float(self.redshift)}

    def __repr__(self) -> str:
        shape = self.wavelength_grid.shape if self.wavelength_grid is not None else None
        redshift_str = f", redshift={self.redshift}" if self.redshift != 0 else ""
        inst_str = f", instrument='{self._instrument_name}'" if self._instrument_name else ""
        name_str = f", name='{self.name}'" if self.name != 'rsp' else ""
        return f"<SpectralResponse(wave={shape}{redshift_str}{inst_str}{name_str})>"


# Public exports
__all__ = [
    'InstrumentResponse',
    'SpectralResponse',
    'is_linear_operator_pipe',
    'load_responses_mapping',
    'add_response_to_archive',
    'list_instruments'
]
