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
When sources are redshifted, the data is typically z-corrected to rest
frame.  The instrument LSF is defined in observer frame.
``SpectralResponse`` handles the mapping via the *z* parameter:
``λ_obs = λ_rest × (1 + z)``.

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
>>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave_rest, z=0.1)
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
from scipy.special import erf
from astropy.io import fits
from .matop import LinearOperatorModel, is_linear_operator_pipe


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


# --- InstrumentResponse class ---

class InstrumentResponse:
    """
    Class for building, loading, saving, and cropping instrumental response matrices.
    
    The response matrix is a square matrix where each row represents the redistribution
    of flux from one wavelength bin to all others due to the instrument's LSF.
    Rows are normalized to sum to 1 (flux conservation).
    
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
        self.wavelength_grid = np.asarray(wavelength_grid)
        self.response_matrix = response_matrix

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
        delta_lam = wavelength_grid / R
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
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Response matrix must be square.")
        if matrix.shape[0] != len(wavelength_grid):
            raise ValueError("Matrix dimensions must match wavelength grid length.")
        if not issparse(matrix):
            matrix = csr_matrix(matrix)
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
        return cls(wavelength_grid, matrix)

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
        _R_interp = interp1d(
            lambda_R, R_values, kind=interp_kind, 
            bounds_error=False, fill_value=(R_values[0], R_values[-1])
        )
        R = np.maximum(_R_interp(wavelength_grid), 1.0)
        delta_lam = wavelength_grid / R
        return delta_lam / (2 * np.sqrt(2 * np.log(2)))

    @staticmethod
    def _build_sparse_gaussian_matrix(wavelength_grid, sigmas):
        """Build sparse Gaussian response matrix."""
        N = len(wavelength_grid)
        wstep = np.diff(wavelength_grid)[0]
        data, row_indices, col_indices = [], [], []
        lower_edges = wavelength_grid - wstep / 2
        upper_edges = wavelength_grid + wstep / 2
        
        for i, (lambda_real, sigma) in enumerate(zip(wavelength_grid, sigmas)):
            a = (lower_edges - lambda_real) / (sigma * np.sqrt(2))
            b = (upper_edges - lambda_real) / (sigma * np.sqrt(2))
            integrals = 0.5 * (erf(b) - erf(a))
            row_sum = integrals.sum()
            if row_sum > 1e-9:
                integrals /= row_sum  # normalize to 1
            non_zero = integrals > 1e-10
            data.extend(integrals[non_zero])
            row_indices.extend([i] * np.sum(non_zero))
            col_indices.extend(np.where(non_zero)[0])
            
        return csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    # ---- SAVE & REGISTER ----
    
    def save_fits(self, filename: str, compress: bool = True) -> None:
        """Save the wavelength grid and response matrix to a FITS file.

        Parameters
        ----------
        filename : str
            The output FITS file path.
        compress : bool, optional
            If True, uses FITS tile compression for the matrix. Default is ``True``.
        """
        if self.response_matrix is None:
            raise ValueError("No response matrix to save.")
        primary_hdu = fits.PrimaryHDU()
        wavelength_hdu = fits.ImageHDU(self.wavelength_grid, name='WAVELENGTH')
        dense_matrix = self.response_matrix.toarray()
        if compress:
            matrix_hdu = fits.CompImageHDU(dense_matrix, name='RESPONSE')
        else:
            matrix_hdu = fits.ImageHDU(dense_matrix, name='RESPONSE')
        hdul = fits.HDUList([primary_hdu, wavelength_hdu, matrix_hdu])
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
        
        self.save_fits(archive_filename, compress=compress)
        add_response_to_archive(instrument_name, filename, clobber=clobber, user=True)
        print(f"Saved and registered '{instrument_name}' to {archive_filename}")

    # ---- CROP ----
    
    def crop(self, new_wavelengths: np.ndarray, renormalize: bool = True) -> "InstrumentResponse":
        """Crop the response matrix to match a new subset wavelength grid.

        Parameters
        ----------
        new_wavelengths : numpy.ndarray
            The target wavelength grid to crop to. Must be a subset of the original grid.
        renormalize : bool, optional
            If True, renormalizes the matrix rows to conserve flux. Default is ``True``.

        Returns
        -------
        InstrumentResponse
            A new cropped response matrix.
        """
        cropped_matrix = _crop_response_matrix(
            self.response_matrix, self.wavelength_grid, new_wavelengths, renormalize=renormalize
        )
        return InstrumentResponse(new_wavelengths, cropped_matrix)

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


def _crop_response_matrix(
    matrix: np.ndarray,
    matrix_wave: np.ndarray,
    target_wave: np.ndarray,
    renormalize: bool = True
) -> np.ndarray:
    """
    Crop a response matrix to match the target wavelengths.
    
    Target wavelengths must be a subset of the matrix wavelengths (within tolerance).
    Gaps in the target grid are allowed (e.g., masked spectral regions).
    
    Args:
        matrix: 2D sparse or dense matrix (N x N)
        matrix_wave: wavelengths corresponding to the matrix
        target_wave: desired output wavelengths (can have gaps)
        renormalize: renormalize rows after cropping (default True)
    
    Returns:
        Cropped (and optionally renormalized) sparse matrix
    """
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
    
    # NOTE: We no longer check step size uniformity.
    # Target wavelengths can have gaps (masked regions) - this is valid.
    # The only requirement is that each target wavelength exists in the matrix grid.
    
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
        Rest-frame wavelength grid.  If *z* > 0 the matrix is cropped at
        λ_obs = wave * (1 + z).
    z : float, default 0
        Redshift used to map rest→observer wavelengths.
    response_matrix : sparse matrix, optional
        Supply a pre-built matrix directly (no archive lookup, no cropping).
    renormalize : bool, default True
        Renormalize rows to unity after cropping.
    flexible : bool, default True
        If ``True``, evaluation on a different wavelength grid is handled by
        evaluating the source model on the native response grid and
        interpolating the convolved spectrum back to the requested grid. This
        keeps same-grid fitting on the fast matrix path while allowing subset
        or plotting grids.
    name : str, default ``'rsp'``
        Name of the response operator model.

    Examples
    --------
    >>> # From archive (serializable — full round-trip supported)
    >>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave, z=0.1)
    >>> model = source | rsp

    >>> # Custom name at construction
    >>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave, z=0.1, name='muse')
    >>> model = source | rsp

    >>> # From InstrumentResponse instance (not serializable from recipe)
    >>> ir = InstrumentResponse.from_fixed_fwhm(wave, fwhm=2.5)
    >>> rsp = SpectralResponse(instrument=ir, wave=wave)
    >>> model = source | rsp
    """
    
    def __init__(
        self,
        instrument: str | InstrumentResponse | None = None,
        wave: np.ndarray | None = None,
        z: float = 0,
        response_matrix=None,
        renormalize: bool = True,
        flexible: bool = True,
        name: str = 'rsp',
    ):
        self.z = z
        self.flexible = flexible
        self._instrument_name = None
        self._interp_cache = None
        
        if response_matrix is not None:
            # Direct matrix mode - use as-is
            self.response_matrix = response_matrix
            self.wavelength_grid = np.asarray(wave, dtype=float) if wave is not None else None
            self._mode = "direct"
            
        elif instrument is not None and wave is not None:
            wave = np.asarray(wave, dtype=float)
            wave_obs = wave * (1 + z) if z != 0 else wave
            
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
            
            # Crop to observer wavelengths
            cropped = response.crop(wave_obs, renormalize=renormalize)
            
            self.response_matrix = cropped.response_matrix
            self.wavelength_grid = wave  # store rest-frame grid
            self._mode = "instrument"
            
        else:
            raise ValueError(
                "Provide either (instrument, wave) or (response_matrix, wave)."
            )

        super().__init__(
            self.response_matrix,
            x=self.wavelength_grid,
            recipe=self._build_recipe(),
            name=name,
        )

    def _same_grid(self, x) -> bool:
        if self.wavelength_grid is None:
            return False
        x = np.asarray(x, dtype=float).ravel()
        if x.shape != self.wavelength_grid.shape:
            return False
        return np.array_equal(x, self.wavelength_grid) or np.allclose(
            x,
            self.wavelength_grid,
            rtol=0.0,
            atol=np.finfo(float).eps * max(1.0, np.abs(self.wavelength_grid).max()),
        )

    def _get_interp_cache(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if self.wavelength_grid is None:
            raise ValueError("Flexible response evaluation requires a wavelength grid.")
        if self._interp_cache is not None and np.array_equal(x, self._interp_cache["x"]):
            return self._interp_cache

        native = self.wavelength_grid
        indices = np.searchsorted(native, x)
        inside = (x >= native[0]) & (x <= native[-1])
        left = np.clip(indices - 1, 0, len(native) - 1)
        right = np.clip(indices, 0, len(native) - 1)

        same = inside & (left == right)
        frac = np.zeros_like(x, dtype=float)

        span = native[right] - native[left]
        valid = inside & (~same)
        frac[valid] = (x[valid] - native[left[valid]]) / span[valid]
        frac[same] = 0.0

        cache = {
            "x": x,
            "inside": inside,
            "left": left,
            "right": right,
            "frac": frac,
        }
        self._interp_cache = cache
        return cache

    def _resample_from_native(self, values, x):
        unit = values.unit if hasattr(values, "unit") else None
        arr = values.to_value(unit) if unit is not None else np.asarray(values)
        arr = np.asarray(arr).ravel()

        if self._same_grid(x):
            return values

        cache = self._get_interp_cache(x)
        result = np.zeros(len(cache["x"]), dtype=arr.dtype)
        inside = cache["inside"]
        left = cache["left"][inside]
        right = cache["right"][inside]
        frac = cache["frac"][inside]
        result[inside] = (1.0 - frac) * arr[left] + frac * arr[right]
        return result * unit if unit is not None else result

    def _native_response(self, values):
        unit = values.unit if hasattr(values, "unit") else None
        arr = values.to_value(unit) if unit is not None else np.asarray(values)
        arr = np.asarray(arr).ravel()
        result = np.asarray(self.response_matrix.dot(arr)).ravel()
        return result * unit if unit is not None else result

    def _prism_pipe_evaluate(
        self,
        leftval,
        left_inputs,
        right_params,
        left_model=None,
        left_params=None,
        **kwargs,
    ):
        x = np.asarray(left_inputs[0], dtype=float).ravel()
        if self.wavelength_grid is None or self._same_grid(x):
            return super()._prism_pipe_evaluate(
                leftval,
                left_inputs,
                right_params,
                left_model=left_model,
                left_params=left_params,
                **kwargs,
            )
        if not self.flexible:
            raise ValueError(
                f"LinearOperatorModel expects {self.response_matrix.shape[0]} samples, got {len(x)}."
            )
        if left_model is None:
            raise ValueError("Flexible response evaluation requires access to the source model.")

        native_x = self.wavelength_grid
        if hasattr(left_inputs[0], "unit"):
            native_x = native_x * left_inputs[0].unit
        if left_params is None:
            native_flux = left_model(native_x, **kwargs)
        else:
            native_flux = left_model.evaluate(native_x, *left_params)
        native_result = self._native_response(native_flux)
        return self._resample_from_native(native_result, x)

    def _prism_pipe_fit_deriv(
        self,
        left_deriv,
        left_inputs,
        left_params,
        right_params,
        left_model,
    ):
        x = np.asarray(left_inputs[0], dtype=float).ravel()
        if self.wavelength_grid is None or self._same_grid(x):
            return super()._prism_pipe_fit_deriv(
                left_deriv,
                left_inputs,
                left_params,
                right_params,
                left_model,
            )
        if not self.flexible:
            raise ValueError(
                f"LinearOperatorModel expects {self.response_matrix.shape[0]} samples, got {len(x)}."
            )

        native_x = self.wavelength_grid
        if hasattr(left_inputs[0], "unit"):
            native_x = native_x * left_inputs[0].unit
        native_deriv = left_model.fit_deriv(native_x, *left_params)
        if native_deriv is None:
            return None
        derivs = np.asanyarray(native_deriv)
        if not left_model.col_fit_deriv:
            derivs = np.moveaxis(derivs, -1, 0)
        derivs = derivs.reshape((derivs.shape[0], -1))

        return np.asarray(
            [
                np.asarray(self._resample_from_native(self._native_response(dparam), x)).ravel()
                for dparam in derivs
            ]
        )

    def _build_recipe(self):
        """Build a reconstruction recipe for serialization.

        The recipe stores enough metadata to reconstruct the operator
        from the instrument archive without saving the full matrix.
        """
        if self._mode == 'instrument' and self._instrument_name:
            return {
                'type': 'instrument',
                'instrument': self._instrument_name,
                'z': float(self.z),
            }
        return {'type': 'direct'}

    def __repr__(self) -> str:
        shape = self.wavelength_grid.shape if self.wavelength_grid is not None else None
        z_str = f", z={self.z}" if self.z != 0 else ""
        inst_str = f", instrument='{self._instrument_name}'" if self._instrument_name else ""
        name_str = f", name='{self.name}'" if self.name != 'rsp' else ""
        return f"<SpectralResponse(wave={shape}{z_str}{inst_str}{name_str})>"


# Public exports
__all__ = [
    'InstrumentResponse',
    'SpectralResponse',
    'is_linear_operator_pipe',
    'load_responses_mapping',
    'add_response_to_archive',
    'list_instruments'
]
