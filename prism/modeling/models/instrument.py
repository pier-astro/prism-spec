"""
Instrumental Response Module

Provides classes for building, loading, and applying spectral instrumental response 
matrices (Line Spread Functions) to Astropy models.

Classes:
--------
- InstrumentResponse: Build/load/save response matrices
- SpectralResponse: Wrap models with instrumental convolution
- ResponseModel: Instrumental response wrapper (inherits from ConvolvedModel)

The key concept is that instruments observe spectra convolved with their LSF. 
This module allows proper modeling of observed spectra by applying the instrumental
response to intrinsic source models.

ResponseModel and ConvolvedModel:
---------------------------------
The base ConvolvedModel class (in convolved.py) handles general linear operators
(useful for velocity broadening, smoothing, etc.). ResponseModel inherits from it
and provides instrument-specific functionality with a default name of 'rsp'.

Astropy's pipe operator `|` does not propagate analytic derivatives (fit_deriv),
forcing numeric jacobian estimation which is slow for many parameters.
ResponseModel/ConvolvedModel solve this by applying the chain rule:

    For h(x) = R @ f(x, θ), the Jacobian is: J_h = R @ J_f

This preserves analytic derivatives since R is a fixed matrix.

Files modified to handle ConvolvedModel unwrapping:
- fantasylab/models/components.py: get_components(), _get_source_model()
- fantasylab/models/flux.py: extract_fluxes(), flux_from_samples()  
- fantasylab/display/model.py: show(), _build_model_expression()

See myresources/convolved_model.md for full documentation.

When working with redshifted sources:
- Data is typically z-corrected to rest frame for modeling convenience
- The LSF matrix is defined in observer frame by the instrument
- SpectralResponse handles the wavelength mapping via the `z` parameter:
  it crops the instrument matrix at λ_obs = λ_rest * (1+z)

Instrument Storage:
- Default instruments are stored in the package directory (responses/)
- Custom instruments are stored in ~/.fantasylab/instruments/
- User instruments override package defaults with the same name

Example:
--------
>>> from fantasylab.models.instrument import SpectralResponse, InstrumentResponse
>>> from astropy.modeling.models import Gaussian1D
>>> 
>>> # Build response from instrument archive
>>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave_rest, z=0.1)
>>> convolved_model = rsp(my_model)
>>> 
>>> # Or pass an InstrumentResponse instance directly
>>> response = InstrumentResponse.from_fixed_fwhm(wave, fwhm=2.5)
>>> rsp = SpectralResponse(instrument=response, wave=wave)
>>> 
>>> # Or build custom response with direct matrix
>>> rsp = SpectralResponse(response_matrix=response.response_matrix, wave=wave)
"""
import numpy as np
import os
import warnings
import yaml

from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.sparse import csr_matrix, issparse
from scipy.special import erf
from astropy.io import fits
from astropy.modeling import Fittable1DModel, Parameter


# --- Path utilities ---

def _get_package_response_dir() -> str:
    """Return the path to the package responses directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources", "responses")


def _get_user_response_dir() -> str:
    """Return the path to the user's local response directory (~/.fantasylab/instruments/)."""
    return os.path.join(os.path.expanduser("~"), ".fantasylab", "instruments")


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
    
    User instruments (~/.fantasylab/instruments/) override package defaults.
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
        """Build from variable resolution R(λ)."""
        sigmas = cls._compute_sigmas_variable(wavelength_grid, lambda_R, R_values, interp_kind)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_fwhm(cls, wavelength_grid: np.ndarray, fwhm: float) -> "InstrumentResponse":
        """Build from fixed FWHM (in same units as wavelength_grid)."""
        if fwhm <= 0:
            raise ValueError("FWHM must be positive.")
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_resolution(cls, wavelength_grid: np.ndarray, R: float) -> "InstrumentResponse":
        """Build from fixed resolution R = λ/Δλ."""
        if R <= 0:
            raise ValueError("Resolution R must be positive.")
        delta_lam = wavelength_grid / R
        sigmas = delta_lam / (2 * np.sqrt(2 * np.log(2)))
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_sigma(cls, wavelength_grid: np.ndarray, sigma: float) -> "InstrumentResponse":
        """Build from fixed sigma (in same units as wavelength_grid)."""
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_array(cls, wavelength_grid: np.ndarray, matrix: np.ndarray) -> "InstrumentResponse":
        """Create from a dense or sparse matrix."""
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
        """Load from FITS file."""
        with fits.open(filename) as hdul:
            wavelength_grid = hdul[1].data
            dense_matrix = hdul[2].data.astype(np.float64)
            matrix = csr_matrix(dense_matrix)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_instrument(cls, instrument: str) -> "InstrumentResponse":
        """
        Load from instrument name in the archive.
        
        Searches user directory (~/.fantasylab/instruments/) first,
        then falls back to package defaults.
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
        """Save wavelength grid and matrix to FITS file."""
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
        """
        Save the response matrix to user directory and register it.
        
        The file is saved to ~/.fantasylab/instruments/<instrument_name>.fits
        and registered in the user's instrument archive.
        
        Parameters
        ----------
        instrument_name : str
            Name to register the instrument under
        compress : bool
            Use FITS compression (default True)
        clobber : bool
            Overwrite existing file/entry (default False)
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
        """Crop the response matrix to match a new wavelength grid."""
        cropped_matrix = _crop_response_matrix(
            self.response_matrix, self.wavelength_grid, new_wavelengths, renormalize=renormalize
        )
        return InstrumentResponse(new_wavelengths, cropped_matrix)

    def __repr__(self) -> str:
        shape = self.wavelength_grid.shape if self.wavelength_grid is not None else None
        wmin = f"{self.wavelength_grid.min():.1f}" if self.wavelength_grid is not None else "?"
        wmax = f"{self.wavelength_grid.max():.1f}" if self.wavelength_grid is not None else "?"
        return f"<InstrumentResponse({shape}, λ=[{wmin}, {wmax}])>"


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
    # Find indices matching target wavelengths (with tolerance)
    indices = []
    atol = np.abs(np.diff(matrix_wave)).min() * 0.1 if len(matrix_wave) > 1 else 0.01
    
    for tw in target_wave:
        matches = np.where(np.isclose(tw, matrix_wave, atol=atol, rtol=1e-6))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Wavelength {tw:.3f} not found in matrix grid "
                f"[{matrix_wave.min():.1f}, {matrix_wave.max():.1f}]."
            )
        indices.append(matches[0])
    indices = np.array(indices)
    
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


# --- ResponseOperator Model ---

class ResponseOperator(Fittable1DModel):
    """
    Applies response matrix multiplication to input flux with grid flexibility.
    
    Used with the pipe operator: source_model | ResponseOperator(matrix, wave)
    
    The result is a proper CompoundModel that fitters handle correctly.
    This model has NO fittable parameters - it only applies the matrix.
    
    Grid Handling
    -------------
    - If `wave` is provided, the operator supports evaluation on arbitrary grids
      through interpolation of the convolved result.
    - The interpolated matrix is cached for efficiency during fitting.
    - Outside the wavelength bounds, the result is zero (instrument is blind).
    - If `wave` is None, evaluation is only allowed on matching-length arrays.
    
    Derivative Handling
    -------------------
    The `fit_deriv` method returns an empty list since this operator has no
    fittable parameters. For pipe compositions, derivatives from the source
    model are transformed through the response matrix via chain rule.
    
    Parameters
    ----------
    response_matrix : sparse matrix or array
        The response matrix (n_out × n_in).
    wave : array-like, optional
        Wavelength grid the matrix was built for. Required for grid interpolation.
    name : str, optional
        Model name for display.
    
    Example
    -------
    >>> wave = np.arange(4500, 5500, 1.0)
    >>> matrix = InstrumentResponse.from_fixed_resolution(wave, R=2000).response_matrix
    >>> rsp = ResponseOperator(matrix, wave=wave, name='response')
    >>> convolved = (continuum + emission_line) | rsp
    >>> 
    >>> # Evaluate on original grid (fast, no interpolation)
    >>> flux = convolved(wave)
    >>> 
    >>> # Evaluate on different grid (uses interpolation, cached)
    >>> flux_subset = convolved(wave[100:200])
    """
    n_inputs = 1
    n_outputs = 1
    
    # No fittable parameters
    
    def __init__(self, response_matrix, wave=None, name=None, **kwargs):
        self._response_matrix = response_matrix
        self._wave = np.asarray(wave) if wave is not None else None
        self._n_wave = response_matrix.shape[0]
        
        # Cache for interpolated results
        self._cache_key = None
        self._cached_interpolator = None
        
        super().__init__(name=name, **kwargs)
    
    @property
    def wave(self):
        """Wavelength grid the matrix was built for."""
        return self._wave
    
    @staticmethod  
    def fit_deriv(flux):
        """
        Return derivatives with respect to parameters.
        
        Since ResponseOperator has no fittable parameters, this returns
        an empty list. This is mathematically correct and allows proper
        derivative chain rule handling in compound models.
        """
        return []
    
    def evaluate(self, flux):
        """
        Apply response matrix to input flux.
        
        If input length matches matrix dimensions, direct multiplication is used.
        Otherwise, if wavelength grid is available, interpolation is performed.
        """
        flux = np.atleast_1d(flux)
        
        # Fast path: input matches matrix dimensions
        if len(flux) == self._n_wave:
            return np.asarray(self._response_matrix.dot(flux)).ravel()
        
        # Need interpolation - requires wavelength grid
        if self._wave is None:
            raise ValueError(
                f"Dimension mismatch: input has {len(flux)} elements but matrix "
                f"expects {self._n_wave}. Provide 'wave' parameter to ResponseOperator "
                "to enable grid interpolation."
            )
        
        # This shouldn't happen in pipe context since we receive flux, not wave
        # But handle gracefully
        raise ValueError(
            f"Dimension mismatch: input flux has {len(flux)} elements but matrix "
            f"expects {self._n_wave}. In pipe context, ensure the source model "
            "returns the expected number of flux values."
        )
    
    def __repr__(self):
        shape = self._response_matrix.shape
        grid_info = f", wave={len(self._wave)}" if self._wave is not None else ""
        return f"<ResponseOperator({shape[0]}x{shape[1]}{grid_info})>"


# --- ResponseModel ---

from .convolved import ConvolvedModel


class ResponseModel(ConvolvedModel):
    """
    Instrumental response model wrapper with analytic derivative propagation.
    
    This is a specialized ConvolvedModel for instrumental response applications.
    It wraps a source model with a response matrix (LSF) and provides:
    
    1. Proper model behavior (via delegation to source)
    2. Analytic derivatives via chain rule: d(R @ f)/d(params) = R @ df/d(params)
    3. Grid flexibility through interpolation (with caching)
    4. Default name 'rsp' for display purposes
    
    Parameters
    ----------
    source_model : Model
        The underlying model to convolve
    response_matrix : sparse matrix
        The instrumental response matrix (n_out × n_in)
    wave : array-like
        Wavelength grid for interpolation support
    name : str, optional
        Name for display (default: 'rsp')
        
    Attributes
    ----------
    source_model : Model
        The underlying source model (read-only access via property)
    response_matrix : sparse matrix
        The instrumental response matrix (alias for operator_matrix)
        
    Example
    -------
    >>> from fantasylab.models.instrument import ResponseModel, InstrumentResponse
    >>> 
    >>> # Build response matrix
    >>> rsp = InstrumentResponse.from_fixed_resolution(wave, R=2000)
    >>> 
    >>> # Create response model
    >>> model = continuum + emission_lines
    >>> rsp_model = ResponseModel(model, rsp.response_matrix, wave=wave)
    >>> 
    >>> # Access source
    >>> rsp_model.source_model  # Returns original compound model
    >>> 
    >>> # Evaluate (applies response)
    >>> flux = rsp_model(wave)
    """
    
    def __init__(self, source_model, response_matrix, wave=None, name=None):
        # Default name for response models is 'rsp'
        super().__init__(
            source_model=source_model,
            operator_matrix=response_matrix,
            wave=wave,
            name=name or 'rsp'
        )
    
    @property
    def response_matrix(self):
        """Alias for operator_matrix (instrument-specific terminology)."""
        return self._operator_matrix
    
    def copy(self):
        """Return a copy of this model."""
        return ResponseModel(
            self._source.copy(),
            self._operator_matrix,
            wave=self._wave.copy() if self._wave is not None else None,
            name=self._name
        )
    
    def __repr__(self):
        return f"<ResponseModel[{self._name}]({self._source!r})>"


# --- SpectralResponse class ---

class SpectralResponse:
    """
    Callable wrapper for applying instrumental response to Astropy models.
    
    Construction modes:
    1. From instrument name: SpectralResponse(instrument='MUSE-WFM', wave=wave_rest, z=0.1)
    2. From InstrumentResponse: SpectralResponse(instrument=response_obj, wave=wave, z=0.1)
    3. From direct matrix: SpectralResponse(response_matrix=matrix, wave=wave)
    
    Parameters
    ----------
    instrument : str or InstrumentResponse, optional
        Instrument name from the archive (e.g., 'MUSE-WFM') or an InstrumentResponse instance
    wave : array-like
        Wavelength grid (rest frame if z>0, observer frame if z=0)
    z : float, default=0
        Redshift. When z>0, the instrument matrix is cropped at λ_obs = wave*(1+z)
    response_matrix : sparse matrix, optional
        Direct matrix input (bypasses instrument loading)
    renormalize : bool, default=True
        Renormalize rows after cropping
    flexible : bool, default=True
        Allow evaluation on arbitrary wavelength grids
        
    Usage
    -----
    >>> # From instrument name
    >>> rsp = SpectralResponse(instrument='MUSE-WFM', wave=wave_rest, z=0.1)
    >>> 
    >>> # From InstrumentResponse instance
    >>> response = InstrumentResponse.from_fixed_resolution(wave_obs, R=2000)
    >>> rsp = SpectralResponse(instrument=response, wave=wave_rest, z=0.1)
    >>> 
    >>> # From direct matrix (no cropping, assumes matrix already matches wave)
    >>> rsp = SpectralResponse(response_matrix=matrix, wave=wave)
    >>> 
    >>> convolved_model = rsp(source_model)
    >>> flux = convolved_model(wave_rest)
    """
    
    def __init__(
        self,
        instrument: str | InstrumentResponse | None = None,
        wave: np.ndarray | None = None,
        z: float = 0,
        response_matrix=None,
        renormalize: bool = True,
        flexible: bool = True
    ):
        self.z = z
        self.flexible = flexible
        self.interpolator = None
        self._instrument_name = None
        
        if response_matrix is not None:
            # Direct matrix mode - use as-is
            self.response_matrix = response_matrix
            self.wavelength_grid = np.asarray(wave) if wave is not None else None
            self._mode = "direct"
            
        elif instrument is not None and wave is not None:
            wave = np.asarray(wave)
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
        
        if self.flexible and self.wavelength_grid is not None:
            self._build_interpolator()

    def _build_interpolator(self) -> None:
        """Build 2D interpolator for flexible grid evaluation."""
        dense = self.response_matrix.toarray() if issparse(self.response_matrix) else self.response_matrix
        self.interpolator = RectBivariateSpline(
            self.wavelength_grid, self.wavelength_grid, dense, kx=1, ky=1
        )

    def __call__(self, source_model, use_analytic_deriv=True, name=None):
        """
        Wrap a source model with instrumental convolution.
        
        Parameters
        ----------
        source_model : Model
            Any Astropy model (simple or compound)
        use_analytic_deriv : bool, default=True
            If True, use ResponseModel which properly propagates
            analytic derivatives through the response matrix (faster fitting).
            If False, use pipe operator (|) which falls back to numeric jacobian.
        name : str, optional
            Name for the response model (default: 'rsp')
            
        Returns
        -------
        Model
            The convolved model (ResponseModel or CompoundModel)
            
        Notes
        -----
        The `use_analytic_deriv=True` option is recommended for fitting as it
        avoids the overhead of numeric jacobian estimation, especially with
        many free parameters.
        """
        if use_analytic_deriv:
            return ResponseModel(
                source_model, 
                self.response_matrix,
                wave=self.wavelength_grid,
                name=name or 'rsp'
            )
        else:
            # Fall back to pipe operator (numeric jacobian)
            rsp_op = ResponseOperator(self.response_matrix, wave=self.wavelength_grid, name='response')
            return source_model | rsp_op

    def __repr__(self) -> str:
        shape = self.wavelength_grid.shape if self.wavelength_grid is not None else None
        z_str = f", z={self.z}" if self.z != 0 else ""
        inst_str = f", instrument='{self._instrument_name}'" if self._instrument_name else ""
        return f"<SpectralResponse(wave={shape}{z_str}{inst_str})>"


# Public exports
__all__ = [
    'InstrumentResponse',
    'SpectralResponse',
    'ResponseOperator',
    'ResponseModel',
    'ConvolvedModel',
    'load_responses_mapping',
    'add_response_to_archive',
    'list_instruments'
]
