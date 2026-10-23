"""Ready-made line sets: shared kinematics, explicit profiles and total fluxes."""
from .component import LineSet
from .profiles import (GaussianProfile, LorentzianProfile, VoigtProfile,
                       _GaussianFluxProfile, _LorentzianFluxProfile, _VoigtFluxProfile)
from . import tables

__all__ = ['GaussianLines', 'LorentzianLines', 'VoigtLines']


class _Lines(LineSet):
    """Common table loading and parameter setup, without numerical profile dispatch."""

    def _initialize(self, profile_class, lines, normalization, flux, amplitude,
                    shared, instfwhm, bounds, name, window):
        if normalization not in ('flux', 'amplitude'):
            raise ValueError("normalization must be 'flux' or 'amplitude'.")
        if ((normalization == 'flux' and amplitude is not None)
                or (normalization == 'amplitude' and flux is not None)):
            raise ValueError('Supply flux= or amplitude= consistently with normalization; no implicit conversion is performed.')
        value = flux if normalization == 'flux' else amplitude
        if value is None:
            value = 1.
        table = tables.load(lines, window=window)
        if normalization == 'flux':
            profile_class = {GaussianProfile: _GaussianFluxProfile,
                             LorentzianProfile: _LorentzianFluxProfile,
                             VoigtProfile: _VoigtFluxProfile}[profile_class]
        profile = profile_class(instfwhm=instfwhm, position_unit=table['position'].unit)
        LineSet.__init__(self, profile, table, shared=shared,
            parameters={str(tag): {normalization: value} for tag in dict.fromkeys(table['name'])},
            scale=normalization, position='position', derivative=profile.derivative, name=name)
        self.component._model_class = type(self)
        self.redshift.fixed = True
        for key, limits in (bounds or {}).items():
            prefix = 'flux_' if normalization == 'flux' else 'amp_'
            names = [n for n in self.param_names if n.startswith(prefix)] if key == normalization else [key]
            for parameter in names:
                self.params[parameter].bounds = limits

    @property
    def normalization(self):
        """Quantity scaled by transition weights: flux or peak amplitude."""
        return self.component._setup['scale']



class GaussianLines(_Lines):
    """Build a Gaussian line set with shared kinematics and template fluxes or peak amplitudes.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Columns ``name``, unit-bearing ``position`` and optional ``weight``.
        Repeated names share one scale parameter. Bare names use the local
        preset directory then packaged defaults; paths are explicit. ECSV records units.
    normalization : {'flux', 'amplitude'}, optional
        Default flux: integral over the full profile in the evaluation coordinate,
        conserved by spectral shifts and local instrumental broadening. Table
        weights multiply this flux, not the peak. Amplitude selects the historical
        peak convention, including division by (1+redshift).
    flux : float, optional
        Initial integrated flux per template (default 1 in flux mode). Parameters
        are named flux_<template>; units are integrated density times coordinate.
    amplitude : float, optional
        Initial peak in amplitude mode (default 1); parameters are amp_<template>.
        Supplying the inactive scale keyword raises; values are not converted.
    offset, fwhm : float, optional
        Shared velocity offset and intrinsic FWHM in km/s.
    redshift : float, optional
        Shared redshift, initially fixed. Flux mode conserves the integral;
        amplitude mode uses amplitude/(1+redshift) for the peak.
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s, constant or tabulated against position
        in the table's spectral unit. Evaluated at the observed center; combined
        in quadrature with the intrinsic Gaussian width.
    bounds : mapping, optional
        Parameter bounds. ``flux`` or ``amplitude`` applies to all active template scales;
        individual parameters use their native names, for example ``amp_hb``.
    name : str, optional
        Component label. Supply a name before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    Weights multiply the selected scale exactly once. Physical measurements
    are accessed through lines(model_or_result), independently of normalization.
    Generic LineSet imposes no normalization.
    """
    def __init__(self, lines, *, normalization='flux', flux=None, amplitude=None, offset=0., fwhm=1000., redshift=0.,
                 instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(GaussianProfile, lines, normalization, flux, amplitude,
            dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name, window)


class LorentzianLines(_Lines):
    """Build a Lorentzian line set with shared kinematics and template fluxes or peak amplitudes.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Named transitions with unit-bearing positions and optional weights;
        bare names search local then packaged presets; paths are explicit.
    normalization : {'flux', 'amplitude'}, optional
        Default flux: integral over the full profile in the evaluation coordinate,
        conserved by spectral shifts and local instrumental broadening. Table
        weights multiply this flux, not the peak. Amplitude selects the historical
        peak convention, including division by (1+redshift).
    flux : float, optional
        Initial integrated flux per template (default 1 in flux mode). Parameters
        are named flux_<template>; units are integrated density times coordinate.
    amplitude : float, optional
        Initial peak in amplitude mode (default 1); parameters are amp_<template>.
        Supplying the inactive scale keyword raises; values are not converted.
    offset, fwhm : float, optional
        Shared offset and intrinsic Lorentzian FWHM, in km/s.
    redshift : float, optional
        Initially fixed. Flux mode preserves the integral when shifting coordinates.
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s; a position-dependent table uses the
        transition coordinate unit. Broadening produces a Voigt with the selected normalization,
        not quadrature addition to the Lorentzian width.
    bounds : mapping, optional
        Parameter bounds; ``flux`` or ``amplitude`` applies to all template scales
        in the corresponding mode.
    name : str, optional
        Component label, required before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    Weights multiply the selected scale exactly once. Physical measurements
    are accessed through lines(model_or_result), independently of normalization.
    Generic LineSet imposes no normalization.
    """
    def __init__(self, lines, *, normalization='flux', flux=None, amplitude=None, offset=0., fwhm=1000., redshift=0.,
                 instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(LorentzianProfile, lines, normalization, flux, amplitude,
            dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name, window)


class VoigtLines(_Lines):
    """Build a Voigt line set with shared Gaussian and Lorentzian widths.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Named transitions with unit-bearing positions and optional weights;
        bare names search local then packaged presets; paths are explicit.
    normalization : {'flux', 'amplitude'}, optional
        Default flux: integral over the full profile in the evaluation coordinate,
        conserved by spectral shifts and local instrumental broadening. Table
        weights multiply this flux, not the peak. Amplitude selects the historical
        peak convention, including division by (1+redshift).
    flux : float, optional
        Initial integrated flux per template (default 1 in flux mode). Parameters
        are named flux_<template>; units are integrated density times coordinate.
    amplitude : float, optional
        Initial peak in amplitude mode (default 1); parameters are amp_<template>.
        Supplying the inactive scale keyword raises; values are not converted.
    offset, fwhm_G, fwhm_L : float, optional
        Shared velocity offset and intrinsic Gaussian/Lorentzian FWHM in km/s.
    redshift : float, optional
        Initially fixed. Flux mode preserves the integral when shifting coordinates.
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s, constant or tabulated in the native
        spectral coordinate. Added in quadrature to the Gaussian part only.
    bounds : mapping, optional
        Parameter bounds; ``flux`` or ``amplitude`` applies to all template scales
        in the corresponding mode.
    name : str, optional
        Component label, required before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    Weights multiply the selected scale exactly once. Physical measurements
    are accessed through lines(model_or_result), independently of normalization.
    Generic LineSet imposes no normalization.
    """
    def __init__(self, lines, *, normalization='flux', flux=None, amplitude=None, offset=0., fwhm_G=1000., fwhm_L=1000.,
                 redshift=0., instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(VoigtProfile, lines, normalization, flux, amplitude,
            dict(offset=offset, fwhm_G=fwhm_G, fwhm_L=fwhm_L, redshift=redshift),
            instfwhm, bounds, name, window)
