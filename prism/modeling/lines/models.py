"""Ready-made line sets: shared kinematics, explicit profiles and total fluxes."""
from astropy.table import QTable
from .component import LineSet
from .profiles import GaussianProfile, LorentzianProfile, VoigtProfile
from . import tables

__all__ = ['GaussianLines', 'LorentzianLines', 'VoigtLines']


class _Lines(LineSet):
    """Common table loading and parameter setup, without numerical profile dispatch."""

    def _initialize(self, profile_class, lines, amplitude, shared, instfwhm, bounds, name, window):
        table = tables.load(lines, window=window)
        profile = profile_class(instfwhm=instfwhm, position_unit=table['position'].unit)
        LineSet.__init__(self, profile, table, shared=shared,
            parameters={str(tag): {'amplitude': amplitude} for tag in dict.fromkeys(table['name'])},
            position='position', derivative=profile.derivative, name=name)
        self.component._model_class = type(self)
        self.redshift.fixed = True
        for key, limits in (bounds or {}).items():
            names = [n for n in self.param_names if n.startswith('amp_')] if key == 'amplitude' else [key]
            for parameter in names:
                self.params[parameter].bounds = limits

    def flux(self):
        """Return total intrinsic flux per transition, before any response operator.

        The table contains name, position, weight and flux. Weights multiply the
        same peak parameter as evaluation, exactly once. Values use the profile's
        complete analytic integral, including profile-local instrumental widths;
        no covariance or uncertainty is inferred from a standalone model.
        """
        from ..fitting.utils import apply_tied
        apply_tied(getattr(self, '_evaluation_parent', self))
        leaf = self.component
        names = [key for key, _ in leaf._setup['parameters']]
        rows = []
        for name, (positions, weights, slots) in zip(names, leaf._groups):
            for position, weight in zip(positions, weights):
                arguments = self.parameters[slots]
                arguments[leaf._position_slot] = position
                arguments[leaf._scale_slot] *= weight
                rows.append((name, position, weight, leaf._profile._integral(*arguments)))
        table = QTable(rows=rows, names=('name', 'position', 'weight', 'flux'))
        table['position'] *= leaf._profile.position_unit
        scale = leaf.param_names[leaf._groups[0][2][leaf._scale_slot]]
        if getattr(leaf, scale).unit is not None:
            table['flux'] *= getattr(leaf, scale).unit*leaf._profile.position_unit
        return table


class GaussianLines(_Lines):
    """Build a Gaussian line set with shared kinematics and template amplitudes.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Columns ``name``, unit-bearing ``position`` and optional ``weight``.
        Repeated names share one amplitude. Bare names use the local
        preset directory then packaged defaults; paths are explicit. ECSV records units.
    amplitude : float, optional
        Initial peak amplitude for each template; weights multiply this value.
    offset, fwhm : float, optional
        Shared velocity offset and intrinsic FWHM in km/s.
    redshift : float, optional
        Shared redshift, initially fixed. Peak scaling is amplitude/(1+redshift).
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s, constant or tabulated against position
        in the table's spectral unit. Evaluated at the observed center; combined
        in quadrature with the intrinsic Gaussian width.
    bounds : mapping, optional
        Parameter bounds. ``amplitude`` applies to all template amplitudes;
        individual parameters use their native names, for example ``amp_hb``.
    name : str, optional
        Component label. Supply a name before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    This thin LineSet subclass provides total analytic fluxes via flux(). Weights
    multiply template peaks, and instrumental broadening follows the profile.
    """
    def __init__(self, lines, *, amplitude=1., offset=0., fwhm=1000., redshift=0.,
                 instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(GaussianProfile, lines, amplitude,
            dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name, window)


class LorentzianLines(_Lines):
    """Build a Lorentzian line set with shared kinematics and template amplitudes.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Named transitions with unit-bearing positions and optional weights;
        bare names search local then packaged presets; paths are explicit.
    amplitude : float, optional
        Initial template peak, multiplied by transition weights.
    offset, fwhm : float, optional
        Shared offset and intrinsic Lorentzian FWHM, in km/s.
    redshift : float, optional
        Initially fixed; peaks scale as amplitude/(1+redshift).
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s; a position-dependent table uses the
        transition coordinate unit. Broadening produces a peak-normalized Voigt,
        not quadrature addition to the Lorentzian width.
    bounds : mapping, optional
        Parameter bounds; ``amplitude`` applies to all template amplitudes.
    name : str, optional
        Component label, required before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    This thin LineSet subclass provides total analytic fluxes via flux(). Weights
    multiply template peaks, and instrumental broadening follows the profile.
    """
    def __init__(self, lines, *, amplitude=1., offset=0., fwhm=1000., redshift=0.,
                 instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(LorentzianProfile, lines, amplitude,
            dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name, window)


class VoigtLines(_Lines):
    """Build a Voigt line set with shared Gaussian and Lorentzian widths.

    Parameters
    ----------
    lines : table, mapping, path or list of paths
        Named transitions with unit-bearing positions and optional weights;
        bare names search local then packaged presets; paths are explicit.
    amplitude : float, optional
        Initial template peak, multiplied by transition weights.
    offset, fwhm_G, fwhm_L : float, optional
        Shared velocity offset and intrinsic Gaussian/Lorentzian FWHM in km/s.
    redshift : float, optional
        Initially fixed; peaks scale as amplitude/(1+redshift).
    instfwhm : float, velocity Quantity or two-column array, optional
        Gaussian instrumental FWHM in km/s, constant or tabulated in the native
        spectral coordinate. Added in quadrature to the Gaussian part only.
    bounds : mapping, optional
        Parameter bounds; ``amplitude`` applies to all template amplitudes.
    name : str, optional
        Component label, required before saving.
    window : pair, optional
        Inclusive rest-position limits in the table's spectral unit, before offset
        and redshift; distinct from a measurement integration window.

    Notes
    -----
    This thin LineSet subclass provides total analytic fluxes via flux(). Weights
    multiply template peaks, and instrumental broadening follows the profile.
    """
    def __init__(self, lines, *, amplitude=1., offset=0., fwhm_G=1000., fwhm_L=1000.,
                 redshift=0., instfwhm=0., bounds=None, name=None, window=None):
        self._initialize(VoigtProfile, lines, amplitude,
            dict(offset=offset, fwhm_G=fwhm_G, fwhm_L=fwhm_L, redshift=redshift),
            instfwhm, bounds, name, window)
