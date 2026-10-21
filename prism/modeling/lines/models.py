"""Ready-to-use native line sets; profiles own evaluation and derivatives."""
from pathlib import Path
from .component import LineSet
from .profiles import GaussianProfile, LorentzianProfile, VoigtProfile
from . import tables

__all__ = ['GaussianLines', 'LorentzianLines', 'VoigtLines']


def _model(profile_class, lines, amplitude, shared, instfwhm, bounds, name):
    if isinstance(lines, (str, Path)):
        path = Path(lines)
        if not path.is_absolute():
            path = Path(tables.linetable_path)/path
        lines = tables.read_linetable(path)
    table = tables.normalize_linetable(lines)
    profile = profile_class(instfwhm=instfwhm, position_unit=table['position'].unit)
    model = LineSet(profile, table, shared=shared,
                    parameters={str(tag): {'amplitude': amplitude} for tag in dict.fromkeys(table['name'])},
                    position='position', derivative=profile.derivative, name=name)
    model.redshift.fixed = True
    for key, limits in (bounds or {}).items():
        names = [n for n in model.param_names if n.startswith('amp_')] if key == 'amplitude' else [key]
        for parameter in names:
            getattr(model, parameter).bounds = limits
    return model


def GaussianLines(lines, *, amplitude=1., offset=0., fwhm=1000., redshift=0.,
                  instfwhm=0., bounds=None, name=None):
    """Build a Gaussian line set with shared kinematics and template amplitudes.

    Parameters
    ----------
    lines : table, mapping or path
        Columns ``name``, unit-bearing ``position`` and optional ``weight``.
        Repeated names share one amplitude. Relative paths use the current
        line-table directory; ECSV records units explicitly.
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

    Returns
    -------
    LineSet
        Native component with analytic derivatives. Parameter attributes can be
        bounded, fixed or linked normally; no Astropy line subclass is created.
    """
    return _model(GaussianProfile, lines, amplitude,
                  dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name)


def LorentzianLines(lines, *, amplitude=1., offset=0., fwhm=1000., redshift=0.,
                    instfwhm=0., bounds=None, name=None):
    """Build a Lorentzian line set with shared kinematics and template amplitudes.

    Parameters
    ----------
    lines : table, mapping or path
        Named transitions with unit-bearing positions and optional weights;
        relative paths use the current line-table directory.
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

    Returns
    -------
    LineSet
        Native component using LorentzianProfile and its analytic derivatives.
    """
    return _model(LorentzianProfile, lines, amplitude,
                  dict(offset=offset, fwhm=fwhm, redshift=redshift), instfwhm, bounds, name)


def VoigtLines(lines, *, amplitude=1., offset=0., fwhm_G=1000., fwhm_L=1000.,
               redshift=0., instfwhm=0., bounds=None, name=None):
    """Build a Voigt line set with shared Gaussian and Lorentzian widths.

    Parameters
    ----------
    lines : table, mapping or path
        Named transitions with unit-bearing positions and optional weights;
        relative paths use the current line-table directory.
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

    Returns
    -------
    LineSet
        Native component using VoigtProfile and its analytic derivatives.
    """
    return _model(VoigtProfile, lines, amplitude,
                  dict(offset=offset, fwhm_G=fwhm_G, fwhm_L=fwhm_L, redshift=redshift),
                  instfwhm, bounds, name)
