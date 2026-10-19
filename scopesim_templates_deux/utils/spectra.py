"""Turn the different ways of naming a spectrum into a ``Spextrum``."""

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
from astropy import units as u
from astropy.constants import c
from synphot import SourceSpectrum, units as su
from synphot.models import BlackBodyNorm1D, Empirical1D
from spextra import Spextrum


__all__ = ["resolve_sed", "emission_line_spectrum", "HII_REGION_LINES"]


# Typical line strengths of a Galactic HII region relative to H-beta:
# Balmer, Paschen and Brackett lines from case B recombination
# (T = 10^4 K, n_e = 10^2 cm^-3, Osterbrock & Ferland 2006); forbidden
# lines are representative values for the Orion nebula.
HII_REGION_LINES = {
    "[OII] 3727": (3727.4, 1.5),
    "H-gamma": (4340.5, 0.47),
    "H-beta": (4861.3, 1.0),
    "[OIII] 4959": (4958.9, 1.0),
    "[OIII] 5007": (5006.8, 3.0),
    "HeI 5876": (5875.6, 0.11),
    "[NII] 6548": (6548.0, 0.13),
    "H-alpha": (6562.8, 2.86),
    "[NII] 6583": (6583.4, 0.4),
    "[SII] 6716": (6716.4, 0.1),
    "[SII] 6731": (6730.8, 0.1),
    "[SIII] 9069": (9068.6, 0.25),
    "[SIII] 9531": (9530.6, 0.6),
    "HeI 10830": (10830.3, 0.3),
    "Pa-beta": (12818.1, 0.163),
    "Pa-alpha": (18751.0, 0.334),
    "Br-gamma": (21655.3, 0.0278),
}


def _blackbody(temperature):
    temperature = u.Quantity(temperature)
    if temperature.unit == u.dimensionless_unscaled:
        temperature = temperature.value * u.K
    if temperature.unit.physical_type != "temperature":
        raise u.UnitTypeError(f"Expected a temperature, got {temperature!r}.")
    temperature = temperature.to_value(u.K, equivalencies=u.temperature())
    return SourceSpectrum(BlackBodyNorm1D, temperature=temperature)


def _file_path(location):
    if location.startswith("file://"):
        return Path(url2pathname(unquote(urlparse(location).path)))
    return Path(location)


def resolve_sed(sed):
    """
    Make a ``Spextrum`` from any of the supported spectrum descriptions.

    Parameters
    ----------
    sed : str, u.Quantity, Spextrum, synphot.SourceSpectrum
        One of:

        - a ``Spextrum`` or ``synphot.SourceSpectrum`` object;
        - a temperature Quantity, e.g. ``5800 * u.K``, for a black body;
        - ``"blackbody:<T>"``, e.g. ``"blackbody:5800"`` or
          ``"blackbody:5800 K"`` (plain numbers are Kelvin);
        - ``"file:<path>"`` or a ``file://`` URI, read with
          ``SourceSpectrum.from_file`` (FITS or ASCII);
        - ``"spex:<name>"`` or just ``"<name>"``, a speXtra database entry
          such as ``"kc96/s0"`` or ``"irtf/Neptune"``.

    Returns
    -------
    Spextrum
        Not rescaled. The black body is only normalised to an arbitrary
        level, so pass an amplitude wherever the spectrum is used.

    Examples
    --------
    >>> from astropy import units as u
    >>> spec = resolve_sed("blackbody:5800")
    >>> spec = resolve_sed(3000 * u.K)
    """
    if isinstance(sed, (Spextrum, SourceSpectrum)):
        return Spextrum(modelclass=sed)
    if isinstance(sed, u.Quantity):
        return Spextrum(modelclass=_blackbody(sed))
    if not isinstance(sed, str):
        raise TypeError("'sed' must be a Spextrum, SourceSpectrum, temperature "
                        f"Quantity or str, found {type(sed)=}.")

    kind, sep, rest = sed.partition(":")
    kind = kind.strip().lower()
    if not sep or kind not in ("blackbody", "file", "spex"):
        return Spextrum(sed)

    rest = rest.strip()
    if kind == "blackbody":
        return Spextrum(modelclass=_blackbody(rest))
    if kind == "file":
        path = _file_path(sed if rest.startswith("//") else rest)
        if not path.exists():
            raise FileNotFoundError(f"Spectrum file not found: {path}")
        return Spextrum(modelclass=SourceSpectrum.from_file(str(path)))
    return Spextrum(rest)


def emission_line_spectrum(lines=None, fwhm=30 * u.km / u.s,
                           continuum=1e-3, wave_min=3000 * u.AA,
                           wave_max=25000 * u.AA):
    """
    Make an emission-line spectrum, by default that of an HII region.

    Parameters
    ----------
    lines : dict, optional
        ``{name: (wavelength [Angstrom], relative flux)}``. Default is
        :data:`HII_REGION_LINES`, which covers the optical and near-IR.
    fwhm : u.Quantity
        Line width, as a velocity (e.g. ``30 * u.km / u.s``) or a
        wavelength. Floats are km/s.
    continuum : float
        Flat continuum (in FLAM) relative to a line flux of 1 (in
        FLAM * Angstrom). It keeps the spectrum non-zero between the lines,
        so it can also be scaled in a filter without strong lines.
    wave_min, wave_max : u.Quantity
        Wavelength range of the spectrum.

    Returns
    -------
    Spextrum
        Not scaled to any absolute flux.

    Examples
    --------
    >>> spec = emission_line_spectrum()
    >>> oiii = emission_line_spectrum({"[OIII]": (5006.8, 1.0)})
    """
    lines = HII_REGION_LINES if lines is None else lines
    if not isinstance(fwhm, u.Quantity):
        fwhm = fwhm * u.km / u.s
    wave_min = u.Quantity(wave_min, u.AA).value
    wave_max = u.Quantity(wave_max, u.AA).value

    centres = np.array([wave for wave, _ in lines.values()], dtype=float)
    fluxes = np.array([flux for _, flux in lines.values()], dtype=float)
    if fwhm.unit.physical_type == "speed":
        sigmas = centres * (fwhm / c).decompose().value / 2.3548
    else:
        sigmas = np.full_like(centres, fwhm.to_value(u.AA) / 2.3548)

    # log-spaced grid with 4 samples per sigma of the narrowest line
    step = np.min(sigmas / centres) / 4
    waves = np.exp(np.arange(np.log(wave_min), np.log(wave_max), step))
    flam = np.full_like(waves, continuum)
    for centre, flux, sigma in zip(centres, fluxes, sigmas):
        near = np.abs(waves - centre) < 8 * sigma
        flam[near] += (flux / (np.sqrt(2 * np.pi) * sigma)
                       * np.exp(-0.5 * ((waves[near] - centre) / sigma) ** 2))

    spec = SourceSpectrum(Empirical1D, points=waves * u.AA,
                          lookup_table=flam * su.FLAM)
    return Spextrum(modelclass=spec)
