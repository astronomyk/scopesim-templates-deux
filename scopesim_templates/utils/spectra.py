"""Turn the different ways of naming a spectrum into a ``Spextrum``."""

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from astropy import units as u
from synphot import SourceSpectrum
from synphot.models import BlackBodyNorm1D
from spextra import Spextrum


__all__ = ["resolve_sed"]


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
