# -*- coding: utf-8 -*-
"""
Binary stars and stars with planets.

Coming from ``scopesim-targets``: ``Binary`` is :func:`binary`, and
``PlanetarySystem`` / ``Exoplanet`` are :func:`planetary_system` and
:func:`exoplanet`. Here ``contrast`` is always the flux ratio
companion / primary (e.g. ``1e-4``), whereas ``scopesim-targets`` divides the
primary flux by the contrast of a planet.
"""

import numpy as np
from astropy import units as u
from astropy.table import Table, vstack

from ..rc import Source
from ..utils.general_utils import RA0, DEC0, add_function_call_str
from ..utils.photometry import as_amplitude, is_magnitude, scale_amplitude
from ..utils.positions import offset_to_xy
from ..utils.spectra import resolve_sed
from .stellar import stars


__all__ = ["binary",
           "planetary_system",
           "exoplanet",
           ]

_PLANET_KEYS = {"separation", "position_angle", "contrast", "sed", "name",
                "distance"}


@add_function_call_str
def binary(filter_name, amplitude, spec_types=("A0V", "A0V"), separation=1,
           position_angle=0, contrast=None, amplitude_secondary=None,
           distance=None, x=0, y=0, library="pyckles",
           ra=RA0, dec=DEC0) -> Source:
    """
    Create a binary star.

    The primary sits at (``x``, ``y``) and the secondary at ``separation``
    and ``position_angle`` from it.

    Parameters
    ----------
    filter_name : str
        Filter in which the amplitudes are given. Common names ("V", "Ks")
        or SVO identifiers ("Paranal/HAWKI.Ks").
    amplitude : float, u.Quantity
        Brightness of the primary. Bare numbers are Vega magnitudes; also
        ``u.ABmag``, ``u.STmag`` or a flux density such as ``u.Jy``.
    spec_types : str or pair of str
        Spectral types of the primary and secondary, e.g. ``("G2V", "M3V")``.
        A single string is used for both.
    separation : float, u.Quantity
        Angular separation (floats are arcsec), or a physical one such as
        ``50 * u.AU``, which needs ``distance``.
    position_angle : float, u.Quantity
        Of the secondary, from North (+y) through East (+x). Floats are
        degrees.
    contrast : float, optional
        Flux ratio secondary / primary in ``filter_name``, e.g. 0.1 for a
        secondary 2.5 mag fainter. Give either this or
        ``amplitude_secondary``. If neither is given, both stars are equally
        bright.
    amplitude_secondary : float, u.Quantity, optional
        Brightness of the secondary, in the same kind of unit as
        ``amplitude``.
    distance : float, u.Quantity, optional
        Only needed for a physical ``separation``. Floats are parsec.
    x, y : float, u.Quantity
        Position of the primary, in arcsec from the field centre.
    library : str
        Spectral library, see :func:`~scopesim_templates_deux.stellar.stars`.
    ra, dec : float, str
        Field centre, stored in the meta data only.

    Returns
    -------
    src : ``scopesim.Source``
        A table source with two rows; row 0 is the primary.

    Examples
    --------
    A G2V + M3V binary at 10 pc, separated by 50 AU towards the East::

        >>> from astropy import units as u
        >>> from scopesim_templates_deux.stellar import binary
        >>> src = binary("Ks", 8, spec_types=("G2V", "M3V"),
        ...              separation=50 * u.AU, position_angle=90,
        ...              contrast=0.05, distance=10 * u.pc)
    """
    if contrast is not None and amplitude_secondary is not None:
        raise ValueError("Give either contrast or amplitude_secondary, "
                         "not both.")
    if isinstance(spec_types, str):
        spec_types = (spec_types, spec_types)
    if len(spec_types) != 2:
        raise ValueError(f"Need two spectral types, got {spec_types!r}.")

    amplitude = as_amplitude(amplitude)
    if amplitude_secondary is None:
        amplitude_secondary = scale_amplitude(
            amplitude, 1 if contrast is None else contrast)
    amplitude_secondary = as_amplitude(amplitude_secondary)
    amplitudes = u.Quantity([amplitude, amplitude_secondary.to(amplitude.unit)])

    dx, dy = offset_to_xy(separation, position_angle, distance)
    x <<= u.arcsec
    y <<= u.arcsec
    src = stars(filter_name, amplitudes, list(spec_types),
                x=u.Quantity([x, x + dx]), y=u.Quantity([y, y + dy]),
                library=library, ra=ra, dec=dec)

    weights = src.fields[0].field["weight"]
    src.meta.update({"separation": np.hypot(dx, dy),
                     "position_angle": u.Quantity(position_angle, u.deg),
                     "contrast": weights[1] / weights[0]})
    return src


def _planet_rows(planets, primary, zero, filter_name, distance):
    """Positions, weights, spectra and names of the planets."""
    rows = []
    spectra = {}
    for planet in planets:
        unknown = set(planet) - _PLANET_KEYS
        if unknown:
            raise ValueError(f"Unknown planet keys {sorted(unknown)}; "
                             f"allowed are {sorted(_PLANET_KEYS)}.")
        if "separation" not in planet or "contrast" not in planet:
            raise ValueError("Each planet needs a separation and a contrast, "
                             f"got {planet!r}.")
        if planet["contrast"] <= 0:
            raise ValueError("contrast must be positive, got "
                             f"{planet['contrast']}.")

        sed = planet.get("sed", "irtf/Neptune")
        key = sed if isinstance(sed, str) else id(sed)
        if key not in spectra:
            spectra[key] = resolve_sed(sed).scale_to_magnitude(
                zero, filter_curve=filter_name)

        dx, dy = offset_to_xy(planet["separation"],
                              planet.get("position_angle", 0),
                              planet.get("distance", distance))
        name = planet.get("name", sed if isinstance(sed, str) else "planet")
        rows.append((primary["x"] + dx.value, primary["y"] + dy.value,
                     list(spectra).index(key),
                     primary["weight"] * planet["contrast"], name))
    return rows, list(spectra.values())


@add_function_call_str
def planetary_system(filter_name, amplitude, spec_type="G2V", planets=(),
                     companion=None, distance=None, x=0, y=0,
                     library="pyckles", ra=RA0, dec=DEC0) -> Source:
    """
    Create a star (or binary) with planets, e.g. for high-contrast imaging.

    Parameters
    ----------
    filter_name : str
        Filter in which the amplitude and the contrasts are given.
    amplitude : float, u.Quantity
        Brightness of the host star. Bare numbers are Vega magnitudes.
    spec_type : str
        Spectral type of the host star.
    planets : iterable of dict
        One mapping per planet with the keys

        - ``separation`` (required): float [arcsec] or Quantity (angle or
          length, a length needs ``distance``),
        - ``contrast`` (required): flux ratio planet / host star in
          ``filter_name``, e.g. ``1e-5``,
        - ``position_angle``: from North through East, default 0 [deg],
        - ``sed``: anything :func:`~scopesim_templates_deux.utils.spectra.resolve_sed`
          accepts, default ``"irtf/Neptune"``,
        - ``name``: label for the ``spec_types`` column,
        - ``distance``: overrides the system ``distance``.

    companion : dict, optional
        Make the host a binary: keyword arguments for
        :func:`binary`, e.g. ``{"spec_type": "M3V", "separation": 0.5,
        "contrast": 0.1}`` (``spec_type`` is that of the companion).
        Planet contrasts and offsets refer to the primary.
    distance : float, u.Quantity, optional
        Distance to the system, needed for physical separations. Floats are
        parsec.
    x, y : float, u.Quantity
        Position of the host star, in arcsec from the field centre.
    library : str
        Spectral library for the star(s), see
        :func:`~scopesim_templates_deux.stellar.stars`.
    ra, dec : float, str
        Field centre, stored in the meta data only.

    Returns
    -------
    src : ``scopesim.Source``
        A single table source. The host star (and companion) come first,
        followed by one row per planet.

    Examples
    --------
    A Sun-like star at 10 pc with two planets::

        >>> from astropy import units as u
        >>> from scopesim_templates_deux.stellar import planetary_system
        >>> src = planetary_system(
        ...     "Ks", 5, "G2V", distance=10 * u.pc,
        ...     planets=[{"separation": 5 * u.AU, "contrast": 1e-5},
        ...              {"separation": 0.9, "position_angle": 120,
        ...               "contrast": 1e-6, "sed": "blackbody:900"}])
    """
    amplitude = as_amplitude(amplitude)
    if companion is not None:
        companion = dict(companion)
        spec_types = (spec_type, companion.pop("spec_type", spec_type))
        companion.setdefault("distance", distance)
        host = binary(filter_name, amplitude, spec_types, x=x, y=y,
                      library=library, ra=ra, dec=dec, **companion)
    else:
        host = stars(filter_name, u.Quantity([amplitude]), [spec_type],
                     x=[x], y=[y], library=library, ra=ra, dec=dec)

    host_table = host.fields[0].field
    host_spectra = list(host.spectra.values())
    zero = (0 if is_magnitude(amplitude.unit) else 1) * amplitude.unit
    rows, planet_spectra = _planet_rows(planets, host_table[0], zero,
                                        filter_name, distance)

    table = host_table.copy()
    if rows:
        planet_table = Table(rows=rows, names=table.colnames)
        planet_table["ref"] += len(host_spectra)
        for col in ("x", "y"):
            planet_table[col].unit = u.arcsec
        table = vstack([table, planet_table])

    src = Source(spectra=host_spectra + planet_spectra, table=table)
    src.meta.update({key: val for key, val in host.meta.items()
                     if key in ("separation", "position_angle", "contrast")})
    src.meta["n_planets"] = len(rows)
    return src


@add_function_call_str
def exoplanet(filter_name, amplitude, separation, contrast, spec_type="G2V",
              position_angle=0, sed="irtf/Neptune", distance=None, x=0, y=0,
              library="pyckles", ra=RA0, dec=DEC0) -> Source:
    """
    Create a star with a single planet.

    A shortcut for :func:`planetary_system` with one planet; see there for
    the parameters.

    Examples
    --------
    ::

        >>> from astropy import units as u
        >>> from scopesim_templates_deux.stellar import exoplanet
        >>> src = exoplanet("Ks", 4, separation=0.3, contrast=1e-4,
        ...                 spec_type="A0V", sed="blackbody:1200")
    """
    planet = {"separation": separation, "contrast": contrast,
              "position_angle": position_angle, "sed": sed}
    return planetary_system(filter_name, amplitude, spec_type, [planet],
                            distance=distance, x=x, y=y, library=library,
                            ra=ra, dec=dec)
