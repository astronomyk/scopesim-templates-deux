"""Helpers to turn sky positions and offsets into (x, y) in arcsec."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from .general_utils import RA0, DEC0


__all__ = ["to_arcsec", "offset_to_xy", "skycoord_to_xy", "as_xy"]


def to_arcsec(value, distance=None):
    """
    Convert an angular or physical size to arcsec.

    Parameters
    ----------
    value : float, array, u.Quantity
        Floats are taken as arcsec. Angles are converted. Lengths (e.g. AU)
        need ``distance`` and are converted with the small-angle
        approximation.
    distance : float, u.Quantity, optional
        Distance to the object. Floats are taken as parsec.

    Returns
    -------
    u.Quantity
        In arcsec.

    Examples
    --------
    >>> from astropy import units as u
    >>> to_arcsec(10 * u.AU, distance=10 * u.pc)
    <Quantity 1. arcsec>
    """
    if not isinstance(value, u.Quantity):
        value = value * u.arcsec
    if value.unit.physical_type == "angle":
        return value.to(u.arcsec)

    if value.unit.physical_type != "length":
        raise u.UnitTypeError(f"Expected an angle or a length, got {value!r}.")
    if distance is None:
        raise ValueError(f"A physical size ({value!r}) needs a distance.")

    distance = u.Quantity(distance, u.pc)
    return (value / distance).to(u.arcsec,
                                  equivalencies=u.dimensionless_angles())


def offset_to_xy(separation, position_angle=0, distance=None):
    """
    Convert a separation and position angle into (x, y) offsets.

    The position angle is measured from North (+y) through East (+x), so a
    companion at 90 deg lies at +x. This follows the package convention of
    East to +x (see the "Units and conventions" docs page).

    Parameters
    ----------
    separation : float, array, u.Quantity
        Angular separation (floats are arcsec), or a physical separation such
        as ``50 * u.AU``, which needs ``distance``.
    position_angle : float, array, u.Quantity, optional
        East of North. Floats are degrees. Default 0 (due North).
    distance : float, u.Quantity, optional
        Needed if ``separation`` is a length. Floats are parsec.

    Returns
    -------
    x, y : u.Quantity
        In arcsec.

    Examples
    --------
    >>> from astropy import units as u
    >>> x, y = offset_to_xy(50 * u.AU, 90 * u.deg, distance=10 * u.pc)
    >>> x.round(6), y.round(6)
    (<Quantity 5. arcsec>, <Quantity 0. arcsec>)
    """
    sep = to_arcsec(separation, distance)
    pa = u.Quantity(position_angle, u.deg).to_value(u.rad)
    return sep * np.sin(pa), sep * np.cos(pa)


def skycoord_to_xy(coords, ra=RA0, dec=DEC0):
    """
    Project sky coordinates onto (x, y) offsets from the field centre.

    East goes to +x and North to +y, as for all templates.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        Scalar or array of positions.
    ra, dec : float, str
        Field centre. Floats are degrees, strings are sexagesimal
        (hourangle for ``ra``).

    Returns
    -------
    x, y : u.Quantity
        In arcsec.
    """
    ra_unit = u.hourangle if isinstance(ra, str) else u.deg
    centre = SkyCoord(ra, dec, unit=(ra_unit, u.deg))
    offsets = coords.transform_to(centre.skyoffset_frame())
    return offsets.lon.to(u.arcsec), offsets.lat.to(u.arcsec)


def as_xy(x, y=None, ra=RA0, dec=DEC0):
    """
    Return positions as (x, y) in arcsec, accepting sky coordinates too.

    Parameters
    ----------
    x : float, array, u.Quantity, SkyCoord
        x offsets (floats are arcsec), or sky positions, which are projected
        around (``ra``, ``dec``) with :func:`skycoord_to_xy`.
    y : float, array, u.Quantity, optional
        y offsets. Must be None (or 0) if ``x`` is a ``SkyCoord``.
    ra, dec : float, str
        Field centre, only used for ``SkyCoord`` input.

    Returns
    -------
    x, y : u.Quantity
        In arcsec.
    """
    if isinstance(x, SkyCoord):
        if y is not None and np.any(y != 0):
            raise ValueError("Give either a SkyCoord or x and y, not both.")
        return skycoord_to_xy(x, ra, dec)
    if y is None:
        raise TypeError("y is needed unless x is a SkyCoord.")
    return u.Quantity(x, u.arcsec), u.Quantity(y, u.arcsec)
