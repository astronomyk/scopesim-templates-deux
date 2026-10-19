# -*- coding: utf-8 -*-
"""
Simple geometric extended sources.

Coming from ``scopesim-targets``: ``Disk(radius, width)`` is
:func:`circumstellar_disk` with ``r_in=radius``. Here the ring can also be
inclined and rotated.
"""

import numpy as np
from astropy import units as u

from ..rc import Source
from ..utils.general_utils import RA0, DEC0, add_function_call_str
from ..utils.positions import to_arcsec
from ..utils.spectra import resolve_sed
from .misc import source_from_array


__all__ = ["ring_profile",
           "circumstellar_disk",
           ]


def ring_profile(shape, r_in, width, inclination=0, angle=0, center=None,
                 oversample=4):
    """
    Make an image of a uniform, optionally inclined ring.

    Face-on, this is the same as ``astropy.modeling.models.Ring2D``. An
    inclined ring is the projection of that circle, i.e. an ellipse whose
    minor axis is shorter by ``cos(inclination)``.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(ny, nx)``.
    r_in : float
        Inner radius in pixels, along the major axis.
    width : float
        Width of the ring in pixels, along the major axis.
    inclination : float, u.Quantity
        0 is face-on. Must be below 90 deg. Floats are degrees.
    angle : float, u.Quantity
        Direction of the major axis, counter-clockwise from +x, as for the
        ``galaxy`` profiles. Floats are degrees.
    center : tuple of float, optional
        ``(x, y)`` in pixels. Default is the image centre.
    oversample : int
        Sub-pixels per pixel and axis, to smooth the edges. Use 1 to sample
        only the pixel centres, like ``Ring2D.render``.

    Returns
    -------
    np.ndarray
        Normalised to a sum of 1.
    """
    inclination = u.Quantity(inclination, u.deg)
    if not 0 * u.deg <= inclination < 90 * u.deg:
        raise ValueError(f"inclination must be in [0, 90) deg, "
                         f"got {inclination}.")
    if r_in < 0 or width <= 0:
        raise ValueError(f"Need r_in >= 0 and width > 0, got {r_in=}, "
                         f"{width=}.")

    ny, nx = shape
    if center is None:
        center = ((nx - 1) / 2, (ny - 1) / 2)
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    yy = (np.arange(ny)[:, None] + sub[None, :]).ravel() - center[1]
    xx = (np.arange(nx)[:, None] + sub[None, :]).ravel() - center[0]

    phi = u.Quantity(angle, u.deg).to_value(u.rad)
    major = xx[None, :] * np.cos(phi) + yy[:, None] * np.sin(phi)
    minor = -xx[None, :] * np.sin(phi) + yy[:, None] * np.cos(phi)
    radius = np.hypot(major, minor / np.cos(inclination))

    ring = ((radius >= r_in) & (radius <= r_in + width)).astype(float)
    ring = ring.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    if ring.sum() == 0:
        raise ValueError("The ring falls between the pixels; increase "
                         "oversample or the ring size.")
    return ring / ring.sum()


@add_function_call_str
def circumstellar_disk(sed, r_in, width, pixel_scale, amplitude=None,
                       filter_curve="V", inclination=0, angle=0,
                       distance=None, extend=1.2, oversample=4,
                       ra=RA0, dec=DEC0) -> Source:
    """
    Create a uniform ring, e.g. a debris disk, centred on the field centre.

    Parameters
    ----------
    sed : str, u.Quantity, Spextrum, synphot.SourceSpectrum
        Anything :func:`~scopesim_templates.utils.spectra.resolve_sed`
        accepts, e.g. ``"blackbody:80"`` or ``"spex:kc96/s0"``.
    r_in, width : float, u.Quantity
        Inner radius and width of the ring. Floats are arcsec; lengths such
        as ``40 * u.AU`` need ``distance``.
    pixel_scale : float, u.Quantity
        Of the image. Floats are arcsec.
    amplitude : float, u.Quantity, optional
        Total brightness of the ring in ``filter_curve``. Bare numbers are
        AB magnitudes. If None, the SED is not rescaled.
    filter_curve : str
        Filter for ``amplitude``.
    inclination : float, u.Quantity
        0 is face-on. Floats are degrees.
    angle : float, u.Quantity
        Direction of the major axis, counter-clockwise from +x. Floats are
        degrees.
    distance : float, u.Quantity, optional
        Needed for physical sizes. Floats are parsec.
    extend : float
        Image half-size as a multiple of the outer radius.
    oversample : int
        Sub-pixels per pixel and axis, see :func:`ring_profile`.
    ra, dec : float, str
        Image centre, see :func:`~scopesim_templates.misc.source_from_array`.

    Returns
    -------
    src : ``scopesim.Source``

    Examples
    --------
    A 40-80 AU debris disk at 20 pc, inclined by 60 deg::

        >>> from astropy import units as u
        >>> from scopesim_templates.misc import circumstellar_disk
        >>> src = circumstellar_disk("blackbody:60", 40 * u.AU, 40 * u.AU,
        ...                          pixel_scale=0.05, amplitude=14,
        ...                          filter_curve="Ks", inclination=60,
        ...                          distance=20 * u.pc)
    """
    pixel_scale = to_arcsec(pixel_scale)
    r_in = (to_arcsec(r_in, distance) / pixel_scale).decompose().value
    width = (to_arcsec(width, distance) / pixel_scale).decompose().value

    size = 2 * int(np.ceil(extend * (r_in + width))) + 1
    img = ring_profile((size, size), r_in, width, inclination=inclination,
                       angle=angle, oversample=oversample)

    src = source_from_array(img, resolve_sed(sed), pixel_scale, amplitude,
                            filter_curve, ra=ra, dec=dec)
    return src
