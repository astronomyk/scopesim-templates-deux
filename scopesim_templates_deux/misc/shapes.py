# -*- coding: utf-8 -*-
"""
Simple geometric extended sources.

Coming from ``scopesim-targets``: ``Disk(radius, width)`` is
:func:`circumstellar_disk` with ``r_in=radius``. Here the ring can also be
inclined and rotated (an elliptical ``Ring`` with ``ellip`` is
``inclination = arccos(1 - ellip)``), and ``r_in=0`` gives a uniform disk.
``Gaussian(x_stddev, y_stddev, theta)`` is :func:`gaussian_source`.
"""

import numpy as np
from astropy import units as u

from ..rc import Source
from ..utils.general_utils import RA0, DEC0, add_function_call_str
from ..utils.positions import to_arcsec
from .misc import source_from_array


__all__ = ["ring_profile",
           "gaussian_profile",
           "circumstellar_disk",
           "gaussian_source",
           ]


def _rotated_grid(shape, angle, center, oversample):
    """Sub-pixel offsets along and across the direction ``angle``."""
    ny, nx = shape
    if center is None:
        center = ((nx - 1) / 2, (ny - 1) / 2)
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    yy = (np.arange(ny)[:, None] + sub[None, :]).ravel() - center[1]
    xx = (np.arange(nx)[:, None] + sub[None, :]).ravel() - center[0]

    phi = u.Quantity(angle, u.deg).to_value(u.rad)
    major = xx[None, :] * np.cos(phi) + yy[:, None] * np.sin(phi)
    minor = -xx[None, :] * np.sin(phi) + yy[:, None] * np.cos(phi)
    return major, minor


def _bin(img, shape, oversample):
    ny, nx = shape
    return img.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))


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

    major, minor = _rotated_grid(shape, angle, center, oversample)
    radius = np.hypot(major, minor / np.cos(inclination))

    ring = ((radius >= r_in) & (radius <= r_in + width)).astype(float)
    ring = _bin(ring, shape, oversample)
    if ring.sum() == 0:
        raise ValueError("The ring falls between the pixels; increase "
                         "oversample or the ring size.")
    return ring / ring.sum()


def gaussian_profile(shape, sigma, sigma_minor=None, angle=0, center=None,
                     oversample=4):
    """
    Make an image of an elliptical 2D Gaussian.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(ny, nx)``.
    sigma : float
        Standard deviation along the major axis, in pixels.
    sigma_minor : float, optional
        Standard deviation along the minor axis. Default ``sigma`` (round).
        An inclined round disk has ``sigma * cos(inclination)``.
    angle : float, u.Quantity
        Direction of the major axis, counter-clockwise from +x. Floats are
        degrees.
    center : tuple of float, optional
        ``(x, y)`` in pixels. Default is the image centre.
    oversample : int
        Sub-pixels per pixel and axis.

    Returns
    -------
    np.ndarray
        Normalised to a sum of 1.
    """
    sigma_minor = sigma if sigma_minor is None else sigma_minor
    if sigma <= 0 or sigma_minor <= 0:
        raise ValueError(f"Need positive widths, got {sigma=}, "
                         f"{sigma_minor=}.")

    major, minor = _rotated_grid(shape, angle, center, oversample)
    img = np.exp(-0.5 * ((major / sigma) ** 2 + (minor / sigma_minor) ** 2))
    img = _bin(img, shape, oversample)
    return img / img.sum()


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
        Anything :func:`~scopesim_templates_deux.utils.spectra.resolve_sed`
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
        Image centre, see :func:`~scopesim_templates_deux.misc.source_from_array`.

    Returns
    -------
    src : ``scopesim.Source``

    Examples
    --------
    A 40-80 AU debris disk at 20 pc, inclined by 60 deg::

        >>> from astropy import units as u
        >>> from scopesim_templates_deux.misc import circumstellar_disk
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

    src = source_from_array(img, sed, pixel_scale, amplitude,
                            filter_curve, ra=ra, dec=dec)
    return src


@add_function_call_str
def gaussian_source(sed, sigma, pixel_scale, amplitude=None,
                    filter_curve="V", sigma_minor=None, angle=0,
                    distance=None, extend=5, ra=RA0, dec=DEC0) -> Source:
    """
    Create an elliptical Gaussian blob, centred on the field centre.

    Parameters
    ----------
    sed : str, u.Quantity, Spextrum, synphot.SourceSpectrum
        Anything :func:`~scopesim_templates_deux.utils.spectra.resolve_sed`
        accepts.
    sigma : float, u.Quantity
        Standard deviation along the major axis. Floats are arcsec; lengths
        need ``distance``.
    pixel_scale : float, u.Quantity
        Of the image. Floats are arcsec.
    amplitude : float, u.Quantity, optional
        Total brightness in ``filter_curve``. Bare numbers are AB
        magnitudes. If None, the SED is not rescaled.
    filter_curve : str
        Filter for ``amplitude``.
    sigma_minor : float, u.Quantity, optional
        Standard deviation along the minor axis. Default ``sigma``.
    angle : float, u.Quantity
        Direction of the major axis, counter-clockwise from +x. Floats are
        degrees.
    distance : float, u.Quantity, optional
        Needed for physical sizes. Floats are parsec.
    extend : float
        Image half-size in units of ``sigma``.
    ra, dec : float, str
        Image centre, see :func:`~scopesim_templates_deux.misc.source_from_array`.

    Returns
    -------
    src : ``scopesim.Source``

    Examples
    --------
    ::

        >>> from scopesim_templates_deux.misc import gaussian_source
        >>> src = gaussian_source("brown/NGC1068", 0.02, pixel_scale=0.004,
        ...                       amplitude=21, filter_curve="H")
    """
    pixel_scale = to_arcsec(pixel_scale)
    sigma = to_arcsec(sigma, distance)
    sigma_minor = sigma if sigma_minor is None else to_arcsec(sigma_minor,
                                                              distance)
    sigma_px = (sigma / pixel_scale).decompose().value
    minor_px = (sigma_minor / pixel_scale).decompose().value

    size = 2 * int(np.ceil(extend * sigma_px)) + 1
    img = gaussian_profile((size, size), sigma_px, minor_px, angle=angle)
    src = source_from_array(img, sed, pixel_scale, amplitude, filter_curve,
                            ra=ra, dec=dec)
    return src
