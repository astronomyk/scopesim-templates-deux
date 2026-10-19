# -*- coding: utf-8 -*-
"""
Young stars with ringed protoplanetary disks.

Covers the "HL Tau-like disk" example of the ELVIS web ETC, which stacks a
scopesim-targets ``Star``, a ``Gaussian`` and six elliptical ``Ring``
objects. Here the disk is one image with a single spectrum.
"""

import numpy as np
from astropy import units as u

from ..rc import Source
from ..utils.general_utils import RA0, DEC0, add_function_call_str
from ..utils.photometry import as_amplitude
from ..utils.positions import to_arcsec
from ..misc.misc import source_from_array
from ..misc.shapes import gaussian_profile, ring_profile
from .stellar import stars


__all__ = ["protoplanetary_disk",
           "HL_TAU_RINGS",
           ]

# Bright rings of HL Tau as (inner radius [AU], width [AU], relative flux),
# following the ALMA Partnership (2015) image, as used in the ELVIS example.
# The inner disk (a Gaussian) has relative flux 1.
HL_TAU_RINGS = (
    (15.4, 7.0, 0.83),
    (29.4, 8.4, 0.63),
    (44.8, 7.0, 0.48),
    (60.2, 8.4, 0.36),
    (78.4, 7.0, 0.28),
    (93.8, 11.2, 0.21),
)


@add_function_call_str
def protoplanetary_disk(filter_name="K", amplitude=13, spec_type="K5V",
                        disk_amplitude=12.55, rings=HL_TAU_RINGS,
                        inner_sigma=5 * u.AU, inner_flux=1.0,
                        inclination=47, position_angle=138,
                        distance=140 * u.pc, pixel_scale=0.005, disk_sed=None,
                        x=0, y=0, library="pyckles",
                        ra=RA0, dec=DEC0) -> Source:
    """
    Create a young star with a ringed disk, by default like HL Tau.

    Parameters
    ----------
    filter_name : str
        Filter for ``amplitude`` and ``disk_amplitude``.
    amplitude : float, u.Quantity
        Brightness of the star. Bare numbers are Vega magnitudes.
    spec_type : str
        Spectral type of the star.
    disk_amplitude : float, u.Quantity
        Total brightness of the disk (inner disk plus rings). Bare numbers
        are Vega magnitudes.
    rings : iterable of (r_in, width, relative_flux)
        The bright rings; the gaps are what lies between them. Radii are
        floats in AU (or any Quantity, arcsec or length). The default is
        :data:`HL_TAU_RINGS`.
    inner_sigma : float, u.Quantity
        Width (standard deviation) of the compact inner disk; floats are AU.
        Use ``inner_flux=0`` to leave it out.
    inner_flux : float
        Flux of the inner disk relative to the ``rings`` fluxes.
    inclination : float, u.Quantity
        0 is face-on. Floats are degrees.
    position_angle : float, u.Quantity
        Of the major axis, from North (+y) through East (+x). Floats are
        degrees.
    distance : float, u.Quantity
        Distance to the star. Floats are parsec.
    pixel_scale : float, u.Quantity
        Of the disk image. Floats are arcsec.
    disk_sed : optional
        Spectrum of the disk light, see
        :func:`~scopesim_templates_deux.utils.spectra.resolve_sed`. Default is
        the stellar spectrum, i.e. scattered starlight.
    x, y : float
        Position of the star, in arcsec from the field centre.
    library : str
        Spectral library of the star, see
        :func:`~scopesim_templates_deux.stellar.stars`.
    ra, dec : float, str
        Field centre, stored in the meta data only.

    Returns
    -------
    src : ``scopesim.Source``
        A one-row table for the star and an image for the disk.

    Examples
    --------
    HL Tau as in the ELVIS example, and a face-on version at 100 pc::

        >>> from astropy import units as u
        >>> from scopesim_templates_deux.stellar import protoplanetary_disk
        >>> src = protoplanetary_disk()
        >>> src = protoplanetary_disk(inclination=0, distance=100 * u.pc)
    """
    def _au(value):
        if not isinstance(value, u.Quantity):
            value = value * u.AU
        return (to_arcsec(value, distance) / pixel_scale).decompose().value

    pixel_scale = to_arcsec(pixel_scale)
    rings = [(_au(r_in), _au(width), flux) for r_in, width, flux in rings]
    sigma = _au(inner_sigma)
    if not rings and inner_flux <= 0:
        raise ValueError("The disk needs rings or an inner disk.")

    outer = max([r_in + width for r_in, width, _ in rings] + [3 * sigma])
    size = 2 * int(np.ceil(1.1 * outer)) + 1
    shape = (size, size)
    # East of North -> counter-clockwise from +x, with East at +x
    angle = 90 * u.deg - u.Quantity(position_angle, u.deg)
    cos_i = np.cos(u.Quantity(inclination, u.deg))

    img = np.zeros(shape)
    for r_in, width, flux in rings:
        img += flux * ring_profile(shape, r_in, width, inclination,
                                   angle=angle)
    if inner_flux > 0:
        img += inner_flux * gaussian_profile(shape, sigma, sigma * cos_i,
                                             angle=angle)

    star = stars(filter_name, u.Quantity([as_amplitude(amplitude)]),
                 [spec_type], x=[x], y=[y], library=library, ra=ra, dec=dec)
    sed = list(star.spectra.values())[0] if disk_sed is None else disk_sed
    disk = source_from_array(img, sed, pixel_scale,
                             as_amplitude(disk_amplitude), filter_name)
    disk.shift(x, y)

    src = star + disk
    src.meta.update({"inclination": u.Quantity(inclination, u.deg),
                     "position_angle": u.Quantity(position_angle, u.deg),
                     "distance": u.Quantity(distance, u.pc),
                     "n_rings": len(rings)})
    return src
