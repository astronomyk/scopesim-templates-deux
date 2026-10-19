# -*- coding: utf-8 -*-
"""
Scenes made of several galaxies and point sources.

These cover the example scenes of the ELVIS web ETC ("Galaxy + supernova",
"Galaxy group", "Einstein ring"), which ELVIS builds from scopesim-targets
``Sersic``, ``Star``, ``Ring`` and ``Gaussian`` objects. Here the supernova
uses a real supernova template, and the Einstein ring comes from an actual
(singular isothermal sphere) lens model instead of a drawn ring.

Positions are offsets in arcsec from the field centre, with East to +x.
Bare-number amplitudes are AB magnitudes, as for :func:`galaxy`.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling.models import GeneralSersic2D
from scipy.special import gamma, gammaincinv

from ..rc import Source
from ..utils.general_utils import add_function_call_str
from ..utils.photometry import as_amplitude, scale_amplitude
from ..utils.positions import as_xy, to_arcsec
from ..utils.spectra import resolve_sed
from ..misc.misc import point_source, source_from_array
from .galaxies import galaxy


__all__ = ["galaxy_with_supernova",
           "galaxy_group",
           "einstein_ring",
           "lensed_sersic",
           "SUPERNOVA_TYPES",
           "DEFAULT_GROUP",
           ]

SUPERNOVA_TYPES = ("sn1a", "sn1b", "sn1c", "sn2l", "sn2p", "sn2n", "hyper")

# Members of the ELVIS "Galaxy group" example: an elliptical, a spiral and a
# compact starburst. Magnitudes are AB in the filter of galaxy_group().
DEFAULT_GROUP = (
    {"sed": "brown/NGC4125", "amplitude": 19.4, "x": 0.0, "y": 0.0,
     "r_eff": 0.8, "n": 4.0, "ellip": 0.2, "theta": 10},
    {"sed": "brown/NGC0628", "amplitude": 21.9, "x": 1.1, "y": 0.6,
     "r_eff": 0.3, "n": 1.0, "ellip": 0.5, "theta": 70},
    {"sed": "brown/Mrk33", "amplitude": 21.5, "x": -0.9, "y": -0.7,
     "r_eff": 0.1, "n": 0.5, "ellip": 0.0, "theta": 0},
)

_GALAXY_KEYS = {"sed", "amplitude", "x", "y", "z", "r_eff", "n", "ellip",
                "theta", "c", "extend"}


def _supernova_sed(sn_type, sed, z):
    if sed is None:
        if sn_type not in SUPERNOVA_TYPES:
            raise ValueError(f"Unknown sn_type {sn_type!r}; choose from "
                             f"{SUPERNOVA_TYPES} or pass sn_sed.")
        sed = f"sne/{sn_type}"
    spec = resolve_sed(sed)
    return spec.redshift(z=z) if z else spec


@add_function_call_str
def galaxy_with_supernova(sn_amplitude=22, sn_offset=(0.35, 0.2),
                          sn_type="sn1a", sn_sed=None,
                          galaxy_sed="brown/NGC0628", galaxy_amplitude=19.5,
                          filter_curve="J", z=0, r_eff=0.6, n=1.2, ellip=0.35,
                          theta=25, pixel_scale=0.01, extend=3) -> Source:
    """
    Create a host galaxy with a supernova in it.

    Parameters
    ----------
    sn_amplitude : float, u.Quantity
        Observed brightness of the supernova in ``filter_curve``. Bare
        numbers are AB magnitudes.
    sn_offset : tuple of float, u.Quantity, or SkyCoord
        Position of the supernova, ``(x, y)`` in arcsec from the galaxy
        centre (which is at the field centre).
    sn_type : str
        speXtra supernova template (library ``sne``), near maximum light:
        ``"sn1a"``, ``"sn1b"``, ``"sn1c"``, ``"sn2l"``, ``"sn2p"``,
        ``"sn2n"`` or ``"hyper"`` (hypernova). Not all templates reach the
        near-IR: ``"sn1a"`` and ``"hyper"`` go to 2.5 um, others stop
        between 1 and 1.5 um (rest frame).
    sn_sed : optional
        Any other spectrum for the transient, see
        :func:`~scopesim_templates_deux.utils.spectra.resolve_sed`. Overrides
        ``sn_type``.
    galaxy_sed : str, Spextrum
        Host galaxy spectrum.
    galaxy_amplitude : float, u.Quantity
        Total brightness of the host. Bare numbers are AB magnitudes.
    filter_curve : str
        Filter for both amplitudes.
    z : float
        Redshift, applied to both spectra before scaling.
    r_eff, n, ellip, theta, pixel_scale, extend
        Sersic profile of the host, see
        :func:`~scopesim_templates_deux.extragalactic.galaxy`.

    Returns
    -------
    src : ``scopesim.Source``
        The host image plus a one-row table for the supernova.

    Examples
    --------
    A type II-P supernova 3 mag fainter than its host::

        >>> from scopesim_templates_deux.extragalactic import galaxy_with_supernova
        >>> src = galaxy_with_supernova(sn_amplitude=22.5, sn_type="sn2p",
        ...                             galaxy_amplitude=19.5, z=0.05)
    """
    host = galaxy(galaxy_sed, z=z, amplitude=galaxy_amplitude,
                  filter_curve=filter_curve, pixel_scale=pixel_scale,
                  r_eff=r_eff, n=n, ellip=ellip, theta=theta, extend=extend)
    if isinstance(sn_offset, SkyCoord):
        x, y = as_xy(sn_offset)
    else:
        x, y = as_xy(*sn_offset)
    spec = _supernova_sed(sn_type, sn_sed, z)
    sn = point_source(spec, amplitude=sn_amplitude, filter_curve=filter_curve,
                      x=x.to_value(u.arcsec), y=y.to_value(u.arcsec))

    src = host + sn
    src.meta.update({"sn_type": sn_type if sn_sed is None else "custom",
                     "sn_offset": (x, y), "z": z})
    return src


@add_function_call_str
def galaxy_group(members=DEFAULT_GROUP, filter_curve="H", z=0,
                 pixel_scale=0.01) -> Source:
    """
    Create a group of galaxies, each a Sersic profile with its own SED.

    Parameters
    ----------
    members : iterable of dict
        One mapping per galaxy with the keys ``x``, ``y`` (position in
        arcsec, default 0), ``sed``, ``amplitude`` (total brightness, bare
        numbers are AB magnitudes) and optionally ``z``, ``r_eff``, ``n``,
        ``ellip``, ``theta``, ``c`` and ``extend`` as for
        :func:`~scopesim_templates_deux.extragalactic.galaxy`. The default,
        :data:`DEFAULT_GROUP`, is the ELVIS example: an elliptical, a
        spiral and a compact starburst within 1.5 arcsec.
    filter_curve : str
        Filter for the amplitudes.
    z : float
        Redshift of the group, unless a member sets its own ``z``.
    pixel_scale : float, u.Quantity
        Of the member images. Floats are arcsec.

    Returns
    -------
    src : ``scopesim.Source``
        One image field per member.

    Examples
    --------
    ::

        >>> from scopesim_templates_deux.extragalactic import galaxy_group
        >>> src = galaxy_group(z=0.1)
        >>> src = galaxy_group([
        ...     {"sed": "brown/NGC4125", "amplitude": 18, "r_eff": 1, "n": 4},
        ...     {"sed": "brown/NGC0628", "amplitude": 20, "x": 2, "y": -1,
        ...      "r_eff": 0.5, "n": 1, "ellip": 0.6, "theta": 45}])
    """
    members = list(members)
    if not members:
        raise ValueError("A galaxy group needs at least one member.")

    src = None
    for member in members:
        unknown = set(member) - _GALAXY_KEYS
        if unknown:
            raise ValueError(f"Unknown galaxy keys {sorted(unknown)}; "
                             f"allowed are {sorted(_GALAXY_KEYS)}.")
        kwargs = dict(member)
        x, y = kwargs.pop("x", 0), kwargs.pop("y", 0)
        kwargs.setdefault("z", z)
        gal = galaxy(filter_curve=filter_curve, pixel_scale=pixel_scale,
                     **kwargs)
        gal.shift(to_arcsec(x).value, to_arcsec(y).value)
        src = gal if src is None else src + gal

    src.meta["n_members"] = len(members)
    return src


def _sersic_total(amplitude, r_eff, n, ellip):
    """Analytic total flux of a Sersic profile with I(r_eff) = amplitude."""
    b_n = gammaincinv(2 * n, 0.5)
    return (amplitude * 2 * np.pi * n * np.exp(b_n) * gamma(2 * n)
            / b_n ** (2 * n) * (1 - ellip) * r_eff ** 2)


def _sis_point_images(beta_x, beta_y, theta_e):
    """Image positions and magnifications of a point source behind an SIS."""
    beta = np.hypot(beta_x, beta_y)
    if beta == 0:
        raise ValueError("A point source exactly behind the lens is "
                         "infinitely magnified; give it an offset.")
    ux, uy = beta_x / beta, beta_y / beta
    images = [(ux * (beta + theta_e), uy * (beta + theta_e),
               1 + theta_e / beta)]
    if beta < theta_e:
        images.append((-ux * (theta_e - beta), -uy * (theta_e - beta),
                       theta_e / beta - 1))
    return images


def lensed_sersic(shape, theta_e, source_x=0., source_y=0., r_eff=1., n=1.,
                  ellip=0., theta=0, oversample=4):
    """
    Image of a Sersic source lensed by a singular isothermal sphere (SIS).

    Every image pixel is traced back to the source plane with the SIS lens
    equation, beta = theta - theta_E * theta / abs(theta), and takes the
    source surface brightness there (lensing conserves surface brightness).
    All lengths are in pixels; the lens is at the image centre.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)``.
    theta_e : float
        Einstein radius.
    source_x, source_y : float
        Source position relative to the lens.
    r_eff, n, ellip, theta
        Sersic parameters of the source; ``theta`` in degrees,
        counter-clockwise from +x.
    oversample : int
        Sub-pixels per pixel and axis.

    Returns
    -------
    image : np.ndarray
        Lensed surface brightness, with I(r_eff) = 1 for the unlensed
        source.
    magnification : float
        Total flux of the image divided by the flux of the unlensed source.
    """
    ny, nx = shape
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    yy = (np.arange(ny)[:, None] + sub).ravel() - (ny - 1) / 2
    xx = (np.arange(nx)[:, None] + sub).ravel() - (nx - 1) / 2
    xx, yy = np.meshgrid(xx, yy)

    radius = np.maximum(np.hypot(xx, yy), 1e-12)
    beta_x = xx - theta_e * xx / radius
    beta_y = yy - theta_e * yy / radius

    model = GeneralSersic2D(amplitude=1, r_eff=r_eff, n=n, ellip=ellip,
                            theta=np.deg2rad(u.Quantity(theta, u.deg).value),
                            x_0=source_x, y_0=source_y, c=0)
    image = model(beta_x, beta_y)
    image = image.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    magnification = image.sum() / _sersic_total(1, r_eff, n, ellip)
    return image, magnification


@add_function_call_str
def einstein_ring(theta_e=0.65, source_offset=(0.12, 0.12),
                  source_amplitude=23.3, source_sed="brown/Mrk33",
                  source_r_eff=0.12, source_n=1.0, source_ellip=0.0,
                  source_theta=0, qso_amplitude=23.7, qso_sed="brown/NGC1068",
                  lens_amplitude=20.4, lens_sed="brown/NGC4125",
                  lens_r_eff=0.25, lens_n=4.0, lens_ellip=0.15, lens_theta=20,
                  filter_curve="H", lens_z=0, source_z=0, pixel_scale=0.01,
                  oversample=4) -> Source:
    """
    Create a strong gravitational lens: a galaxy, its arcs and AGN images.

    The lens is a singular isothermal sphere (SIS) at the field centre. A
    background Sersic galaxy is ray-traced through it, which gives an
    Einstein ring (or arcs, for a larger offset). An optional point-like
    AGN at the centre of that galaxy appears as two images, with the SIS
    magnifications ``1 + theta_E/beta`` and ``theta_E/beta - 1``.

    Parameters
    ----------
    theta_e : float, u.Quantity
        Einstein radius. Floats are arcsec.
    source_offset : tuple of float
        ``(x, y)`` of the background source relative to the lens, in
        arcsec. ``(0, 0)`` gives a complete ring (then the AGN is not
        allowed).
    source_amplitude : float, u.Quantity
        *Intrinsic* (unlensed) brightness of the background galaxy. The
        lensed total is brighter by the computed magnification.
    source_sed : optional
        Spectrum of the background galaxy.
    source_r_eff, source_n, source_ellip, source_theta
        Sersic profile of the background galaxy (``r_eff`` in arcsec,
        ``theta`` in degrees counter-clockwise from +x).
    qso_amplitude : float, u.Quantity, None
        Intrinsic brightness of the background AGN; None for no AGN.
    qso_sed : optional
        Spectrum of the AGN. The default is the Seyfert 2 galaxy NGC 1068
        (Brown et al. 2014), because the speXtra ``agn/*`` templates stop
        below 1.1 um; use e.g. ``"agn/qso"`` for optical filters.
    lens_amplitude : float, u.Quantity
        Total brightness of the lens galaxy.
    lens_sed, lens_r_eff, lens_n, lens_ellip, lens_theta
        Lens galaxy, see :func:`~scopesim_templates_deux.extragalactic.galaxy`.
    filter_curve : str
        Filter for all amplitudes.
    lens_z, source_z : float
        Redshifts applied to the spectra. They do not change the geometry,
        which is set by ``theta_e``.
    pixel_scale : float, u.Quantity
        Of the images. Floats are arcsec.
    oversample : int
        Sub-pixels per pixel and axis for the ray tracing.

    Returns
    -------
    src : ``scopesim.Source``
        The lens galaxy image, the lensed galaxy image and (if any) a table
        with the AGN images. ``meta`` holds the magnifications.

    Examples
    --------
    The default is the ELVIS example, an Einstein ring of 0.65 arcsec with
    two AGN images::

        >>> from scopesim_templates_deux.extragalactic import einstein_ring
        >>> src = einstein_ring()
        >>> round(float(src.meta["arc_magnification"]))
        8
    """
    pixel_scale = to_arcsec(pixel_scale).value
    theta_e = to_arcsec(theta_e).value
    beta_x, beta_y = (to_arcsec(v).value for v in source_offset)

    lens = galaxy(lens_sed, z=lens_z, amplitude=lens_amplitude,
                  filter_curve=filter_curve, pixel_scale=pixel_scale,
                  r_eff=lens_r_eff, n=lens_n, ellip=lens_ellip,
                  theta=lens_theta)

    reach = theta_e + np.hypot(beta_x, beta_y) + 5 * source_r_eff
    size = 2 * int(np.ceil(1.2 * reach / pixel_scale)) + 1
    image, mu_arc = lensed_sersic(
        (size, size), theta_e / pixel_scale, beta_x / pixel_scale,
        beta_y / pixel_scale, source_r_eff / pixel_scale, source_n,
        source_ellip, source_theta, oversample)

    source_spec = resolve_sed(source_sed)
    if source_z:
        source_spec = source_spec.redshift(z=source_z)
    amplitude = scale_amplitude(as_amplitude(source_amplitude, u.ABmag),
                                mu_arc)
    arcs = source_from_array(image, source_spec, pixel_scale, amplitude,
                             filter_curve)
    src = lens + arcs
    meta = {"theta_e": theta_e * u.arcsec, "arc_magnification": mu_arc}

    if qso_amplitude is not None:
        qso_spec = resolve_sed(qso_sed)
        if source_z:
            qso_spec = qso_spec.redshift(z=source_z)
        qso_amplitude = as_amplitude(qso_amplitude, u.ABmag)
        images = _sis_point_images(beta_x, beta_y, theta_e)
        for x, y, mu in images:
            amp = scale_amplitude(qso_amplitude, mu)
            src = src + point_source(qso_spec, amp, filter_curve, x=x, y=y)
        meta["qso_images"] = [(x, y) for x, y, _ in images]
        meta["qso_magnifications"] = [mu for _, _, mu in images]

    src.meta.update(meta)
    return src
