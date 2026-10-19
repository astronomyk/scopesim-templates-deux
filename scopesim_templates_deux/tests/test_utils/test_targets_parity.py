"""
One test per feature of ``scopesim-targets``, using this package's templates.

This is the executable form of the table in ``docs/from_scopesim_targets.md``.
When ``scopesim-targets`` gains a feature, add a row there and a test here.
``scopesim-targets`` itself is never imported.
"""

import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling.models import GeneralSersic2D, Sersic2D

from scopesim_templates_deux import (star, stars, galaxy, binary,
                                planetary_system, exoplanet,
                                circumstellar_disk, gaussian_source,
                                cluster, galaxy_with_supernova, galaxy_group,
                                einstein_ring, protoplanetary_disk)
from scopesim_templates_deux.extragalactic.exgal_models import GalaxyBase
from scopesim_templates_deux.misc import point_source


def _table(src):
    return src.fields[0].field


@pytest.mark.webtest
class TestPointSources:
    def test_star(self):
        """Star(spectrum="G2V", brightness=("Ks", 10))"""
        assert len(_table(star("Ks", 10, "G2V"))) == 1

    def test_star_from_bosz(self):
        """scopesim-targets uses bosz/lr; exact type names only."""
        assert len(_table(star("Ks", 10, "A0V", library="bosz/lr"))) == 1

    def test_star_field_with_skycoords(self):
        """StarField(positions=[SkyCoord, ...], ...)"""
        coords = SkyCoord([0, 1 / 3600], [0, 0], unit="deg")
        tbl = _table(stars("Ks", [10, 11], ["A0V", "A0V"], coords))
        assert np.allclose(tbl["x"], [0, 1])
        assert len(set(tbl["ref"])) == 1

    def test_point_source_with_skycoord_and_blackbody(self):
        """Star(position=SkyCoord, spectrum="blackbody:5800 K")"""
        src = point_source("blackbody:5800", 10, "Ks",
                           SkyCoord(0, 2 / 3600, unit="deg"))
        assert _table(src)["y"][0] == pytest.approx(2)

    def test_binary(self):
        """Binary(offset={separation: 50 AU, position_angle: 90 deg}, ...)"""
        src = binary("Ks", 8, separation=50 * u.AU, position_angle=90,
                     contrast=0.1, distance=10 * u.pc)
        assert _table(src)["x"][1] == pytest.approx(5)

    def test_planetary_system_with_binary_host(self):
        """PlanetarySystem(primary=Binary(...), components=[Exoplanet, ...])"""
        src = planetary_system(
            "Ks", 5, companion={"separation": 1, "contrast": 0.5},
            planets=[{"separation": 0.3, "contrast": 1e-5}])
        assert len(_table(src)) == 3

    def test_exoplanet(self):
        """Exoplanet(spectrum=irtf/Neptune) next to a star."""
        src = exoplanet("Ks", 5, separation=0.3, contrast=1e-5)
        assert "irtf/Neptune" in _table(src)["spec_types"]


@pytest.mark.webtest
class TestExtendedSources:
    def test_sersic_with_boxiness(self):
        """Sersic(params={r_eff, n, ellip, theta, c})"""
        boxy = galaxy("kc96/s0", r_eff=1, n=1, ellip=0.5, c=1)
        plain = galaxy("kc96/s0", r_eff=1, n=1, ellip=0.5)
        assert not np.allclose(_table(boxy).data, _table(plain).data)

    def test_disk(self):
        """Disk(radius=1 arcsec, width=0.5 arcsec)"""
        src = circumstellar_disk("blackbody:100", 1, 0.5, pixel_scale=0.1)
        assert _table(src).data.sum() == pytest.approx(1)

    def test_elliptical_ring(self):
        """Ring(r_in, width, ellip=0.5, theta) (ELVIS)"""
        incl = np.rad2deg(np.arccos(1 - 0.5))
        src = circumstellar_disk("blackbody:100", 1, 0.2, pixel_scale=0.05,
                                 inclination=incl, angle=30)
        assert _table(src).data.sum() == pytest.approx(1)

    def test_uniform_disk(self):
        """Disk(R_0=0.5 arcsec) (ELVIS)"""
        img = _table(circumstellar_disk("blackbody:100", 0, 0.5,
                                        pixel_scale=0.1)).data
        centre = img.shape[0] // 2
        assert img[centre, centre] == pytest.approx(img.max())

    def test_gaussian(self):
        """Gaussian(x_stddev, y_stddev, theta) (ELVIS)"""
        src = gaussian_source("blackbody:5000", 0.1, pixel_scale=0.02,
                              sigma_minor=0.05, angle=45)
        assert _table(src).data.sum() == pytest.approx(1)

    def test_zero_age_cluster(self):
        """ZeroAgeCluster (ELVIS)"""
        assert len(_table(cluster(mass=100, seed=1))) > 0


@pytest.mark.webtest
class TestElvisExampleScenes:
    def test_galaxy_with_supernova(self):
        assert len(galaxy_with_supernova().fields) == 2

    def test_galaxy_group(self):
        assert len(galaxy_group().fields) == 3

    def test_einstein_ring(self):
        assert len(einstein_ring().fields) == 4

    def test_hl_tau_disk(self):
        assert len(protoplanetary_disk().fields) == 2


class TestOffline:
    def test_boxiness_zero_is_plain_sersic(self):
        y, x = np.mgrid[:51, :51]
        kwargs = dict(x_0=25, y_0=25, amplitude=1, r_eff=8, n=2.5,
                      ellip=0.3, theta=30)
        gal = GalaxyBase(x=x, y=y, **kwargs)
        theta = np.deg2rad(kwargs.pop("theta"))
        ref = Sersic2D(**kwargs, theta=theta)(x, y)
        assert np.allclose(gal.intensity, ref)
        boxy = GalaxyBase(x=x, y=y, theta=30, c=0.5,
                          **{k: v for k, v in kwargs.items()})
        ref = GeneralSersic2D(**kwargs, theta=theta, c=0.5)(x, y)
        assert np.allclose(boxy.intensity, ref)
