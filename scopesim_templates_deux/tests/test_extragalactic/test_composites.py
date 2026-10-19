import pytest
import numpy as np
from astropy import units as u

from scopesim_templates_deux.extragalactic import (galaxy_with_supernova,
                                              galaxy_group, einstein_ring,
                                              lensed_sersic)
from scopesim_templates_deux.extragalactic.composites import _sis_point_images


class TestLensedSersic:
    def test_distant_source_has_weak_lensing_magnification(self):
        # SIS: mu = 1 + theta_E / beta for the outer image only
        _, mu = lensed_sersic((801, 801), 50, 300, 0, r_eff=10)
        assert mu == pytest.approx(1 + 50 / 300, rel=1e-3)

    def test_compact_source_matches_point_source_magnification(self):
        # two images: (1 + tE/b) + (tE/b - 1) = 2 tE / b
        _, mu = lensed_sersic((801, 801), 50, 30, 0, r_eff=0.5, oversample=8)
        assert mu == pytest.approx(2 * 50 / 30, rel=0.01)

    def test_centred_source_makes_a_ring(self):
        img, _ = lensed_sersic((201, 201), 50, 0, 0, r_eff=5)
        yy, xx = np.indices(img.shape)
        rr = np.hypot(xx - 100, yy - 100)
        ring = img[np.abs(rr - 50) < 2]
        assert ring.min() > 0.5 * ring.max()
        assert img[rr < 20].max() < 0.01 * ring.max()


class TestSisPointImages:
    def test_two_images_inside_the_einstein_radius(self):
        (x1, y1, mu1), (x2, y2, mu2) = _sis_point_images(0.3, 0, 1)
        assert (x1, x2) == pytest.approx((1.3, -0.7))
        assert (mu1, mu2) == pytest.approx((1 + 1 / 0.3, 1 / 0.3 - 1))

    def test_one_image_outside(self):
        assert len(_sis_point_images(0, 1.5, 1)) == 1

    def test_source_on_axis_raises(self):
        with pytest.raises(ValueError):
            _sis_point_images(0, 0, 1)


@pytest.mark.webtest
class TestGalaxyWithSupernova:
    def test_supernova_is_at_offset_with_its_own_spectrum(self):
        src = galaxy_with_supernova(sn_offset=(0.35, 0.2))
        tbl = src.fields[1].field
        assert (tbl["x"][0], tbl["y"][0]) == pytest.approx((0.35, 0.2))
        assert len(src.spectra) == 2
        assert src.meta["object"] == "galaxy_with_supernova"
        assert src.meta["sn_type"] == "sn1a"

    def test_supernova_is_brighter_than_fainter_one(self):
        def sn_flux(mag):
            src = galaxy_with_supernova(sn_amplitude=mag, filter_curve="V")
            return src.spectra[1](5500 * u.AA).value
        assert sn_flux(20) / sn_flux(22.5) == pytest.approx(10, rel=1e-3)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            galaxy_with_supernova(sn_type="kilonova")


@pytest.mark.webtest
class TestGalaxyGroup:
    def test_default_members_are_placed(self):
        src = galaxy_group()
        offsets = [(f.header["CRVAL1"] * 3600, f.header["CRVAL2"] * 3600)
                   for f in src.fields]
        assert np.allclose(offsets, [(0, 0), (1.1, 0.6), (-0.9, -0.7)])
        assert src.meta["n_members"] == 3

    def test_custom_members(self):
        src = galaxy_group([{"sed": "brown/NGC4125", "amplitude": 18,
                             "x": 2, "y": -1, "r_eff": 0.5}], z=0.1)
        assert len(src.fields) == 1
        assert src.fields[0].header["CRVAL1"] * 3600 == pytest.approx(2)

    @pytest.mark.parametrize("members", [[], [{"sed": "brown/NGC4125",
                                               "amplitude": 18, "r": 1}]])
    def test_bad_members_raise(self, members):
        with pytest.raises(ValueError):
            galaxy_group(members)


@pytest.mark.webtest
class TestEinsteinRing:
    def test_default_has_lens_arcs_and_two_agn_images(self):
        src = einstein_ring()
        assert len(src.fields) == 4
        mus = src.meta["qso_magnifications"]
        assert mus[0] / mus[1] == pytest.approx(
            (1 + 0.65 / np.hypot(0.12, 0.12)) / (0.65 / np.hypot(0.12, 0.12)
                                                  - 1))
        assert src.meta["arc_magnification"] > 1

    def test_no_agn(self):
        src = einstein_ring(source_offset=(0, 0), qso_amplitude=None)
        assert len(src.fields) == 2
        assert "qso_images" not in src.meta
