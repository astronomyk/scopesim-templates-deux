import pytest
import numpy as np
from astropy import units as u

from scopesim_templates.rc import Source
from scopesim_templates.stellar import binary, planetary_system, exoplanet


def _table(src):
    return src.fields[0].field


@pytest.mark.webtest
class TestBinary:
    def test_secondary_is_at_separation_and_position_angle(self):
        src = binary("Ks", 8, ("G2V", "M3V"), separation=50 * u.AU,
                     position_angle=90, distance=10 * u.pc, x=1, y=2)
        tbl = _table(src)
        assert isinstance(src, Source)
        assert len(tbl) == 2
        assert np.allclose(tbl["x"], [1, 6])
        assert np.allclose(tbl["y"], [2, 2])
        assert src.meta["separation"].to_value(u.arcsec) == pytest.approx(5)
        assert src.meta["object"] == "binary"

    def test_contrast_is_the_weight_ratio(self):
        tbl = _table(binary("V", 10, contrast=0.1))
        assert tbl["weight"][1] / tbl["weight"][0] == pytest.approx(0.1)

    def test_contrast_with_flux_units(self):
        tbl = _table(binary("V", 2 * u.Jy, contrast=0.25))
        assert tbl["weight"][1] / tbl["weight"][0] == pytest.approx(0.25)

    def test_amplitude_secondary(self):
        src = binary("V", 10, amplitude_secondary=12.5)
        assert src.meta["contrast"] == pytest.approx(0.1)

    def test_default_is_equal_brightness(self):
        tbl = _table(binary("V", 10, "A0V"))
        assert tbl["weight"][0] == tbl["weight"][1]
        assert list(tbl["spec_types"]) == ["A0V", "A0V"]

    def test_contrast_and_amplitude_secondary_raises(self):
        with pytest.raises(ValueError):
            binary("V", 10, contrast=0.1, amplitude_secondary=12)


class TestBinaryOffline:
    @pytest.mark.parametrize("spec_types", [("A0V",), ("A0V", "G2V", "K0V")])
    def test_needs_two_spec_types(self, spec_types):
        with pytest.raises(ValueError):
            binary("V", 10, spec_types)

    @pytest.mark.parametrize("contrast", [0, -1])
    def test_contrast_must_be_positive(self, contrast):
        with pytest.raises(ValueError):
            binary("V", 10, contrast=contrast)

    def test_physical_separation_needs_distance(self):
        with pytest.raises(ValueError):
            binary("V", 10, separation=10 * u.AU)


@pytest.mark.webtest
class TestPlanetarySystem:
    def test_planets_have_right_positions_and_contrasts(self):
        src = planetary_system(
            "Ks", 5, "G2V", distance=10 * u.pc,
            planets=[{"separation": 5 * u.AU, "contrast": 1e-5},
                     {"separation": 0.9, "position_angle": 90,
                      "contrast": 1e-6, "sed": "blackbody:900",
                      "name": "c"}])
        tbl = _table(src)
        assert len(tbl) == 3
        assert np.allclose(tbl["x"], [0, 0, 0.9], atol=1e-12)
        assert np.allclose(tbl["y"], [0, 0.5, 0], atol=1e-12)
        assert np.allclose(tbl["weight"][1:] / tbl["weight"][0], [1e-5, 1e-6])
        assert list(tbl["spec_types"]) == ["G2V", "irtf/Neptune", "c"]
        assert len(src.spectra) == 3
        assert src.meta["n_planets"] == 2

    def test_planets_share_identical_seds(self):
        planets = [{"separation": s, "contrast": 1e-4} for s in (1, 2, 3)]
        src = planetary_system("Ks", 5, planets=planets)
        assert len(src.spectra) == 2
        assert list(_table(src)["ref"]) == [0, 1, 1, 1]

    def test_binary_host(self):
        src = planetary_system(
            "Ks", 5, "G2V", x=1,
            companion={"spec_type": "M3V", "separation": 2, "contrast": 0.1},
            planets=[{"separation": 0.5, "position_angle": 180,
                      "contrast": 1e-5}])
        tbl = _table(src)
        assert list(tbl["spec_types"][:2]) == ["G2V", "M3V"]
        assert np.allclose(tbl["x"], [1, 1, 1])
        assert np.allclose(tbl["y"], [0, 2, -0.5])
        assert tbl["weight"][2] / tbl["weight"][0] == pytest.approx(1e-5)
        assert src.meta["contrast"] == pytest.approx(0.1)

    def test_no_planets_gives_the_star(self):
        assert len(_table(planetary_system("Ks", 5))) == 1

    def test_exoplanet(self):
        src = exoplanet("Ks", 4, separation=0.3, contrast=1e-4,
                        spec_type="A0V", sed="blackbody:1200")
        tbl = _table(src)
        assert np.allclose(tbl["y"], [0, 0.3])
        assert tbl["weight"][1] / tbl["weight"][0] == pytest.approx(1e-4)
        assert src.meta["object"] == "exoplanet"

    @pytest.mark.parametrize("planet", [{"contrast": 1e-4},
                                        {"separation": 1},
                                        {"separation": 1, "contrast": 0},
                                        {"separation": 1, "contrast": 1e-4,
                                         "seperation": 2}])
    def test_bad_planets_raise(self, planet):
        with pytest.raises(ValueError):
            planetary_system("Ks", 5, planets=[planet])
