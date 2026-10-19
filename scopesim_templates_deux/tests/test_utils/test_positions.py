import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from scopesim_templates_deux.utils.positions import (to_arcsec, offset_to_xy,
                                                skycoord_to_xy, as_xy)


class TestToArcsec:
    def test_floats_are_arcsec(self):
        assert to_arcsec(1.5) == 1.5 * u.arcsec

    def test_angles_are_converted(self):
        assert to_arcsec(2 * u.mas).to_value(u.arcsec) == pytest.approx(2e-3)

    def test_au_at_10pc_is_a_tenth_arcsec_per_au(self):
        assert to_arcsec(1 * u.AU, 10 * u.pc).value == pytest.approx(0.1)

    def test_float_distance_is_parsec(self):
        assert to_arcsec(1 * u.AU, 10) == to_arcsec(1 * u.AU, 10 * u.pc)

    def test_length_without_distance_raises(self):
        with pytest.raises(ValueError):
            to_arcsec(1 * u.AU)

    def test_wrong_unit_raises(self):
        with pytest.raises(u.UnitTypeError):
            to_arcsec(1 * u.kg)


class TestOffsetToXY:
    @pytest.mark.parametrize("pa, xy", [(0, (0, 1)), (90, (1, 0)),
                                        (180, (0, -1)), (270, (-1, 0))])
    def test_position_angle_is_east_of_north(self, pa, xy):
        x, y = offset_to_xy(1, pa)
        assert x.value == pytest.approx(xy[0], abs=1e-12)
        assert y.value == pytest.approx(xy[1], abs=1e-12)

    def test_physical_separation(self):
        x, y = offset_to_xy(50 * u.AU, 90 * u.deg, distance=10 * u.pc)
        assert x.to_value(u.arcsec) == pytest.approx(5)

    def test_position_angle_in_radians(self):
        x, _ = offset_to_xy(1, np.pi / 2 * u.rad)
        assert x.value == pytest.approx(1)

    def test_accepts_arrays(self):
        x, y = offset_to_xy([1, 2], [0, 90])
        assert x.shape == (2,)
        assert np.allclose(np.hypot(x, y).value, [1, 2])


class TestSkyCoordToXY:
    def test_east_is_plus_x_and_north_is_plus_y(self):
        coords = SkyCoord([1 / 3600, 0], [0, 1 / 3600], unit="deg")
        x, y = skycoord_to_xy(coords)
        assert np.allclose(x.value, [1, 0], atol=1e-9)
        assert np.allclose(y.value, [0, 1], atol=1e-9)

    def test_sexagesimal_field_centre(self):
        centre = SkyCoord("17:45:40.04", "-29:00:28.2",
                          unit=("hourangle", "deg"))
        x, y = skycoord_to_xy(centre, "17:45:40.04", "-29:00:28.2")
        assert abs(x.value) < 1e-9 and abs(y.value) < 1e-9


class TestAsXY:
    def test_floats_and_lists(self):
        x, y = as_xy([1, 2], [3, 4])
        assert x.unit == u.arcsec and list(y.value) == [3, 4]

    def test_skycoord(self):
        x, y = as_xy(SkyCoord(1 / 3600, 0, unit="deg"))
        assert x.value == pytest.approx(1) and y.value == pytest.approx(0)

    def test_skycoord_with_y_raises(self):
        with pytest.raises(ValueError):
            as_xy(SkyCoord(0, 0, unit="deg"), 1)

    def test_missing_y_raises(self):
        with pytest.raises(TypeError):
            as_xy(1)
