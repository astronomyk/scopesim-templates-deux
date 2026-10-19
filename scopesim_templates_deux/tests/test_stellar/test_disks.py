import pytest
import numpy as np
from astropy import units as u

from scopesim_templates_deux.stellar import protoplanetary_disk


def _extent(mask, axis):
    return np.any(mask, axis=axis).sum()


@pytest.mark.webtest
class TestProtoplanetaryDisk:
    def test_default_has_star_and_disk(self):
        src = protoplanetary_disk()
        assert len(src.fields) == 2
        assert len(src.fields[0].field) == 1
        assert src.fields[1].field.data.sum() == pytest.approx(1)
        assert src.meta["n_rings"] == 6

    def test_face_on_rings_have_the_right_radius(self):
        # outermost ring ends at 105 AU = 1.05 arcsec at 100 pc
        src = protoplanetary_disk(inclination=0, distance=100 * u.pc,
                                  inner_flux=0, pixel_scale=0.01)
        img = src.fields[1].field.data
        assert _extent(img > 0, 0) == pytest.approx(211, abs=2)
        assert _extent(img > 0, 1) == pytest.approx(211, abs=2)

    def test_inclined_disk_major_axis_follows_position_angle(self):
        # PA 90 = East = +x, so the major axis is horizontal
        src = protoplanetary_disk(inclination=60, position_angle=90,
                                  inner_flux=0, distance=100 * u.pc,
                                  pixel_scale=0.01)
        img = src.fields[1].field.data
        assert _extent(img > 0, 0) == pytest.approx(211, abs=2)
        assert _extent(img > 0, 1) == pytest.approx(106, abs=3)

    def test_custom_rings_and_no_inner_disk(self):
        src = protoplanetary_disk(rings=[(0.3 * u.arcsec, 0.05 * u.arcsec,
                                          1)], inner_flux=0, inclination=0)
        img = src.fields[1].field.data
        centre = img.shape[0] // 2
        assert img[centre, centre] == 0

    def test_empty_disk_raises(self):
        with pytest.raises(ValueError):
            protoplanetary_disk(rings=[], inner_flux=0)
