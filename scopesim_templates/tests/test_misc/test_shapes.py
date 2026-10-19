import pytest
import numpy as np
from astropy import units as u
from astropy.modeling.models import Ring2D

from scopesim_templates.misc import ring_profile, circumstellar_disk


def _extent(mask, axis):
    return np.any(mask, axis=axis).sum()


class TestRingProfile:
    def test_is_normalised(self):
        assert ring_profile((101, 101), 20, 10).sum() == pytest.approx(1)

    @pytest.mark.parametrize("shape", [(101, 101), (80, 121)])
    def test_face_on_matches_ring2d(self, shape):
        ny, nx = shape
        ring = Ring2D(amplitude=1, x_0=(nx - 1) / 2, y_0=(ny - 1) / 2,
                      r_in=20, width=10)
        ref = ring.render(np.zeros(shape))
        img = ring_profile(shape, 20, 10, oversample=1)
        assert np.allclose(img, ref / ref.sum())

    def test_inclination_shrinks_the_minor_axis(self):
        img = ring_profile((201, 201), 60, 20, inclination=60)
        mask = img > 0
        assert _extent(mask, 0) == pytest.approx(161, abs=2)
        assert _extent(mask, 1) == pytest.approx(81, abs=2)

    def test_angle_rotates_the_major_axis(self):
        img = ring_profile((201, 201), 60, 20, inclination=60, angle=90)
        mask = img > 0
        assert _extent(mask, 0) == pytest.approx(81, abs=2)
        assert _extent(mask, 1) == pytest.approx(161, abs=2)

    def test_center(self):
        img = ring_profile((101, 101), 0, 5, center=(20, 70))
        yy, xx = np.indices(img.shape)
        assert (img * xx).sum() == pytest.approx(20, abs=0.1)
        assert (img * yy).sum() == pytest.approx(70, abs=0.1)

    @pytest.mark.parametrize("kwargs", [{"inclination": 90},
                                        {"inclination": -5},
                                        {"width": 0}, {"r_in": -1}])
    def test_bad_input_raises(self, kwargs):
        kwargs = {"r_in": 10, "width": 5} | kwargs
        with pytest.raises(ValueError):
            ring_profile((51, 51), **kwargs)

    def test_ring_between_pixels_raises(self):
        with pytest.raises(ValueError):
            ring_profile((11, 11), 2.1, 0.01, oversample=1)


@pytest.mark.webtest
class TestCircumstellarDisk:
    def test_physical_sizes(self):
        src = circumstellar_disk("blackbody:60", 40 * u.AU, 40 * u.AU,
                                 pixel_scale=0.05, amplitude=14,
                                 filter_curve="Ks", distance=20 * u.pc)
        img = src.fields[0].field.data
        assert img.sum() == pytest.approx(1)
        # outer radius 4 arcsec = 80 pixels
        assert _extent(img > 0, 0) == pytest.approx(161, abs=2)
        assert src.meta["object"] == "circumstellar_disk"

    def test_float_sizes_are_arcsec(self):
        src = circumstellar_disk(300 * u.K, 1, 0.5, pixel_scale=0.1)
        img = src.fields[0].field.data
        assert img.shape == (37, 37)
