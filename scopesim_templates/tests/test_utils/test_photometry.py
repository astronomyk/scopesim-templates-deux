import pytest
from astropy import units as u

from scopesim_templates.utils.photometry import (as_amplitude, is_magnitude,
                                                 scale_amplitude)


def test_as_amplitude():
    assert as_amplitude(5) == 5 * u.mag
    assert as_amplitude(5, u.ABmag) == 5 * u.ABmag
    assert as_amplitude(2 * u.Jy) == 2 * u.Jy


@pytest.mark.parametrize("unit, expected", [(u.mag, True), (u.ABmag, True),
                                            (u.STmag, True), (u.Jy, False)])
def test_is_magnitude(unit, expected):
    assert is_magnitude(unit) is expected


def test_scale_amplitude():
    assert scale_amplitude(20 * u.ABmag, 100).value == pytest.approx(15)
    assert scale_amplitude(2 * u.Jy, 0.5) == 1 * u.Jy
    with pytest.raises(ValueError):
        scale_amplitude(20 * u.mag, 0)
