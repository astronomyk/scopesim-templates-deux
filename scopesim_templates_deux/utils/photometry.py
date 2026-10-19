"""Small helpers for amplitudes given as magnitudes or fluxes."""

import numpy as np
from astropy import units as u


__all__ = ["as_amplitude", "is_magnitude", "scale_amplitude"]


def as_amplitude(amplitude, default_unit=u.mag):
    """Attach ``default_unit`` to a bare number; Quantities pass through."""
    if isinstance(amplitude, u.Quantity):
        return amplitude
    return amplitude * default_unit


def is_magnitude(unit):
    """True for the magnitude units the templates accept."""
    return unit in (u.mag, u.ABmag, u.STmag)


def scale_amplitude(amplitude, factor):
    """
    Return the amplitude of something ``factor`` times as bright.

    Works for magnitudes (``u.mag``, ``u.ABmag``, ``u.STmag``) and fluxes.

    Examples
    --------
    >>> from astropy import units as u
    >>> scale_amplitude(20 * u.ABmag, 100)
    <Magnitude 15. mag(AB)>
    """
    if factor <= 0:
        raise ValueError(f"The flux factor must be positive, got {factor}.")
    if is_magnitude(amplitude.unit):
        return (amplitude.value - 2.5 * np.log10(factor)) * amplitude.unit
    return amplitude * factor
