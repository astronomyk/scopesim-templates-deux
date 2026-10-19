import pytest
import numpy as np
from astropy import units as u
from synphot import SourceSpectrum
from synphot.models import ConstFlux1D
from spextra import Spextrum

from scopesim_templates_deux.utils.spectra import resolve_sed

WAVES = np.linspace(2000, 60000, 20000) * u.AA


def _photon_peak(spec):
    return WAVES[np.argmax(spec(WAVES))].to_value(u.AA)


class TestResolveSed:
    @pytest.mark.parametrize("sed", ["blackbody:5000", "blackbody:5000 K",
                                     "BlackBody: 5000K", 5000 * u.K,
                                     4726.85 * u.deg_C])
    def test_blackbody_peaks_where_wien_says(self, sed):
        spec = resolve_sed(sed)
        assert isinstance(spec, Spextrum)
        # Wien's law for photon flux: lambda_max = 3.6697e7 AA K / T
        assert _photon_peak(spec) == pytest.approx(3.6697e7 / 5000, rel=0.01)

    def test_spectrum_objects_are_wrapped(self):
        spec = resolve_sed(SourceSpectrum(ConstFlux1D, amplitude=1))
        assert isinstance(spec, Spextrum)

    def test_reads_files(self, tmp_path):
        path = tmp_path / "spec.dat"
        waves = np.linspace(5000, 6000, 11)
        np.savetxt(path, np.c_[waves, np.ones_like(waves)])
        for sed in (f"file:{path}", path.as_uri()):
            spec = resolve_sed(sed)
            # constant FLAM, so the photon flux grows with wavelength
            ratio = spec(5500 * u.AA) / spec(5000 * u.AA)
            assert ratio.value == pytest.approx(1.1)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_sed(f"file:{tmp_path / 'nope.dat'}")

    @pytest.mark.parametrize("sed", [42, None, [1, 2]])
    def test_wrong_type_raises(self, sed):
        with pytest.raises(TypeError):
            resolve_sed(sed)

    def test_non_temperature_quantity_raises(self):
        with pytest.raises(u.UnitTypeError):
            resolve_sed(5 * u.m)

    @pytest.mark.webtest
    @pytest.mark.parametrize("sed", ["spex:kc96/s0", "kc96/s0"])
    def test_spextra_names(self, sed):
        assert repr(resolve_sed(sed)) == repr(Spextrum("kc96/s0"))
