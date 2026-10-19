# Coming from scopesim-targets

[scopesim-targets](https://github.com/AstarVienna/scopesim-targets) describes
targets as classes (or YAML) with a `to_source()` method. Everything it can
describe can also be made with the functions in this package, which return a
`Source` directly. The two packages are independent; this one does not use
scopesim-targets.

| scopesim-targets | ScopeSim Templates |
|---|---|
| `Star(spectrum="G2V", brightness=("Ks", 10))` | `star("Ks", 10, "G2V")` (Pickles; `library="bosz/lr"` for the scopesim-targets library) |
| `StarField(positions, spectra, brightnesses, band)` | `stars(band, mags, spec_types, x, y)`, or `stars(band, mags, spec_types, skycoords)` |
| `Star(position=SkyCoord(...))` | `stars(..., skycoords, ra=..., dec=...)` or `point_source(sed, mag, band, skycoord, ra=..., dec=...)` |
| `Binary(spectra=[p, s], brightness=..., contrast=c, offset={"separation": 50 * u.AU, "position_angle": 90 * u.deg})` | `binary(band, mag, (p, s), separation=50 * u.AU, position_angle=90, contrast=c, distance=...)` |
| `Exoplanet(...)` around a `Star` | `exoplanet(band, mag, separation, contrast, spec_type=..., sed="irtf/Neptune")` |
| `PlanetarySystem(primary=Star or Binary, components=[Exoplanet, ...])` | `planetary_system(band, mag, spec_type, planets=[{...}, ...], companion={...})` |
| `Sersic(params={"r_eff", "n", "ellip", "theta", "c"})` | `galaxy(sed, r_eff=..., n=..., ellip=..., theta=..., c=...)` |
| `Disk(radius=r, width=w)` | `circumstellar_disk(sed, r_in=r, width=w, pixel_scale=...)`, which can also be inclined |
| `Ring(r_in, width, ellip, theta)` | `circumstellar_disk(sed, r_in, width, inclination=arccos(1 - ellip), angle=theta, ...)` |
| `Disk(R_0)` (uniform disk, `Disk2D`) | `circumstellar_disk(sed, r_in=0, width=R_0, ...)` |
| `Gaussian(x_stddev, y_stddev, theta)` | `gaussian_source(sed, sigma, sigma_minor=..., angle=theta, ...)` |
| `ZeroAgeCluster(...)` | `cluster(mass, distance, core_radius)` |
| spectrum `"spex:kc96/s0"`, `"file:spec.fits"`, `"blackbody:5800 K"` | the same strings work for every `sed` argument (see `utils.spectra.resolve_sed`) |
| separation in AU plus the parent's distance | any separation or size given as a length plus `distance=` (see `utils.positions`) |

The `Ring`, `Disk(R_0)`, `Gaussian` and `ZeroAgeCluster` rows come from the
copy of scopesim-targets that the ELVIS web ETC ships.

## ELVIS example scenes

The example scenes of the ELVIS web ETC each have a one-call equivalent:

| ELVIS example | ScopeSim Templates |
|---|---|
| Galaxy + supernova | `galaxy_with_supernova()`, with a real `sne/...` template instead of a B0V star |
| Binary star | `binary("K", 14.05, ("G2V", "M2V"), separation=0.04, position_angle=60, contrast=0.05)` |
| Crowded star field | `stars("K", mags, spec_types, x, y)` |
| Galaxy group | `galaxy_group()` |
| Einstein ring | `einstein_ring()`, ray-traced through an isothermal lens instead of a drawn ring |
| HL Tau-like disk (rings) | `protoplanetary_disk()` |
| Galactic Centre S-stars | `galactic_centre("2026-09-27")` |

A nebula, which ELVIS does not have, is `nebula_source()`.

## Differences to keep in mind

- **Contrast** is always the flux ratio companion / primary (`1e-5` for a
  faint planet). scopesim-targets multiplies by the contrast for `Binary` but
  divides by it for planets.
- **Bare magnitudes** are Vega for the stellar templates (`star`, `binary`,
  `planetary_system`, ...) and AB for the others, see
  [Units and conventions](conventions.md). scopesim-targets uses Vega only.
- **Positions** are offsets in arcsec from the field centre. `SkyCoord`
  positions are projected around the `ra`/`dec` of the call.
- Extended sources only need a `pixel_scale`, not an optical train.
- The YAML syntax (`!Star`, `!qty`, `!Coord`) is not supported; call the
  functions from Python instead.

Each row above has a test in `scopesim_templates/tests/test_utils/test_targets_parity.py`.
When scopesim-targets gains a new target type, add a row here and a test there.
