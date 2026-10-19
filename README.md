# ScopeSim Templates Deux

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**An interim fork of [ScopeSim_Templates](https://github.com/AstarVienna/ScopeSim_Templates)**,
made so its refactor and new templates can be used before the upstream review is done. The two
packages will be reconciled later (or this one discarded). It installs as `scopesim-templates-deux` and
imports as `scopesim_templates_deux`, so it can sit next to the official package without clashing.

What is different from upstream 0.9 is summarised in
[`docs/audit/release-notes-0.10.0-draft.md`](docs/audit/release-notes-0.10.0-draft.md)
and the audit in [`docs/audit/`](docs/audit/).

This package contains a number of templates to generate `Source` objects to be used in simulations with the [ScopeSim Simulator](https://scopesim.readthedocs.io/en/latest/)

## Installation

Install straight from GitHub:

``` bash
pip install git+https://github.com/astronomyk/scopesim-templates-deux
```

To install the development version you can clone the repository

``` bash
git clone https://github.com/astronomyk/scopesim-templates-deux
cd scopesim-templates-deux
pip install -e .
```

Then use it like the original, with the new import name:

``` python
from scopesim_templates_deux.stellar import star_field
src = star_field(n=100, mmin=15, mmax=20, width=10, seed=42)
```

## Requirements

- numpy
- astropy
- [synphot](https://synphot.readthedocs.io/en/latest/index.html)
- [ScopeSim](https://scopesim.readthedocs.io)
- [speXtra](https://spextra.readthedocs.io)
- [pyckles](https://pyckles.readthedocs.io)

## The `Source` object

The above functions are created to ease the creation of standard sources but the power of the `Source` object doesn't end there and can be used to create endless source possibilities.

In a nutshell a `Source` object contains a spatial and spectral description of the sources. The spectral description is in the form of `synphot` spectra and the spatial description can be an `astropy` table referencing the spectra or a fits image. `Source` can also accept data cubes. The `speXtra` package contains an extensive library of spectral templates that can be used with the sources. Please check the relevant documentation.

## `Source` templates included

All templates return a `scopesim.Source`. Positions are offsets in arcsec from
the field centre. Stochastic templates take a `seed` argument (an `int` or a
`numpy.random.Generator`); without it every call gives a new realisation (see
[Units and conventions](docs/conventions.md)).

- `stellar`
  - `star`, `stars`: one or more stars with given spectral types and magnitudes
  - `star_field`: a field of stars at random positions
  - `star_grid`: a field of stars on a regular grid
  - `binary`: a binary star, with the separation in arcsec or AU
  - `planetary_system`, `exoplanet`: a star (or binary) with planets of given contrast
  - `protoplanetary_disk`: a young star with a ringed disk, by default like HL Tau
  - `cluster`: a young (age = 0) cluster of a given mass
  - `globular_cluster`: a globular cluster around a central IMBH, with time evolution
  - `omega_cen`, `evolve_omega_cen`: an ω Centauri model (King profile, Jeans
    velocities, IMBH, RGB stars) and its Keplerian time evolution
  - `galactic_centre`: the Gillessen+2017 S-stars around Sgr A* at any epoch,
    optionally with a background population (`background_mass=`)
  - `galactic_centre_background`: a hypothetical unseen main-sequence population
    around Sgr A* (up to the GRAVITY 2022 limit of ~3000 Msun)
  - `s301`, `s301_orbit`: the S301 star (Mang et al. 2026) at any epoch, for either
    orbit solution

- `extragalactic`
  - `galaxy`: a Sérsic model with a user-selected SED from the `speXtra` database
  - `galaxy3d`: like `galaxy`, plus a velocity field and velocity dispersion field
  - `spiral_two_component`: a two-component spiral with a young disk SED and an old bulge SED
  - `elliptical`: a Sérsic model using the Brown et al. (2014) SEDs
  - `hubble_ultra_deep_field`: the HUDF from published catalogues as Sérsic
    profiles, optionally with star-forming clumps
  - `galaxy_with_supernova`: a host galaxy with a supernova (speXtra SN templates)
  - `galaxy_group`: several Sérsic galaxies with their own SEDs and positions
  - `einstein_ring`: a lens galaxy with a ray-traced Einstein ring and lensed AGN images

- `misc`
  - `point_source`: like `star`, but using any SED from the `speXtra` database
  - `uniform_source`: a uniform source with any SED from `speXtra`
  - `source_from_array`: a source from a 2D `numpy` array
  - `source_from_file`: a source from an image in a FITS file
  - `source_from_imagehdu`: a source from an `ImageHDU` with an arbitrary flux and scale
  - `source_from_imagehdu_with_flux`: a source from an `ImageHDU` whose flux per pixel is known
  - `source_from_cube`: a source from a 3D datacube
  - `nebulosity`: a procedural, semi-realistic nebula image
  - `nebula_source`: a `Source` of an emission nebula (`nebulosity` image, HII-region line spectrum)
  - `circumstellar_disk`: a uniform, optionally inclined ring, e.g. a debris disk
  - `gaussian_source`: an elliptical Gaussian blob

- `calibration`
  - `lamp`: a calibration lamp, i.e. a homogeneous source with emission lines
  - `flat_field`: a flat field
  - `empty_sky`: an empty sky with no sources

- `micado`, `metis`: instrument-specific calibration sources and test fields
  (flat lamps, pinhole masks, line lists, lasers, VIKING fields)

Every `sed` argument also accepts `"blackbody:<T>"` and `"file:<path>"`. If you
are used to [scopesim-targets](https://github.com/AstarVienna/scopesim-targets),
see [Coming from scopesim-targets](docs/from_scopesim_targets.md).

Please see the documentation for how to use each source, and contact us (raise
an issue or submit a pull request) if more specialized sources are needed.
