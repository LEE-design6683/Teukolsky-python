# Teukolsky for Python

Python implementation of Teukolsky-equation solvers for Kerr perturbations.

This repository ports the main public functionality of the
Black Hole Perturbation Toolkit Mathematica `Teukolsky` package into a Python
package with:

- radial solvers: `NumericalIntegration`, `MST`, `SasakiNakamura`, `HeunC`
- point-particle mode solvers for circular, spherical, eccentric-equatorial,
  and generic bound Kerr orbits
- PN helper APIs and symbolic utilities
- optional DCU acceleration for the expensive source-convolution path on
  generic and eccentric-equatorial modes

## Install

```bash
pip install -e .
```

Dependencies:

- `numpy`
- `scipy`
- `mpmath`
- `sympy`

## Quick Start

```python
from teukolsky import KerrGeoOrbit, TeukolskyPointParticleMode

orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
mode = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)

print(mode.amplitudes)
print(mode.fluxes)
```

DCU acceleration:

```python
from teukolsky import KerrGeoOrbit
from teukolsky.modes import solve_point_particle_mode

orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
mode = solve_point_particle_mode(-2, 2, 2, orbit, accelerator="dcu", device_id=0)

print(mode["Acceleration"])
```

## Package Layout

```text
src/teukolsky/
├── accelerated/
├── angular/
├── geodesics/
├── modes/
├── mst/
├── pn/
└── radial/
```

## Tests

```bash
python -m pytest -q
```

## Upstream

The original Mathematica package is part of the
Black Hole Perturbation Toolkit:

- https://bhptoolkit.org/Teukolsky/

This repository is a Python port and keeps the original MIT license.

## License

MIT. See [LICENSE](LICENSE).
