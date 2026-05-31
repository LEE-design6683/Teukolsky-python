# Teukolsky for Python

Python implementation of Teukolsky-equation solvers for Kerr perturbations.

This repository ports the main public interface of the Black Hole
Perturbation Toolkit Mathematica `Teukolsky` package into a Python package,
with additional engineering around testing and GPU execution.

## Features

- radial solvers: `NumericalIntegration`, `MST`, `SasakiNakamura`, `HeunC`
- point-particle mode solvers for circular, spherical,
  eccentric-equatorial, and generic bound Kerr orbits
- PN helper APIs and symbolic utilities
- optional PyTorch-based GPU acceleration for the expensive source-convolution
  path in generic and eccentric-equatorial modes
- validation helpers for CPU vs GPU timing and accuracy checks

## Scope

### What this package covers

| Component | Supported range |
|---|---|
| Spin weights for point-particle modes | `s = -2, -1, 0, 1, 2` |
| Orbit types | circular equatorial, spherical, eccentric equatorial, generic bound Kerr |
| Radial methods | `NumericalIntegration`, `MST`, `SasakiNakamura`, `HeunC` |
| `SasakiNakamura` | only `s = -2` |
| Explicit radial `Domain` option | only `NumericalIntegration` and `SasakiNakamura` |
| Static modes (`omega = 0`) | supported, but explicit `Domain` is not supported |
| Derivative orders for numerical radial objects | `1`, `2`, and `4` |

### PN and symbolic layer

| Component | Supported range |
|---|---|
| PN radial objects | `TeukolskyRadialPN`, `TeukolskyRadialFunctionPN` |
| PN point-particle modes | `TeukolskyPointParticleModePN` supports circular equatorial orbits |
| Symbolic tools | `SeriesTake`, `SeriesCollect`, `MSTCoefficients`, `InvariantWronskian`, `TeukolskyEquation`, `TeukolskyPointParticleSource`, and related helpers |

### Current library limits

| Area | Current limit |
|---|---|
| Point-particle source implementation | only `s = -2, -1, 0, 1, 2` |
| `SasakiNakamura` radial solver | only `s = -2` |
| Explicit finite radial domains | not supported for `MST`, `HeunC`, or static modes |
| PN point-particle modes | only circular equatorial orbits |
| Public accelerated mode solve | only `generic` and `eccentric-equatorial` point-particle modes |

## Installation

### CPU-only install

```bash
python -m pip install -e .
```

Core dependencies:

- `numpy`
- `scipy`
- `mpmath`
- `sympy`

Python requirement:

- `>=3.10`

### GPU / DCU install

Acceleration uses the PyTorch `torch.cuda` device interface.

That means the same accelerated code path can run in:

- ROCm / DCU environments where `torch.cuda.is_available()` is `True`
- CUDA GPU environments where `torch.cuda.is_available()` is `True`

Install a PyTorch build that matches your hardware platform, then install this
package:

```bash
python -m pip install torch
python -m pip install -e .
```

## Quick start

Solve a point-particle mode on CPU:

```python
from teukolsky import KerrGeoOrbit, TeukolskyPointParticleMode

orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
mode = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)

print(mode.amplitudes)
print(mode.fluxes)
```

Use the accelerated path:

```python
from teukolsky import KerrGeoOrbit
from teukolsky.modes import solve_point_particle_mode

orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
mode = solve_point_particle_mode(
    -2,
    2,
    2,
    orbit,
    accelerator="gpu",
    device_id=0,
)

print(mode["Acceleration"])
```

Supported accelerator spellings:

- `accelerator="gpu"`: preferred public interface
- `accelerator="dcu"`: backward-compatible alias

## Acceleration support

### What is accelerated

| Path | Status |
|---|---|
| Generic bound-orbit point-particle source convolution | accelerated |
| Eccentric-equatorial point-particle source convolution | accelerated |
| Public accelerated entrypoint | `solve_point_particle_mode(..., accelerator="gpu")` |
| Low-level validation helpers | `dcu_status()`, `gpu_status()`, `dcu_execution_report()`, `benchmark_mode()` |

### Current acceleration limits

| Limit | Status |
|---|---|
| Accelerated orbit kinds | only `generic` and `eccentric-equatorial` |
| Accelerated circular modes | not implemented |
| Accelerated spherical modes | not implemented |

### Runtime checks

```python
from teukolsky.accelerated import dcu_status, dcu_execution_report

print(dcu_status())
print(dcu_execution_report())
```

## Reproducible GPU environment

A dedicated conda environment file is included for GPU benchmarking:

```bash
conda env create -f environment.gpu.yml
conda activate teukolsky-gpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

This repository was validated in a dedicated environment named
`teukolsky-gpu`.

## Benchmarking

Run the predefined benchmark cases:

```bash
python scripts/benchmark_gpu.py --case all
```

Available cases:

- `generic`
- `eccentric`
- `all`

Validated example results on `NVIDIA GeForce RTX 5060` with CUDA 12.8:

- generic case (`a=0.5, p=10, e=0.2, x=0.7`):
  CPU `16.54s`, GPU `4.33s`, speedup `3.82x`
- eccentric-equatorial case (`a=0.5, p=10, e=0.2, x=1.0`):
  CPU `2.18s`, GPU `1.75s`, speedup `1.24x`
- amplitude agreement for both cases:
  `I` and `H` relative differences at `~1e-16`

Interpretation:

- generic modes benefit substantially from GPU acceleration
- eccentric-equatorial modes remain accurate, but the speedup is smaller
  because the integral is only one-dimensional

## Testing

Run the full test suite:

```bash
python -m pytest -q
```

GPU-focused checks:

```bash
python -m pytest -q tests/test_dcu_validation.py
python scripts/benchmark_gpu.py --case all
```

## Package layout

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

Important modules:

- `src/teukolsky/modes/point_particle.py`: public point-particle mode solver
- `src/teukolsky/accelerated/convolution.py`: GPU source-convolution kernels
- `src/teukolsky/accelerated/validation.py`: CPU vs GPU validation helpers
- `scripts/benchmark_gpu.py`: reproducible benchmark entry point

## Upstream and attribution

The original Mathematica package is part of the Black Hole Perturbation
Toolkit:

- https://bhptoolkit.org/Teukolsky/
- https://github.com/BlackHolePerturbationToolkit/Teukolsky

This repository is a standalone Python port and engineering implementation of
the public `Teukolsky` package interface. The original attribution and MIT
license are preserved.

## License

MIT. See [LICENSE](LICENSE).
