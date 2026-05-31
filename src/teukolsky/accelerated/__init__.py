"""GPU-accelerated Teukolsky computations.

Uses PyTorch CUDA/ROCm backends for GPU-accelerated:
- Source convolution integrals (point-particle modes)
- Batch radial ODE integration
- Angular eigenvalue computation

All accelerated functions accept and return numpy arrays or Python scalars,
converting to/from torch tensors internally on the GPU.
"""

from .backend import dcu_status, gpu_status, require_dcu, require_gpu
from .convolution import accelerated_generic_alpha, accelerated_eccentric_alpha
from .radial_dcu import batch_solve_radial, batch_solve_point_particle_modes
from .validation import benchmark_mode, validate_precision, benchmark_batch_modes, dcu_execution_report

__all__ = [
    "dcu_status",
    "gpu_status",
    "require_dcu",
    "require_gpu",
    "accelerated_generic_alpha",
    "accelerated_eccentric_alpha",
    "batch_solve_radial",
    "batch_solve_point_particle_modes",
    "dcu_execution_report",
    "benchmark_mode",
    "benchmark_batch_modes",
    "validate_precision",
]
