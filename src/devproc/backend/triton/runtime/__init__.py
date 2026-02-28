"""
Runtime components for executing AOT-compiled kernels.

Provides CUDA Driver API wrapper and kernel launcher.
"""

from devproc.backend.triton.runtime.cuda_driver import (
    CUDADriver,
    get_driver,
    init_cuda,
    CUDARuntimeError,
)

from devproc.backend.triton.runtime.kernel_launcher import (
    KernelLauncher,
    AOTKernel,
    KernelSpec,
    compute_grid_for_tensor,
    compute_block_for_warps,
)

__all__ = [
    # CUDA Driver
    "CUDADriver",
    "get_driver",
    "init_cuda",
    "CUDARuntimeError",
    # Kernel Launcher
    "KernelLauncher",
    "AOTKernel",
    "KernelSpec",
    "compute_grid_for_tensor",
    "compute_block_for_warps",
]
