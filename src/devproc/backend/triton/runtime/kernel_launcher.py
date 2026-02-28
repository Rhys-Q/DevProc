"""
Kernel Launcher for AOT-compiled kernels.

Provides high-level interface for loading and executing
AOT-compiled CUDA kernels using the CUDA Driver API.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from ctypes import c_void_p, c_int32, c_int64, c_float, c_double
import torch
import logging
import numpy as np

from devproc.backend.triton.runtime.cuda_driver import (
    CUDADriver,
    get_driver,
    CUDARuntimeError,
)

logger = logging.getLogger(__name__)


@dataclass
class KernelSpec:
    """Specification for a compiled kernel."""
    name: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    num_warps: int
    num_stages: int
    shared_memory: int = 0


class AOTKernel:
    """AOT-compiled kernel wrapper."""

    def __init__(
        self,
        name: str,
        cubin: bytes,
        function_handle: Any,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        num_warps: int = 4,
        num_stages: int = 2,
        shared_memory: int = 0,
    ):
        self.name = name
        self.cubin = cubin
        self.function_handle = function_handle
        self.grid = grid
        self.block = block
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.shared_memory = shared_memory


class KernelLauncher:
    """High-level kernel launcher using CUDA Driver API."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.driver = get_driver()

        # Initialize CUDA if not already done
        try:
            self.driver.init()
        except CUDARuntimeError:
            pass  # Already initialized

        # Get device and create context
        self.device = self.driver.get_device(device_id)

        # Try to use existing context or create new one
        self.context = self.driver.get_context()
        if self.context is None:
            self.context = self.driver.create_context(self.device)
        else:
            self.driver.set_context(self.context)

        # Track loaded modules and kernels
        self.modules: Dict[str, Any] = {}
        self.kernels: Dict[str, AOTKernel] = {}

        # Memory tracking
        self.allocated_buffers: Dict[str, Tuple[Any, int]] = {}

    def load_module(self, name: str, cubin: bytes) -> Any:
        """Load a module from cubin binary."""
        try:
            module = self.driver.load_module_from_memory(cubin)
            self.modules[name] = module
            return module
        except CUDARuntimeError as e:
            logger.error(f"Failed to load module {name}: {e}")
            raise

    def get_function(self, module: Any, name: str) -> Any:
        """Get a kernel function from a module."""
        try:
            return self.driver.get_function(module, name)
        except CUDARuntimeError as e:
            logger.error(f"Failed to get function {name}: {e}")
            raise

    def register_kernel(
        self,
        name: str,
        cubin: bytes,
        grid: Tuple[int, int, int],
        block: Optional[Tuple[int, int, int]] = None,
        num_warps: int = 4,
        num_stages: int = 2,
        shared_memory: int = 0,
    ) -> None:
        """Register a kernel for execution."""
        # Load module
        module = self.load_module(name, cubin)

        # Get function
        function = self.get_function(module, name)

        # Use default block size based on num_warps
        if block is None:
            block = (32 * num_warps, 1, 1)

        kernel = AOTKernel(
            name=name,
            cubin=cubin,
            function_handle=function,
            grid=grid,
            block=block,
            num_warps=num_warps,
            num_stages=num_stages,
            shared_memory=shared_memory,
        )

        self.kernels[name] = kernel

    def allocate_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Allocate a GPU tensor."""
        tensor = torch.empty(shape, dtype=dtype, device=device)
        size = tensor.numel() * tensor.element_size()
        self.allocated_buffers[name] = (tensor, size)
        return tensor

    def allocate_from_tensor(self, name: str, tensor: torch.Tensor) -> Tuple[Any, int]:
        """Register an existing tensor for tracking."""
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        size = tensor.numel() * tensor.element_size()
        self.allocated_buffers[name] = (tensor, size)
        return tensor, size

    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        """Get a registered buffer by name."""
        if name in self.allocated_buffers:
            return self.allocated_buffers[name][0]
        return None

    def launch(
        self,
        kernel_name: str,
        grid: Optional[Tuple[int, int, int]] = None,
        block: Optional[Tuple[int, int, int]] = None,
        shared_memory: int = 0,
        args: Optional[List[Tuple[Any, int]]] = None,
    ) -> None:
        """Launch a kernel.

        Args:
            kernel_name: Name of the kernel to launch
            grid: Grid dimensions (x, y, z)
            block: Block dimensions (x, y, z)
            shared_memory: Shared memory size in bytes
            args: List of (pointer, size) tuples for kernel arguments
        """
        kernel = self.kernels.get(kernel_name)
        if kernel is None:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # Use provided or default grid/block
        if grid is None:
            grid = kernel.grid
        if block is None:
            block = kernel.block
        if shared_memory == 0:
            shared_memory = kernel.shared_memory

        # Prepare kernel arguments
        # All arguments must be passed as pointers. For scalars, we need to
        # store them in a buffer and pass the pointer.
        if args is None:
            args = []

        # Store scalar values in a buffer
        scalar_buffers = []

        arg_ptrs = []
        for arg in args:
            if isinstance(arg, tuple):
                # Tuple of (tensor, size) or (pointer, size)
                ptr = arg[0]
                if hasattr(ptr, 'data_ptr'):
                    # It's a torch.Tensor
                    arg_ptrs.append(c_void_p(ptr.data_ptr()))
                else:
                    # It's already a pointer
                    arg_ptrs.append(ptr)
            elif isinstance(arg, torch.Tensor):
                # Direct tensor - pass pointer to its data
                arg_ptrs.append(c_void_p(arg.data_ptr()))
            elif isinstance(arg, int):
                # Integer scalar - allocate buffer and store value
                import numpy as np
                buf = np.array([arg], dtype=np.int32)
                buf_gpu = torch.from_numpy(buf).cuda()
                scalar_buffers.append(buf_gpu)
                arg_ptrs.append(c_void_p(buf_gpu.data_ptr()))
            elif isinstance(arg, float):
                # Float scalar - allocate buffer and store value
                import numpy as np
                buf = np.array([arg], dtype=np.float32)
                buf_gpu = torch.from_numpy(buf).cuda()
                scalar_buffers.append(buf_gpu)
                arg_ptrs.append(c_void_p(buf_gpu.data_ptr()))
            else:
                # Try to convert directly
                try:
                    arg_ptrs.append(c_void_p(int(arg)))
                except (TypeError, ValueError):
                    arg_ptrs.append(arg)

        # Launch kernel
        try:
            self.driver.launch_kernel(
                kernel.function_handle,
                grid,
                block,
                shared_memory,
                arg_ptrs,
            )
        except CUDARuntimeError as e:
            logger.error(f"Failed to launch kernel {kernel_name}: {e}")
            raise

    def synchronize(self) -> None:
        """Synchronize CUDA context."""
        # Note: CUDA Driver API doesn't have a direct synchronize
        # We can use a simple CUDA synchronization via torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)

    def clear(self) -> None:
        """Clear all allocated resources."""
        self.allocated_buffers.clear()
        self.kernels.clear()

        # Unload modules
        for module in self.modules.values():
            try:
                self.driver.unload_module(module)
            except Exception:
                pass
        self.modules.clear()


def compute_grid_for_tensor(
    shape: Tuple[int, ...],
    block_size: int = 128,
) -> Tuple[int, int, int]:
    """Compute grid dimensions for a tensor operation."""
    total_elements = int(np.prod(shape))
    grid_x = (total_elements + block_size - 1) // block_size
    return (grid_x, 1, 1)


def compute_block_for_warps(num_warps: int = 4) -> Tuple[int, int, int]:
    """Compute block dimensions based on number of warps."""
    return (32 * num_warps, 1, 1)
