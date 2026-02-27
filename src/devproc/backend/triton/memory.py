"""
GPU Memory Management for Triton Backend.

Provides GPU memory pool and tensor management.
"""

from typing import Dict, Optional, Tuple
import torch


class GPUMemoryPool:
    """GPU memory pool for tensor allocation."""

    def __init__(self, device_id: int = 0):
        """Initialize the memory pool.

        Args:
            device_id: CUDA device ID.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.allocated_tensors: Dict[str, torch.Tensor] = {}
        self.total_allocated = 0

    def allocate(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate a GPU tensor.

        Args:
            name: Name for the tensor.
            shape: Tensor shape.
            dtype: Torch data type.

        Returns:
            The allocated tensor.
        """
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.allocated_tensors[name] = tensor
        self.total_allocated += tensor.element_size() * tensor.nelement()
        return tensor

    def deallocate(self, name: str) -> None:
        """Deallocate a tensor by name.

        Args:
            name: Name of the tensor.
        """
        if name in self.allocated_tensors:
            tensor = self.allocated_tensors[name]
            self.total_allocated -= tensor.element_size() * tensor.nelement()
            del self.allocated_tensors[name]

    def clear(self) -> None:
        """Clear all allocated tensors."""
        self.allocated_tensors.clear()
        self.total_allocated = 0


class TensorManager:
    """Manages CPU-GPU tensor transfers and allocation."""

    def __init__(self, memory_pool: GPUMemoryPool):
        """Initialize the tensor manager.

        Args:
            memory_pool: The GPU memory pool to use.
        """
        self.memory_pool = memory_pool
        self.transfer_cache: Dict[str, torch.Tensor] = {}

    def upload(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Upload a CPU tensor to GPU.

        Args:
            name: Name for the tensor.
            tensor: CPU tensor to upload.

        Returns:
            The GPU tensor.
        """
        # Determine device
        if tensor.is_cuda:
            device = tensor.device
        else:
            device = self.memory_pool.device

        # Allocate on target device
        gpu_tensor = self.memory_pool.allocate(
            name,
            tensor.shape,
            tensor.dtype,
        )

        # Copy data
        gpu_tensor.copy_(tensor.to(device))
        return gpu_tensor

    def download(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """Download a GPU tensor to CPU.

        Args:
            gpu_tensor: GPU tensor to download.

        Returns:
            CPU tensor copy.
        """
        return gpu_tensor.cpu().clone()

    def allocate_output(
        self, name: str, shape: Tuple[int, ...], dtype: str
    ) -> torch.Tensor:
        """Allocate an output tensor on GPU.

        Args:
            name: Name for the tensor.
            shape: Tensor shape.
            dtype: Data type string.

        Returns:
            GPU tensor.
        """
        torch_dtype = self._convert_dtype(dtype)
        return self.memory_pool.allocate(name, shape, torch_dtype)

    @staticmethod
    def _convert_dtype(dtype: str) -> torch.dtype:
        """Convert dtype string to torch.dtype.

        Args:
            dtype: Data type string.

        Returns:
            Torch data type.
        """
        dtype_map = {
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bool": torch.bool,
        }
        return dtype_map.get(dtype, torch.float32)

    @staticmethod
    def convertTorchToStr(dtype: torch.dtype) -> str:
        """Convert torch.dtype to dtype string.

        Args:
            dtype: Torch data type.

        Returns:
            Data type string.
        """
        str_map = {
            torch.uint8: "uint8",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.bool: "bool",
        }
        return str_map.get(dtype, "float32")
