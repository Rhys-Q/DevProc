"""
Code Generator for Triton Backend.

Generates Triton kernels from IR operations.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import torch
import triton

from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType
from devproc.ir.function import Function


@dataclass
class TritonKernelSpec:
    """Specification for a Triton kernel."""

    name: str
    grid: tuple
    num_warps: int
    num_stages: int
    kernel_fn: Callable
    args: tuple
    kwargs: dict


class KernelGenerator:
    """Generates Triton kernels from IR operations."""

    def __init__(self, device_id: int = 0):
        """Initialize the kernel generator.

        Args:
            device_id: CUDA device ID.
        """
        self.device_id = device_id
        self.op_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default operation handlers."""
        from devproc.backend.triton.ops import (
            handle_normalize,
            handle_relu,
            handle_sigmoid,
            handle_softmax,
            handle_argmax,
            handle_matmul,
            handle_linear,
            handle_add,
            handle_to,
            handle_resize,
            handle_transpose,
        )

        self.op_handlers = {
            "normalize": handle_normalize,
            "relu": handle_relu,
            "sigmoid": handle_sigmoid,
            "softmax": handle_softmax,
            "argmax": handle_argmax,
            "matmul": handle_matmul,
            "linear": handle_linear,
            "add": handle_add,
            "to": handle_to,
            "resize": handle_resize,
            "transpose": handle_transpose,
        }

    def register_handler(self, op_name: str, handler: Callable) -> None:
        """Register a custom handler for an operation.

        Args:
            op_name: Name of the operation.
            handler: Handler function.
        """
        self.op_handlers[op_name] = handler

    def generate(self, op: Op, ctx: "LoweringContext") -> TritonKernelSpec:
        """Generate a Triton kernel for an IR operation.

        Args:
            op: The IR operation.
            ctx: Lowering context.

        Returns:
            Triton kernel specification.
        """
        handler = self.op_handlers.get(op.name)
        if handler is None:
            raise ValueError(f"No Triton handler for op: {op.name}")

        return handler(op, ctx, self.device_id)

    def generate_launch_config(
        self, op: Op, tensor_type: TensorType
    ) -> Dict[str, Any]:
        """Generate kernel launch configuration.

        Args:
            op: The IR operation.
            tensor_type: Input tensor type.

        Returns:
            Launch configuration dict.
        """
        # Calculate total elements
        total_elements = 1
        for dim in tensor_type.shape:
            total_elements *= dim

        # Default configuration
        BLOCK_SIZE = 128
        grid = (torch.cuda.current_device(),)
        num_warps = 4
        num_stages = 2

        # Adjust based on operation type
        if op.name in ("matmul", "linear"):
            # Matmul uses different grid
            M = tensor_type.shape[0]
            N = tensor_type.shape[1] if len(tensor_type.shape) > 1 else 1
            BLOCK_M = 128
            BLOCK_N = 256
            grid = (
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            )
            num_warps = 8

        elif op.name in ("argmax", "softmax"):
            # Reduce operations
            M = tensor_type.shape[0] if len(tensor_type.shape) > 1 else 1
            N = tensor_type.shape[-1]
            BLOCK_M = 16
            BLOCK_N = 1024
            grid = (M,)
            num_warps = 4

        else:
            # Element-wise operations
            grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        return {
            "grid": grid,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "BLOCK_SIZE": BLOCK_SIZE,
        }


class TritonLoweringContext:
    """Context for lowering IR to Triton."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.tensor_map: Dict[str, torch.Tensor] = {}
        self.kernel_specs: list = []

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Store a tensor by IR value name."""
        self.tensor_map[name] = tensor

    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Get a tensor by IR value name."""
        return self.tensor_map.get(name)

    def add_kernel(self, spec: TritonKernelSpec) -> None:
        """Add a kernel specification to execute."""
        self.kernel_specs.append(spec)

    def clear(self) -> None:
        """Clear the context."""
        self.tensor_map.clear()
        self.kernel_specs.clear()
