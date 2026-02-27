"""
Operation Lowering to Triton Kernels.

Maps IR operations to Triton kernel executions.
"""

from typing import Any, Dict, TYPE_CHECKING
import torch

from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType
from devproc.backend.triton.codegen import TritonKernelSpec
from devproc.backend.triton.memory import TensorManager


# Forward declaration for type hints
if TYPE_CHECKING:
    from devproc.backend.triton.ops import TritonLoweringContext


def handle_normalize(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle normalize operation.

    Args:
        op: The normalize operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    # Store kernel info
    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_normalize

        launch_normalize(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="normalize",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_relu(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle ReLU operation.

    Args:
        op: The relu operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_relu

        launch_relu(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="relu",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_sigmoid(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle sigmoid operation.

    Args:
        op: The sigmoid operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_sigmoid

        launch_sigmoid(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="sigmoid",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_softmax(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle softmax operation.

    Args:
        op: The softmax operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    M, N = input_tensor.shape

    def kernel_fn():
        from devproc.backend.triton.templates.reduce import launch_softmax

        launch_softmax(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="softmax",
        grid=(M,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_argmax(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle argmax operation.

    Args:
        op: The argmax operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    M, N = input_tensor.shape

    def kernel_fn():
        from devproc.backend.triton.templates.reduce import launch_argmax

        launch_argmax(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="argmax",
        grid=(M,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_matmul(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle matmul operation.

    Args:
        op: The matmul operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    a_tensor = ctx.get_tensor(op.inputs[0].name)
    b_tensor = ctx.get_tensor(op.inputs[1].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    M, K = a_tensor.shape
    K2, N = b_tensor.shape

    def kernel_fn():
        from devproc.backend.triton.templates.matmul import launch_matmul

        launch_matmul(a_tensor, b_tensor, output_tensor)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    spec = TritonKernelSpec(
        name="matmul",
        grid=grid,
        num_warps=8,
        num_stages=3,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_linear(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle linear (fully connected) operation.

    Args:
        op: The linear operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    weight_tensor = ctx.get_tensor(op.inputs[1].name)

    # Check if bias is provided
    has_bias = len(op.inputs) > 2
    bias_tensor = ctx.get_tensor(op.inputs[2].name) if has_bias else None

    output_tensor = ctx.get_tensor(op.outputs[0].name)

    M, K = input_tensor.shape
    N = weight_tensor.shape[0]

    def kernel_fn():
        from devproc.backend.triton.templates.matmul import launch_linear

        launch_linear(input_tensor, weight_tensor, bias_tensor, output_tensor)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    spec = TritonKernelSpec(
        name="linear",
        grid=grid,
        num_warps=8,
        num_stages=3,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_add(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle element-wise addition operation.

    Args:
        op: The add operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    a_tensor = ctx.get_tensor(op.inputs[0].name)
    b_tensor = ctx.get_tensor(op.inputs[1].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_add

        launch_add(a_tensor, b_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="add",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_to(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle type/device conversion operation.

    Args:
        op: The to operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    # Get target dtype
    output_dtype = op._dtype if hasattr(op, "_dtype") else "float32"

    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_to_dtype

        launch_to_dtype(input_tensor, output_tensor, output_dtype)

    spec = TritonKernelSpec(
        name="to",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        num_stages=2,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_resize(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle resize operation.

    Note: For MVP, resize is not implemented as a Triton kernel.
    We'll use torch.nn.functional.interpolate instead.

    Args:
        op: The resize operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)
    size = op._size if hasattr(op, "_size") else (224, 224)

    def kernel_fn():
        import torch.nn.functional as F

        # Use torch for resize (Triton doesn't have built-in resize)
        output_tensor.copy_(F.interpolate(
            input_tensor.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0))

    spec = TritonKernelSpec(
        name="resize",
        grid=(1,),
        num_warps=1,
        num_stages=1,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


def handle_transpose(
    op: Op, ctx: "TritonLoweringContext", device_id: int
) -> TritonKernelSpec:
    """Handle transpose operation.

    Args:
        op: The transpose operation.
        ctx: Lowering context.
        device_id: CUDA device ID.

    Returns:
        Kernel specification.
    """
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)
    dims = op._dims if hasattr(op, "_dims") else None

    def kernel_fn():
        # Use torch for transpose
        output_tensor.copy_(input_tensor.permute(dims))

    spec = TritonKernelSpec(
        name="transpose",
        grid=(1,),
        num_warps=1,
        num_stages=1,
        kernel_fn=kernel_fn,
        args=(),
        kwargs={},
    )
    ctx.add_kernel(spec)
    return spec


# Import triton for grid function in handle_matmul and handle_linear
import triton


class TritonLoweringContext:
    """Simplified lowering context for ops handlers."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.tensor_map: Dict[str, torch.Tensor] = {}
        self.kernel_specs = []

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        self.tensor_map[name] = tensor

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensor_map.get(name)

    def add_kernel(self, spec: TritonKernelSpec) -> None:
        self.kernel_specs.append(spec)
