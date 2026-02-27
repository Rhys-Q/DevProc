"""
Element-wise Kernel Templates for Triton.

Triton kernels for element-wise operations.
"""

import triton
import triton.language as tl


@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize kernel - converts input to float32."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)

    # Convert to float32
    output = input.to(tl.float32)

    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU kernel - max(0, x)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)

    # ReLU
    output = tl.maximum(input, 0.0)

    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sigmoid kernel - 1 / (1 + exp(-x))."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)

    # Sigmoid
    output = tl.sigmoid(input)

    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


@triton.jit
def to_dtype_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Type conversion kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    input = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)

    # Convert dtype
    output = input.to(OUTPUT_DTYPE)

    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


@triton.jit
def add_kernel(
    input_ptr_a,
    input_ptr_b,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    a = tl.load(input_ptr_a + block_start + offsets, mask=mask, other=0.0)
    b = tl.load(input_ptr_b + block_start + offsets, mask=mask, other=0.0)

    # Add
    output = a + b

    # Store output
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


# Helper functions to launch kernels with proper configuration


def launch_normalize(input_tensor, output_tensor, BLOCK_SIZE: int = 128):
    """Launch normalize kernel."""
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    normalize_kernel[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )


def launch_relu(input_tensor, output_tensor, BLOCK_SIZE: int = 128):
    """Launch relu kernel."""
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )


def launch_sigmoid(input_tensor, output_tensor, BLOCK_SIZE: int = 128):
    """Launch sigmoid kernel."""
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sigmoid_kernel[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )


def launch_to_dtype(input_tensor, output_tensor, output_dtype, BLOCK_SIZE: int = 128):
    """Launch type conversion kernel."""
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Map dtype string to tl dtype
    dtype_map = {
        "float32": tl.float32,
        "float16": tl.float16,
        "int32": tl.int32,
        "int64": tl.int64,
    }
    tl_dtype = dtype_map.get(output_dtype, tl.float32)

    to_dtype_kernel[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE, OUTPUT_DTYPE=tl_dtype
    )


def launch_add(input_a, input_b, output_tensor, BLOCK_SIZE: int = 128):
    """Launch add kernel."""
    n_elements = input_a.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](
        input_a, input_b, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
