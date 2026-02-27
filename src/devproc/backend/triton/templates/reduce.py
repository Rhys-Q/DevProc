"""
Reduction Kernel Templates for Triton.

Triton kernels for reduction operations like argmax and softmax.
"""

import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Argmax kernel - computes the index of the maximum value along the last dimension.

    Args:
        input_ptr: Input tensor pointer (M, N)
        output_ptr: Output tensor pointer (M,)
        M: Number of rows
        N: Number of columns
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    # Row offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Column offsets
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize max values and indices
    max_vals = tl.zeros((BLOCK_M,), dtype=tl.float32) + float("-inf")
    max_indices = tl.zeros((BLOCK_M,), dtype=tl.int64)

    # Loop over columns
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        # Load input
        input_ptrs = input_ptr + offs_m[:, None] * N + (n * BLOCK_N + offs_n[None, :])
        input = tl.load(
            input_ptrs,
            mask=mask_m[:, None] & (n * BLOCK_N + offs_n < N)[None, :],
            other=float("-inf"),
        )

        # Compute max and indices
        current_max, current_idx = tl.max(input, axis=1, return_indices=True)

        # Update global max if needed
        update_mask = current_max > max_vals
        max_vals = tl.where(update_mask, current_max, max_vals)
        max_indices = tl.where(update_mask, n * BLOCK_N + current_idx, max_indices)

    # Store results
    output_ptrs = output_ptr + offs_m
    tl.store(output_ptrs, max_indices, mask=mask_m)


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Softmax kernel - computes softmax along the last dimension.

    Args:
        input_ptr: Input tensor pointer (M, N)
        output_ptr: Output tensor pointer (M, N)
        M: Number of rows
        N: Number of columns
    """
    pid = tl.program_id(axis=0)

    # Row offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load row data
    row_ptrs = input_ptr + offs_m[:, None] * N + offs_n[None, :]
    row = tl.load(
        row_ptrs,
        mask=mask_m[:, None] & mask_n[None, :],
        other=float("-inf"),
    )

    # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    row_minus_max = row - tl.max(row, axis=1, keep_dims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    softmax_output = numerator / denominator

    # Store results
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        output_ptrs,
        softmax_output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def reduce_sum_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum reduction kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize sum
    sums = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_N)):
        input_ptrs = input_ptr + offs_m[:, None] * N + (n * BLOCK_N + offs_n[None, :])
        input = tl.load(
            input_ptrs,
            mask=mask_m[:, None] & (n * BLOCK_N + offs_n < N)[None, :],
            other=0.0,
        )
        sums += tl.sum(input, axis=1)

    # Store results
    output_ptrs = output_ptr + offs_m
    tl.store(output_ptrs, sums, mask=mask_m)


# Helper functions to launch kernels


def launch_argmax(input_tensor, output_tensor):
    """Launch argmax kernel.

    Args:
        input_tensor: Input tensor (M, N)
        output_tensor: Output tensor (M,)
    """
    M, N = input_tensor.shape
    BLOCK_M = 16
    BLOCK_N = 1024

    grid = (M,)
    argmax_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
    )


def launch_softmax(input_tensor, output_tensor):
    """Launch softmax kernel.

    Args:
        input_tensor: Input tensor (M, N)
        output_tensor: Output tensor (M, N)
    """
    M, N = input_tensor.shape
    BLOCK_M = 1
    BLOCK_N = 1024

    grid = (M,)
    softmax_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
    )


def launch_reduce_sum(input_tensor, output_tensor):
    """Launch sum reduction kernel.

    Args:
        input_tensor: Input tensor (M, N)
        output_tensor: Output tensor (M,)
    """
    M, N = input_tensor.shape
    BLOCK_M = 16
    BLOCK_N = 1024

    grid = (M,)
    reduce_sum_kernel[grid](
        input_tensor,
        output_tensor,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
    )
