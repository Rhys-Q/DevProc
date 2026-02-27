"""
Matrix Multiplication Kernel Templates for Triton.

Triton kernels for matmul and linear operations with autotuning.
"""

import torch
import triton
import triton.language as tl


# Autotune configurations for different matrix sizes
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=2),
]


@triton.autotune(
    key=["M", "N", "K"],
    configs=AUTOTUNE_CONFIGS,
)
@triton.jit
def matmul_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matrix multiplication kernel with autotuning.

    Computes C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    """
    # Get program ID
    pid = tl.program_id(axis=0)

    # Number of program ids along the M and N axis
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Number of proids in a group
    num_pid_in_group = num_pid_m * num_pid_n

    # Group id
    group_id = pid // num_pid_in_group

    # Id within the group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Load pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load a and b
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_K,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_K,
            other=0.0,
        )

        # Compute dot product
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    # Convert to output dtype and store
    c = accumulator.to(tl.float32)

    # Compute output offsets
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Store result
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    key=["M", "N", "K"],
    configs=AUTOTUNE_CONFIGS,
)
@triton.jit
def linear_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Matrix dimensions
    M,  # batch size
    N,  # output features
    K,  # input features
    # Strides
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Linear layer kernel (with optional bias).

    Computes: output = input @ weight.T + bias (if bias is provided)
    - input is (M, K)
    - weight is (N, K)
    - bias is (N,)
    - output is (M, N)
    """
    # Get program ID
    pid = tl.program_id(axis=0)

    # Number of program ids along the M and N axis
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n

    # Group id
    group_id = pid // num_pid_in_group

    # Id within the group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Load pointers
    # input: (M, K), weight: (N, K) -> use weight.T
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load input and weight
        input = tl.load(
            input_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_K,
            other=0.0,
        )
        weight = tl.load(
            weight_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_K,
            other=0.0,
        )

        # Compute dot product
        accumulator += tl.dot(input, weight)

        # Advance pointers
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n)
        accumulator += bias

    # Convert to output dtype and store
    output = accumulator.to(tl.float32)

    # Compute output offsets
    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Store result
    output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    tl.store(output_ptrs, output, mask=(offs_om[:, None] < M) & (offs_on[None, :] < N))


# Helper functions to launch kernels


def launch_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Launch matmul kernel.

    Args:
        a: Input tensor A (M, K)
        b: Input tensor B (K, N)
        c: Output tensor C (M, N)
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {K} vs {K2}"

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )


def launch_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, output: torch.Tensor):
    """Launch linear kernel.

    Args:
        input: Input tensor (M, K)
        weight: Weight tensor (N, K)
        bias: Bias tensor (N,) or None
        output: Output tensor (M, N)
    """
    M, K = input.shape
    N = weight.shape[0]
    assert weight.shape == (N, K), f"Incompatible weight shape: {weight.shape}"

    has_bias = bias is not None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
        input,
        weight,
        bias if bias is not None else 0,  # Pass null pointer if no bias
        output,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_BIAS=has_bias,
    )
