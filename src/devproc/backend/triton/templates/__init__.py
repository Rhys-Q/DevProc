"""
Triton Kernel Templates.

Pre-defined Triton kernel templates for different operation types.
"""

from devproc.backend.triton.templates.elementwise import (
    normalize_kernel,
    relu_kernel,
    sigmoid_kernel,
    to_dtype_kernel,
    add_kernel,
)
from devproc.backend.triton.templates.matmul import matmul_kernel
from devproc.backend.triton.templates.reduce import argmax_kernel, softmax_kernel

__all__ = [
    "normalize_kernel",
    "relu_kernel",
    "sigmoid_kernel",
    "to_dtype_kernel",
    "add_kernel",
    "matmul_kernel",
    "argmax_kernel",
    "softmax_kernel",
]
