"""
Triton GPU Backend for DevProc.

Compiles IR to Triton kernels for NVIDIA GPU execution.
"""

from devproc.backend.triton.compiler import TritonCompiler, TritonRuntime, TritonCompiledProgram

__all__ = ["TritonCompiler", "TritonRuntime", "TritonCompiledProgram"]
