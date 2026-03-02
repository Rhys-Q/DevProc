"""
CPU Backend for DevProc.

Provides CPU code generation and execution for DevProc IR.
"""

from devproc.backend.cpu.codegen import CPUCodeGenerator, CPULoweringContext

__all__ = ["CPUCodeGenerator", "CPULoweringContext"]
