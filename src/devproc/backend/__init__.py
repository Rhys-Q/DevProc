"""
DevProc Backend Module.

Provides backend compilation support for different targets (CPU, GPU, etc.).
"""

from devproc.backend.base import Backend, CompiledProgram

__all__ = ["Backend", "CompiledProgram"]
