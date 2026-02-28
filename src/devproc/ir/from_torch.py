"""
Torch FX Frontend.

Provides utilities to convert torch.nn.Module or Python functions to DevProc IR
using torchdynamo for tracing.
"""

from typing import Union, Callable, Tuple, Any, Optional
import torch

# Try importing torchdynamo (older) or torch._dynamo (newer)
try:
    import torchdynamo
except ImportError:
    import torch._dynamo as torchdynamo

from devproc.ir.function import Function
from devproc.ir.fx_converter import FXToIRConverter


class DevProcDynamoBackend:
    """
    Custom dynamo backend for DevProc.

    This backend captures the FX graph from torchdynamo and converts it to
    DevProc IR. It can be used with torch.compile() or torchdynamo.optimize().

    Usage:
        # Using torch.compile
        compiled = torch.compile(model, backend="devproc")
        result = compiled(x)

        # Using from_torch
        ir = devproc.from_torch(model, x)
    """

    def __init__(self):
        self.graph_module: Optional[torch.fx.GraphModule] = None
        self.ir_function: Optional[Function] = None
        self.example_inputs: Optional[Tuple[Any, ...]] = None

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...]
    ) -> "DevProcCompiledProgram":
        """
        Called by dynamo when compiling.

        Args:
            gm: FX GraphModule from dynamo
            example_inputs: Example inputs used for tracing

        Returns:
            Compiled program wrapper
        """
        self.graph_module = gm
        self.example_inputs = example_inputs

        # Convert FX to IR
        converter = FXToIRConverter()
        self.ir_function = converter.convert(gm, example_inputs)

        # Compile the IR
        from devproc.backend.triton import TritonCompiler
        compiler = TritonCompiler()
        compiled_program = compiler.compile(self.ir_function)

        return DevProcCompiledProgram(compiled_program)


class DevProcCompiledProgram:
    """Wrapper for compiled DevProc program."""

    def __init__(self, compiled_program):
        self._compiled = compiled_program

    def __call__(self, *args, **kwargs):
        """Execute the compiled program."""
        return self._compiled.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Execute the compiled program."""
        return self._compiled.run(*args, **kwargs)


def from_torch(
    target: Union[torch.nn.Module, Callable],
    *example_inputs: Any,
    backend: str = "devproc",
) -> Union[Function, "DevProcCompiledProgram"]:
    """
    Convert torch.nn.Module or Python function to DevProc IR.

    This function uses torchdynamo to trace the model/function and converts
    the resulting FX graph to DevProc IR.

    Args:
        target: torch.nn.Module or Python function to convert
        *example_inputs: Example inputs for tracing
        backend: Either "devproc" to get compiled program, or "ir" to get IR

    Returns:
        If backend="devproc": Compiled program that can be called
        If backend="ir": IR Function

    Examples:
        # Convert a model and get compiled program
        model = torch.nn.Linear(128, 256)
        compiled = devproc.from_torch(model, torch.randn(1, 128))
        result = compiled(torch.randn(1, 128).cuda())

        # Convert and get IR
        ir = devproc.from_torch(model, torch.randn(1, 128), backend="ir")
        print(ir)
    """
    if backend == "devproc":
        # Use torch.compile with custom backend to get compiled program
        compiled = torch.compile(target, backend=DevProcDynamoBackend())
        return compiled(*example_inputs)

    elif backend == "ir":
        # Capture the graph and convert to IR
        captured = {"gm": None, "inputs": example_inputs}

        class CaptureBackend:
            def __call__(self, gm, example_inputs):
                captured["gm"] = gm
                # Return a dummy function
                return lambda *args: args[0] if args else None

        compiled = torch.compile(target, backend=CaptureBackend())
        compiled(*example_inputs)

        if captured["gm"] is None:
            raise ValueError("Failed to capture FX graph")

        # Convert to IR
        converter = FXToIRConverter()
        return converter.convert(captured["gm"], captured["inputs"])

    else:
        raise ValueError(f"Unknown backend: {backend}")


# Register as a torch.compile backend
def devproc_backend(target: Union[torch.nn.Module, Callable]):
    """
    Decorator to compile a model/function with DevProc.

    Usage:
        @devproc_backend
        def model(x):
            return torch.nn.functional.relu(x)

        result = model(x)
    """
    compiled = torch.compile(target, backend="devproc")
    return compiled
