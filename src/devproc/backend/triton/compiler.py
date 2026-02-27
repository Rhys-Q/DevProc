"""
Triton Compiler and Runtime.

Compiles IR to Triton kernels and executes them.
"""

from typing import Dict, Any, List, Optional
import torch

from devproc.ir.function import Function
from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType
from devproc.backend.base import Backend, CompiledProgram
from devproc.backend.triton.memory import GPUMemoryPool, TensorManager
from devproc.backend.triton.ops import TritonLoweringContext


class TritonCompiledProgram(CompiledProgram):
    """Compiled Triton program."""

    def __init__(
        self,
        ir_function: Function,
        kernels: List[Any],
        tensor_allocations: Dict[str, Any],
        device_id: int = 0,
    ):
        """Initialize the compiled program.

        Args:
            ir_function: The original IR function.
            kernels: List of kernel specifications.
            tensor_allocations: Tensor allocation info.
            device_id: CUDA device ID.
        """
        self.ir_function = ir_function
        self.kernels = kernels
        self.tensor_allocations = tensor_allocations
        self.device_id = device_id
        self.memory_pool = GPUMemoryPool(device_id)
        self.tensor_manager = TensorManager(self.memory_pool)

    def run(self, **kwargs) -> List[torch.Tensor]:
        """Execute the compiled program.

        Args:
            **kwargs: Input tensors.

        Returns:
            List of output tensors.
        """
        # Prepare input tensors
        inputs = {}
        for name, tensor in kwargs.items():
            if tensor.is_cuda:
                inputs[name] = tensor
            else:
                inputs[name] = tensor.to(f"cuda:{self.device_id}")

        # Allocate and copy inputs
        ctx = TritonLoweringContext(self.device_id)

        for input_param in self.ir_function.inputs:
            param_name = input_param.name
            if param_name in inputs:
                tensor = inputs[param_name]
            else:
                # Use default tensor from kwargs if name doesn't match
                # Try to find by position
                continue

            # Allocate GPU tensor
            gpu_tensor = self.tensor_manager.allocate_output(
                param_name,
                tensor.shape,
                TensorManager.convertTorchToStr(tensor.dtype),
            )
            gpu_tensor.copy_(tensor)
            ctx.set_tensor(param_name, gpu_tensor)

        # Allocate output tensors
        for name, (shape, dtype) in self.tensor_allocations.items():
            if name not in ctx.tensor_map:
                output_tensor = self.tensor_manager.allocate_output(name, shape, dtype)
                ctx.set_tensor(name, output_tensor)

        # Execute kernels in order
        for kernel in self.kernels:
            kernel.kernel_fn()

        # Collect outputs
        output_names = [out.name for out in self.ir_function.block.ops[-1].outputs]
        outputs = []
        for name in output_names:
            tensor = ctx.get_tensor(name)
            if tensor is not None:
                outputs.append(tensor.cpu())

        return outputs

    def export(self, path: str) -> None:
        """Export the compiled program to a .so file.

        Args:
            path: Path to export the .so file.
        """
        raise NotImplementedError(".so export not implemented yet")

    @staticmethod
    def load(path: str, device_id: int = 0) -> "TritonCompiledProgram":
        """Load a compiled program from a .so file.

        Args:
            path: Path to the .so file.
            device_id: CUDA device ID.

        Returns:
            Loaded TritonCompiledProgram instance.
        """
        raise NotImplementedError(".so load not implemented yet")


class TritonCompiler(Backend):
    """Triton backend compiler."""

    def __init__(self, device_id: int = 0):
        """Initialize the compiler.

        Args:
            device_id: CUDA device ID.
        """
        self.device_id = device_id

    @property
    def name(self) -> str:
        return "triton"

    def is_available(self) -> bool:
        """Check if Triton is available."""
        try:
            import triton

            return torch.cuda.is_available()
        except ImportError:
            return False

    def compile(self, ir_function: Function) -> TritonCompiledProgram:
        """Compile an IR Function to Triton.

        Args:
            ir_function: The IR Function to compile.

        Returns:
            Compiled Triton program.
        """
        # Analyze IR and generate kernels
        kernels = []
        tensor_allocations = {}

        # Track value names to output values
        output_values = {}

        # Process each operation
        for op in ir_function.block.ops:
            # Get or create lowering context
            ctx = TritonLoweringContext(self.device_id)

            # Register tensor placeholders
            # (will be filled at runtime)
            for input_val in op.inputs:
                if input_val.name not in ctx.tensor_map:
                    ctx.tensor_map[input_val.name] = None

            for output_val in op.outputs:
                if isinstance(output_val.type, TensorType):
                    shape = output_val.type.shape
                    dtype = output_val.type.dtype
                    tensor_allocations[output_val.name] = (shape, dtype)
                    output_values[output_val.name] = output_val

            # Generate kernel (placeholder for now, will be filled at runtime)
            from devproc.backend.triton.codegen import TritonKernelSpec

            kernel_spec = TritonKernelSpec(
                name=op.name,
                grid=(1,),
                num_warps=4,
                num_stages=2,
                kernel_fn=lambda: None,
                args=(),
                kwargs={},
            )
            kernels.append(kernel_spec)

        return TritonCompiledProgram(
            ir_function=ir_function,
            kernels=kernels,
            tensor_allocations=tensor_allocations,
            device_id=self.device_id,
        )


class TritonRuntime:
    """Triton runtime for executing compiled programs."""

    def __init__(self, device_id: int = 0):
        """Initialize the runtime.

        Args:
            device_id: CUDA device ID.
        """
        self.device_id = device_id
        self.compiler = TritonCompiler(device_id)
        self.compiled_program: Optional[TritonCompiledProgram] = None

    def build(self, ir_function: Function) -> "TritonRuntime":
        """Build the runtime from an IR function.

        Args:
            ir_function: The IR function to build.

        Returns:
            Self.
        """
        self.compiled_program = self.compiler.compile(ir_function)
        return self

    def __call__(self, **kwargs) -> List[torch.Tensor]:
        """Execute the compiled program.

        Args:
            **kwargs: Input tensors.

        Returns:
            List of output tensors.
        """
        if self.compiled_program is None:
            raise RuntimeError("Must call build() first")

        # Ensure inputs are on the correct device
        inputs = {}
        for name, tensor in kwargs.items():
            if tensor.is_cuda:
                inputs[name] = tensor
            else:
                inputs[name] = tensor.to(f"cuda:{self.device_id}")

        return self.compiled_program.run(**inputs)


# Convenience function for quick usage
def compile(ir_function: Function, device_id: int = 0) -> TritonCompiledProgram:
    """Compile an IR function to Triton.

    Args:
        ir_function: The IR function.
        device_id: CUDA device ID.

    Returns:
        Compiled program.
    """
    compiler = TritonCompiler(device_id)
    return compiler.compile(ir_function)


def run(ir_function: Function, device_id: int = 0, **kwargs) -> List[torch.Tensor]:
    """Compile and run an IR function.

    Args:
        ir_function: The IR function.
        device_id: CUDA device ID.
        **kwargs: Input tensors.

    Returns:
        List of output tensors.
    """
    program = compile(ir_function, device_id)
    return program.run(**kwargs)
