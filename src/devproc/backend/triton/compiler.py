"""
Triton Compiler and Runtime.

Compiles IR to Triton kernels and executes them.
"""

from typing import Dict, Any, List, Optional
import torch
import json
import logging

from devproc.ir.function import Function
from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType
from devproc.backend.base import Backend, CompiledProgram
from devproc.backend.triton.memory import GPUMemoryPool, TensorManager
from devproc.backend.triton.ops import TritonLoweringContext
from devproc.backend.triton.codegen import TritonKernelSpec

logger = logging.getLogger(__name__)


class TritonCompiledProgram(CompiledProgram):
    """Compiled Triton program with AOT support."""

    def __init__(
        self,
        ir_function: Function,
        kernels: List[Any],
        tensor_allocations: Dict[str, Any],
        device_id: int = 0,
        aot_kernels: Optional[List[Any]] = None,
        launcher: Optional[Any] = None,
    ):
        """Initialize the compiled program.

        Args:
            ir_function: The original IR function.
            kernels: List of kernel specifications.
            tensor_allocations: Tensor allocation info.
            device_id: CUDA device ID.
            aot_kernels: List of AOT compiled kernels (optional).
            launcher: Kernel launcher instance (optional).
        """
        self.ir_function = ir_function
        self.kernels = kernels
        self.tensor_allocations = tensor_allocations
        self.device_id = device_id
        self.aot_kernels = aot_kernels or []
        self.launcher = launcher
        self.memory_pool = GPUMemoryPool(device_id)
        self.tensor_manager = TensorManager(self.memory_pool)

    def run(self, **kwargs) -> List[torch.Tensor]:
        """Execute the compiled program.

        Args:
            **kwargs: Input tensors.

        Returns:
            List of output tensors.
        """
        # Try AOT execution first if launcher is available
        if self.launcher is not None and self.aot_kernels:
            return self._run_aot(**kwargs)

        # Fallback to JIT execution
        return self._run_jit(**kwargs)

    def _run_aot(self, **kwargs) -> List[torch.Tensor]:
        """Execute using AOT compiled kernels with CUDA Driver API."""
        # For now, fall back to JIT as AOT has issues
        logger.warning("AOT execution not fully working, falling back to JIT")
        return self._run_jit(**kwargs)

        # Prepare inputs
        inputs = {}
        for name, tensor in kwargs.items():
            if tensor.is_cuda:
                inputs[name] = tensor
            else:
                inputs[name] = tensor.to(f"cuda:{self.device_id}")

        # Register input tensors
        input_ptrs = {}
        for name, tensor in inputs.items():
            ptr, size = self.launcher.allocate_from_tensor(name, tensor)
            input_ptrs[name] = (ptr, size)

        # Allocate output tensors
        output_tensors = {}
        output_ptrs = {}
        for name, (shape, dtype_str) in self.tensor_allocations.items():
            dtype = self._str_to_dtype(dtype_str)
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            output_tensors[name] = tensor
            ptr, size = self.launcher.allocate_from_tensor(name, tensor)
            output_ptrs[name] = (ptr, size)

        # Execute kernels
        for aot_kernel in self.aot_kernels:
            # Build kernel arguments
            args = []

            # Add arguments based on signature
            for param_name, param_dtype in aot_kernel.signature:
                if param_name in input_ptrs:
                    # Tensor argument - pass as (tensor, size) tuple
                    args.append(input_ptrs[param_name])
                elif param_name in output_ptrs:
                    # Tensor argument - pass as (tensor, size) tuple
                    args.append(output_ptrs[param_name])
                elif param_name == "n_elements":
                    # Scalar - pass as int
                    # Find from input tensors
                    n_elements = 1
                    for name, tensor in inputs.items():
                        n_elements = tensor.numel()
                        break
                    args.append(n_elements)
                elif param_name == "BLOCK_SIZE":
                    # Scalar - pass as int
                    args.append(128)
                elif param_name in ("M", "N", "K"):
                    # Matmul dimensions - extract from input tensors
                    if param_name == "M" and inputs:
                        for name, tensor in inputs.items():
                            args.append(tensor.shape[0])
                            break
                    elif param_name == "N" and len(inputs) > 1:
                        for name, tensor in list(inputs.items())[1:]:
                            args.append(tensor.shape[0])
                            break
                    elif param_name == "K" and inputs:
                        for name, tensor in inputs.items():
                            args.append(tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0])
                            break
                    else:
                        args.append(1)
                elif "stride" in param_name:
                    # Stride values - extract from input tensors
                    if inputs:
                        for name, tensor in inputs.items():
                            strides = list(tensor.stride())
                            if param_name == "stride_am":
                                args.append(strides[0] if len(strides) > 0 else 1)
                            elif param_name == "stride_ak":
                                args.append(strides[1] if len(strides) > 1 else 1)
                            elif param_name == "stride_bk":
                                args.append(strides[0] if len(strides) > 0 else 1)
                            elif param_name == "stride_bn":
                                args.append(strides[1] if len(strides) > 1 else 1)
                            elif param_name == "stride_cm":
                                args.append(1)
                            elif param_name == "stride_cn":
                                args.append(1)
                            break
                    else:
                        args.append(1)
                else:
                    # Unknown parameter, skip
                    logger.warning(f"Unknown kernel parameter: {param_name}")

            # Launch kernel
            try:
                self.launcher.launch(
                    aot_kernel.name,
                    grid=aot_kernel.grid,
                    block=aot_kernel.block,
                    args=args,
                )
            except Exception as e:
                logger.warning(f"AOT kernel launch failed for {aot_kernel.name}, trying JIT: {e}")
                # Fallback to JIT if AOT fails
                return self._run_jit(**kwargs)

        # Synchronize
        self.launcher.synchronize()

        # Collect outputs
        output_names = [out.name for out in self.ir_function.block.ops[-1].outputs]
        outputs = []
        for name in output_names:
            if name in output_tensors:
                outputs.append(output_tensors[name].cpu())

        return outputs

    def _run_jit(self, **kwargs) -> List[torch.Tensor]:
        """Execute using JIT compiled kernels (fallback)."""
        # Prepare input tensors
        inputs = {}
        for name, tensor in kwargs.items():
            if tensor.is_cuda:
                inputs[name] = tensor
            else:
                inputs[name] = tensor.to(f"cuda:{self.device_id}")

        # If we have ir_function, use the handler-based execution
        if self.ir_function:
            return self._run_jit_with_ir(inputs)

        # Otherwise, use aot_kernels directly
        return self._run_jit_direct(inputs)

    def _run_jit_with_ir(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Execute using IR function and handlers."""
        ctx = TritonLoweringContext(self.device_id)

        # Map IR function inputs to kwargs
        for input_param in self.ir_function.inputs:
            param_name = input_param.name
            if param_name in inputs:
                tensor = inputs[param_name]
            else:
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

        # Execute kernels using handlers at runtime
        for op in self.ir_function.block.ops:
            self._execute_op_jit(op, ctx)

        # Collect outputs
        if self.ir_function.block.ops:
            last_op = self.ir_function.block.ops[-1]
            output_names = [out.name for out in last_op.outputs]
        else:
            output_names = list(self.tensor_allocations.keys())

        outputs = []
        for name in output_names:
            tensor = ctx.get_tensor(name)
            if tensor is not None:
                outputs.append(tensor.cpu())

        return outputs

    def _run_jit_direct(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Execute using aot_kernels directly when ir_function is not available."""
        output_tensors = {}
        output_names = []

        # Allocate output tensors
        for name, (shape, dtype_str) in self.tensor_allocations.items():
            dtype = self._str_to_dtype(dtype_str)
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            output_tensors[name] = tensor
            output_names.append(name)

        # Execute each aot_kernel directly
        for aot_kernel in self.aot_kernels:
            kernel_fn = aot_kernel.kernel_fn
            if kernel_fn is None:
                logger.warning(f"No kernel function for {aot_kernel.name}")
                continue

            # Build arguments based on signature
            args = []
            for param_name, param_dtype in aot_kernel.signature:
                if param_name in inputs:
                    # Input tensor
                    args.append(inputs[param_name])
                elif param_name in output_tensors:
                    # Output tensor
                    args.append(output_tensors[param_name])
                elif param_name == "n_elements":
                    # Compute n_elements from first input
                    for inp in inputs.values():
                        args.append(inp.numel())
                        break
                elif param_name == "BLOCK_SIZE":
                    args.append(128)
                else:
                    # Default value
                    args.append(1)

            # Determine grid
            grid = aot_kernel.grid
            if grid == (1, 1, 1):
                # Compute grid from input size
                for inp in inputs.values():
                    n_elements = inp.numel()
                    grid = ((n_elements + 127) // 128, 1, 1)
                    break

            # Launch kernel
            try:
                kernel_fn[grid](*args)
            except Exception as e:
                logger.warning(f"Failed to execute kernel {aot_kernel.name}: {e}")

        # Collect outputs
        outputs = []
        for name in output_names:
            outputs.append(output_tensors[name].cpu())

        return outputs

    def _execute_op_jit(self, op: Op, ctx) -> None:
        """Execute a single operation using JIT."""
        from devproc.backend.triton.ops import (
            handle_normalize,
            handle_relu,
            handle_sigmoid,
            handle_softmax,
            handle_argmax,
            handle_matmul,
            handle_linear,
            handle_add,
            handle_to,
            handle_resize,
            handle_transpose,
        )

        handlers = {
            "normalize": handle_normalize,
            "relu": handle_relu,
            "sigmoid": handle_sigmoid,
            "softmax": handle_softmax,
            "argmax": handle_argmax,
            "matmul": handle_matmul,
            "linear": handle_linear,
            "add": handle_add,
            "to": handle_to,
            "resize": handle_resize,
            "transpose": handle_transpose,
        }

        handler = handlers.get(op.name)
        if handler is None:
            logger.warning(f"No handler for op: {op.name}")
            return

        try:
            spec = handler(op, ctx, self.device_id)
            # The handler registers the kernel in ctx
            # Execute any registered kernels
            if hasattr(ctx, "kernel_specs"):
                for kernel_spec in ctx.kernel_specs:
                    if kernel_spec.kernel_fn is not None:
                        kernel_spec.kernel_fn()
        except Exception as e:
            logger.warning(f"JIT execution failed for {op.name}: {e}")
            raise

    def _str_to_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def export(self, path: str) -> None:
        """Export the compiled program to files.

        Args:
            path: Base path for export (will create path.cubin and path.meta.json)
        """
        from devproc.backend.triton.serialization import SerializationManager

        # Export using serialization manager
        SerializationManager.export(
            kernels=self.aot_kernels,
            output_path=path,
            tensor_allocations=self.tensor_allocations,
            device_id=self.device_id,
        )

        logger.info(f"Exported compiled program to {path}")

    @staticmethod
    def load(path: str, device_id: int = 0) -> "TritonCompiledProgram":
        """Load a compiled program from files.

        Args:
            path: Base path for the compiled program
            device_id: CUDA device ID.

        Returns:
            Loaded TritonCompiledProgram instance.
        """
        from devproc.backend.triton.serialization import SerializationManager
        from devproc.backend.triton.runtime import KernelLauncher

        # Load kernels and metadata
        kernels, metadata = SerializationManager.load(path, device_id=device_id)

        # Reconstruct tensor allocations
        tensor_allocations = {}
        for name, info in metadata.tensor_allocations.items():
            tensor_allocations[name] = (tuple(info["shape"]), info["dtype"])

        # Reconstruct kernel_fn from templates
        for kernel in kernels:
            if kernel.kernel_fn is None:
                kernel.kernel_fn = _reconstruct_kernel_fn(kernel.name)

        # Create launcher and register kernels
        launcher = None
        if kernels:
            try:
                launcher = KernelLauncher(device_id)
                for kernel in kernels:
                    if kernel.cubin:
                        launcher.register_kernel(
                            name=kernel.name,
                            cubin=kernel.cubin,
                            grid=kernel.grid,
                            block=kernel.block,
                            num_warps=kernel.num_warps,
                            num_stages=kernel.num_stages,
                        )
            except Exception as e:
                logger.warning(f"Failed to create launcher: {e}")

        # Create compiled program
        program = TritonCompiledProgram(
            ir_function=None,  # IR not preserved in load
            kernels=kernels,
            tensor_allocations=tensor_allocations,
            device_id=device_id,
            aot_kernels=kernels,
            launcher=launcher,
        )

        logger.info(f"Loaded compiled program from {path}")
        return program


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
        # Import AOT compiler
        from devproc.backend.triton.aot import AOTCompiler, AOTCompiledKernel
        from devproc.backend.triton.codegen import TritonKernelSpec

        # Try AOT compilation first
        aot_kernels = []
        try:
            aot_compiler = AOTCompiler(self.device_id)
            aot_kernels = aot_compiler.compile_function(ir_function)
            logger.info(f"AOT compiled {len(aot_kernels)} kernels")
        except Exception as e:
            logger.warning(f"AOT compilation failed: {e}")
            aot_kernels = []

        # Analyze IR and generate kernels (for JIT fallback)
        kernels = []
        tensor_allocations = {}

        # Track value names to output values
        output_values = {}

        # Process each operation - just collect metadata for now
        # Kernel execution (JIT) will be done at runtime
        for op in ir_function.block.ops:
            # Collect tensor allocations
            for output_val in op.outputs:
                if isinstance(output_val.type, TensorType):
                    shape = output_val.type.shape
                    dtype = output_val.type.dtype
                    tensor_allocations[output_val.name] = (shape, dtype)
                    output_values[output_val.name] = output_val

            # Create a simple kernel spec with op info
            # The actual kernel_fn will be created at runtime
            kernel_spec = TritonKernelSpec(
                name=op.name,
                grid=(1,),
                num_warps=4,
                num_stages=2,
                kernel_fn=None,  # Will be created at runtime
                args=(),
                kwargs={},
            )
            kernels.append(kernel_spec)

        return TritonCompiledProgram(
            ir_function=ir_function,
            kernels=kernels,
            tensor_allocations=tensor_allocations,
            device_id=self.device_id,
            aot_kernels=aot_kernels,
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


def _reconstruct_kernel_fn(kernel_name: str):
    """Reconstruct kernel function from template based on kernel name."""
    from devproc.backend.triton.templates import elementwise, matmul, reduce

    # Map kernel names to template functions
    kernel_map = {
        "normalize_kernel": elementwise.normalize_kernel,
        "relu_kernel": elementwise.relu_kernel,
        "sigmoid_kernel": elementwise.sigmoid_kernel,
        "to_dtype_kernel": elementwise.to_dtype_kernel,
        "add_kernel": elementwise.add_kernel,
        "matmul_kernel": matmul.matmul_kernel,
        "softmax_kernel": reduce.softmax_kernel,
        "argmax_kernel": reduce.argmax_kernel,
    }

    return kernel_map.get(kernel_name)


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
