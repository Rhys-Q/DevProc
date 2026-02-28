"""
AOT Compilation Core Logic.

Handles compilation of IR operations to AOT kernels.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import torch
import triton
import logging

from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType
from devproc.ir.function import Function

logger = logging.getLogger(__name__)


@dataclass
class AOTCompiledKernel:
    """AOT compiled kernel with binary and metadata."""

    name: str
    # Compiled binary (cubin)
    cubin: Optional[bytes] = None
    # PTX code (fallback)
    ptx: Optional[str] = None
    # Kernel function (JIT compiled - for fallback execution)
    kernel_fn: Optional[Any] = None
    # Grid configuration
    grid: Tuple[int, int, int] = (1, 1, 1)
    block: Tuple[int, int, int] = (128, 1, 1)
    num_warps: int = 4
    num_stages: int = 2
    shared_memory: int = 0
    # Signature: list of (param_name, dtype_str)
    signature: List[Tuple[str, str]] = field(default_factory=list)
    # Constants extracted from IR
    constants: Dict[str, Any] = field(default_factory=dict)
    # Input/output names
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    # Shape information for allocation
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_dtypes: Dict[str, str] = field(default_factory=dict)


class AOTCompiler:
    """AOT Compiler for Triton kernels."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.compiled_kernels: List[AOTCompiledKernel] = []

        # Import kernel templates
        self._load_kernel_templates()

    def _load_kernel_templates(self) -> None:
        """Load kernel templates from templates directory."""
        # These are JIT kernels that we'll use to extract cubin
        from devproc.backend.triton.templates import elementwise, matmul, reduce

        self._elementwise_templates = elementwise
        self._matmul_templates = matmul
        self._reduce_templates = reduce

        # Map op names to template functions
        self._op_to_template = {
            "normalize": (elementwise.normalize_kernel, "elementwise"),
            "relu": (elementwise.relu_kernel, "elementwise"),
            "sigmoid": (elementwise.sigmoid_kernel, "elementwise"),
            "to": (elementwise.to_dtype_kernel, "elementwise"),
            "add": (elementwise.add_kernel, "elementwise"),
            "matmul": (matmul.matmul_kernel, "matmul"),
            "linear": (matmul.linear_kernel, "matmul") if hasattr(matmul, "linear_kernel") else None,
            "softmax": (reduce.softmax_kernel, "reduce"),
            "argmax": (reduce.argmax_kernel, "reduce"),
        }

    def compile_op(self, op: Op, input_tensors: Dict[str, torch.Tensor]) -> AOTCompiledKernel:
        """Compile a single IR operation to AOT kernel.

        Args:
            op: The IR operation
            input_tensors: Dictionary of input tensors for shape inference

        Returns:
            AOTCompiledKernel with compiled binary
        """
        op_name = op.name

        # Get template kernel
        template_info = self._op_to_template.get(op_name)
        if template_info is None:
            raise ValueError(f"No template for op: {op_name}")

        kernel_fn, _ = template_info
        if kernel_fn is None:
            raise ValueError(f"Op {op_name} not yet implemented for AOT")

        # Handle Autotuner wrapper - get the underlying function
        if hasattr(kernel_fn, "fn"):
            actual_fn = kernel_fn.fn
        else:
            actual_fn = kernel_fn

        # Use the actual kernel function name (e.g., "relu_kernel") for registration
        kernel_name = actual_fn.__name__

        # Extract metadata
        aot_kernel = AOTCompiledKernel(name=kernel_name)
        aot_kernel.signature = self._extract_signature(op, input_tensors)
        aot_kernel.input_names = [v.name for v in op.inputs]
        aot_kernel.output_names = [v.name for v in op.outputs]

        # Extract output shapes and dtypes
        for val in op.outputs:
            if isinstance(val.type, TensorType):
                aot_kernel.output_shapes[val.name] = val.type.shape
                aot_kernel.output_dtypes[val.name] = val.type.dtype

        # Generate JIT compiled kernel and try to extract cubin
        try:
            compiled = self._compile_kernel_jit(kernel_fn, op, input_tensors)

            # Try to extract cubin from compiled kernel
            # Triton stores compiled artifacts in the compiled object
            if hasattr(compiled, "asm"):
                asm = compiled.asm
                if asm is not None:
                    aot_kernel.cubin = asm.get("cubin")
                    aot_kernel.ptx = asm.get("ptx")

            # Get launch config
            if hasattr(compiled, "grid"):
                aot_kernel.grid = compiled.grid
            if hasattr(compiled, "num_warps"):
                aot_kernel.num_warps = compiled.num_warps
            if hasattr(compiled, "num_stages"):
                aot_kernel.num_stages = compiled.num_stages

            # Compute block from num_warps
            aot_kernel.block = (32 * aot_kernel.num_warps, 1, 1)

        except Exception as e:
            logger.warning(f"JIT compilation for {op_name} failed: {e}")

        # Always keep the kernel_fn for JIT fallback execution
        # Use actual_fn if available (for Autotuner), otherwise use kernel_fn
        aot_kernel.kernel_fn = actual_fn if "actual_fn" in locals() else kernel_fn

        self.compiled_kernels.append(aot_kernel)
        return aot_kernel

    def _compile_kernel_jit(
        self,
        kernel_fn: Any,
        op: Op,
        input_tensors: Dict[str, torch.Tensor],
    ) -> Any:
        """Compile a kernel using Triton JIT and extract cubin from the compiled result."""
        # Determine grid and block size based on operation
        grid, num_warps, num_stages = self._compute_launch_config(op, input_tensors)

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            # Prepare all arguments including scalars
            args = self._prepare_all_args(kernel_fn, op, input_tensors, grid, num_warps, num_stages)

            if args is None:
                raise ValueError("Failed to prepare kernel arguments")

            # Launch kernel to trigger compilation and get compiled result
            compiled = kernel_fn[grid](*args)

            # Synchronize to ensure compilation is complete
            torch.cuda.synchronize(self.device_id)

            # The compiled object should now have asm with cubin and ptx
            if hasattr(compiled, "asm") and compiled.asm:
                logger.info(
                    f"AOT compiled {op.name}: cubin={bool(compiled.asm.get('cubin'))}, ptx={bool(compiled.asm.get('ptx'))}"
                )
                return compiled
            else:
                logger.warning(f"No asm in compiled kernel for {op.name}")

        except Exception as e:
            logger.warning(f"JIT compilation for {op.name} failed: {e}")

        # Fallback: return mock
        class CompiledKernel:
            def __init__(self, grid, num_warps, num_stages):
                self.grid = grid
                self.num_warps = num_warps
                self.num_stages = num_stages
                self.asm = {}

        return CompiledKernel(grid, num_warps, num_stages)

    def _prepare_all_args(
        self,
        kernel_fn: Any,
        op: Op,
        input_tensors: Dict[str, torch.Tensor],
        grid: Tuple[int, int, int],
        num_warps: int,
        num_stages: int,
    ) -> Optional[List[Any]]:
        """Prepare all arguments for kernel launch including scalars."""
        args = []

        # Get scalar values based on operation type
        op_name = op.name
        BLOCK_SIZE = 128

        if op_name in ("matmul", "linear"):
            # Matmul: a, b, c, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
            if len(op.inputs) >= 2:
                # Get input tensor a
                a_tensor = None
                for val in op.inputs:
                    if isinstance(val.type, TensorType):
                        a_tensor = input_tensors.get(val.name)
                        if a_tensor is None:
                            a_tensor = torch.empty(
                                val.type.shape, dtype=self._str_to_dtype(val.type.dtype), device="cuda"
                            )
                        break

                # Get weight tensor b
                b_tensor = None
                for val in op.inputs[1:]:
                    if isinstance(val.type, TensorType):
                        b_tensor = input_tensors.get(val.name)
                        if b_tensor is None:
                            b_tensor = torch.empty(
                                val.type.shape, dtype=self._str_to_dtype(val.type.dtype), device="cuda"
                            )
                        break

                if a_tensor is None or b_tensor is None:
                    return None

                M, K = a_tensor.shape
                N = b_tensor.shape[0]

                args.extend([a_tensor, b_tensor])  # a_ptr, b_ptr
                args.append(torch.empty(M, N, dtype=a_tensor.dtype, device="cuda"))  # c_ptr
                args.extend([M, N, K])  # M, N, K
                args.extend([a_tensor.stride(0), a_tensor.stride(1)])  # stride_am, stride_ak
                args.extend([b_tensor.stride(0), b_tensor.stride(1)])  # stride_bk, stride_bn
                args.append(torch.empty(M, N, dtype=a_tensor.dtype, device="cuda").stride(0))  # stride_cm
                args.append(1)  # stride_cn

                return args

        # Element-wise and reduce operations: input_ptr, output_ptr, n_elements, BLOCK_SIZE
        # Get input tensor
        input_tensor = None
        for val in op.inputs:
            if isinstance(val.type, TensorType):
                input_tensor = input_tensors.get(val.name)
                if input_tensor is None:
                    input_tensor = torch.empty(val.type.shape, dtype=self._str_to_dtype(val.type.dtype), device="cuda")
                break

        if input_tensor is None:
            return None

        n_elements = input_tensor.numel()

        # Output tensor
        output_tensor = torch.empty_like(input_tensor)

        # Build args based on operation
        args.append(input_tensor)
        args.append(output_tensor)
        args.append(n_elements)
        args.append(BLOCK_SIZE)

        return args

    def _compile_kernel_jit_fallback(
        self,
        kernel_fn: Any,
        op: Op,
        input_tensors: Dict[str, torch.Tensor],
        grid: Tuple[int, int, int],
        num_warps: int,
        num_stages: int,
    ) -> Any:
        """Fallback: JIT warmup to get launch config."""
        try:
            # Prepare dummy inputs for compilation
            dummy_args = self._prepare_dummy_args(kernel_fn, op, input_tensors)

            # Create output tensors on GPU
            outputs = self._create_dummy_outputs(op)

            # Launch kernel to trigger compilation
            kernel_fn[grid](
                *dummy_args,
                *outputs,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # Synchronize to ensure compilation is complete
            torch.cuda.synchronize(self.device_id)

        except Exception as e:
            logger.debug(f"Warmup compilation attempt: {e}")

        # Return compiled object without asm (JIT fallback)
        class CompiledKernel:
            def __init__(self, grid, num_warps, num_stages):
                self.grid = grid
                self.num_warps = num_warps
                self.num_stages = num_stages
                self.asm = {}

        return CompiledKernel(grid, num_warps, num_stages)

    def _get_op_shapes(self, op: Op, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Tuple]:
        """Get shapes for all tensors in an operation."""
        shapes = {}

        # Input shapes
        for val in op.inputs:
            if isinstance(val.type, TensorType):
                if val.name in input_tensors:
                    shapes[val.name] = input_tensors[val.name].shape
                else:
                    shapes[val.name] = val.type.shape

        # Output shapes
        for val in op.outputs:
            if isinstance(val.type, TensorType):
                shapes[val.name] = val.type.shape

        return shapes

    def _build_triton_signature(self, op: Op, shapes: Dict[str, Tuple]) -> Dict[str, str]:
        """Build Triton signature from operation."""
        signature = {}

        # Map dtype strings to Triton types
        dtype_to_triton = {
            "float32": "*fp32",
            "float16": "*fp16",
            "float64": "*fp64",
            "int32": "*i32",
            "int64": "*i64",
            "int16": "*i16",
            "int8": "*i8",
            "uint8": "*u8",
            "bool": "*i1",
        }

        # Add input pointers
        idx = 0
        for val in op.inputs:
            if isinstance(val.type, TensorType):
                dtype = dtype_to_triton.get(val.type.dtype, "*fp32")
                signature[idx] = dtype
                idx += 1

        # Add output pointers
        for val in op.outputs:
            if isinstance(val.type, TensorType):
                dtype = dtype_to_triton.get(val.type.dtype, "*fp32")
                signature[idx] = dtype
                idx += 1

        # Add scalar constants
        # For now, add common ones
        signature[idx] = "i32"  # n_elements or similar

        return signature

    def _compute_launch_config(
        self,
        op: Op,
        input_tensors: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[int, int, int], int, int]:
        """Compute kernel launch configuration."""
        op_name = op.name

        if op_name in ("matmul", "linear"):
            # Matmul uses 2D grid
            # Get shapes from output
            if op.outputs and isinstance(op.outputs[0].type, TensorType):
                shape = op.outputs[0].type.shape
                M, N = shape[0], shape[1]
                BLOCK_M = 128
                BLOCK_N = 256
                grid = (
                    triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
                    1,
                    1,
                )
                return grid, 8, 2

        elif op_name in ("argmax", "softmax"):
            # Reduce operations
            if op.inputs and isinstance(op.inputs[0].type, TensorType):
                shape = op.inputs[0].type.shape
                M = shape[0] if len(shape) > 1 else 1
                N = shape[-1]
                BLOCK_N = 1024
                grid = (M, 1, 1)
                return grid, 4, 2

        else:
            # Element-wise operations
            if op.inputs and isinstance(op.inputs[0].type, TensorType):
                shape = op.inputs[0].type.shape
                total_elements = 1
                for dim in shape:
                    total_elements *= dim

                BLOCK_SIZE = 128
                grid = (triton.cdiv(total_elements, BLOCK_SIZE), 1, 1)
                return grid, 4, 2

        # Default config
        return (1, 1, 1), 4, 2

    def _prepare_dummy_args(
        self,
        kernel_fn: Any,
        op: Op,
        input_tensors: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """Prepare dummy arguments for kernel compilation."""
        args = []

        # Add input tensors
        for val in op.inputs:
            if isinstance(val.type, TensorType):
                shape = val.type.shape
                dtype_str = val.type.dtype
                dtype = self._str_to_dtype(dtype_str)

                if val.name in input_tensors:
                    tensor = input_tensors[val.name]
                else:
                    # Create dummy tensor
                    tensor = torch.zeros(shape, dtype=dtype, device="cuda")

                args.append(tensor)

        return args

    def _create_dummy_outputs(self, op: Op) -> List[torch.Tensor]:
        """Create dummy output tensors for kernel compilation."""
        outputs = []

        for val in op.outputs:
            if isinstance(val.type, TensorType):
                shape = val.type.shape
                dtype_str = val.type.dtype
                dtype = self._str_to_dtype(dtype_str)

                tensor = torch.empty(shape, dtype=dtype, device="cuda")
                outputs.append(tensor)

        return outputs

    def _extract_signature(self, op: Op, input_tensors: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
        """Extract parameter signature from operation, including scalar parameters."""
        signature = []
        op_name = op.name

        # Input parameters
        for val in op.inputs:
            if isinstance(val.type, TensorType):
                signature.append((val.name, val.type.dtype))

        # Output parameters
        for val in op.outputs:
            if isinstance(val.type, TensorType):
                signature.append((val.name, val.type.dtype))

        # Add scalar parameters based on operation type
        if op_name in ("matmul", "linear"):
            # Matmul: M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
            signature.extend(
                [
                    ("M", "int32"),
                    ("N", "int32"),
                    ("K", "int32"),
                    ("stride_am", "int32"),
                    ("stride_ak", "int32"),
                    ("stride_bk", "int32"),
                    ("stride_bn", "int32"),
                    ("stride_cm", "int32"),
                    ("stride_cn", "int32"),
                ]
            )
        else:
            # Element-wise and reduce: n_elements, BLOCK_SIZE
            # Get n_elements from input tensor
            if op.inputs and isinstance(op.inputs[0].type, TensorType):
                shape = op.inputs[0].type.shape
                n_elements = 1
                for dim in shape:
                    n_elements *= dim
                signature.append(("n_elements", "int32"))
                signature.append(("BLOCK_SIZE", "int32"))

        return signature

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

    def compile_function(self, ir_function: Function) -> List[AOTCompiledKernel]:
        """Compile all operations in an IR function.

        Args:
            ir_function: The IR function to compile

        Returns:
            List of AOT compiled kernels
        """
        # For now, we'll need example tensors to compile
        # In a full implementation, we'd have shape inference
        kernels = []

        # Create dummy input tensors based on IR function signature
        input_tensors = {}
        for param in ir_function.inputs:
            if isinstance(param.type, TensorType):
                input_tensors[param.name] = torch.empty(
                    param.type.shape, dtype=self._str_to_dtype(param.type.dtype), device="cuda"
                )

        # Compile each operation
        for op in ir_function.block.ops:
            try:
                kernel = self.compile_op(op, input_tensors)
                kernels.append(kernel)
            except Exception as e:
                logger.warning(f"Failed to compile op {op.name}: {e}")
                # Continue with other operations

        return kernels


def compile_aot(ir_function: Function, device_id: int = 0) -> List[AOTCompiledKernel]:
    """Convenience function to compile IR function to AOT kernels.

    Args:
        ir_function: The IR function to compile
        device_id: CUDA device ID

    Returns:
        List of AOT compiled kernels
    """
    compiler = AOTCompiler(device_id)
    return compiler.compile_function(ir_function)
