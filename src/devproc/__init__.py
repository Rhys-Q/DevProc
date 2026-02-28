# DevProc IR
from devproc.ir.types import Type, TensorType, ScalarType
from devproc.ir.base import Value, Op
from devproc.ir.function import Function, Block
from devproc.ir.ops import OpBuilder
from devproc.ir.verifier import IRVerifier

# DevProc DSL (Pipeline style)
from devproc.dsl.pipeline import Pipeline, Tensor as PipelineTensor

# DevProc DSL (Kernel decorator style)
from devproc.dsl.kernel import kernel, KernelTensor, parse_ir
from devproc.dsl import ops as _ops
from devproc.dsl.types import (
    String,
    Tensor,
    Float32,
    Float16,
    Int32,
    Int64,
    UInt8,
    Bool,
    Tokenizer,
    TorchModel,
    KVCache,
)

# Module-level devproc functions
load_image = _ops.load_image
input = _ops.input
resize = _ops.resize
to = _ops.to
normalize = _ops.normalize
transpose = _ops.transpose
matmul = _ops.matmul
linear = _ops.linear
relu = _ops.relu
sigmoid = _ops.sigmoid
softmax = _ops.softmax
add = _ops.add
argmax = _ops.argmax

# DevProc Backend (Triton GPU)
from devproc.backend.triton import TritonCompiler, TritonRuntime, TritonCompiledProgram

# DevProc Runtime
from devproc.runtime import Runtime

# DevProc compile function
def compile(target, *example_inputs, backend: str = "triton", **kwargs):
    """AOT 编译函数

    Args:
        target: IR Function 或 @kernel 装饰的函数
        *example_inputs: 示例输入（用于解析 IR）
        backend: 目标后端 ("triton")
        **kwargs: 后端特定配置（如 device_id）

    Returns:
        CompiledProgram 对象
    """
    # 获取 IR Function
    ir_function = None

    if hasattr(target, 'ir_function'):
        # Kernel 函数对象（需要先获取 example_inputs）
        ir_function = target.ir_function
    elif isinstance(target, Function):
        # 直接是 IR Function
        ir_function = target
    else:
        # 可能是普通函数，需要用 example_inputs 解析
        ir_function = target(*example_inputs)

    # 根据 backend 选择编译器
    if backend == "triton":
        from devproc.backend.triton import TritonCompiler as _TritonCompiler
        compiler = _TritonCompiler(**kwargs)
        return compiler.compile(ir_function)
    raise ValueError(f"Unknown backend: {backend}")

__all__ = [
    # IR
    "Type",
    "TensorType",
    "ScalarType",
    "Value",
    "Op",
    "Function",
    "Block",
    "OpBuilder",
    "IRVerifier",
    # DSL Pipeline style
    "Pipeline",
    "PipelineTensor",
    # DSL Kernel decorator style
    "kernel",
    "KernelTensor",
    "parse_ir",
    # DSL Types
    "String",
    "Tensor",
    "Float32",
    "Float16",
    "Int32",
    "Int64",
    "UInt8",
    "Bool",
    "Tokenizer",
    "TorchModel",
    "KVCache",
    # Module-level functions
    "load_image",
    "input",
    "resize",
    "to",
    "normalize",
    "transpose",
    "matmul",
    "linear",
    "relu",
    "sigmoid",
    "softmax",
    "add",
    "argmax",
    # Backend
    "TritonCompiler",
    "TritonRuntime",
    "TritonCompiledProgram",
    # Runtime and compile
    "Runtime",
    "compile",
]
