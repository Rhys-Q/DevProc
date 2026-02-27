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
load_torch_model = _ops.load_torch_model
load_tokenizer = _ops.load_tokenizer
tokenize_encode = _ops.tokenize_encode
tokenize_decode = _ops.tokenize_decode
prefill = _ops.prefill
decode = _ops.decode

# DevProc Backend (Triton GPU)
from devproc.backend.triton import TritonCompiler, TritonRuntime, TritonCompiledProgram

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
    "load_torch_model",
    "load_tokenizer",
    "tokenize_encode",
    "tokenize_decode",
    "prefill",
    "decode",
    # Backend
    "TritonCompiler",
    "TritonRuntime",
    "TritonCompiledProgram",
]
