"""
IR Verifier for DevProc.

Validates IR correctness:
- SSA property: each Value is defined exactly once
- Def-use integrity: all used Values are defined
- Type consistency: shape/dtype match
- Device validity: device must be cpu or cuda
"""

from typing import List, Set
from devproc.ir.function import Function
from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType, ScalarType, StringType, TokenizerType, DictType


class IRVerifier:
    """
    Verifies IR correctness.

    Usage:
        verifier = IRVerifier(function)
        if verifier.verify():
            print("IR is valid")
        else:
            for error in verifier.get_errors():
                print(f"Error: {error}")
    """

    def __init__(self, function: Function):
        """
        Args:
            function: The Function to verify
        """
        self.function = function
        self.errors: List[str] = []

    def verify(self) -> bool:
        """
        Run all verification checks.

        Returns:
            True if IR is valid, False otherwise
        """
        self.errors = []

        self._check_ssa()
        self._check_def_use()
        self._check_types()
        self._check_devices()

        return len(self.errors) == 0

    def _check_ssa(self) -> None:
        """Check SSA property: each Value is defined exactly once."""
        defined_values: Set[str] = set()

        # Check function inputs (they are defined)
        for input_val in self.function.inputs:
            if input_val.name in defined_values:
                self.errors.append(
                    f"SSA violation: input '{input_val.name}' defined multiple times"
                )
            defined_values.add(input_val.name)

        # Check operations' outputs
        for op in self.function.block.ops:
            for output in op.outputs:
                if output.name in defined_values:
                    self.errors.append(
                        f"SSA violation: value '{output.name}' defined multiple times "
                        f"(at op: {op.name})"
                    )
                defined_values.add(output.name)

    def _check_def_use(self) -> None:
        """Check that all used Values are defined."""
        defined_values: Set[str] = set()

        # Function inputs are defined
        for input_val in self.function.inputs:
            defined_values.add(input_val.name)

        # Check each operation's inputs
        for op in self.function.block.ops:
            # First, register this op's outputs as defined
            for output in op.outputs:
                defined_values.add(output.name)

            # Then check inputs are defined
            for input_val in op.inputs:
                if input_val.name not in defined_values:
                    self.errors.append(
                        f"Def-use error: value '{input_val.name}' used in op '{op.name}' "
                        f"but never defined"
                    )

    def _check_types(self) -> None:
        """Check type consistency."""
        for op in self.function.block.ops:
            # Check each operation's type constraints
            if op.name == "normalize":
                # normalize: output should be float32
                if len(op.outputs) != 1:
                    self.errors.append(
                        f"normalize should have exactly 1 output, got {len(op.outputs)}"
                    )
                    continue

                output = op.outputs[0]
                if not isinstance(output.type, TensorType):
                    self.errors.append(
                        f"normalize output must be TensorType, got {type(output.type)}"
                    )
                    continue

                if output.type.dtype != "float32":
                    self.errors.append(
                        f"normalize output dtype must be float32, got {output.type.dtype}"
                    )

            elif op.name == "argmax":
                # argmax: output should be scalar int64
                if len(op.outputs) != 1:
                    self.errors.append(
                        f"argmax should have exactly 1 output, got {len(op.outputs)}"
                    )
                    continue

                output = op.outputs[0]
                if not isinstance(output.type, ScalarType):
                    self.errors.append(
                        f"argmax output must be ScalarType, got {type(output.type)}"
                    )
                    continue

                if output.type.dtype != "int64":
                    self.errors.append(
                        f"argmax output dtype must be int64, got {output.type.dtype}"
                    )

            elif op.name == "matmul":
                # matmul: check 2D tensors
                if len(op.inputs) != 2:
                    self.errors.append(f"matmul requires 2 inputs, got {len(op.inputs)}")
                    continue

                a_type = op.inputs[0].type
                b_type = op.inputs[1].type

                if not isinstance(a_type, TensorType) or not isinstance(b_type, TensorType):
                    self.errors.append("matmul requires TensorType inputs")
                    continue

                if len(a_type.shape) != 2 or len(b_type.shape) != 2:
                    self.errors.append("matmul requires 2D tensors")
                    continue

                # Check K dimension matches
                if a_type.shape[1] != b_type.shape[0]:
                    self.errors.append(
                        f"matmul dimension mismatch: {a_type.shape[1]} != {b_type.shape[0]}"
                    )

            elif op.name == "relu":
                # relu: output shape/dtype should match input
                if len(op.inputs) != 1 or len(op.outputs) != 1:
                    self.errors.append("relu requires 1 input and 1 output")
                    continue

                input_type = op.inputs[0].type
                output_type = op.outputs[0].type

                if not isinstance(input_type, TensorType) or not isinstance(output_type, TensorType):
                    self.errors.append("relu requires TensorType")
                    continue

                if input_type.shape != output_type.shape:
                    self.errors.append(
                        f"relu: input shape {input_type.shape} != output shape {output_type.shape}"
                    )

            # ========== Tokenizer Operations ==========

            elif op.name == "load_tokenizer":
                # load_tokenizer: no inputs, output should be TokenizerType
                if len(op.outputs) != 1:
                    self.errors.append("load_tokenizer should have 1 output")
                    continue

                output = op.outputs[0]
                if not isinstance(output.type, TokenizerType):
                    self.errors.append(
                        f"load_tokenizer output must be TokenizerType, got {type(output.type)}"
                    )

            elif op.name == "tokenize_encode":
                # tokenize_encode: inputs = TokenizerType + StringType, output = TensorType int32
                if len(op.inputs) != 2:
                    self.errors.append("tokenize_encode requires 2 inputs")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("tokenize_encode requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, TokenizerType):
                    self.errors.append("tokenize_encode input 0 must be TokenizerType")
                if not isinstance(op.inputs[1].type, StringType):
                    self.errors.append("tokenize_encode input 1 must be StringType")

                output = op.outputs[0]
                if not isinstance(output.type, TensorType):
                    self.errors.append(
                        f"tokenize_encode output must be TensorType, got {type(output.type)}"
                    )
                elif output.type.dtype != "int32":
                    self.errors.append(
                        f"tokenize_encode output dtype must be int32, got {output.type.dtype}"
                    )

            elif op.name == "tokenize_decode":
                # tokenize_decode: inputs = TokenizerType + TensorType, output = StringType
                if len(op.inputs) != 2:
                    self.errors.append("tokenize_decode requires 2 inputs")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("tokenize_decode requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, TokenizerType):
                    self.errors.append("tokenize_decode input 0 must be TokenizerType")
                if not isinstance(op.inputs[1].type, TensorType):
                    self.errors.append("tokenize_decode input 1 must be TensorType")

                output = op.outputs[0]
                if not isinstance(output.type, StringType):
                    self.errors.append(
                        f"tokenize_decode output must be StringType, got {type(output.type)}"
                    )

            # ========== String Operations ==========

            elif op.name == "string_length":
                # string_length: input = StringType, output = ScalarType int32
                if len(op.inputs) != 1:
                    self.errors.append("string_length requires 1 input")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("string_length requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, StringType):
                    self.errors.append("string_length input must be StringType")

                output = op.outputs[0]
                if not isinstance(output.type, ScalarType):
                    self.errors.append("string_length output must be ScalarType")

            elif op.name == "string_concat":
                # string_concat: inputs = StringType + StringType, output = StringType
                if len(op.inputs) != 2:
                    self.errors.append("string_concat requires 2 inputs")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("string_concat requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, StringType):
                    self.errors.append("string_concat input 0 must be StringType")
                if not isinstance(op.inputs[1].type, StringType):
                    self.errors.append("string_concat input 1 must be StringType")

                output = op.outputs[0]
                if not isinstance(output.type, StringType):
                    self.errors.append("string_concat output must be StringType")

            elif op.name == "string_slice":
                # string_slice: input = StringType, output = StringType
                if len(op.inputs) != 1:
                    self.errors.append("string_slice requires 1 input")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("string_slice requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, StringType):
                    self.errors.append("string_slice input must be StringType")

                output = op.outputs[0]
                if not isinstance(output.type, StringType):
                    self.errors.append("string_slice output must be StringType")

            elif op.name == "string_format":
                # string_format: inputs = StringType + args, output = StringType
                if len(op.inputs) < 1:
                    self.errors.append("string_format requires at least 1 input")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("string_format requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, StringType):
                    self.errors.append("string_format input 0 must be StringType")

                output = op.outputs[0]
                if not isinstance(output.type, StringType):
                    self.errors.append("string_format output must be StringType")

            # ========== Dict Operations ==========

            elif op.name == "dict_create":
                # dict_create: no inputs, output = DictType
                if len(op.outputs) != 1:
                    self.errors.append("dict_create requires 1 output")
                    continue

                output = op.outputs[0]
                if not isinstance(output.type, DictType):
                    self.errors.append(
                        f"dict_create output must be DictType, got {type(output.type)}"
                    )

            elif op.name == "dict_get":
                # dict_get: inputs = DictType + key, output = value type
                if len(op.inputs) != 2:
                    self.errors.append("dict_get requires 2 inputs")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("dict_get requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, DictType):
                    self.errors.append("dict_get input 0 must be DictType")

            elif op.name == "dict_set":
                # dict_set: inputs = DictType + key + value, output = DictType
                if len(op.inputs) != 3:
                    self.errors.append("dict_set requires 3 inputs")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("dict_set requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, DictType):
                    self.errors.append("dict_set input 0 must be DictType")

                output = op.outputs[0]
                if not isinstance(output.type, DictType):
                    self.errors.append("dict_set output must be DictType")

            elif op.name == "dict_size":
                # dict_size: input = DictType, output = ScalarType int32
                if len(op.inputs) != 1:
                    self.errors.append("dict_size requires 1 input")
                    continue
                if len(op.outputs) != 1:
                    self.errors.append("dict_size requires 1 output")
                    continue

                if not isinstance(op.inputs[0].type, DictType):
                    self.errors.append("dict_size input must be DictType")

                output = op.outputs[0]
                if not isinstance(output.type, ScalarType):
                    self.errors.append("dict_size output must be ScalarType")

    def _check_devices(self) -> None:
        """Check device validity."""
        valid_devices = {"cpu", "cuda"}

        # Check all values in the function
        for input_val in self.function.inputs:
            if isinstance(input_val.type, TensorType):
                if input_val.type.device not in valid_devices:
                    self.errors.append(
                        f"Invalid device '{input_val.type.device}' for value '{input_val.name}'"
                    )

        for op in self.function.block.ops:
            for output in op.outputs:
                if isinstance(output.type, TensorType):
                    if output.type.device not in valid_devices:
                        self.errors.append(
                            f"Invalid device '{output.type.device}' for value '{output.name}'"
                        )

    def get_errors(self) -> List[str]:
        """Get list of verification errors."""
        return self.errors
