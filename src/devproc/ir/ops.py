"""
Op definitions for DevProc IR.

Provides builders for standard operations:
- input: Pipeline input
- normalize: Normalization
- argmax: Argmax operation
- matmul, linear, relu, add: Torch export compatible ops
"""

from typing import List, Tuple, Any, Optional
from devproc.ir.base import Value, Op
from devproc.ir.types import TensorType, ScalarType, Type, StringType, TokenizerType, DictType
from devproc.ir.function import Function


class OpBuilder:
    """Builder class for creating IR operations."""

    @staticmethod
    def input(name: str, tensor_type: TensorType) -> Op:
        """
        Create an input operation (pipeline input).

        Args:
            name: Name for the input value
            tensor_type: Type of the input tensor

        Returns:
            Op with a single output
        """
        output_value = Value(name, tensor_type)
        return Op("input", [], [output_value])

    @staticmethod
    def normalize(input_value: Value) -> Op:
        """
        Create a normalize operation.

        Args:
            input_value: Input tensor value

        Returns:
            Op with normalized output (same shape/dtype as input)
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"normalize requires TensorType, got {type(input_type)}")

        # Output has same shape, converted to float32
        output_type = TensorType(
            shape=input_type.shape,
            dtype="float32",
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)
        return Op("normalize", [input_value], [output_value])

    @staticmethod
    def argmax(input_value: Value, dim: int = -1) -> Op:
        """
        Create an argmax operation.

        Args:
            input_value: Input tensor value
            dim: Dimension to reduce (default: -1)

        Returns:
            Op with scalar output (int64)
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"argmax requires TensorType, got {type(input_type)}")

        # Output is a scalar (int64)
        output_type = ScalarType("int64")
        output_value = Value(Function.generate_name(), output_type)

        # Store dim as attribute (stored in op for now)
        op = Op("argmax", [input_value], [output_value])
        op._dim = dim
        return op

    @staticmethod
    def matmul(a: Value, b: Value) -> Op:
        """
        Create a matrix multiplication operation.

        Args:
            a: First input tensor (M x K)
            b: Second input tensor (K x N)

        Returns:
            Op with output tensor (M x N), float32
        """
        a_type = a.type
        b_type = b.type

        if not isinstance(a_type, TensorType) or not isinstance(b_type, TensorType):
            raise TypeError("matmul requires TensorType inputs")

        # Compute output shape
        if len(a_type.shape) != 2 or len(b_type.shape) != 2:
            raise ValueError("matmul requires 2D tensors")

        output_shape = (a_type.shape[0], b_type.shape[1])
        output_type = TensorType(output_shape, "float32", a_type.device)

        output_value = Value(Function.generate_name(), output_type)
        return Op("matmul", [a, b], [output_value])

    @staticmethod
    def linear(input_value: Value, weight: Value, bias: Optional[Value] = None) -> Op:
        """
        Create a linear (fully connected) operation.

        Args:
            input_value: Input tensor
            weight: Weight tensor (out_features x in_features)
            bias: Optional bias tensor (out_features)

        Returns:
            Op with output tensor
        """
        input_type = input_value.type
        weight_type = weight.type

        if not isinstance(input_type, TensorType) or not isinstance(weight_type, TensorType):
            raise TypeError("linear requires TensorType inputs")

        # Output features from weight
        out_features = weight_type.shape[0]
        batch_size = input_type.shape[0] if len(input_type.shape) > 1 else 1
        output_shape = (batch_size, out_features)
        output_type = TensorType(output_shape, "float32", input_type.device)

        output_value = Value(Function.generate_name(), output_type)

        if bias is not None:
            return Op("linear", [input_value, weight, bias], [output_value])
        else:
            return Op("linear", [input_value, weight], [output_value])

    @staticmethod
    def relu(input_value: Value) -> Op:
        """
        Create a ReLU activation operation.

        Args:
            input_value: Input tensor

        Returns:
            Op with output tensor (same shape/dtype as input)
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"relu requires TensorType, got {type(input_type)}")

        output_type = TensorType(
            shape=input_type.shape,
            dtype=input_type.dtype,
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)
        return Op("relu", [input_value], [output_value])

    @staticmethod
    def add(a: Value, b: Value) -> Op:
        """
        Create an element-wise addition operation.

        Args:
            a: First input tensor
            b: Second input tensor

        Returns:
            Op with output tensor (broadcasted shape)
        """
        a_type = a.type
        b_type = b.type

        if not isinstance(a_type, TensorType) or not isinstance(b_type, TensorType):
            raise TypeError("add requires TensorType inputs")

        # Simple broadcasting: use max of each dimension
        # (simplified for MVP)
        if a_type.shape == b_type.shape:
            output_shape = a_type.shape
        else:
            raise NotImplementedError(
                "Broadcasting not fully implemented in MVP"
            )

        output_type = TensorType(output_shape, a_type.dtype, a_type.device)
        output_value = Value(Function.generate_name(), output_type)
        return Op("add", [a, b], [output_value])

    @staticmethod
    def return_op(value: Value) -> Op:
        """
        Create a return operation.

        Args:
            value: Value to return

        Returns:
            Op with no outputs
        """
        return Op("return", [value], [])

    @staticmethod
    def resize(input_value: Value, size: Tuple[int, ...]) -> Op:
        """
        Create a resize operation.

        Args:
            input_value: Input tensor value
            size: Target size (e.g., (224, 224))

        Returns:
            Op with resized output
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"resize requires TensorType, got {type(input_type)}")

        # Output shape: for 2D images, height x width
        # For 3D (HWC), becomes H x W x C
        if len(input_type.shape) == 3 and len(size) == 2:
            # HWC -> HWC (resize H, W)
            output_shape = (*size, input_type.shape[2])
        else:
            output_shape = size

        output_type = TensorType(
            shape=output_shape,
            dtype=input_type.dtype,
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)

        op = Op("resize", [input_value], [output_value])
        op._size = size
        return op

    @staticmethod
    def transpose(input_value: Value, dims: Tuple[int, ...]) -> Op:
        """
        Create a transpose operation.

        Args:
            input_value: Input tensor value
            dims: Permutation of dimensions

        Returns:
            Op with transposed output
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"transpose requires TensorType, got {type(input_type)}")

        # Compute output shape based on permutation
        original_shape = input_type.shape
        output_shape = tuple(original_shape[d] for d in dims)

        output_type = TensorType(
            shape=output_shape,
            dtype=input_type.dtype,
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)

        op = Op("transpose", [input_value], [output_value])
        op._dims = dims
        return op

    @staticmethod
    def to(input_value: Value, dtype: str, device: Optional[str] = None) -> Op:
        """
        Create a type/device conversion operation.

        Args:
            input_value: Input tensor value
            dtype: Target dtype
            device: Target device (optional)

        Returns:
            Op with converted output
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"to requires TensorType, got {type(input_type)}")

        output_device = device if device else input_type.device

        output_type = TensorType(
            shape=input_type.shape,
            dtype=dtype,
            device=output_device
        )
        output_value = Value(Function.generate_name(), output_type)
        op = Op("to", [input_value], [output_value])
        # Store dtype/device for type inference
        op._dtype = dtype
        op._device = output_device
        return op

    @staticmethod
    def sigmoid(input_value: Value) -> Op:
        """
        Create a sigmoid activation operation.

        Args:
            input_value: Input tensor

        Returns:
            Op with sigmoid output (same shape/dtype as input)
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"sigmoid requires TensorType, got {type(input_type)}")

        # Sigmoid typically outputs float32
        output_type = TensorType(
            shape=input_type.shape,
            dtype="float32",
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)
        return Op("sigmoid", [input_value], [output_value])

    @staticmethod
    def softmax(input_value: Value, dim: int = -1) -> Op:
        """
        Create a softmax activation operation.

        Args:
            input_value: Input tensor
            dim: Dimension to apply softmax

        Returns:
            Op with softmax output (same shape as input)
        """
        input_type = input_value.type
        if not isinstance(input_type, TensorType):
            raise TypeError(f"softmax requires TensorType, got {type(input_type)}")

        # Softmax typically outputs float32
        output_type = TensorType(
            shape=input_type.shape,
            dtype="float32",
            device=input_type.device
        )
        output_value = Value(Function.generate_name(), output_type)

        op = Op("softmax", [input_value], [output_value])
        op._dim = dim
        return op

    # ==================== Tokenizer Operations ====================

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> Op:
        """
        Create a load_tokenizer operation.

        Args:
            tokenizer_path: Path or name of the tokenizer

        Returns:
            Op with TokenizerType output
        """
        output_type = TokenizerType(tokenizer_path)
        output_value = Value(Function.generate_name(), output_type)

        op = Op("load_tokenizer", [], [output_value])
        op._tokenizer_path = tokenizer_path
        return op

    @staticmethod
    def tokenize_encode(tokenizer: Value, text: Value) -> Op:
        """
        Create a tokenize_encode operation (text -> token_ids).

        Args:
            tokenizer: Tokenizer value
            text: Input text value

        Returns:
            Op with TensorType output (int32, fixed length)
        """
        tokenizer_type = tokenizer.type
        if not isinstance(tokenizer_type, TokenizerType):
            raise TypeError(f"tokenize_encode requires TokenizerType, got {type(tokenizer_type)}")

        text_type = text.type
        if not isinstance(text_type, StringType):
            raise TypeError(f"tokenize_encode requires StringType input, got {type(text_type)}")

        # Output: fixed length int32 tensor (MVP uses 2048)
        max_length = 2048
        output_type = TensorType((max_length,), "int32", "cpu")
        output_value = Value(Function.generate_name(), output_type)

        op = Op("tokenize_encode", [tokenizer, text], [output_value])
        op._max_length = max_length
        return op

    @staticmethod
    def tokenize_decode(tokenizer: Value, token_ids: Value) -> Op:
        """
        Create a tokenize_decode operation (token_ids -> text).

        Args:
            tokenizer: Tokenizer value
            token_ids: Input token IDs (TensorType int32)

        Returns:
            Op with StringType output
        """
        tokenizer_type = tokenizer.type
        if not isinstance(tokenizer_type, TokenizerType):
            raise TypeError(f"tokenize_decode requires TokenizerType, got {type(tokenizer_type)}")

        token_ids_type = token_ids.type
        if not isinstance(token_ids_type, TensorType):
            raise TypeError(f"tokenize_decode requires TensorType input, got {type(token_ids_type)}")

        output_type = StringType()
        output_value = Value(Function.generate_name(), output_type)

        return Op("tokenize_decode", [tokenizer, token_ids], [output_value])

    # ==================== String Operations ====================

    @staticmethod
    def string_length(text: Value) -> Op:
        """
        Create a string_length operation.

        Args:
            text: Input text value (StringType)

        Returns:
            Op with ScalarType output (int32)
        """
        text_type = text.type
        if not isinstance(text_type, StringType):
            raise TypeError(f"string_length requires StringType, got {type(text_type)}")

        output_type = ScalarType("int32")
        output_value = Value(Function.generate_name(), output_type)

        return Op("string_length", [text], [output_value])

    @staticmethod
    def string_concat(a: Value, b: Value) -> Op:
        """
        Create a string_concat operation.

        Args:
            a: First string value
            b: Second string value

        Returns:
            Op with StringType output
        """
        a_type = a.type
        b_type = b.type

        if not isinstance(a_type, StringType):
            raise TypeError(f"string_concat requires StringType, got {type(a_type)}")
        if not isinstance(b_type, StringType):
            raise TypeError(f"string_concat requires StringType, got {type(b_type)}")

        # Compute max length
        max_len = None
        if a_type.max_length and b_type.max_length:
            max_len = a_type.max_length + b_type.max_length

        output_type = StringType(max_len)
        output_value = Value(Function.generate_name(), output_type)

        return Op("string_concat", [a, b], [output_value])

    @staticmethod
    def string_slice(text: Value, start: int, end: Optional[int] = None) -> Op:
        """
        Create a string_slice operation.

        Args:
            text: Input text value
            start: Start index
            end: End index (optional)

        Returns:
            Op with StringType output
        """
        text_type = text.type
        if not isinstance(text_type, StringType):
            raise TypeError(f"string_slice requires StringType, got {type(text_type)}")

        # Compute output length
        out_len = None
        if text_type.max_length and end:
            out_len = end - start
        elif text_type.max_length:
            out_len = text_type.max_length - start

        output_type = StringType(out_len)
        output_value = Value(Function.generate_name(), output_type)

        op = Op("string_slice", [text], [output_value])
        op._start = start
        op._end = end
        return op

    @staticmethod
    def string_format(template: Value, *args: Value) -> Op:
        """
        Create a string_format operation.

        Args:
            template: Format template string
            args: Format arguments

        Returns:
            Op with StringType output
        """
        template_type = template.type
        if not isinstance(template_type, StringType):
            raise TypeError(f"string_format requires StringType template, got {type(template_type)}")

        # Simple implementation: output length based on template
        output_type = StringType(template_type.max_length)
        output_value = Value(Function.generate_name(), output_type)

        inputs = [template] + list(args)
        return Op("string_format", inputs, [output_value])

    # ==================== Dict Operations ====================

    @staticmethod
    def dict_create(key_type: Type, value_type: Type) -> Op:
        """
        Create a dict_create operation.

        Args:
            key_type: Type of dictionary keys
            value_type: Type of dictionary values

        Returns:
            Op with DictType output
        """
        output_type = DictType(key_type, value_type)
        output_value = Value(Function.generate_name(), output_type)

        return Op("dict_create", [], [output_value])

    @staticmethod
    def dict_get(dict_val: Value, key: Value) -> Op:
        """
        Create a dict_get operation.

        Args:
            dict_val: Dictionary value
            key: Key value

        Returns:
            Op with value type output
        """
        dict_type = dict_val.type
        if not isinstance(dict_type, DictType):
            raise TypeError(f"dict_get requires DictType, got {type(dict_type)}")

        output_value = Value(Function.generate_name(), dict_type.value_type)
        return Op("dict_get", [dict_val, key], [output_value])

    @staticmethod
    def dict_set(dict_val: Value, key: Value, value: Value) -> Op:
        """
        Create a dict_set operation.

        Args:
            dict_val: Dictionary value
            key: Key value
            value: Value to set

        Returns:
            Op with DictType output
        """
        dict_type = dict_val.type
        if not isinstance(dict_type, DictType):
            raise TypeError(f"dict_set requires DictType, got {type(dict_type)}")

        # Output type is the same dict type
        output_value = Value(Function.generate_name(), dict_type)
        return Op("dict_set", [dict_val, key, value], [output_value])

    @staticmethod
    def dict_size(dict_val: Value) -> Op:
        """
        Create a dict_size operation.

        Args:
            dict_val: Dictionary value

        Returns:
            Op with ScalarType output (int32)
        """
        dict_type = dict_val.type
        if not isinstance(dict_type, DictType):
            raise TypeError(f"dict_size requires DictType, got {type(dict_type)}")

        output_type = ScalarType("int32")
        output_value = Value(Function.generate_name(), output_type)

        return Op("dict_size", [dict_val], [output_value])
