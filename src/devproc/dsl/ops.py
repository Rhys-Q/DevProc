"""
DevProc DSL Operations - Module-level functions for @kernel decorated functions.

These functions are designed to be used inside @devproc.kernel decorated functions.
They build IR operations instead of executing computation.
"""

from typing import Optional, Tuple, Any, Union
from devproc.ir.ops import OpBuilder
from devproc.ir.types import TensorType, StringType, TokenizerType, DictType
from devproc.ir.base import Value

from devproc.dsl.kernel import KernelContext, KernelTensor


# ==================== Input/Output Operations ====================

def load_image(path: str) -> KernelTensor:
    """
    Load an image from path.

    Note: This is a placeholder that creates a tensor input.
    The actual image loading will be handled in the runtime.

    Args:
        path: Path to the image file

    Returns:
        KernelTensor representing the loaded image
    """
    ctx = KernelContext.get_current()

    # For MVP, create a placeholder input
    # In real implementation, this would be tied to the actual image
    tensor_type = TensorType((224, 224, 3), "uint8", "cpu")
    input_op = OpBuilder.input(f"img_{path}", tensor_type)
    input_value = input_op.outputs[0]

    # Add as parameter if not already added
    if input_value.name not in ctx.param_map:
        ctx.ir_function.add_input(input_value)

    return KernelTensor(input_value, input_value.name)


def input(name: str, dtype: str, shape: Tuple[int, ...], device: str = "cpu") -> KernelTensor:
    """
    Define a kernel input.

    Args:
        name: Name of the input
        dtype: Data type (e.g., "uint8", "float32")
        shape: Tensor shape
        device: Device ("cpu" or "cuda")

    Returns:
        KernelTensor representing the input
    """
    ctx = KernelContext.get_current()

    tensor_type = TensorType(shape, dtype, device)
    input_op = OpBuilder.input(name, tensor_type)
    input_value = input_op.outputs[0]

    # Add as parameter
    ctx.ir_function.add_input(input_value)
    ctx.param_map[name] = input_value

    return KernelTensor(input_value, name)


# ==================== Type Conversion ====================

def to(tensor: KernelTensor, dtype: Union[str, type], device: Optional[str] = None) -> KernelTensor:
    """
    Convert tensor to different dtype or device.

    Args:
        tensor: Input tensor
        dtype: Target dtype (e.g., "float32", Float32)
        device: Target device (optional)

    Returns:
        Converted tensor
    """
    ctx = KernelContext.get_current()

    # Handle type annotation objects
    if isinstance(dtype, type):
        dtype_map = {
            "Float32": "float32",
            "Float16": "float16",
            "Int32": "int32",
            "Int64": "int64",
            "UInt8": "uint8",
            "Bool": "bool",
        }
        dtype = dtype_map.get(dtype.__name__, str(dtype))

    # Create IR to op
    to_op = OpBuilder.to(tensor.ir_value, dtype, device)
    to_value = to_op.outputs[0]

    # Add to function
    ctx.ir_function.add_op(to_op)

    return KernelTensor(to_value)


# ==================== Preprocessing Operations ====================

def normalize(tensor: KernelTensor) -> KernelTensor:
    """
    Normalize tensor (convert to float32).

    Args:
        tensor: Input tensor

    Returns:
        Normalized tensor (float32)
    """
    ctx = KernelContext.get_current()

    normalize_op = OpBuilder.normalize(tensor.ir_value)
    normalize_value = normalize_op.outputs[0]

    ctx.ir_function.add_op(normalize_op)

    return KernelTensor(normalize_value)


def resize(tensor: KernelTensor, size: Tuple[int, ...]) -> KernelTensor:
    """
    Resize tensor to target size.

    Args:
        tensor: Input tensor
        size: Target size (e.g., (224, 224))

    Returns:
        Resized tensor
    """
    ctx = KernelContext.get_current()

    resize_op = OpBuilder.resize(tensor.ir_value, size)
    resize_value = resize_op.outputs[0]

    ctx.ir_function.add_op(resize_op)

    return KernelTensor(resize_value)


def transpose(tensor: KernelTensor, dims: Tuple[int, ...]) -> KernelTensor:
    """
    Transpose tensor along given dimensions.

    Args:
        tensor: Input tensor
        dims: Permutation of dimensions

    Returns:
        Transposed tensor
    """
    ctx = KernelContext.get_current()

    transpose_op = OpBuilder.transpose(tensor.ir_value, dims)
    transpose_value = transpose_op.outputs[0]

    ctx.ir_function.add_op(transpose_op)

    return KernelTensor(transpose_value)


# ==================== Neural Network Operations ====================

def matmul(a: KernelTensor, b: KernelTensor) -> KernelTensor:
    """
    Matrix multiplication.

    Args:
        a: First input tensor (M x K)
        b: Second input tensor (K x N)

    Returns:
        Output tensor (M x N)
    """
    ctx = KernelContext.get_current()

    matmul_op = OpBuilder.matmul(a.ir_value, b.ir_value)
    matmul_value = matmul_op.outputs[0]

    ctx.ir_function.add_op(matmul_op)

    return KernelTensor(matmul_value)


def linear(input: KernelTensor, weight: KernelTensor, bias: Optional[KernelTensor] = None) -> KernelTensor:
    """
    Linear (fully connected) layer.

    Args:
        input: Input tensor
        weight: Weight tensor (out_features x in_features)
        bias: Optional bias tensor (out_features)

    Returns:
        Output tensor
    """
    ctx = KernelContext.get_current()

    if bias is not None:
        linear_op = OpBuilder.linear(input.ir_value, weight.ir_value, bias.ir_value)
    else:
        linear_op = OpBuilder.linear(input.ir_value, weight.ir_value)

    linear_value = linear_op.outputs[0]

    ctx.ir_function.add_op(linear_op)

    return KernelTensor(linear_value)


def relu(tensor: KernelTensor) -> KernelTensor:
    """
    ReLU activation.

    Args:
        tensor: Input tensor

    Returns:
        Output tensor with ReLU applied
    """
    ctx = KernelContext.get_current()

    relu_op = OpBuilder.relu(tensor.ir_value)
    relu_value = relu_op.outputs[0]

    ctx.ir_function.add_op(relu_op)

    return KernelTensor(relu_value)


def sigmoid(tensor: KernelTensor) -> KernelTensor:
    """
    Sigmoid activation.

    Args:
        tensor: Input tensor

    Returns:
        Output tensor with sigmoid applied
    """
    ctx = KernelContext.get_current()

    sigmoid_op = OpBuilder.sigmoid(tensor.ir_value)
    sigmoid_value = sigmoid_op.outputs[0]

    ctx.ir_function.add_op(sigmoid_op)

    return KernelTensor(sigmoid_value)


def softmax(tensor: KernelTensor, dim: int = -1) -> KernelTensor:
    """
    Softmax activation.

    Args:
        tensor: Input tensor
        dim: Dimension to apply softmax

    Returns:
        Output tensor with softmax applied
    """
    ctx = KernelContext.get_current()

    softmax_op = OpBuilder.softmax(tensor.ir_value, dim)
    softmax_value = softmax_op.outputs[0]

    ctx.ir_function.add_op(softmax_op)

    return KernelTensor(softmax_value)


def add(a: KernelTensor, b: KernelTensor) -> KernelTensor:
    """
    Element-wise addition.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Output tensor (broadcasted shape)
    """
    ctx = KernelContext.get_current()

    add_op = OpBuilder.add(a.ir_value, b.ir_value)
    add_value = add_op.outputs[0]

    ctx.ir_function.add_op(add_op)

    return KernelTensor(add_value)


# ==================== Postprocessing Operations ====================

def argmax(tensor: KernelTensor, dim: int = -1) -> KernelTensor:
    """
    Argmax operation.

    Args:
        tensor: Input tensor
        dim: Dimension to reduce

    Returns:
        Scalar tensor with argmax result (int64)
    """
    ctx = KernelContext.get_current()

    argmax_op = OpBuilder.argmax(tensor.ir_value, dim)
    argmax_value = argmax_op.outputs[0]

    ctx.ir_function.add_op(argmax_op)

    return KernelTensor(argmax_value)


# ==================== Kernel Wrapper Classes ====================

class KernelString:
    """String wrapper for DSL kernel functions."""

    def __init__(self, ir_value: Value, name: str = None):
        self._ir_value = ir_value
        self._name = name or ir_value.name

    @property
    def ir_value(self) -> Value:
        return self._ir_value

    @property
    def name(self) -> str:
        return self._name


class KernelTokenizer:
    """Tokenizer wrapper for DSL kernel functions."""

    def __init__(self, ir_value: Value, name: str = None):
        self._ir_value = ir_value
        self._name = name or ir_value.name

    @property
    def ir_value(self) -> Value:
        return self._ir_value

    @property
    def name(self) -> str:
        return self._name


class KernelDict:
    """Dict wrapper for DSL kernel functions."""

    def __init__(self, ir_value: Value, name: str = None):
        self._ir_value = ir_value
        self._name = name or ir_value.name

    @property
    def ir_value(self) -> Value:
        return self._ir_value

    @property
    def name(self) -> str:
        return self._name


# ==================== Tokenizer Operations ====================

def load_tokenizer(tokenizer_path: str) -> KernelTokenizer:
    """
    Load a tokenizer from path or name.

    Args:
        tokenizer_path: Path or name of the tokenizer (e.g., "gpt2", "./my-tokenizer")

    Returns:
        KernelTokenizer representing the loaded tokenizer
    """
    ctx = KernelContext.get_current()

    # Create IR operation
    tokenizer_op = OpBuilder.load_tokenizer(tokenizer_path)
    tokenizer_value = tokenizer_op.outputs[0]

    # Add to function
    ctx.ir_function.add_op(tokenizer_op)

    return KernelTokenizer(tokenizer_value, f"tokenizer_{tokenizer_path}")


def tokenize_encode(tokenizer: KernelTokenizer, text: KernelString) -> KernelTensor:
    """
    Encode text to token IDs.

    Args:
        tokenizer: The tokenizer
        text: Input text string

    Returns:
        KernelTensor of token IDs (int32, fixed length 2048)
    """
    ctx = KernelContext.get_current()

    encode_op = OpBuilder.tokenize_encode(tokenizer.ir_value, text.ir_value)
    encode_value = encode_op.outputs[0]

    ctx.ir_function.add_op(encode_op)

    return KernelTensor(encode_value)


def tokenize_decode(tokenizer: KernelTokenizer, token_ids: KernelTensor) -> KernelString:
    """
    Decode token IDs to text.

    Args:
        tokenizer: The tokenizer
        token_ids: Input token IDs

    Returns:
        KernelString of decoded text
    """
    ctx = KernelContext.get_current()

    decode_op = OpBuilder.tokenize_decode(tokenizer.ir_value, token_ids.ir_value)
    decode_value = decode_op.outputs[0]

    ctx.ir_function.add_op(decode_op)

    return KernelString(decode_value)


# ==================== String Operations ====================

def string_input(name: str, max_length: Optional[int] = None) -> KernelString:
    """
    Define a string input.

    Args:
        name: Name of the input
        max_length: Optional maximum length constraint

    Returns:
        KernelString representing the input
    """
    ctx = KernelContext.get_current()

    string_type = StringType(max_length)
    input_op = OpBuilder.input(name, TensorType((), "int8", "cpu"))
    input_value = input_op.outputs[0]

    # Override the type to StringType for semantic meaning
    input_value._type = string_type

    # Add as parameter
    ctx.ir_function.add_input(input_value)
    ctx.param_map[name] = input_value

    return KernelString(input_value, name)


def string_length(text: KernelString) -> KernelTensor:
    """
    Get string length.

    Args:
        text: Input string

    Returns:
        KernelTensor of length (int32 scalar)
    """
    ctx = KernelContext.get_current()

    length_op = OpBuilder.string_length(text.ir_value)
    length_value = length_op.outputs[0]

    ctx.ir_function.add_op(length_op)

    return KernelTensor(length_value)


def string_concat(a: KernelString, b: KernelString) -> KernelString:
    """
    Concatenate two strings.

    Args:
        a: First string
        b: Second string

    Returns:
        KernelString of concatenated result
    """
    ctx = KernelContext.get_current()

    concat_op = OpBuilder.string_concat(a.ir_value, b.ir_value)
    concat_value = concat_op.outputs[0]

    ctx.ir_function.add_op(concat_op)

    return KernelString(concat_value)


def string_slice(text: KernelString, start: int, end: Optional[int] = None) -> KernelString:
    """
    Slice a string.

    Args:
        text: Input string
        start: Start index
        end: End index (optional)

    Returns:
        KernelString of sliced result
    """
    ctx = KernelContext.get_current()

    slice_op = OpBuilder.string_slice(text.ir_value, start, end)
    slice_value = slice_op.outputs[0]

    ctx.ir_function.add_op(slice_op)

    return KernelString(slice_value)


def string_format(template: KernelString, *args: KernelTensor) -> KernelString:
    """
    Format a string with arguments.

    Args:
        template: Format template string
        args: Format arguments (as tensors)

    Returns:
        KernelString of formatted result
    """
    ctx = KernelContext.get_current()

    arg_values = [arg.ir_value for arg in args]
    format_op = OpBuilder.string_format(template.ir_value, *arg_values)
    format_value = format_op.outputs[0]

    ctx.ir_function.add_op(format_op)

    return KernelString(format_value)


# ==================== Dict Operations ====================

def dict_create(key_dtype: str, value_dtype: str) -> KernelDict:
    """
    Create an empty dictionary.

    Args:
        key_dtype: Data type of keys ("string" or tensor dtype like "float32")
        value_dtype: Data type of values (tensor dtype like "float32")

    Returns:
        KernelDict representing the created dictionary
    """
    ctx = KernelContext.get_current()

    # Determine key and value types
    if key_dtype == "string":
        key_type = StringType()
    else:
        key_type = TensorType((), key_dtype, "cpu")

    if value_dtype == "string":
        value_type = StringType()
    else:
        value_type = TensorType((), value_dtype, "cpu")

    dict_op = OpBuilder.dict_create(key_type, value_type)
    dict_value = dict_op.outputs[0]

    ctx.ir_function.add_op(dict_op)

    return KernelDict(dict_value)


def dict_get(d: KernelDict, key: KernelString) -> KernelTensor:
    """
    Get value from dictionary.

    Args:
        d: Dictionary
        key: Key string

    Returns:
        KernelTensor of the value
    """
    ctx = KernelContext.get_current()

    get_op = OpBuilder.dict_get(d.ir_value, key.ir_value)
    get_value = get_op.outputs[0]

    ctx.ir_function.add_op(get_op)

    return KernelTensor(get_value)


def dict_set(d: KernelDict, key: KernelString, value: KernelTensor) -> KernelDict:
    """
    Set value in dictionary.

    Args:
        d: Dictionary
        key: Key string
        value: Value tensor

    Returns:
        KernelDict (updated dictionary)
    """
    ctx = KernelContext.get_current()

    set_op = OpBuilder.dict_set(d.ir_value, key.ir_value, value.ir_value)
    set_value = set_op.outputs[0]

    ctx.ir_function.add_op(set_op)

    return KernelDict(set_value)


def dict_size(d: KernelDict) -> KernelTensor:
    """
    Get dictionary size.

    Args:
        d: Dictionary

    Returns:
        KernelTensor of size (int32 scalar)
    """
    ctx = KernelContext.get_current()

    size_op = OpBuilder.dict_size(d.ir_value)
    size_value = size_op.outputs[0]

    ctx.ir_function.add_op(size_op)

    return KernelTensor(size_value)


# ==================== VLA Model Operations ====================

def load_vla_module(model_path: str) -> "torch.nn.Module":
    """
    Load a VLA model from LeRobot and return the underlying torch.nn.Module.

    This function loads a VLA (Vision-Language-Action) model from the LeRobot
    repository and returns the underlying PyTorch model for conversion to
    DevProc IR using devproc.from_torch().

    Args:
        model_path: Path or name of the VLA model (e.g., "lerobot/pi05_base")

    Returns:
        torch.nn.Module: The underlying PyTorch model

    Example:
        >>> import devproc
        >>> import torch
        >>>
        >>> # Load VLA module
        >>> vla_module = devproc.load_vla_module("lerobot/pi05_base")
        >>>
        >>> # Prepare example inputs
        >>> images = torch.randn(1, 3, 224, 224)
        >>> state = torch.randn(1, 32)
        >>> language = "pick up the cup"
        >>>
        >>> # Convert to DevProc IR
        >>> vla_ir = devproc.from_torch(vla_module, example_inputs=(images, state, language))
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load VLA models. Install with: pip install torch")

    # Try to load from LeRobot
    # The actual implementation depends on the LeRobot API
    # For π0.5, we load the policy and extract the model
    try:
        from lerobot.policies.pi05 import PI05Policy

        policy = PI05Policy.from_pretrained(model_path)
        # Return the underlying model for torch -> IR conversion
        return policy.model
    except ImportError:
        raise ImportError(
            "LeRobot is required to load VLA models. "
            "Install with: pip install 'lerobot[pi]@git+https://github.com/huggingface/lerobot.git'"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load VLA model from {model_path}: {e}")
