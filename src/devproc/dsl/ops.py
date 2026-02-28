"""
DevProc DSL Operations - Module-level functions for @kernel decorated functions.

These functions are designed to be used inside @devproc.kernel decorated functions.
They build IR operations instead of executing computation.
"""

from typing import Optional, Tuple, Any, Union
from devproc.ir.ops import OpBuilder
from devproc.ir.types import TensorType
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
