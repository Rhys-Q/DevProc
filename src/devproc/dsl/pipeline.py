"""
DevProc DSL - Pipeline builder for constructing IR.

This module provides a Python embedded DSL for building DevProc IR pipelines.
The DSL does NOT execute any computation - it only constructs IR nodes.
"""

from typing import Optional, Any, List, Tuple
from devproc.ir.function import Function
from devproc.ir.types import TensorType, ScalarType
from devproc.ir.base import Value, Op
from devproc.ir.ops import OpBuilder


class Tensor:
    """
    DSL Tensor - wraps IR Value for use in DSL pipeline.

    This is a lightweight wrapper that provides a convenient API
    while preserving the underlying IR Value.
    """

    def __init__(self, ir_value: Value, name: str = None):
        """
        Args:
            ir_value: The underlying IR Value
            name: Optional name for the tensor
        """
        self._ir_value = ir_value
        self._name = name or ir_value.name

    @property
    def ir_value(self) -> Value:
        """Get the underlying IR Value."""
        return self._ir_value

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self._ir_value.type.shape

    @property
    def dtype(self) -> str:
        """Get tensor dtype."""
        return self._ir_value.type.dtype

    @property
    def device(self) -> str:
        """Get tensor device."""
        return self._ir_value.type.device

    @property
    def name(self) -> str:
        """Get tensor name."""
        return self._name

    def __repr__(self) -> str:
        return f"Tensor({self._name} : {self._ir_value.type})"


class Pipeline:
    """
    DSL Pipeline - builds IR from Python code.

    This is a Python embedded DSL that constructs DevProc IR without
    executing any computation. Each method call creates an IR Op.

    Usage:
        pipe = Pipeline()
        img = pipe.input("image", "uint8", (224, 224, 3))
        x = pipe.normalize(img)
        out = pipe.argmax(x)
        pipe.output(out)

        # Build IR
        ir_func = pipe.build()
    """

    # Valid dtypes for DSL
    VALID_DTYPES = frozenset([
        "uint8", "int8", "int16", "int32", "int64",
        "float16", "float32", "float64", "bool"
    ])

    # Valid devices
    VALID_DEVICES = frozenset(["cpu", "cuda"])

    def __init__(self, name: str = "main"):
        """
        Args:
            name: Name of the pipeline function
        """
        self._name = name
        self._ir_function = Function(name)
        self._tensors: List[Tensor] = []  # Track all created tensors
        self._output_tensor: Optional[Tensor] = None

    @property
    def name(self) -> str:
        """Get pipeline name."""
        return self._name

    def _validate_dtype(self, dtype: str) -> None:
        """Validate dtype."""
        if dtype not in self.VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{dtype}'. Valid dtypes: {self.VALID_DTYPES}"
            )

    def _validate_device(self, device: str) -> None:
        """Validate device."""
        if device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. Valid devices: {self.VALID_DEVICES}"
            )

    def _validate_tensor(self, tensor: Tensor, op_name: str) -> None:
        """Validate that input is a Tensor."""
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{op_name} requires Tensor, got {type(tensor)}"
            )

    # ==================== Input/Output ====================

    def input(self, name: str, dtype: str, shape: tuple, device: str = "cpu") -> Tensor:
        """
        Define a pipeline input.

        Args:
            name: Name of the input
            dtype: Data type (e.g., "uint8", "float32")
            shape: Tensor shape (e.g., (224, 224, 3))
            device: Device ("cpu" or "cuda")

        Returns:
            Tensor representing the input
        """
        self._validate_dtype(dtype)
        self._validate_device(device)

        # Create IR input op
        tensor_type = TensorType(shape, dtype, device)
        input_op = OpBuilder.input(name, tensor_type)
        input_value = input_op.outputs[0]

        # Add to function as parameter
        self._ir_function.add_input(input_value)

        # Create DSL tensor
        tensor = Tensor(input_value, name)
        self._tensors.append(tensor)

        return tensor

    def output(self, tensor: Tensor) -> None:
        """
        Set the pipeline output.

        Args:
            tensor: The tensor to return as output
        """
        self._validate_tensor(tensor, "output")
        self._output_tensor = tensor
        self._ir_function.output = tensor.ir_value

    # ==================== Preprocessing Ops ====================

    def normalize(self, tensor: Tensor) -> Tensor:
        """
        Normalize tensor (convert to float32).

        Args:
            tensor: Input tensor

        Returns:
            Normalized tensor (float32)
        """
        self._validate_tensor(tensor, "normalize")

        # Create IR normalize op
        normalize_op = OpBuilder.normalize(tensor.ir_value)
        normalize_value = normalize_op.outputs[0]

        # Add to function
        self._ir_function.add_op(normalize_op)

        # Create DSL tensor
        result_tensor = Tensor(normalize_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def resize(self, tensor: Tensor, size: tuple) -> Tensor:
        """
        Resize tensor to target size.

        Args:
            tensor: Input tensor
            size: Target size (e.g., (224, 224))

        Returns:
            Resized tensor
        """
        self._validate_tensor(tensor, "resize")

        # Create IR resize op
        resize_op = OpBuilder.resize(tensor.ir_value, size)
        resize_value = resize_op.outputs[0]

        # Add to function
        self._ir_function.add_op(resize_op)

        # Create DSL tensor
        result_tensor = Tensor(resize_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def transpose(self, tensor: Tensor, dims: tuple) -> Tensor:
        """
        Transpose tensor along given dimensions.

        Args:
            tensor: Input tensor
            dims: Permutation of dimensions

        Returns:
            Transposed tensor
        """
        self._validate_tensor(tensor, "transpose")

        # Create IR transpose op
        transpose_op = OpBuilder.transpose(tensor.ir_value, dims)
        transpose_value = transpose_op.outputs[0]

        # Add to function
        self._ir_function.add_op(transpose_op)

        # Create DSL tensor
        result_tensor = Tensor(transpose_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def to(self, tensor: Tensor, dtype: str, device: Optional[str] = None) -> Tensor:
        """
        Convert tensor to different dtype or device.

        Args:
            tensor: Input tensor
            dtype: Target dtype
            device: Target device (optional)

        Returns:
            Converted tensor
        """
        self._validate_tensor(tensor, "to")
        self._validate_dtype(dtype)
        if device:
            self._validate_device(device)

        # Create IR to op
        to_op = OpBuilder.to(tensor.ir_value, dtype, device)
        to_value = to_op.outputs[0]

        # Add to function
        self._ir_function.add_op(to_op)

        # Create DSL tensor
        result_tensor = Tensor(to_value)
        self._tensors.append(result_tensor)

        return result_tensor

    # ==================== Neural Network Ops ====================

    def model(self, exported_model: Any, *args: Tensor) -> Tensor:
        """
        Apply a torch.export model.

        Note: This is a placeholder. Phase 3 will implement the actual
        torch.export importer.

        Args:
            exported_model: torch.export.ExportedProgram
            *args: Input tensors

        Returns:
            Output tensor from model
        """
        # Placeholder: for now, just pass through the last input
        # Phase 3 will implement actual model loading
        if not args:
            raise ValueError("model requires at least one input tensor")

        # For MVP, just return the input as a placeholder
        # Real implementation will come in Phase 3
        return args[-1]

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Matrix multiplication.

        Args:
            a: First input tensor (M x K)
            b: Second input tensor (K x N)

        Returns:
            Output tensor (M x N)
        """
        self._validate_tensor(a, "matmul")
        self._validate_tensor(b, "matmul")

        # Create IR matmul op
        matmul_op = OpBuilder.matmul(a.ir_value, b.ir_value)
        matmul_value = matmul_op.outputs[0]

        # Add to function
        self._ir_function.add_op(matmul_op)

        # Create DSL tensor
        result_tensor = Tensor(matmul_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def linear(self, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        """
        Linear (fully connected) layer.

        Args:
            input: Input tensor
            weight: Weight tensor (out_features x in_features)
            bias: Optional bias tensor (out_features)

        Returns:
            Output tensor
        """
        self._validate_tensor(input, "linear")
        self._validate_tensor(weight, "linear")

        # Create IR linear op
        if bias is not None:
            self._validate_tensor(bias, "linear")
            linear_op = OpBuilder.linear(input.ir_value, weight.ir_value, bias.ir_value)
        else:
            linear_op = OpBuilder.linear(input.ir_value, weight.ir_value)

        linear_value = linear_op.outputs[0]

        # Add to function
        self._ir_function.add_op(linear_op)

        # Create DSL tensor
        result_tensor = Tensor(linear_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def relu(self, tensor: Tensor) -> Tensor:
        """
        ReLU activation.

        Args:
            tensor: Input tensor

        Returns:
            Output tensor with ReLU applied
        """
        self._validate_tensor(tensor, "relu")

        # Create IR relu op
        relu_op = OpBuilder.relu(tensor.ir_value)
        relu_value = relu_op.outputs[0]

        # Add to function
        self._ir_function.add_op(relu_op)

        # Create DSL tensor
        result_tensor = Tensor(relu_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Element-wise addition.

        Args:
            a: First input tensor
            b: Second input tensor

        Returns:
            Output tensor (broadcasted shape)
        """
        self._validate_tensor(a, "add")
        self._validate_tensor(b, "add")

        # Create IR add op
        add_op = OpBuilder.add(a.ir_value, b.ir_value)
        add_value = add_op.outputs[0]

        # Add to function
        self._ir_function.add_op(add_op)

        # Create DSL tensor
        result_tensor = Tensor(add_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def sigmoid(self, tensor: Tensor) -> Tensor:
        """
        Sigmoid activation.

        Args:
            tensor: Input tensor

        Returns:
            Output tensor with sigmoid applied
        """
        self._validate_tensor(tensor, "sigmoid")

        # Create IR sigmoid op
        sigmoid_op = OpBuilder.sigmoid(tensor.ir_value)
        sigmoid_value = sigmoid_op.outputs[0]

        # Add to function
        self._ir_function.add_op(sigmoid_op)

        # Create DSL tensor
        result_tensor = Tensor(sigmoid_value)
        self._tensors.append(result_tensor)

        return result_tensor

    def softmax(self, tensor: Tensor, dim: int = -1) -> Tensor:
        """
        Softmax activation.

        Args:
            tensor: Input tensor
            dim: Dimension to apply softmax

        Returns:
            Output tensor with softmax applied
        """
        self._validate_tensor(tensor, "softmax")

        # Create IR softmax op
        softmax_op = OpBuilder.softmax(tensor.ir_value, dim)
        softmax_value = softmax_op.outputs[0]

        # Add to function
        self._ir_function.add_op(softmax_op)

        # Create DSL tensor
        result_tensor = Tensor(softmax_value)
        self._tensors.append(result_tensor)

        return result_tensor

    # ==================== Postprocessing Ops ====================

    def argmax(self, tensor: Tensor, dim: int = -1) -> Tensor:
        """
        Argmax operation.

        Args:
            tensor: Input tensor
            dim: Dimension to reduce

        Returns:
            Scalar tensor with argmax result (int64)
        """
        self._validate_tensor(tensor, "argmax")

        # Create IR argmax op
        argmax_op = OpBuilder.argmax(tensor.ir_value, dim)
        argmax_value = argmax_op.outputs[0]

        # Add to function
        self._ir_function.add_op(argmax_op)

        # Create DSL tensor (scalar)
        result_tensor = Tensor(argmax_value)
        self._tensors.append(result_tensor)

        return result_tensor

    # ==================== Build ====================

    def build(self) -> Function:
        """
        Build and return the IR Function.

        Returns:
            The constructed IR Function
        """
        # If output not explicitly set, use the last tensor
        if self._output_tensor is None and self._tensors:
            self._output_tensor = self._tensors[-1]
            self._ir_function.output = self._output_tensor.ir_value

        return self._ir_function

    def __repr__(self) -> str:
        return f"Pipeline({self._name})"
