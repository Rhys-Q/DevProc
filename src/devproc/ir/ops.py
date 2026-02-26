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
from devproc.ir.types import TensorType, ScalarType, Type
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
