"""
Type system for DevProc IR.

Strict type system - None/Any types are explicitly forbidden.
"""

from typing import Tuple


class Type:
    """Base class for all types in DevProc IR."""

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._equals(other)

    def _equals(self, other) -> bool:
        raise NotImplementedError


class TensorType(Type):
    """
    Tensor type with explicit shape, dtype, and device.

    Args:
        shape: Tuple of integers representing tensor shape (e.g., (224, 224, 3))
        dtype: Data type (e.g., "uint8", "float32", "int32")
        device: Device (e.g., "cpu", "cuda")
    """

    VALID_DTYPES = frozenset([
        "uint8", "int8", "int16", "int32", "int64",
        "float16", "float32", "float64", "bool"
    ])

    VALID_DEVICES = frozenset(["cpu", "cuda"])

    def __init__(self, shape: Tuple[int, ...], dtype: str, device: str = "cpu"):
        super().__init__()
        self._validate_shape(shape)
        self._validate_dtype(dtype)
        self._validate_device(device)

        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device

    def _validate_shape(self, shape: Tuple[int, ...]) -> None:
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape).__name__}")
        if not all(isinstance(s, int) and s > 0 for s in shape):
            raise ValueError(f"Shape must be positive integers, got {shape}")

    def _validate_dtype(self, dtype: str) -> None:
        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be a string, got {type(dtype).__name__}")
        if dtype not in self.VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{dtype}'. Valid dtypes: {self.VALID_DTYPES}"
            )

    def _validate_device(self, device: str) -> None:
        if not isinstance(device, str):
            raise TypeError(f"device must be a string, got {type(device).__name__}")
        if device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. Valid devices: {self.VALID_DEVICES}"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    def _equals(self, other: "TensorType") -> bool:
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def __repr__(self) -> str:
        shape_str = ",".join(str(s) for s in self.shape)
        return f"{self.dtype}[{shape_str}]"


class ScalarType(Type):
    """
    Scalar type for scalar values.

    Args:
        dtype: Data type (e.g., "int32", "float32")
    """

    VALID_DTYPES = frozenset([
        "int8", "int16", "int32", "int64",
        "float16", "float32", "float64", "bool"
    ])

    def __init__(self, dtype: str):
        super().__init__()
        self._validate_dtype(dtype)
        self._dtype = dtype

    def _validate_dtype(self, dtype: str) -> None:
        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be a string, got {type(dtype).__name__}")
        if dtype not in self.VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{dtype}'. Valid dtypes: {self.VALID_DTYPES}"
            )

    @property
    def dtype(self) -> str:
        return self._dtype

    def _equals(self, other: "ScalarType") -> bool:
        return self.dtype == other.dtype

    def __repr__(self) -> str:
        return f"scalar({self.dtype})"
