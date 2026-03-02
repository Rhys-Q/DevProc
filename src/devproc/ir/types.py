"""
Type system for DevProc IR.

Strict type system - None/Any types are explicitly forbidden.
"""

from typing import Tuple, Optional


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


class StringType(Type):
    """
    String type for text values.

    Args:
        max_length: Optional maximum length constraint
    """

    def __init__(self, max_length: Optional[int] = None):
        super().__init__()
        self._max_length = max_length

    @property
    def max_length(self) -> Optional[int]:
        return self._max_length

    def _equals(self, other: "StringType") -> bool:
        return self.max_length == other.max_length

    def __repr__(self) -> str:
        if self.max_length:
            return f"string[{self.max_length}]"
        return "string"


class TokenizerType(Type):
    """
    Tokenizer type for LLM preprocessing.

    Args:
        tokenizer_name: Name or path of the tokenizer
    """

    def __init__(self, tokenizer_name: str = "unknown"):
        super().__init__()
        self._tokenizer_name = tokenizer_name

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def _equals(self, other: "TokenizerType") -> bool:
        return self.tokenizer_name == other.tokenizer_name

    def __repr__(self) -> str:
        return f"tokenizer[{self.tokenizer_name}]"


class DictType(Type):
    """
    Dict type for key-value storage.

    Args:
        key_type: Type of keys
        value_type: Type of values
    """

    def __init__(self, key_type: Type, value_type: Type):
        super().__init__()
        self._key_type = key_type
        self._value_type = value_type

    @property
    def key_type(self) -> Type:
        return self._key_type

    @property
    def value_type(self) -> Type:
        return self._value_type

    def _equals(self, other: "DictType") -> bool:
        return (self.key_type == other.key_type and
                self.value_type == other.value_type)

    def __repr__(self) -> str:
        return f"dict<{self.key_type}, {self.value_type}>"


class ListType(Type):
    """
    List type for sequences.

    Args:
        element_type: Type of list elements
    """

    def __init__(self, element_type: Type):
        super().__init__()
        self._element_type = element_type

    @property
    def element_type(self) -> Type:
        return self._element_type

    def _equals(self, other: "ListType") -> bool:
        return self.element_type == other.element_type

    def __repr__(self) -> str:
        return f"list<{self.element_type}>"


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
