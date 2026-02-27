"""
Backend abstract base classes for DevProc.

Defines the interface for compilation backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from devproc.ir.function import Function


class Backend(ABC):
    """Abstract base class for compilation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @abstractmethod
    def compile(self, ir_function: Function) -> "CompiledProgram":
        """Compile an IR Function to the target backend.

        Args:
            ir_function: The IR Function to compile.

        Returns:
            A compiled program that can be executed.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available on this system.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass


class CompiledProgram(ABC):
    """Abstract base class for compiled programs."""

    @abstractmethod
    def run(self, **kwargs) -> List[Any]:
        """Execute the compiled program.

        Args:
            **kwargs: Input tensors as keyword arguments.

        Returns:
            List of output tensors.
        """
        pass


class LoweringContext:
    """Context for lowering IR operations to backend-specific operations."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.tensor_map: Dict[str, Any] = {}
        self.kernel_cache: Dict[str, Any] = {}

    def allocate_tensor(self, name: str, shape: tuple, dtype: str, device: str) -> Any:
        """Allocate a tensor for the given IR value.

        Args:
            name: Name of the tensor.
            shape: Tensor shape.
            dtype: Data type.
            device: Device string.

        Returns:
            The allocated tensor.
        """
        raise NotImplementedError

    def get_tensor(self, name: str) -> Any:
        """Get a tensor by name.

        Args:
            name: Name of the tensor.

        Returns:
            The tensor, or None if not found.
        """
        return self.tensor_map.get(name)

    def set_tensor(self, name: str, tensor: Any) -> None:
        """Set a tensor by name.

        Args:
            name: Name of the tensor.
            tensor: The tensor to store.
        """
        self.tensor_map[name] = tensor


class BackendError(Exception):
    """Base exception for backend errors."""
    pass


class UnsupportedOpError(BackendError):
    """Exception raised when an operation is not supported by the backend."""

    def __init__(self, op_name: str):
        self.op_name = op_name
        super().__init__(f"Operation '{op_name}' is not supported by this backend")


class CompilationError(BackendError):
    """Exception raised during compilation."""
    pass
