"""
Base classes for DevProc IR: Value and Op.

Implements SSA semantics - each Value is defined exactly once.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from devproc.ir.types import Type
    from devproc.ir.base import Op


class Value:
    """
    SSA Value - represents a typed value in IR.

    Each Value can only be defined once (SSA property).
    Maintains def-use chains through the `uses` list.
    """

    _counter = 0

    def __init__(self, name: str, type: "Type", op: Optional["Op"] = None):
        """
        Args:
            name: Name of the value (e.g., "%0", "x")
            type: Type of the value
            op: The Op that produces this value (None for function inputs)
        """
        self._name = name
        self._type = type
        self._op = op
        self._uses: List["Op"] = []

        # Register this value as an output of the op
        if op is not None:
            op._add_output(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> "Type":
        return self._type

    @property
    def op(self) -> Optional["Op"]:
        return self._op

    @property
    def uses(self) -> List["Op"]:
        return self._uses

    def add_use(self, op: "Op") -> None:
        """Register an op as using this value."""
        if op not in self._uses:
            self._uses.append(op)

    def __repr__(self) -> str:
        return self._name

    def __str__(self) -> str:
        type_repr = repr(self.type)
        return f"{self.name} : {type_repr}"


class Op:
    """
    Operation in the IR.

    Has inputs (Values) and outputs (Values).
    Maintains use-def chains.
    """

    _counter = 0

    def __init__(self, name: str, inputs: List["Value"], outputs: List["Value"]):
        """
        Args:
            name: Name of the operation (e.g., "normalize", "matmul")
            inputs: List of input Values
            outputs: List of output Values
        """
        self._name = name
        self._inputs = list(inputs)
        self._outputs = list(outputs)

        # Register this op as a user of all inputs
        for input_value in inputs:
            input_value.add_use(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> List["Value"]:
        return self._inputs

    @property
    def outputs(self) -> List["Value"]:
        return self._outputs

    def _add_output(self, value: Value) -> None:
        """Internal: register an output value."""
        # This is called by Value constructor
        pass

    def _get_attrs_str(self) -> str:
        """Get attribute string for the op (only user-defined attrs)."""
        # Only show specific known attributes
        known_attrs = []
        if hasattr(self, '_size'):
            known_attrs.append(f"size={self._size}")
        if hasattr(self, '_dims'):
            known_attrs.append(f"dims={self._dims}")
        if hasattr(self, '_dtype'):
            known_attrs.append(f"dtype={self._dtype}")
        if hasattr(self, '_device'):
            known_attrs.append(f"device={self._device}")
        if hasattr(self, '_dim'):
            known_attrs.append(f"dim={self._dim}")

        if known_attrs:
            return f"[{', '.join(known_attrs)}]"
        return ""

    def __repr__(self) -> str:
        inputs_str = ", ".join(v.name for v in self.inputs)
        outputs_str = ", ".join(f"{v.name} : {repr(v.type)}" for v in self.outputs)
        attrs_str = self._get_attrs_str()
        return f"{outputs_str} = {self.name}{attrs_str}({inputs_str})"

    def __str__(self) -> str:
        return self.__repr__()
