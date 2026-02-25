"""
Function and Block classes for DevProc IR.
"""

from typing import List, Optional, Dict
from devproc.ir.base import Value, Op
from devproc.ir.types import Type


class Block:
    """
    Basic block containing operations.

    MVP only supports a single block per function.
    """

    def __init__(self):
        self._ops: List[Op] = []
        self._values: Dict[str, Value] = {}  # name -> Value

    @property
    def ops(self) -> List[Op]:
        return self._ops

    @property
    def values(self) -> Dict[str, Value]:
        return self._values

    def add_op(self, op: Op) -> None:
        """Add an operation to this block."""
        self._ops.append(op)
        # Register output values
        for output in op.outputs:
            self._values[output.name] = output

    def get_value(self, name: str) -> Optional[Value]:
        """Get a value by name."""
        return self._values.get(name)

    def __repr__(self) -> str:
        lines = ["Block {"]
        for op in self._ops:
            lines.append(f"  {op}")
        lines.append("}")
        return "\n".join(lines)


class Function:
    """
    Represents a function in the IR.

    Contains input parameters, a single block (MVP), and an output.
    """

    _name_counter = 0

    def __init__(self, name: str):
        """
        Args:
            name: Name of the function
        """
        self._name = name
        self._inputs: List[Value] = []
        self._output: Optional[Value] = None
        self._block = Block()

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> List[Value]:
        return self._inputs

    @property
    def output(self) -> Optional[Value]:
        return self._output

    @output.setter
    def output(self, value: Value) -> None:
        self._output = value

    @property
    def block(self) -> Block:
        return self._block

    def add_input(self, value: Value) -> None:
        """Add an input parameter to this function.

        Note: Function inputs are not added to block.values because they are
        not produced by any op (they are function parameters).
        """
        self._inputs.append(value)

    def add_op(self, op: Op) -> None:
        """Add an operation to this function's block.

        Note: 'input' ops are special - their outputs are function parameters,
        not actual operations in the block. They are skipped.
        """
        # Skip 'input' ops - they represent function parameters, not operations
        if op.name == "input":
            return
        self._block.add_op(op)

    def get_value(self, name: str) -> Optional[Value]:
        """Get a value by name (from inputs or block)."""
        # Check inputs first
        for input_val in self._inputs:
            if input_val.name == name:
                return input_val
        # Then check block
        return self._block.get_value(name)

    @staticmethod
    def generate_name(prefix: str = "v") -> str:
        """Generate a unique name for a value."""
        Function._name_counter += 1
        return f"%{prefix}{Function._name_counter}"

    def __repr__(self) -> str:
        lines = [f"func @{self.name}("]

        # Print inputs
        if self._inputs:
            input_strs = [f"{v.name} : {repr(v.type)}" for v in self._inputs]
            lines.append(f"  inputs: {', '.join(input_strs)}")
        else:
            lines.append("  inputs: ()")

        # Print operations
        lines.append("  {")
        for op in self._block.ops:
            lines.append(f"    {op}")
        lines.append("  }")

        # Print output
        if self._output:
            lines.append(f"  return {self._output.name}")
        else:
            lines.append("  return (none)")

        lines.append(")")
        return "\n".join(lines)
