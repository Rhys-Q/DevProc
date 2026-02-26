"""
Type inference for DevProc IR.

Provides type inference/verification by propagating types from known
inputs through the IR graph.
"""

from typing import Dict, List, Optional, Tuple
from devproc.ir.function import Function
from devproc.ir.base import Value, Op
from devproc.ir.types import TensorType, ScalarType, Type


class TypeInferencer:
    """
    IR Type Inferencer - propagates types from known inputs through the IR.

    Algorithm:
    1. Initialize input ops as known (their output types are declared)
    2. Iterate: for each op where all inputs are known, infer output types
    3. Repeat until fixed point (no new types inferred)
    4. Return all inferred types

    This provides independent type verification separate from OpBuilder.
    """

    def __init__(self, func: Function):
        self.func = func
        self.inferred_types: Dict[Value, Type] = {}
        self.conflicts: List[Tuple[Value, Type, Type]] = []  # (value, declared, inferred)

    def infer(self, update_values: bool = True) -> Dict[Value, Type]:
        """
        Execute type inference.

        Args:
            update_values: If True, update Value.type with inferred types

        Returns:
            Dict mapping Values to their inferred types.
        """
        # Get all ops in order
        ops = self._get_all_ops()

        # Initialize: input ops have known types (from their declarations)
        for op in ops:
            if op.name == "input":
                for out in op.outputs:
                    self.inferred_types[out] = out.type

        # Also include function inputs (these are the parameters)
        for inp in self.func.inputs:
            self.inferred_types[inp] = inp.type

        # Iterative inference
        changed = True
        while changed:
            changed = False
            for op in ops:
                if op.name == "input":
                    continue

                if self._can_infer(op):
                    inferred = self._infer_op(op)
                    for out, inferred_type in zip(op.outputs, inferred):
                        if out not in self.inferred_types:
                            self.inferred_types[out] = inferred_type
                            changed = True
                        elif self.inferred_types[out] != inferred_type:
                            # Type conflict detected
                            self.conflicts.append((out, out.type, inferred_type))

        # Optionally update Value types with inferred types
        # Skip function inputs - they already have correct declared types
        if update_values:
            func_input_names = {inp.name for inp in self.func.inputs}
            for value, inferred_type in self.inferred_types.items():
                if value.name not in func_input_names:
                    value._type = inferred_type

        return self.inferred_types

    def _get_all_ops(self) -> List[Op]:
        """Get all ops in the function (including input ops from block)."""
        # Block ops don't include 'input' ops (they're handled specially)
        # So we need to look at function inputs to find 'input' ops
        # Actually, looking at Function.add_op, 'input' ops are skipped in block
        # The function inputs ARE the outputs of 'input' ops

        # Get ops from block
        ops = list(self.func.block.ops)

        # Add 'input' ops by reconstructing them from function inputs
        # Each function input came from an 'input' op
        return ops

    def _can_infer(self, op: Op) -> bool:
        """Check if we can infer output types for this op."""
        return all(inp in self.inferred_types for inp in op.inputs)

    def _infer_op(self, op: Op) -> List[Type]:
        """Infer output types based on input types.

        Returns empty list if inference fails (e.g., type error in IR).
        """
        method_name = f"_infer_{op.name}"
        if hasattr(self, method_name):
            result = getattr(self, method_name)(op)
            # If result is empty, inference failed (invalid IR)
            return result if result else []
        # Unknown op - return empty list
        return []

    # --- Operation-specific inference methods ---

    def _infer_normalize(self, op: Op) -> List[Type]:
        """normalize: same shape, float32 dtype, same device."""
        inp_type = self.inferred_types[op.inputs[0]]
        if not isinstance(inp_type, TensorType):
            return []  # Cannot infer
        return [TensorType(inp_type.shape, "float32", inp_type.device)]

    def _infer_argmax(self, op: Op) -> List[Type]:
        """argmax: returns ScalarType int64."""
        return [ScalarType("int64")]

    def _infer_matmul(self, op: Op) -> List[Type]:
        """matmul: (M x K) @ (K x N) -> (M x N), float32."""
        a = self.inferred_types[op.inputs[0]]
        b = self.inferred_types[op.inputs[1]]
        if not isinstance(a, TensorType) or not isinstance(b, TensorType):
            return []
        if len(a.shape) != 2 or len(b.shape) != 2:
            return []
        if a.shape[1] != b.shape[0]:
            return []  # Dimension mismatch - cannot infer valid output
        shape = (a.shape[0], b.shape[1])
        return [TensorType(shape, "float32", a.device)]

    def _infer_linear(self, op: Op) -> List[Type]:
        """linear: (batch, in) @ (out, in).T -> (batch, out), float32."""
        inp = self.inferred_types[op.inputs[0]]
        weight = self.inferred_types[op.inputs[1]]
        if not isinstance(inp, TensorType) or not isinstance(weight, TensorType):
            return []
        out_features = weight.shape[0]
        batch = inp.shape[0] if len(inp.shape) > 1 else 1
        shape = (batch, out_features)
        return [TensorType(shape, "float32", inp.device)]

    def _infer_relu(self, op: Op) -> List[Type]:
        """relu: same shape/dtype/device as input."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []
        return [TensorType(inp.shape, inp.dtype, inp.device)]

    def _infer_add(self, op: Op) -> List[Type]:
        """add: element-wise, same shape/dtype/device (no broadcasting in MVP)."""
        a = self.inferred_types[op.inputs[0]]
        b = self.inferred_types[op.inputs[1]]
        if not isinstance(a, TensorType) or not isinstance(b, TensorType):
            return []
        if a.shape != b.shape:
            return []  # Shape mismatch
        return [TensorType(a.shape, a.dtype, a.device)]

    def _infer_resize(self, op: Op) -> List[Type]:
        """resize: output shape from op._size, same dtype/device."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []
        size = getattr(op, '_size', None)
        if size is None:
            return []

        # Compute output shape based on size param
        if len(inp.shape) == 3 and len(size) == 2:
            # HWC -> HWC (resize H, W)
            shape = (*size, inp.shape[2])
        else:
            shape = size

        return [TensorType(shape, inp.dtype, inp.device)]

    def _infer_transpose(self, op: Op) -> List[Type]:
        """transpose: permute shape according to op._dims."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []
        dims = getattr(op, '_dims', None)
        if dims is None:
            return []

        # Compute output shape from dimension permutation
        shape = tuple(inp.shape[d] for d in dims)
        return [TensorType(shape, inp.dtype, inp.device)]

    def _infer_to(self, op: Op) -> List[Type]:
        """to: convert dtype/device."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []

        # Get target dtype/device from op attributes
        dtype = getattr(op, '_dtype', None) or inp.dtype
        device = getattr(op, '_device', None) or inp.device

        return [TensorType(inp.shape, dtype, device)]

    def _infer_sigmoid(self, op: Op) -> List[Type]:
        """sigmoid: same shape, float32 dtype."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []
        return [TensorType(inp.shape, "float32", inp.device)]

    def _infer_softmax(self, op: Op) -> List[Type]:
        """softmax: same shape, float32 dtype."""
        inp = self.inferred_types[op.inputs[0]]
        if not isinstance(inp, TensorType):
            return []
        return [TensorType(inp.shape, "float32", inp.device)]


def infer_types(func: Function) -> Dict[Value, Type]:
    """
    Convenience function to infer all types in a function.

    Args:
        func: The function to analyze

    Returns:
        Dict mapping each Value to its inferred type
    """
    inferrer = TypeInferencer(func)
    return inferrer.infer()
