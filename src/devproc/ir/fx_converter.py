"""
FX to IR Converter.

Converts torch.fx GraphModule to DevProc IR Function.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import torch
import torch.fx
from dataclasses import dataclass, field

from devproc.ir.base import Value, Op
from devproc.ir.types import TensorType, ScalarType
from devproc.ir.function import Function, Block
from devproc.ir.ops import OpBuilder
from devproc.ir.fx_op_map import get_fx_op_map


# Type alias for IR op builders
IROpBuilder = Callable[..., Op]


@dataclass
class FXToIRConverter:
    """
    Converts FX GraphModule to DevProc IR Function.

    Usage:
        converter = FXToIRConverter()
        ir_function = converter.convert(gm, example_inputs)
    """

    # FX target to IR op name mapping
    op_map: Dict[str, str] = field(default_factory=get_fx_op_map)

    # Node to IR Value mapping
    value_map: Dict[torch.fx.Node, Value] = field(default_factory=dict)

    # FX GraphModule being converted
    gm: Optional[torch.fx.GraphModule] = field(default=None, init=False)

    # Example inputs
    example_inputs: Tuple[Any, ...] = field(default_factory=tuple, init=False)

    # Counter for generating unique names
    _name_counter: int = field(default=0, init=False)

    def convert(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Tuple[Any, ...],
    ) -> Function:
        """
        Convert FX GraphModule to IR Function.

        Args:
            gm: FX GraphModule to convert
            example_inputs: Example inputs for shape/dtype inference

        Returns:
            IR Function
        """
        self.gm = gm
        self.example_inputs = example_inputs
        self.value_map.clear()
        self._name_counter = 0

        # Create IR function
        func_name = gm.__class__.__name__ if hasattr(gm, '__class__') else "fx_graph"
        ir_function = Function(name=func_name)

        # Process nodes in order
        nodes = list(gm.graph.nodes)

        for node in nodes:
            self._process_node(node, ir_function)

        return ir_function

    def _process_node(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process a single FX node and convert to IR."""
        if node.op == "placeholder":
            self._process_placeholder(node, ir_function)
        elif node.op == "get_attr":
            self._process_get_attr(node, ir_function)
        elif node.op == "call_function":
            self._process_call_function(node, ir_function)
        elif node.op == "call_method":
            self._process_call_method(node, ir_function)
        elif node.op == "call_module":
            self._process_call_module(node, ir_function)
        elif node.op == "output":
            self._process_output(node, ir_function)
        else:
            raise ValueError(f"Unsupported FX node op: {node.op}")

    def _get_next_name(self, prefix: str = "v") -> str:
        """Generate a unique name."""
        name = f"{prefix}{self._name_counter}"
        self._name_counter += 1
        return name

    def _get_tensor_type(self, node: torch.fx.Node) -> TensorType:
        """Extract TensorType from FX node."""
        # Try to get shape and dtype from node's meta
        if "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            shape = tuple(meta.shape)
            dtype = str(meta.dtype).replace("torch.", "")
            # Handle device
            device = "cuda" if meta.is_cuda else "cpu"
            return TensorType(shape=shape, dtype=dtype, device=device)

        # Try to infer from example inputs
        if node.op == "placeholder":
            idx = list(self.gm.graph.nodes).index(node)
            if idx < len(self.example_inputs):
                tensor = self.example_inputs[idx]
                if isinstance(tensor, torch.Tensor):
                    return TensorType(
                        shape=tuple(tensor.shape),
                        dtype=str(tensor.dtype).replace("torch.", ""),
                        device="cuda" if tensor.is_cuda else "cpu"
                    )

        # Try to get from shape/env
        if hasattr(node, 'shape') and node.shape is not None:
            shape = tuple(node.shape)
        else:
            # Default to dynamic shape
            shape = (1,)

        # Default dtype
        dtype = "float32"
        if hasattr(node, 'dtype') and node.dtype is not None:
            dtype = str(node.dtype).replace("torch.", "")

        return TensorType(shape=shape, dtype=dtype, device="cpu")

    def _process_placeholder(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process placeholder node (function input)."""
        tensor_type = self._get_tensor_type(node)
        name = node.name if hasattr(node, 'name') else self._get_next_name("input")

        # Create IR input operation
        input_op = OpBuilder.input(name, tensor_type)
        ir_function.add_input(input_op.outputs[0])

        # Map FX node to IR Value
        self.value_map[node] = input_op.outputs[0]

    def _process_get_attr(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process get_attr node (module attribute/weight)."""
        # Get the attribute from the module
        attr_path = node.target  # e.g., "linear.weight"
        parts = attr_path.split(".")

        # Navigate to the attribute
        obj = self.gm
        for part in parts:
            obj = getattr(obj, part)

        if isinstance(obj, torch.Tensor):
            # It's a weight tensor
            tensor_type = TensorType(
                shape=tuple(obj.shape),
                dtype=str(obj.dtype).replace("torch.", ""),
                device="cuda" if obj.is_cuda else "cpu"
            )
        elif isinstance(obj, torch.nn.Parameter):
            tensor_type = TensorType(
                shape=tuple(obj.data.shape),
                dtype=str(obj.data.dtype).replace("torch.", ""),
                device="cuda" if obj.is_cuda else "cpu"
            )
        else:
            raise ValueError(f"Unsupported get_attr target: {attr_path}, type: {type(obj)}")

        # Create input op for the weight (treat as additional function input)
        name = node.name if hasattr(node, 'name') else self._get_next_name("weight")
        input_op = OpBuilder.input(name, tensor_type)
        ir_function.add_input(input_op.outputs[0])

        # Map FX node to IR Value
        self.value_map[node] = input_op.outputs[0]

    def _process_call_function(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process call_function node."""
        target = node.target
        target_name = target.__name__ if callable(target) else str(target)

        # Handle .default suffix (e.g., relu.default, sigmoid.default)
        if target_name.endswith('.default'):
            target_name = target_name[:-8]  # Remove .default

        # Map to IR op
        ir_op_name = self.op_map.get(target_name)
        if ir_op_name is None:
            raise ValueError(f"Unsupported FX function: {target_name}")

        # Get input IR Values
        input_values = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in self.value_map:
                    input_values.append(self.value_map[arg])

        # Handle kwargs
        kwargs = {}
        for key, val in node.kwargs.items():
            if isinstance(val, torch.fx.Node) and val in self.value_map:
                kwargs[key] = self.value_map[val]

        # Create IR op
        op = self._create_ir_op(ir_op_name, input_values, kwargs, node)
        ir_function.block.add_op(op)

        # Map output
        if op.outputs:
            self.value_map[node] = op.outputs[0]

    def _process_call_method(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process call_method node."""
        # Get the method name
        method_name = node.target

        # For now, treat methods as functions
        ir_op_name = self.op_map.get(method_name)
        if ir_op_name is None:
            raise ValueError(f"Unsupported FX method: {method_name}")

        # Get input IR Values (first arg is the tensor)
        input_values = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in self.value_map:
                    input_values.append(self.value_map[arg])

        # Handle kwargs
        kwargs = {}
        for key, val in node.kwargs.items():
            if isinstance(val, torch.fx.Node) and val in self.value_map:
                kwargs[key] = self.value_map[val]

        # Create IR op
        op = self._create_ir_op(ir_op_name, input_values, kwargs, node)
        ir_function.block.add_op(op)

        # Map output
        if op.outputs:
            self.value_map[node] = op.outputs[0]

    def _process_call_module(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process call_module node (calling a submodule)."""
        # Get the submodule
        submodule = self.gm.get_submodule(node.target)

        if isinstance(submodule, torch.nn.Linear):
            self._process_linear(node, submodule, ir_function)
        elif isinstance(submodule, torch.nn.ReLU):
            self._process_relu(node, ir_function)
        elif isinstance(submodule, torch.nn.Conv2d):
            raise NotImplementedError(f"Conv2d not yet supported")
        else:
            raise ValueError(f"Unsupported module type: {type(submodule)}")

    def _process_linear(
        self,
        node: torch.fx.Node,
        linear_module: torch.nn.Linear,
        ir_function: Function,
    ) -> None:
        """Process torch.nn.Linear as IR linear op."""
        # Get input
        input_values = []
        if node.args and isinstance(node.args[0], torch.fx.Node):
            input_values.append(self.value_map[node.args[0]])

        # Get weight
        weight_tensor = linear_module.weight
        weight_type = TensorType(
            shape=tuple(weight_tensor.shape),
            dtype=str(weight_tensor.dtype).replace("torch.", ""),
            device="cuda" if weight_tensor.is_cuda else "cpu"
        )
        weight_name = self._get_next_name("weight")
        weight_op = OpBuilder.input(weight_name, weight_type)
        ir_function.add_input(weight_op.outputs[0])
        input_values.append(weight_op.outputs[0])

        # Get bias if exists
        bias_value = None
        if linear_module.bias is not None:
            bias_tensor = linear_module.bias
            bias_type = TensorType(
                shape=tuple(bias_tensor.shape),
                dtype=str(bias_tensor.dtype).replace("torch.", ""),
                device="cuda" if bias_tensor.is_cuda else "cpu"
            )
            bias_name = self._get_next_name("bias")
            bias_op = OpBuilder.input(bias_name, bias_type)
            ir_function.add_input(bias_op.outputs[0])
            bias_value = bias_op.outputs[0]

        # Create linear op
        if bias_value is not None:
            op = OpBuilder.linear(input_values[0], input_values[1], bias_value)
        else:
            op = OpBuilder.linear(input_values[0], input_values[1])

        ir_function.block.add_op(op)

        # Map output
        if op.outputs:
            self.value_map[node] = op.outputs[0]

    def _process_relu(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process torch.nn.ReLU as IR relu op."""
        # Get input
        input_values = []
        if node.args and isinstance(node.args[0], torch.fx.Node):
            input_values.append(self.value_map[node.args[0]])

        if not input_values:
            raise ValueError("ReLU has no input")

        # Create relu op
        op = OpBuilder.relu(input_values[0])
        ir_function.block.add_op(op)

        # Map output
        if op.outputs:
            self.value_map[node] = op.outputs[0]

    def _process_output(self, node: torch.fx.Node, ir_function: Function) -> None:
        """Process output node."""
        # Get the return value
        if node.args:
            arg = node.args[0]
            if isinstance(arg, torch.fx.Node) and arg in self.value_map:
                return_value = self.value_map[arg]
            else:
                # Direct tensor output
                tensor_type = self._get_tensor_type(node)
                return_value = Value(self._get_next_name("output"), tensor_type)

            # Set as function output
            ir_function.output = return_value

    def _create_ir_op(
        self,
        op_name: str,
        input_values: List[Value],
        kwargs: Dict[str, Any],
        fx_node: torch.fx.Node,
    ) -> Op:
        """Create IR op based on op name."""
        # Infer output type from input types
        output_type = None
        if input_values:
            first_input = input_values[0]
            if isinstance(first_input.type, TensorType):
                # Default: output has same shape/dtype as input
                input_type = first_input.type
                output_type = TensorType(
                    shape=input_type.shape,
                    dtype=input_type.dtype,
                    device=input_type.device
                )

        # Handle specific ops
        if op_name == "matmul":
            if len(input_values) >= 2:
                return OpBuilder.matmul(input_values[0], input_values[1])
            raise ValueError("matmul requires 2 inputs")

        elif op_name == "linear":
            if len(input_values) >= 2:
                bias = kwargs.get("bias") or (input_values[2] if len(input_values) > 2 else None)
                return OpBuilder.linear(input_values[0], input_values[1], bias)
            raise ValueError("linear requires at least 2 inputs")

        elif op_name == "relu":
            if input_values:
                return OpBuilder.relu(input_values[0])
            raise ValueError("relu requires 1 input")

        elif op_name == "add":
            if len(input_values) >= 2:
                return OpBuilder.add(input_values[0], input_values[1])
            raise ValueError("add requires 2 inputs")

        elif op_name == "sigmoid":
            if input_values:
                return OpBuilder.sigmoid(input_values[0])
            raise ValueError("sigmoid requires 1 input")

        elif op_name == "softmax":
            if input_values:
                dim = kwargs.get("dim", -1)
                return OpBuilder.softmax(input_values[0], dim)
            raise ValueError("softmax requires 1 input")

        elif op_name == "argmax":
            if input_values:
                dim = kwargs.get("dim", -1)
                return OpBuilder.argmax(input_values[0], dim)
            raise ValueError("argmax requires 1 input")

        elif op_name == "transpose":
            if input_values:
                dim0 = kwargs.get("dim0", 0)
                dim1 = kwargs.get("dim1", 1)
                return OpBuilder.transpose(input_values[0], (dim0, dim1))
            raise ValueError("transpose requires 1 input")

        elif op_name == "resize":
            if input_values:
                size = kwargs.get("size")
                if size is None and fx_node.kwargs:
                    size = fx_node.kwargs.get("size")
                if size:
                    return OpBuilder.resize(input_values[0], size)
            raise ValueError("resize requires 1 input and size")

        elif op_name == "to":
            if input_values:
                dtype = kwargs.get("dtype", "float32")
                device = kwargs.get("device", "cpu")
                return OpBuilder.to(input_values[0], dtype, device)
            raise ValueError("to requires 1 input")

        elif op_name == "normalize":
            if input_values:
                return OpBuilder.normalize(input_values[0])
            raise ValueError("normalize requires 1 input")

        elif op_name == "clone":
            if input_values:
                return OpBuilder.normalize(input_values[0])
            raise ValueError("clone requires 1 input")

        else:
            # Generic fallback
            if output_type is None:
                output_type = TensorType(shape=(1,), dtype="float32", device="cpu")
            output_value = Value(self._get_next_name(op_name), output_type)
            return Op(op_name, input_values, [output_value])
