"""
CPU Code Generator.

Generates executable Python code from DevProc IR for CPU execution.
"""

from typing import Any, Dict, Optional, Set
import torch

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from devproc.ir.function import Function
from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType, StringType, TokenizerType, DictType


class CPULoweringContext:
    """Context for lowering IR operations to CPU code."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.tensor_map: Dict[str, torch.Tensor] = {}
        self.string_map: Dict[str, str] = {}
        self.dict_map: Dict[str, Dict[str, torch.Tensor]] = {}
        self.tokenizer_paths: Dict[str, str] = {}
        self.tokenizer_cache: Dict[str, Any] = {}

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Set a tensor by name."""
        self.tensor_map[name] = tensor

    def get_tensor(self, name: str) -> torch.Tensor:
        """Get a tensor by name."""
        return self.tensor_map.get(name)

    def set_string(self, name: str, text: str) -> None:
        """Set a string by name."""
        self.string_map[name] = text

    def get_string(self, name: str) -> str:
        """Get a string by name."""
        return self.string_map.get(name, "")

    def set_dict(self, name: str, d: Dict[str, torch.Tensor]) -> None:
        """Set a dict by name."""
        self.dict_map[name] = d

    def get_dict(self, name: str) -> Dict[str, torch.Tensor]:
        """Get a dict by name."""
        return self.dict_map.get(name, {})

    def set_tokenizer(self, name: str, path: str) -> None:
        """Set a tokenizer path by name."""
        self.tokenizer_paths[name] = path

    def get_tokenizer(self, name: str) -> str:
        """Get a tokenizer path by name."""
        return self.tokenizer_paths.get(name, "unknown")

    def get_cached_tokenizer(self, path: str) -> Any:
        """Get a cached tokenizer or load it."""
        if path not in self.tokenizer_cache:
            try:
                self.tokenizer_cache[path] = AutoTokenizer.from_pretrained(path)
            except Exception:
                # Return a dummy tokenizer for testing
                self.tokenizer_cache[path] = None
        return self.tokenizer_cache[path]


class CPUCodeGenerator:
    """Generates CPU executable code from IR."""

    def __init__(self):
        self.context = CPULoweringContext()

    def generate(self, ir_function: Function) -> str:
        """
        Generate Python code from IR function.

        Args:
            ir_function: The IR Function to generate code for

        Returns:
            Generated Python code as string
        """
        code_lines = [
            "'''",
            "Generated CPU Code for DevProc",
            "'''",
            "",
            "import torch",
            "from typing import Dict",
            "",
        ]

        # Add input handling
        for input_val in ir_function.inputs:
            code_lines.append(f"# Input: {input_val.name} : {input_val.type}")

        # Generate main execution code
        code_lines.append("")
        code_lines.append("def execute(**inputs):")
        code_lines.append("    '''Execute the compiled function.'''")
        code_lines.append("")

        # Initialize context
        code_lines.append("    # Initialize context")
        code_lines.append("    ctx = CPULoweringContext()")
        code_lines.append("")

        # Process inputs
        for input_val in ir_function.inputs:
            input_type = input_val.type
            if isinstance(input_type, TensorType):
                code_lines.append(f"    ctx.set_tensor('{input_val.name}', inputs['{input_val.name}'])")
            elif isinstance(input_type, StringType):
                code_lines.append(f"    ctx.set_string('{input_val.name}', inputs['{input_val.name}'])")
            elif isinstance(input_type, DictType):
                code_lines.append(f"    ctx.set_dict('{input_val.name}', inputs['{input_val.name}'])")
            elif isinstance(input_type, TokenizerType):
                code_lines.append(f"    ctx.set_tokenizer('{input_val.name}', '{input_type.tokenizer_name}')")

        code_lines.append("")

        # Process operations
        for op in ir_function.block.ops:
            code_lines.extend(self._generate_op_code(op))

        code_lines.append("")
        code_lines.append("    # Collect outputs")
        code_lines.append("    return {")

        # Collect outputs
        for output_val in ir_function.outputs:
            output_type = output_val.type
            if isinstance(output_type, TensorType):
                code_lines.append(f"        '{output_val.name}': ctx.get_tensor('{output_val.name}'),")
            elif isinstance(output_type, StringType):
                code_lines.append(f"        '{output_val.name}': ctx.get_string('{output_val.name}'),")
            elif isinstance(output_type, DictType):
                code_lines.append(f"        '{output_val.name}': ctx.get_dict('{output_val.name}'),")

        code_lines.append("    }")

        return "\n".join(code_lines)

    def _generate_op_code(self, op: Op) -> list:
        """Generate code for a single operation."""
        code_lines = [f"    # {op.name}"]

        if op.name == "load_tokenizer":
            path = op._tokenizer_path if hasattr(op, "_tokenizer_path") else "unknown"
            code_lines.append(f"    ctx.set_tokenizer('{op.outputs[0].name}', '{path}')")

        elif op.name == "tokenize_encode":
            code_lines.append(f"    # tokenize_encode: {op.inputs[0].name} + {op.inputs[1].name} -> {op.outputs[0].name}")
            code_lines.append(f"    tokenizer = ctx.get_cached_tokenizer(ctx.get_tokenizer('{op.inputs[0].name}'))")
            code_lines.append(f"    text = ctx.get_string('{op.inputs[1].name}')")
            code_lines.append(f"    max_length = {op._max_length if hasattr(op, '_max_length') else 2048}")
            code_lines.append(f"    token_ids = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')")
            code_lines.append(f"    ctx.set_tensor('{op.outputs[0].name}', torch.tensor(token_ids, dtype=torch.int32))")

        elif op.name == "tokenize_decode":
            code_lines.append(f"    # tokenize_decode: {op.inputs[0].name} + {op.inputs[1].name} -> {op.outputs[0].name}")
            code_lines.append(f"    tokenizer = ctx.get_cached_tokenizer(ctx.get_tokenizer('{op.inputs[0].name}'))")
            code_lines.append(f"    token_ids = ctx.get_tensor('{op.inputs[1].name}')")
            code_lines.append(f"    text = tokenizer.decode(token_ids.tolist())")
            code_lines.append(f"    ctx.set_string('{op.outputs[0].name}', text)")

        elif op.name == "string_length":
            code_lines.append(f"    text = ctx.get_string('{op.inputs[0].name}')")
            code_lines.append(f"    length = len(text)")
            code_lines.append(f"    ctx.set_tensor('{op.outputs[0].name}', torch.tensor(length, dtype=torch.int32))")

        elif op.name == "string_concat":
            code_lines.append(f"    a = ctx.get_string('{op.inputs[0].name}')")
            code_lines.append(f"    b = ctx.get_string('{op.inputs[1].name}')")
            code_lines.append(f"    ctx.set_string('{op.outputs[0].name}', a + b)")

        elif op.name == "string_slice":
            start = op._start if hasattr(op, "_start") else 0
            end = op._end if hasattr(op, "_end") else "None"
            code_lines.append(f"    text = ctx.get_string('{op.inputs[0].name}')")
            code_lines.append(f"    ctx.set_string('{op.outputs[0].name}', text[{start}:{end}])")

        elif op.name == "dict_create":
            code_lines.append(f"    ctx.set_dict('{op.outputs[0].name}', {{}})")

        elif op.name == "dict_get":
            code_lines.append(f"    d = ctx.get_dict('{op.inputs[0].name}')")
            code_lines.append(f"    key = ctx.get_string('{op.inputs[1].name}')")
            code_lines.append(f"    value = d.get(key)")
            code_lines.append(f"    if value is not None:")
            code_lines.append(f"        ctx.set_tensor('{op.outputs[0].name}', value)")

        elif op.name == "dict_set":
            code_lines.append(f"    d = ctx.get_dict('{op.inputs[0].name}').copy()")
            code_lines.append(f"    key = ctx.get_string('{op.inputs[1].name}')")
            code_lines.append(f"    value = ctx.get_tensor('{op.inputs[2].name}')")
            code_lines.append(f"    d[key] = value.clone()")
            code_lines.append(f"    ctx.set_dict('{op.outputs[0].name}', d)")

        elif op.name == "dict_size":
            code_lines.append(f"    d = ctx.get_dict('{op.inputs[0].name}')")
            code_lines.append(f"    ctx.set_tensor('{op.outputs[0].name}', torch.tensor(len(d), dtype=torch.int32))")

        code_lines.append("")

        return code_lines

    def execute(self, ir_function: Function, **inputs) -> Dict[str, Any]:
        """
        Execute the IR function directly using the lowering context.

        Args:
            ir_function: The IR Function to execute
            **inputs: Input tensors and values

        Returns:
            Dictionary of output tensors and values
        """
        from devproc.backend.cpu import ops as cpu_ops

        ctx = CPULoweringContext()

        # Process inputs
        for input_val in ir_function.inputs:
            input_name = input_val.name
            if input_name in inputs:
                input_value = inputs[input_name]
                input_type = input_val.type
                if isinstance(input_type, TensorType):
                    ctx.set_tensor(input_name, input_value)
                elif isinstance(input_type, StringType):
                    ctx.set_string(input_name, input_value)
                elif isinstance(input_type, DictType):
                    ctx.set_dict(input_name, input_value)
                elif isinstance(input_type, TokenizerType):
                    ctx.set_tokenizer(input_name, input_value)

        # Execute operations
        for op in ir_function.block.ops:
            cpu_ops.handle_op(op, ctx)

        # Collect outputs
        outputs = {}
        for output_val in ir_function.outputs:
            output_type = output_val.type
            if isinstance(output_type, TensorType):
                outputs[output_val.name] = ctx.get_tensor(output_val.name)
            elif isinstance(output_type, StringType):
                outputs[output_val.name] = ctx.get_string(output_val.name)
            elif isinstance(output_type, DictType):
                outputs[output_val.name] = ctx.get_dict(output_val.name)

        return outputs
