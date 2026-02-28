"""
FX Node to IR Op Mapping.

Maps torch.fx node target names to DevProc IR operation names.
"""

from typing import Dict, Optional, Callable


def get_fx_op_map() -> Dict[str, str]:
    """Get the mapping from FX node target names to IR op names.

    Returns:
        Dictionary mapping FX targets to IR op names.
    """
    return {
        # Linear / Matmul
        "linear": "linear",
        "aten.linear": "linear",
        "matmul": "matmul",
        "aten.matmul": "matmul",
        "mm": "matmul",
        "aten.mm": "matmul",
        "bmm": "bmm",
        "aten.bmm": "bmm",
        "addmm": "addmm",
        "aten.addmm": "addmm",

        # Shape ops
        "t": "transpose",
        "aten.t": "transpose",
        "transpose": "transpose",
        "aten.transpose": "transpose",

        # Activations (with and without aten. prefix)
        "relu": "relu",
        "aten.relu": "relu",
        "aten.relu_": "relu",
        "gelu": "gelu",
        "aten.gelu": "gelu",
        "sigmoid": "sigmoid",
        "aten.sigmoid": "sigmoid",
        "aten.sigmoid_": "sigmoid",
        "tanh": "tanh",
        "aten.tanh": "tanh",
        "aten.tanh_": "tanh",
        "softmax": "softmax",
        "aten.softmax": "softmax",
        "softplus": "softplus",
        "aten.softplus": "softplus",

        # Element-wise operations
        "add": "add",
        "aten.add": "add",
        "aten.add_": "add",
        "sub": "sub",
        "aten.sub": "sub",
        "aten.sub_": "sub",
        "mul": "mul",
        "aten.mul": "mul",
        "aten.mul_": "mul",
        "div": "div",
        "aten.div": "div",
        "aten.div_": "div",
        "pow": "pow",
        "aten.pow": "pow",
        "neg": "neg",
        "aten.neg": "neg",
        "abs": "abs",
        "aten.abs": "abs",

        # Shape operations
        "aten.transpose": "transpose",
        "aten.t": "transpose",
        "aten.view": "resize",
        "aten.reshape": "resize",
        "aten.flatten": "flatten",
        "aten.squeeze": "squeeze",
        "aten.unsqueeze": "unsqueeze",
        "aten.cat": "concat",
        "aten.stack": "stack",

        # Type conversion
        "aten.to": "to",
        "aten.clone": "clone",
        "aten.contiguous": "contiguous",
        "aten.type_as": "to",

        # Reduction operations
        "aten.argmax": "argmax",
        "aten.argmin": "argmin",
        "aten.sum": "sum",
        "aten.mean": "mean",
        "aten.max": "max",
        "aten.min": "min",

        # Comparison
        "aten.eq": "eq",
        "aten.ne": "ne",
        "aten.lt": "lt",
        "aten.le": "le",
        "aten.gt": "gt",
        "aten.ge": "ge",

        # Other
        "aten.select": "select",
        "aten.slice": "slice",
        "aten.where": "where",
        "aten.zeros": "zeros",
        "aten.ones": "ones",
        "aten.full": "full",
    }


def get_ir_op_handler(op_name: str) -> Optional[Callable]:
    """Get the handler function for a given IR op name.

    Args:
        op_name: The IR operation name

    Returns:
        Handler function or None if not supported
    """
    # This will be implemented in fx_converter.py
    return None
