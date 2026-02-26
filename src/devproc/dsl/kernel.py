"""
DevProc DSL Kernel - @devproc.kernel decorator and context.

This module provides the @kernel decorator that converts a Python function
into a DevProc IR Function.
"""

from typing import Any, Callable, Optional, Dict
from devproc.ir.function import Function
from devproc.ir.base import Value
from devproc.ir.types import TensorType
from devproc.ir.ops import OpBuilder


class KernelContext:
    """
    Global kernel build context.

    This context is used during kernel function execution to track
    operations and build the IR.
    """

    _current: Optional["KernelContext"] = None

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.ir_function = Function(func_name)
        self.param_map: Dict[str, Value] = {}  # param_name -> IR Value
        self._output_tensor: Optional["KernelTensor"] = None

    @staticmethod
    def get_current() -> "KernelContext":
        """Get the current kernel context."""
        if KernelContext._current is None:
            raise RuntimeError("Not inside a @kernel function")
        return KernelContext._current

    def set_output(self, tensor: "KernelTensor") -> None:
        """Set the output tensor."""
        self._output_tensor = tensor


class KernelTensor:
    """
    Lightweight Tensor wrapper for DSL kernel functions.

    This is different from the Pipeline Tensor - it's designed for
    use in @kernel decorated functions.
    """

    def __init__(self, ir_value: Value, name: str = None):
        self._ir_value = ir_value
        self._name = name or ir_value.name

    @property
    def ir_value(self) -> Value:
        return self._ir_value

    @property
    def shape(self):
        return self._ir_value.type.shape

    @property
    def dtype(self):
        return self._ir_value.type.dtype

    @property
    def device(self):
        return self._ir_value.type.device

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"KernelTensor({self._name})"


def kernel(func: Callable) -> Callable:
    """
    @devproc.kernel decorator.

    Converts a Python function into a DevProc IR Function.

    Usage:
        @devproc.kernel
        def vision_preproc(img_path: devproc.String):
            img = devproc.load_image(img_path)
            img = devproc.resize(img, (224, 224))
            img = devproc.to(img, devproc.Float32)
            return img

    Args:
        func: The function to decorate

    Returns:
        A wrapper function that builds and returns the IR Function
    """
    def wrapper(*args, **kwargs):
        # Create the kernel context
        ctx = KernelContext(func.__name__)
        KernelContext._current = ctx

        try:
            # Get parameter names from function signature
            param_names = list(func.__code__.co_varnames[:func.__code__.co_argcount])

            # Now execute the function body
            # We need to wrap arguments in KernelTensor objects
            wrapped_args = []
            for i, param_name in enumerate(param_names):
                if i < len(args):
                    arg = args[i]
                    if isinstance(arg, KernelTensor):
                        wrapped_args.append(arg)
                    else:
                        # Need to create a KernelTensor for this arg
                        # Handle different arg types
                        if isinstance(arg, tuple) and all(isinstance(x, int) for x in arg):
                            # Shape tuple - create input
                            tensor_type = TensorType(arg, "float32", "cpu")
                            input_op = OpBuilder.input(param_name, tensor_type)
                            input_value = input_op.outputs[0]
                            ctx.ir_function.add_input(input_value)
                            ctx.param_map[param_name] = input_value
                            wrapped_args.append(KernelTensor(input_value, param_name))
                        elif isinstance(arg, str):
                            # String - create input
                            tensor_type = TensorType((1,), "int8", "cpu")
                            input_op = OpBuilder.input(param_name, tensor_type)
                            input_value = input_op.outputs[0]
                            ctx.ir_function.add_input(input_value)
                            ctx.param_map[param_name] = input_value
                            wrapped_args.append(KernelTensor(input_value, param_name))
                        elif isinstance(arg, (int, float)):
                            # Numeric - create scalar input
                            tensor_type = TensorType((1,), "float32", "cpu")
                            input_op = OpBuilder.input(param_name, tensor_type)
                            input_value = input_op.outputs[0]
                            ctx.ir_function.add_input(input_value)
                            ctx.param_map[param_name] = input_value
                            wrapped_args.append(KernelTensor(input_value, param_name))
                        else:
                            # Other - create placeholder
                            tensor_type = TensorType((1,), "float32", "cpu")
                            input_op = OpBuilder.input(param_name, tensor_type)
                            input_value = input_op.outputs[0]
                            ctx.ir_function.add_input(input_value)
                            ctx.param_map[param_name] = input_value
                            wrapped_args.append(KernelTensor(input_value, param_name))
                else:
                    wrapped_args.append(None)

            # Call the function with wrapped arguments
            result = func(*wrapped_args, **kwargs)

            # Set output if result is a KernelTensor
            if isinstance(result, KernelTensor):
                ctx.set_output(result)
                ctx.ir_function.output = result.ir_value

            return ctx.ir_function

        finally:
            KernelContext._current = None

    return wrapper
