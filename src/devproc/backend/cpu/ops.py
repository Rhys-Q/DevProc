"""
CPU Operation Lowering.

Maps IR operations to CPU executable code.
"""

from typing import Any, Dict, TYPE_CHECKING, Optional
import torch

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from devproc.ir.base import Op, Value
from devproc.ir.types import TensorType, StringType, TokenizerType, DictType

if TYPE_CHECKING:
    from devproc.backend.cpu.codegen import CPULoweringContext


# ==================== Tokenizer Operations ====================

def handle_load_tokenizer(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle load_tokenizer operation.

    Loads a tokenizer from the specified path or name.
    """
    tokenizer_path = op._tokenizer_path if hasattr(op, "_tokenizer_path") else "unknown"

    # Store tokenizer in context
    ctx.set_tokenizer(op.outputs[0].name, tokenizer_path)


def handle_tokenize_encode(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle tokenize_encode operation.

    Encodes text to token IDs using the tokenizer.
    """
    tokenizer_path = ctx.get_tokenizer(op.inputs[0].name)
    text = ctx.get_string(op.inputs[1].name)

    # Get tokenizer from cache or load
    tokenizer = ctx.get_cached_tokenizer(tokenizer_path)

    # Encode text
    max_length = op._max_length if hasattr(op, "_max_length") else 2048
    token_ids = tokenizer.encode(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    # Convert to tensor
    output_tensor = torch.tensor(token_ids, dtype=torch.int32)
    ctx.set_tensor(op.outputs[0].name, output_tensor)


def handle_tokenize_decode(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle tokenize_decode operation.

    Decodes token IDs to text using the tokenizer.
    """
    tokenizer_path = ctx.get_tokenizer(op.inputs[0].name)
    token_ids_tensor = ctx.get_tensor(op.inputs[1].name)

    # Get tokenizer from cache
    tokenizer = ctx.get_cached_tokenizer(tokenizer_path)

    # Decode (handle padding)
    token_ids = token_ids_tensor.tolist()
    # Remove padding tokens
    token_ids = [t for t in token_ids if t != tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else token_ids

    text = tokenizer.decode(token_ids)
    ctx.set_string(op.outputs[0].name, text)


# ==================== String Operations ====================

def handle_string_length(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle string_length operation.

    Gets the length of a string.
    """
    text = ctx.get_string(op.inputs[0].name)
    length = len(text)

    # Create scalar tensor
    output_tensor = torch.tensor(length, dtype=torch.int32)
    ctx.set_tensor(op.outputs[0].name, output_tensor)


def handle_string_concat(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle string_concat operation.

    Concatenates two strings.
    """
    a = ctx.get_string(op.inputs[0].name)
    b = ctx.get_string(op.inputs[1].name)

    result = a + b
    ctx.set_string(op.outputs[0].name, result)


def handle_string_slice(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle string_slice operation.

    Slices a string.
    """
    text = ctx.get_string(op.inputs[0].name)
    start = op._start if hasattr(op, "_start") else 0
    end = op._end if hasattr(op, "_end") else None

    result = text[start:end]
    ctx.set_string(op.outputs[0].name, result)


def handle_string_format(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle string_format operation.

    Formats a string with arguments (simplified implementation).
    """
    template = ctx.get_string(op.inputs[0].name)

    # For MVP, just return the template
    # A full implementation would handle format arguments
    ctx.set_string(op.outputs[0].name, template)


# ==================== Dict Operations ====================

def handle_dict_create(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle dict_create operation.

    Creates an empty dictionary.
    """
    # Get dict type from output
    dict_type = op.outputs[0].type

    # Create empty dict
    d = {}
    ctx.set_dict(op.outputs[0].name, d)


def handle_dict_get(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle dict_get operation.

    Gets a value from dictionary by key.
    """
    d = ctx.get_dict(op.inputs[0].name)
    key = ctx.get_string(op.inputs[1].name)

    # Get value (returns None if key not found)
    value = d.get(key)

    if value is not None:
        ctx.set_tensor(op.outputs[0].name, value)


def handle_dict_set(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle dict_set operation.

    Sets a value in dictionary by key.
    """
    d = ctx.get_dict(op.inputs[0].name)
    key = ctx.get_string(op.inputs[1].name)
    value = ctx.get_tensor(op.inputs[2].name)

    # Update dict (create a new dict for immutability)
    d[key] = value.clone()
    ctx.set_dict(op.outputs[0].name, d)


def handle_dict_size(
    op: Op, ctx: "CPULoweringContext", device_id: int = 0
) -> None:
    """
    Handle dict_size operation.

    Gets the size of dictionary.
    """
    d = ctx.get_dict(op.inputs[0].name)
    size = len(d)

    output_tensor = torch.tensor(size, dtype=torch.int32)
    ctx.set_tensor(op.outputs[0].name, output_tensor)


# ==================== Dispatcher ====================

OP_HANDLERS = {
    # Tokenizer
    "load_tokenizer": handle_load_tokenizer,
    "tokenize_encode": handle_tokenize_encode,
    "tokenize_decode": handle_tokenize_decode,

    # String
    "string_length": handle_string_length,
    "string_concat": handle_string_concat,
    "string_slice": handle_string_slice,
    "string_format": handle_string_format,

    # Dict
    "dict_create": handle_dict_create,
    "dict_get": handle_dict_get,
    "dict_set": handle_dict_set,
    "dict_size": handle_dict_size,
}


def handle_op(op: Op, ctx: "CPULoweringContext", device_id: int = 0) -> None:
    """
    Dispatch operation to appropriate handler.
    """
    handler = OP_HANDLERS.get(op.name)
    if handler is None:
        raise NotImplementedError(f"CPU operation '{op.name}' not implemented")

    handler(op, ctx, device_id)
