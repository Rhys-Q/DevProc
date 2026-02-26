"""
DSL Type Annotations for DevProc.

These types are used for type annotations in @devproc.kernel functions.
They are NOT actual runtime types - they are used for documentation and
static analysis.
"""

#: String type for DSL
class String:
    """String type for DSL function parameters and return values."""
    pass


#: Tensor type for DSL
class Tensor:
    """Tensor type for DSL function parameters and return values."""
    pass


#: Float32 type annotation
class Float32:
    """Float32 type annotation."""
    pass


#: Float16 type annotation
class Float16:
    """Float16 type annotation."""
    pass


#: Int32 type annotation
class Int32:
    """Int32 type annotation."""
    pass


#: Int64 type annotation
class Int64:
    """Int64 type annotation."""
    pass


#: UInt8 type annotation
class UInt8:
    """UInt8 type annotation."""
    pass


#: Bool type annotation
class Bool:
    """Bool type annotation."""
    pass


#: Tokenizer type annotation
class Tokenizer:
    """Tokenizer type for DSL."""
    pass


#: TorchModel type annotation
class TorchModel:
    """Torch model type for DSL."""
    pass


#: KVCache type annotation
class KVCache:
    """KVCache type for LLM inference."""
    pass
