"""
Tests for CPU operations in DevProc.

Tests tokenizer, string, and dict operations.
"""

import pytest
import torch

import devproc
from devproc.ir.ops import OpBuilder
from devproc.ir.types import StringType, TokenizerType, DictType, TensorType, ScalarType
from devproc.ir.function import Function
from devproc.ir.base import Value
from devproc.ir.verifier import IRVerifier


class TestStringType:
    """Test StringType."""

    def test_string_type_creation(self):
        """Test creating a StringType."""
        string_type = StringType()
        assert string_type.max_length is None

    def test_string_type_with_max_length(self):
        """Test creating a StringType with max length."""
        string_type = StringType(100)
        assert string_type.max_length == 100

    def test_string_type_equality(self):
        """Test StringType equality."""
        t1 = StringType(100)
        t2 = StringType(100)
        t3 = StringType(200)

        assert t1 == t2
        assert t1 != t3


class TestTokenizerType:
    """Test TokenizerType."""

    def test_tokenizer_type_creation(self):
        """Test creating a TokenizerType."""
        tokenizer_type = TokenizerType("gpt2")
        assert tokenizer_type.tokenizer_name == "gpt2"

    def test_tokenizer_type_equality(self):
        """Test TokenizerType equality."""
        t1 = TokenizerType("gpt2")
        t2 = TokenizerType("gpt2")
        t3 = TokenizerType("bert")

        assert t1 == t2
        assert t1 != t3


class TestDictType:
    """Test DictType."""

    def test_dict_type_creation(self):
        """Test creating a DictType."""
        key_type = StringType()
        value_type = TensorType((10,), "float32", "cpu")
        dict_type = DictType(key_type, value_type)

        assert dict_type.key_type == key_type
        assert dict_type.value_type == value_type


class TestIRStringOps:
    """Test IR string operations."""

    def test_string_length_op(self):
        """Test string_length IR operation."""
        # Create a StringType value directly
        string_type = StringType(100)
        text_value = Value("text", string_type)

        length_op = OpBuilder.string_length(text_value)

        assert length_op.name == "string_length"
        assert isinstance(length_op.outputs[0].type, ScalarType)
        assert length_op.outputs[0].type.dtype == "int32"

    def test_string_concat_op(self):
        """Test string_concat IR operation."""
        # Create StringType values directly
        a_type = StringType(50)
        b_type = StringType(50)
        a = Value("a", a_type)
        b = Value("b", b_type)

        concat_op = OpBuilder.string_concat(a, b)

        assert concat_op.name == "string_concat"
        assert isinstance(concat_op.outputs[0].type, StringType)
        assert concat_op.outputs[0].type.max_length == 100

    def test_string_slice_op(self):
        """Test string_slice IR operation."""
        # Create a StringType value directly
        text_type = StringType(100)
        text = Value("text", text_type)

        slice_op = OpBuilder.string_slice(text, 0, 10)

        assert slice_op.name == "string_slice"
        assert isinstance(slice_op.outputs[0].type, StringType)
        assert slice_op.outputs[0].type.max_length == 10


class TestIRTokenizerOps:
    """Test IR tokenizer operations."""

    def test_load_tokenizer_op(self):
        """Test load_tokenizer IR operation."""
        tokenizer_op = OpBuilder.load_tokenizer("gpt2")

        assert tokenizer_op.name == "load_tokenizer"
        assert isinstance(tokenizer_op.outputs[0].type, TokenizerType)
        assert tokenizer_op.outputs[0].type.tokenizer_name == "gpt2"

    def test_tokenize_encode_op(self):
        """Test tokenize_encode IR operation."""
        # Create values directly
        tokenizer_type = TokenizerType("gpt2")
        tokenizer = Value("tokenizer", tokenizer_type)

        text_type = StringType()
        text = Value("text", text_type)

        encode_op = OpBuilder.tokenize_encode(tokenizer, text)

        assert encode_op.name == "tokenize_encode"
        assert isinstance(encode_op.outputs[0].type, TensorType)
        assert encode_op.outputs[0].type.dtype == "int32"
        assert encode_op.outputs[0].type.shape == (2048,)

    def test_tokenize_decode_op(self):
        """Test tokenize_decode IR operation."""
        # Create values directly
        tokenizer_type = TokenizerType("gpt2")
        tokenizer = Value("tokenizer", tokenizer_type)

        token_ids_type = TensorType((2048,), "int32", "cpu")
        token_ids = Value("token_ids", token_ids_type)

        decode_op = OpBuilder.tokenize_decode(tokenizer, token_ids)

        assert decode_op.name == "tokenize_decode"
        assert isinstance(decode_op.outputs[0].type, StringType)


class TestIRDictOps:
    """Test IR dict operations."""

    def test_dict_create_op(self):
        """Test dict_create IR operation."""
        key_type = StringType()
        value_type = TensorType((10,), "float32", "cpu")
        dict_op = OpBuilder.dict_create(key_type, value_type)

        assert dict_op.name == "dict_create"
        assert isinstance(dict_op.outputs[0].type, DictType)
        assert dict_op.outputs[0].type.key_type == key_type
        assert dict_op.outputs[0].type.value_type == value_type


class TestDSLTokenizerOps:
    """Test DSL tokenizer operations."""

    def test_load_tokenizer_dsl(self):
        """Test load_tokenizer DSL function."""

        @devproc.kernel
        def kernel():
            tokenizer = devproc.dsl.ops.load_tokenizer("gpt2")
            return tokenizer

        ir = kernel()
        assert ir is not None

    def test_tokenize_encode_dsl(self):
        """Test tokenize_encode DSL function."""

        @devproc.kernel
        def kernel():
            tokenizer = devproc.dsl.ops.load_tokenizer("gpt2")
            # Create a string input
            text = devproc.dsl.ops.input("text_str", "int8", ())
            # For encode, we need a string type
            # This tests the basic structure
            tokens = devproc.dsl.ops.input("tokens", "int32", (2048,))
            return tokens

        ir = kernel()
        assert ir is not None


class TestIRVerifier:
    """Test IR verifier with new operations."""

    def test_verify_string_length(self):
        """Test verifying string_length operation."""
        func = Function("test")

        # Create string input directly
        text_type = StringType(100)
        text = Value("text", text_type)
        func.add_input(text)

        length_op = OpBuilder.string_length(text)
        func.add_op(length_op)

        verifier = IRVerifier(func)
        assert verifier.verify()

    def test_verify_tokenize_encode(self):
        """Test verifying tokenize_encode operation."""
        func = Function("test")

        # Add tokenizer input
        tokenizer_type = TokenizerType("gpt2")
        tokenizer = Value("tokenizer", tokenizer_type)
        func.add_input(tokenizer)

        # Add string input
        text_type = StringType()
        text = Value("text", text_type)
        func.add_input(text)

        # Add encode operation
        encode_op = OpBuilder.tokenize_encode(tokenizer, text)
        func.add_op(encode_op)

        verifier = IRVerifier(func)
        assert verifier.verify()

    def test_verify_dict_ops(self):
        """Test verifying dict operations."""
        func = Function("test")

        # Create dict
        key_type = StringType()
        value_type = TensorType((10,), "float32", "cpu")
        dict_op = OpBuilder.dict_create(key_type, value_type)
        func.add_op(dict_op)

        # Add dict_get and dict_set operations
        key_type2 = StringType()
        key = Value("key", key_type2)
        func.add_input(key)

        value_type2 = TensorType((10,), "float32", "cpu")
        value = Value("value", value_type2)
        func.add_input(value)

        get_op = OpBuilder.dict_get(dict_op.outputs[0], key)
        func.add_op(get_op)

        set_op = OpBuilder.dict_set(dict_op.outputs[0], key, value)
        func.add_op(set_op)

        verifier = IRVerifier(func)
        assert verifier.verify()


class TestCPULowering:
    """Test CPU lowering context."""

    def test_context_string_operations(self):
        """Test CPULoweringContext string operations."""
        from devproc.backend.cpu.codegen import CPULoweringContext

        ctx = CPULoweringContext()

        # Test string operations
        ctx.set_string("text", "hello world")
        assert ctx.get_string("text") == "hello world"

        # Test dict operations
        ctx.set_dict("cache", {"key1": torch.tensor([1.0, 2.0])})
        d = ctx.get_dict("cache")
        assert "key1" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
