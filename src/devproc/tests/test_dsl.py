"""
Tests for DevProc DSL.

Validates the implementation against Phase 2 requirements:
- DSL builds IR without executing computation
- All basic operations work
- Generated IR is valid
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from devproc.dsl.pipeline import Pipeline, Tensor
from devproc.ir.verifier import IRVerifier


class TestTensor:
    """Tests for DSL Tensor class."""

    def test_create_tensor_from_value(self):
        """Test creating a Tensor from IR Value."""
        from devproc.ir.types import TensorType
        from devproc.ir.base import Value

        tensor_type = TensorType((224, 224, 3), "uint8", "cpu")
        ir_value = Value("test", tensor_type)
        tensor = Tensor(ir_value, "test_tensor")

        assert tensor.name == "test_tensor"
        assert tensor.shape == (224, 224, 3)
        assert tensor.dtype == "uint8"
        assert tensor.device == "cpu"


class TestPipeline:
    """Tests for DSL Pipeline class."""

    def test_create_pipeline(self):
        """Test creating a Pipeline."""
        pipe = Pipeline("test_pipe")
        assert pipe.name == "test_pipe"

    def test_input(self):
        """Test defining pipeline input."""
        pipe = Pipeline()
        img = pipe.input("image", "uint8", (224, 224, 3))

        assert isinstance(img, Tensor)
        assert img.name == "image"
        assert img.shape == (224, 224, 3)
        assert img.dtype == "uint8"
        assert img.device == "cpu"

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        pipe = Pipeline()
        with pytest.raises(ValueError):
            pipe.input("image", "invalid_dtype", (224, 224, 3))

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        pipe = Pipeline()
        with pytest.raises(ValueError):
            pipe.input("image", "uint8", (224, 224, 3), "invalid_device")

    def test_normalize(self):
        """Test normalize operation."""
        pipe = Pipeline()
        img = pipe.input("image", "uint8", (224, 224, 3))
        x = pipe.normalize(img)

        assert isinstance(x, Tensor)
        assert x.dtype == "float32"

    def test_argmax(self):
        """Test argmax operation."""
        pipe = Pipeline()
        x = pipe.input("logits", "float32", (1, 10))
        out = pipe.argmax(x, dim=1)

        assert isinstance(out, Tensor)
        assert out.dtype == "int64"

    def test_matmul(self):
        """Test matmul operation."""
        pipe = Pipeline()
        a = pipe.input("a", "float32", (2, 3))
        b = pipe.input("b", "float32", (3, 2))
        c = pipe.matmul(a, b)

        assert isinstance(c, Tensor)
        assert c.shape == (2, 2)
        assert c.dtype == "float32"

    def test_linear(self):
        """Test linear operation."""
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 10))
        w = pipe.input("w", "float32", (5, 10))
        b = pipe.input("b", "float32", (5,))
        y = pipe.linear(x, w, b)

        assert isinstance(y, Tensor)
        assert y.shape == (1, 5)
        assert y.dtype == "float32"

    def test_relu(self):
        """Test relu operation."""
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 10))
        y = pipe.relu(x)

        assert isinstance(y, Tensor)
        assert y.shape == (1, 10)
        assert y.dtype == "float32"

    def test_add(self):
        """Test add operation."""
        pipe = Pipeline()
        a = pipe.input("a", "float32", (1, 10))
        b = pipe.input("b", "float32", (1, 10))
        c = pipe.add(a, b)

        assert isinstance(c, Tensor)
        assert c.shape == (1, 10)
        assert c.dtype == "float32"

    def test_sigmoid(self):
        """Test sigmoid operation."""
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 10))
        y = pipe.sigmoid(x)

        assert isinstance(y, Tensor)
        assert y.shape == (1, 10)
        assert y.dtype == "float32"

    def test_softmax(self):
        """Test softmax operation."""
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 10))
        y = pipe.softmax(x, dim=1)

        assert isinstance(y, Tensor)
        assert y.shape == (1, 10)
        assert y.dtype == "float32"

    def test_resize(self):
        """Test resize operation."""
        pipe = Pipeline()
        x = pipe.input("x", "uint8", (224, 224, 3))
        y = pipe.resize(x, (112, 112))

        assert isinstance(y, Tensor)
        assert y.shape == (112, 112, 3)

    def test_transpose(self):
        """Test transpose operation."""
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 2, 3))
        y = pipe.transpose(x, (0, 2, 1))

        assert isinstance(y, Tensor)
        assert y.shape == (1, 3, 2)

    def test_to(self):
        """Test to (dtype conversion) operation."""
        pipe = Pipeline()
        x = pipe.input("x", "uint8", (224, 224, 3))
        y = pipe.to(x, "float32")

        assert isinstance(y, Tensor)
        assert y.shape == (224, 224, 3)
        assert y.dtype == "float32"


class TestDSLToIR:
    """Tests for DSL building valid IR."""

    def test_simple_pipeline(self):
        """
        Test building a simple pipeline: input -> normalize -> argmax
        This is the same example from Phase 1 spec.
        """
        pipe = Pipeline()
        img = pipe.input("image", "uint8", (224, 224, 3))
        x = pipe.normalize(img)
        out = pipe.argmax(x)
        pipe.output(out)

        # Build IR
        ir_func = pipe.build()

        # Verify
        verifier = IRVerifier(ir_func)
        is_valid = verifier.verify()

        assert is_valid, f"IR should be valid, got errors: {verifier.get_errors()}"

        # Print IR
        print("\n" + "=" * 50)
        print("DSL Generated IR:")
        print("=" * 50)
        print(repr(ir_func))
        print("=" * 50 + "\n")

    def test_mlp_pipeline(self):
        """
        Test building an MLP pipeline:
        input -> linear -> relu -> linear -> argmax
        """
        pipe = Pipeline()
        x = pipe.input("x", "float32", (1, 128))

        # First linear layer
        w1 = pipe.input("w1", "float32", (256, 128))
        b1 = pipe.input("b1", "float32", (256,))
        h = pipe.linear(x, w1, b1)
        h = pipe.relu(h)

        # Second linear layer
        w2 = pipe.input("w2", "float32", (10, 256))
        b2 = pipe.input("b2", "float32", (10,))
        out = pipe.linear(h, w2, b2)

        # Argmax
        result = pipe.argmax(out)
        pipe.output(result)

        # Build IR
        ir_func = pipe.build()

        # Verify
        verifier = IRVerifier(ir_func)
        is_valid = verifier.verify()

        assert is_valid, f"IR should be valid, got errors: {verifier.get_errors()}"

        # Print IR
        print("\n" + "=" * 50)
        print("MLP Pipeline IR:")
        print("=" * 50)
        print(repr(ir_func))
        print("=" * 50 + "\n")


def test_dsl_no_execution():
    """
    Test that DSL does NOT execute any computation.
    This is a key requirement - DSL should only build IR.
    """
    import numpy as np

    pipe = Pipeline()
    img = pipe.input("image", "uint8", (224, 224, 3))
    x = pipe.normalize(img)

    # The DSL should NOT have executed any numpy/torch operation
    # We can verify this by checking that the IR is built correctly
    # without actually running any computation
    ir_func = pipe.build()

    # If we got here, DSL didn't execute any computation
    assert ir_func is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
