"""
Tests for @devproc.kernel decorator style DSL.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import devproc
from devproc.ir.verifier import IRVerifier


class TestKernelDecorator:
    """Tests for @devproc.kernel decorator."""

    def test_simple_kernel(self):
        """Test a simple kernel function."""

        @devproc.kernel
        def test_func(x):
            return x

        # Call the kernel
        ir = test_func((10,))

        assert ir is not None
        print("\n" + "=" * 50)
        print("Simple kernel IR:")
        print("=" * 50)
        print(repr(ir))
        print("=" * 50)

    def test_vision_preproc(self):
        """Test the vision_preproc example from README."""

        @devproc.kernel
        def vision_preproc(img_path):
            img = devproc.load_image(img_path)
            img = devproc.resize(img, (224, 224))
            img = devproc.to(img, devproc.Float32)
            return img

        # Call the kernel
        ir = vision_preproc("test.jpg")

        assert ir is not None
        print("\n" + "=" * 50)
        print("Vision preproc IR:")
        print("=" * 50)
        print(repr(ir))
        print("=" * 50)

    def test_normalize_argmax(self):
        """Test normalize -> argmax pipeline."""

        @devproc.kernel
        def simple_pipeline(x):
            y = devproc.normalize(x)
            out = devproc.argmax(y)
            return out

        # Call the kernel with a tensor shape
        ir = simple_pipeline((224, 224, 3))

        # Verify
        verifier = IRVerifier(ir)
        is_valid = verifier.verify()

        print("\n" + "=" * 50)
        print("Normalize-Argmax IR:")
        print("=" * 50)
        print(repr(ir))
        print(f"Valid: {is_valid}")
        if not is_valid:
            print("Errors:", verifier.get_errors())
        print("=" * 50)

        assert is_valid, f"IR should be valid, got: {verifier.get_errors()}"

    def test_mlp_kernel(self):
        """Test MLP: linear -> relu -> linear -> argmax."""

        @devproc.kernel
        def mlp(x):
            # First layer
            w1 = devproc.input("w1", "float32", (256, 128))
            b1 = devproc.input("b1", "float32", (256,))
            h = devproc.linear(x, w1, b1)
            h = devproc.relu(h)

            # Second layer
            w2 = devproc.input("w2", "float32", (10, 256))
            b2 = devproc.input("b2", "float32", (10,))
            out = devproc.linear(h, w2, b2)

            # Argmax
            result = devproc.argmax(out)
            return result

        # Call with input shape
        ir = mlp((1, 128))

        # Verify
        verifier = IRVerifier(ir)
        is_valid = verifier.verify()

        print("\n" + "=" * 50)
        print("MLP IR:")
        print("=" * 50)
        print(repr(ir))
        print(f"Valid: {is_valid}")
        if not is_valid:
            print("Errors:", verifier.get_errors())
        print("=" * 50)

        assert is_valid, f"IR should be valid, got: {verifier.get_errors()}"


def test_types_import():
    """Test that DSL types can be imported."""
    from devproc import String, Tensor, Float32, Int32
    assert String is not None
    assert Tensor is not None
    assert Float32 is not None
    assert Int32 is not None


def test_module_functions():
    """Test that module-level functions can be imported."""
    import devproc
    assert devproc.load_image is not None
    assert devproc.resize is not None
    assert devproc.to is not None
    assert devproc.normalize is not None
    assert devproc.argmax is not None
    assert devproc.relu is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
