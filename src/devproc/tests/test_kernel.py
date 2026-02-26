"""
Tests for @devproc.kernel decorator style DSL.
"""

import pytest
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import devproc
from devproc.ir.verifier import IRVerifier


class TestKernelDecorator:
    """Tests for @devproc.kernel decorator."""

    def test_simple_kernel(self):
        """Test a simple kernel function."""

        @devproc.kernel
        def test_func(x: torch.Tensor):
            return x

        # Example input
        x = torch.randn(10,)

        # Use parse_ir to get IR
        ir = devproc.parse_ir(test_func, x)

        assert ir is not None
        print("\n" + "=" * 50)
        print("Simple kernel IR:")
        print("=" * 50)
        print(repr(ir))
        print("=" * 50)

    def test_vision_preproc(self):
        """Test the vision_preproc example from README."""

        @devproc.kernel
        def vision_preproc(img: torch.Tensor):
            img = devproc.resize(img, (224, 224))
            img = devproc.to(img, devproc.Float32)
            return img

        # Example input
        img = torch.randn(256, 256, 3)

        # Use parse_ir to get IR
        ir = devproc.parse_ir(vision_preproc, img)

        assert ir is not None
        print("\n" + "=" * 50)
        print("Vision preproc IR:")
        print("=" * 50)
        print(repr(ir))
        print("=" * 50)

    def test_normalize_argmax(self):
        """Test normalize -> argmax pipeline."""

        @devproc.kernel
        def simple_pipeline(x: torch.Tensor):
            y = devproc.normalize(x)
            out = devproc.argmax(y)
            return out

        # Example input
        x = torch.randn(224, 224, 3)

        # Use parse_ir to get IR
        ir = devproc.parse_ir(simple_pipeline, x)

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
        def mlp(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                w2: torch.Tensor, b2: torch.Tensor):
            # First layer
            h = devproc.linear(x, w1, b1)
            h = devproc.relu(h)

            # Second layer
            out = devproc.linear(h, w2, b2)

            # Argmax
            result = devproc.argmax(out)
            return result

        # Example inputs
        x = torch.randn(1, 128)
        w1 = torch.randn(256, 128)
        b1 = torch.randn(256,)
        w2 = torch.randn(10, 256)
        b2 = torch.randn(10,)

        # Use parse_ir to get IR
        ir = devproc.parse_ir(mlp, x, w1, b1, w2, b2)
        ir.infer_types()
        breakpoint()

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
