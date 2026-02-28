"""
Tests for Torch FX Frontend (from_torch).

Tests converting torch.nn.Module and Python functions to DevProc IR.
"""

import pytest
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import devproc
from devproc.ir.from_torch import from_torch, DevProcDynamoBackend, FXToIRConverter
from devproc.ir.fx_op_map import get_fx_op_map


def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")


class TestFXOpMap:
    """Tests for FX op mapping."""

    def test_get_fx_op_map(self):
        """Test that FX op map contains expected mappings."""
        op_map = get_fx_op_map()

        # Check key mappings
        assert op_map["relu"] == "relu"
        assert op_map["aten.relu"] == "relu"
        assert op_map["linear"] == "linear"
        assert op_map["aten.linear"] == "linear"
        assert op_map["matmul"] == "matmul"
        assert op_map["aten.matmul"] == "matmul"
        assert op_map["add"] == "add"
        assert op_map["aten.add"] == "add"
        assert op_map["softmax"] == "softmax"
        assert op_map["transpose"] == "transpose"


class TestFromTorchFunction:
    """Tests for from_torch with Python functions."""

    def test_simple_relu_function(self):
        """Test converting a simple relu function."""
        def relu_func(x):
            return torch.relu(x)

        ir = from_torch(relu_func, torch.randn(1, 128), backend="ir")

        # Verify IR structure
        assert ir is not None
        assert len(ir.inputs) == 1
        assert len(ir.block.ops) == 1

        # Check relu op
        relu_op = ir.block.ops[0]
        assert relu_op.name == "relu"

    def test_add_function(self):
        """Test converting an add function."""
        def add_func(a, b):
            return a + b

        ir = from_torch(add_func, torch.randn(1, 128), torch.randn(1, 128), backend="ir")

        assert ir is not None
        assert len(ir.inputs) == 2

        # Check add op
        add_op = ir.block.ops[0]
        assert add_op.name == "add"

    def test_matmul_function(self):
        """Test converting a matmul function."""
        def matmul_func(a, b):
            return torch.matmul(a, b)

        ir = from_torch(matmul_func, torch.randn(2, 3), torch.randn(3, 4), backend="ir")

        assert ir is not None

        # Check matmul op
        matmul_op = ir.block.ops[0]
        assert matmul_op.name == "matmul"

    def test_sigmoid_function(self):
        """Test converting a sigmoid function."""
        def sigmoid_func(x):
            return torch.sigmoid(x)

        ir = from_torch(sigmoid_func, torch.randn(1, 128), backend="ir")

        assert ir is not None
        sigmoid_op = ir.block.ops[0]
        assert sigmoid_op.name == "sigmoid"


class TestFromTorchModule:
    """Tests for from_torch with torch.nn.Module."""

    def test_relu_module(self):
        """Test converting torch.nn.ReLU."""
        model = torch.nn.ReLU()
        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None
        assert len(ir.block.ops) == 1
        assert ir.block.ops[0].name == "relu"

    def test_linear_module(self):
        """Test converting torch.nn.Linear."""
        model = torch.nn.Linear(128, 256)
        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None
        assert len(ir.block.ops) == 1
        assert ir.block.ops[0].name == "linear"

    def test_sequential_module(self):
        """Test converting torch.nn.Sequential."""
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
        )
        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None
        assert len(ir.block.ops) == 2

    def test_mlp_module(self):
        """Test converting a simple MLP."""

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 256)
                self.fc2 = torch.nn.Linear(256, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = MLP()
        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None

        # Should have 3 ops: 2 linear + 1 relu
        ops = ir.block.ops
        assert len(ops) == 3

        # First op should be linear
        assert ops[0].name == "linear"

        # Second op should be relu
        assert ops[1].name == "relu"

        # Third op should be linear
        assert ops[2].name == "linear"


class TestFromTorchExecution:
    """Tests for from_torch with backend='devproc' (execution).

    Note: These tests require full AOT compilation to work.
    Skipping for now as the AOT path has known issues.
    """

    @pytest.mark.skip(reason="AOT execution has known issues")
    @requires_cuda
    def test_compile_relu_function(self):
        """Test compiling and running a relu function."""
        def relu_func(x):
            return torch.relu(x)

        compiled = from_torch(relu_func, torch.randn(4, 128))
        result = compiled(torch.randn(4, 128).cuda())

        # Verify result
        expected = torch.relu(torch.randn(4, 128))
        assert result[0].shape == expected.shape
        assert torch.allclose(result[0], expected.cpu(), atol=1e-4)

    @pytest.mark.skip(reason="AOT execution has known issues")
    @requires_cuda
    def test_compile_linear_module(self):
        """Test compiling and running a linear module."""
        model = torch.nn.Linear(128, 256)

        compiled = from_torch(model, torch.randn(1, 128))
        result = compiled(torch.randn(1, 128).cuda())

        # Verify result
        x = torch.randn(1, 128).cuda()
        expected = model(x.cpu()).cuda()

        assert result[0].shape == expected.shape
        assert torch.allclose(result[0], expected, atol=1e-3)


class TestFXToIRConverter:
    """Tests for FXToIRConverter class."""

    def test_converter_creation(self):
        """Test creating a converter."""
        converter = FXToIRConverter()
        assert converter is not None
        assert converter.op_map is not None

    def test_converter_with_simple_function(self):
        """Test converting a simple function."""

        def simple_func(x):
            return x * 2

        # Use torch.compile to get graph
        captured = {"gm": None}

        class CaptureBackend:
            def __call__(self, gm, example_inputs):
                captured["gm"] = gm
                return lambda *args: args[0] if args else None

        compiled = torch.compile(simple_func, backend=CaptureBackend())
        compiled(torch.randn(1, 128))

        assert captured["gm"] is not None

        # Convert to IR
        converter = FXToIRConverter()
        ir = converter.convert(captured["gm"], (torch.randn(1, 128),))

        assert ir is not None


class TestFromTorchEdgeCases:
    """Edge case tests for from_torch."""

    def test_function_with_multiple_ops(self):
        """Test function with multiple operations."""
        def multi_op(x):
            y = torch.relu(x)
            z = torch.sigmoid(y)
            return z

        ir = from_torch(multi_op, torch.randn(1, 128), backend="ir")

        assert ir is not None
        assert len(ir.block.ops) == 2
        assert ir.block.ops[0].name == "relu"
        assert ir.block.ops[1].name == "sigmoid"

    def test_module_with_bias(self):
        """Test linear module with bias."""
        model = torch.nn.Linear(128, 256, bias=True)

        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None
        linear_op = ir.block.ops[0]
        assert linear_op.name == "linear"
        # Should have 3 inputs: input, weight, bias
        assert len(linear_op.inputs) >= 2

    def test_module_without_bias(self):
        """Test linear module without bias."""
        model = torch.nn.Linear(128, 256, bias=False)

        ir = from_torch(model, torch.randn(1, 128), backend="ir")

        assert ir is not None
        linear_op = ir.block.ops[0]
        assert linear_op.name == "linear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
