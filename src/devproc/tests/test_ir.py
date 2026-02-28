"""
Tests for DevProc IR.

Validates the implementation against Phase 1 requirements:
- IR structure (Function, Block, Value, Op)
- Type system (TensorType, ScalarType)
- SSA & Def-Use
- Verifier functionality
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from devproc.ir.types import TensorType, ScalarType
from devproc.ir.base import Value, Op
from devproc.ir.function import Function
from devproc.ir.ops import OpBuilder
from devproc.ir.verifier import IRVerifier


class TestTensorType:
    """Tests for TensorType."""

    def test_create_tensor_type(self):
        """Test creating a valid TensorType."""
        t = TensorType((224, 224, 3), "uint8", "cpu")
        assert t.shape == (224, 224, 3)
        assert t.dtype == "uint8"
        assert t.device == "cpu"

    def test_invalid_shape(self):
        """Test that invalid shape raises error."""
        with pytest.raises(ValueError):
            TensorType((224, -1, 3), "uint8", "cpu")

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        with pytest.raises(ValueError):
            TensorType((224, 224, 3), "invalid_dtype", "cpu")

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError):
            TensorType((224, 224, 3), "uint8", "invalid_device")

    def test_tensor_type_repr(self):
        """Test TensorType string representation."""
        t = TensorType((224, 224, 3), "uint8", "cpu")
        assert repr(t) == "uint8[224,224,3]"


class TestScalarType:
    """Tests for ScalarType."""

    def test_create_scalar_type(self):
        """Test creating a valid ScalarType."""
        s = ScalarType("int32")
        assert s.dtype == "int32"

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        with pytest.raises(ValueError):
            ScalarType("invalid")


class TestValueAndOp:
    """Tests for Value and Op."""

    def test_create_value(self):
        """Test creating a Value."""
        t = TensorType((224, 224, 3), "uint8", "cpu")
        v = Value("%0", t)
        assert v.name == "%0"
        assert v.type == t
        assert v.op is None

    def test_value_uses(self):
        """Test Value use tracking."""
        t = TensorType((224, 224, 3), "uint8", "cpu")
        v1 = Value("%0", t)
        v2 = Value("%1", t)
        op = Op("test_op", [v1], [v2])

        # v1 should be used by op
        assert op in v1.uses


class TestFunction:
    """Tests for Function and Block."""

    def test_create_function(self):
        """Test creating a Function."""
        f = Function("test_func")
        assert f.name == "test_func"
        assert len(f.inputs) == 0
        assert len(f.block.ops) == 0

    def test_add_input(self):
        """Test adding inputs to Function."""
        f = Function("test_func")
        t = TensorType((224, 224, 3), "uint8", "cpu")
        v = Value("input0", t)
        f.add_input(v)

        assert len(f.inputs) == 1
        assert f.inputs[0] == v

    def test_add_op(self):
        """Test adding operations to Function."""
        f = Function("test_func")
        t = TensorType((224, 224, 3), "uint8", "cpu")

        # Add an input first (required for normalize)
        input_op = OpBuilder.input("input0", t)
        input_value = input_op.outputs[0]
        f.add_input(input_value)
        # Note: add_op skips 'input' ops

        # Now add a normalize op (not skipped)
        normalize_op = OpBuilder.normalize(input_value)
        f.add_op(normalize_op)

        assert len(f.block.ops) == 1
        assert f.block.ops[0] == normalize_op

    def test_function_repr(self):
        """Test Function string representation."""
        f = Function("test_func")
        t = TensorType((224, 224, 3), "uint8", "cpu")
        input_op = OpBuilder.input("input0", t)
        f.add_input(input_op.outputs[0])
        f.add_op(input_op)

        repr_str = repr(f)
        assert "func @test_func" in repr_str
        assert "input0" in repr_str


class TestOpBuilder:
    """Tests for OpBuilder."""

    def test_input_op(self):
        """Test creating an input operation."""
        t = TensorType((224, 224, 3), "uint8", "cpu")
        op = OpBuilder.input("input0", t)

        assert op.name == "input"
        assert len(op.inputs) == 0
        assert len(op.outputs) == 1
        assert op.outputs[0].name == "input0"
        assert op.outputs[0].type == t

    def test_normalize_op(self):
        """Test creating a normalize operation."""
        input_t = TensorType((224, 224, 3), "uint8", "cpu")
        input_val = Value("input0", input_t)
        op = OpBuilder.normalize(input_val)

        assert op.name == "normalize"
        assert len(op.inputs) == 1
        assert len(op.outputs) == 1
        # Output should be float32
        assert op.outputs[0].type.dtype == "float32"

    def test_argmax_op(self):
        """Test creating an argmax operation."""
        input_t = TensorType((1, 10), "float32", "cpu")
        input_val = Value("input0", input_t)
        op = OpBuilder.argmax(input_val, dim=1)

        assert op.name == "argmax"
        assert len(op.outputs) == 1
        # Output should be scalar int64
        assert isinstance(op.outputs[0].type, ScalarType)
        assert op.outputs[0].type.dtype == "int64"


class TestVerifier:
    """Tests for IRVerifier."""

    def test_valid_ir(self):
        """
        Test that a valid IR passes verification.

        This is the example from the Phase 1 spec:
        %0 = input uint8[224,224,3]
        %1 = normalize %0
        %2 = argmax %1
        return %2
        """
        f = Function("test_pipeline")

        # %0 = input uint8[224,224,3]
        input_type = TensorType((224, 224, 3), "uint8", "cpu")
        input_op = OpBuilder.input("%0", input_type)
        input_value = input_op.outputs[0]
        f.add_input(input_value)

        # %1 = normalize %0
        normalize_op = OpBuilder.normalize(input_value)
        normalize_value = normalize_op.outputs[0]
        f.add_op(normalize_op)

        # %2 = argmax %1
        argmax_op = OpBuilder.argmax(normalize_value)
        argmax_value = argmax_op.outputs[0]
        f.add_op(argmax_op)

        # return %2
        f.output = argmax_value

        # Verify
        verifier = IRVerifier(f)
        is_valid = verifier.verify()

        assert is_valid, f"Expected valid IR but got errors: {verifier.get_errors()}"

    def test_ssa_violation(self):
        """Test that SSA violation is detected."""
        f = Function("test_func")

        t = TensorType((224, 224, 3), "uint8", "cpu")

        # Create two ops with the same output name (SSA violation)
        v1 = Value("%0", t)
        v2 = Value("%0", t)  # Same name - SSA violation!

        f.add_input(v1)
        f.add_input(v2)

        verifier = IRVerifier(f)
        is_valid = verifier.verify()

        assert not is_valid
        errors = verifier.get_errors()
        assert any("SSA violation" in e for e in errors)

    def test_undefined_value(self):
        """Test that using undefined value is detected."""
        f = Function("test_func")

        input_t = TensorType((224, 224, 3), "uint8", "cpu")
        input_op = OpBuilder.input("%0", input_t)
        input_value = input_op.outputs[0]
        f.add_input(input_value)

        # Create an op that uses an undefined value
        # This simulates a manually created op with a reference to an undefined value
        # We need to bypass OpBuilder and directly create the Op
        # Since Op constructor validates inputs, we create it directly
        undefined_val = Value("undefined", input_t)
        # Manually create op to bypass input validation
        fake_op = Op.__new__(Op)
        fake_op._name = "fake_op"
        fake_op._inputs = [undefined_val]
        fake_op._outputs = []

        # Add the op to function's block
        f.block.add_op(fake_op)

        verifier = IRVerifier(f)
        is_valid = verifier.verify()

        assert not is_valid
        errors = verifier.get_errors()
        assert any("never defined" in e for e in errors)

    def test_invalid_device(self):
        """Test that invalid device is detected.

        Since TensorType validates device in constructor, we test this by
        directly manipulating the internal type after creation.
        """
        f = Function("test_func")

        # Create valid tensor first
        t = TensorType((224, 224, 3), "uint8", "cpu")
        input_op = OpBuilder.input("%0", t)
        input_value = input_op.outputs[0]

        # Directly set an invalid device (bypassing validation)
        input_value._type._device = "invalid_device"

        f.add_input(input_value)

        verifier = IRVerifier(f)
        is_valid = verifier.verify()

        assert not is_valid
        errors = verifier.get_errors()
        assert any("Invalid device" in e for e in errors)

    def test_matmul_dimension_mismatch(self):
        """Test that matmul dimension mismatch is detected."""
        f = Function("test_func")

        # A: M x K (2 x 3)
        a_t = TensorType((2, 3), "float32", "cpu")
        a_op = OpBuilder.input("a", a_t)
        a_val = a_op.outputs[0]
        f.add_input(a_val)

        # B: K x N but wrong K dimension (4 x 2 instead of 3 x 2)
        b_t = TensorType((4, 2), "float32", "cpu")
        b_op = OpBuilder.input("b", b_t)
        b_val = b_op.outputs[0]
        f.add_input(b_val)

        # matmul
        matmul_op = OpBuilder.matmul(a_val, b_val)
        f.add_op(a_op)
        f.add_op(b_op)
        f.add_op(matmul_op)

        verifier = IRVerifier(f)
        is_valid = verifier.verify()

        assert not is_valid
        errors = verifier.get_errors()
        assert any("dimension mismatch" in e for e in errors)


def test_manual_ir_example():
    """
    Test the manual IR example from Phase 1 spec.
    This is the primary completion criterion.
    """
    # Create function
    f = Function("main")

    # %0 = input uint8[224,224,3]
    input_type = TensorType((224, 224, 3), "uint8", "cpu")
    input_op = OpBuilder.input("%0", input_type)
    input_value = input_op.outputs[0]
    f.add_input(input_value)
    f.add_op(input_op)

    # %1 = normalize %0
    normalize_op = OpBuilder.normalize(input_value)
    normalize_value = normalize_op.outputs[0]
    f.add_op(normalize_op)

    # %2 = argmax %1
    argmax_op = OpBuilder.argmax(normalize_value)
    argmax_value = argmax_op.outputs[0]
    f.add_op(argmax_op)

    # return %2
    f.output = argmax_value

    # Print the IR
    print("\n" + "=" * 50)
    print("Generated IR:")
    print("=" * 50)
    print(repr(f))

    # Verify
    verifier = IRVerifier(f)
    is_valid = verifier.verify()

    print("\n" + "=" * 50)
    print(f"Verification result: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print("Errors:")
        for error in verifier.get_errors():
            print(f"  - {error}")
    print("=" * 50 + "\n")

    assert is_valid, "IR should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
