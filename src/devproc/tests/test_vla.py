"""
Tests for DevProc VLA (Vision-Language-Action) support.

Validates:
- IRFunction type import
- load_vla_module function existence
- Error handling when dependencies are missing
"""

import pytest
import sys
import os
import importlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestIRFunctionType:
    """Tests for IRFunction type."""

    def test_ir_function_type_import_from_dsl_types(self):
        """Test IRFunction can be imported from dsl.types."""
        from devproc.dsl.types import IRFunction
        assert IRFunction is not None

    def test_ir_function_type_import_from_devproc(self):
        """Test IRFunction can be imported from devproc main module."""
        import devproc
        assert devproc.IRFunction is not None

    def test_ir_function_is_class(self):
        """Test IRFunction is a class."""
        from devproc.dsl.types import IRFunction
        assert isinstance(IRFunction, type)

    def test_ir_function_in_devproc_all(self):
        """Test IRFunction is in devproc.__all__."""
        import devproc
        assert "IRFunction" in devproc.__all__


class TestLoadVLAModule:
    """Tests for load_vla_module function."""

    def test_load_vla_module_exists(self):
        """Test load_vla_module function exists in devproc."""
        import devproc
        assert hasattr(devproc, "load_vla_module")

    def test_load_vla_module_is_callable(self):
        """Test load_vla_module is callable."""
        from devproc import load_vla_module
        assert callable(load_vla_module)

    def test_load_vla_module_in_devproc_all(self):
        """Test load_vla_module is in devproc.__all__."""
        import devproc
        assert "load_vla_module" in devproc.__all__


class TestVLAErrorHandling:
    """Tests for VLA error handling."""

    def test_load_vla_module_requires_lerobot(self):
        """Test that load_vla_module raises ImportError when LeRobot is not installed."""
        # Save original modules
        original_lerobot = sys.modules.get("lerobot")
        original_pi05 = sys.modules.get("lerobot.policies.pi05")

        # Remove lerobot modules if loaded
        if "lerobot" in sys.modules:
            del sys.modules["lerobot"]
        if "lerobot.policies.pi05" in sys.modules:
            del sys.modules["lerobot.policies.pi05"]

        try:
            # Also need to handle the case where from_torch might have imported it
            # Reload the ops module to force re-import
            from devproc import dsl
            import devproc.dsl.ops as ops_module

            # Reload to pick up fresh state
            importlib.reload(ops_module)

            from devproc import load_vla_module

            # Attempt to load a non-existent model path to trigger the import error
            # This tests the error message
            with pytest.raises(ImportError, match="LeRobot"):
                load_vla_module("nonexistent/model")
        finally:
            # Restore original modules
            if original_lerobot is not None:
                sys.modules["lerobot"] = original_lerobot
            if original_pi05 is not None:
                sys.modules["lerobot.policies.pi05"] = original_pi05


class TestVLAIntegration:
    """Integration tests for VLA workflow (without actually loading models)."""

    def test_vla_module_function_signature(self):
        """Test load_vla_module has correct function signature."""
        import inspect
        from devproc import load_vla_module

        sig = inspect.signature(load_vla_module)
        params = list(sig.parameters.keys())

        assert "model_path" in params

    def test_vla_workflow_components_exist(self):
        """Test all VLA workflow components exist."""
        import devproc

        # Check from_torch exists
        assert hasattr(devproc, "from_torch")

        # Check kernel decorator exists
        assert hasattr(devproc, "kernel")

        # Check IRFunction type exists
        assert hasattr(devproc, "IRFunction")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
