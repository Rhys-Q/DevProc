"""
Tests for Triton GPU Backend.

These tests require CUDA and Triton to be available.
"""

import pytest
import torch

import devproc
from devproc import compile, Runtime


def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def triton_available():
    """Check if Triton is available."""
    try:
        import triton

        return True
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
requires_triton = pytest.mark.skipif(
    not triton_available(), reason="Triton not available"
)


class TestTritonBackend:
    """Test suite for Triton backend."""

    @requires_cuda
    @requires_triton
    def test_normalize_op(self):
        """Test normalize operation."""
        from devproc.backend.triton.templates.elementwise import launch_normalize

        # Create test tensors
        input_tensor = torch.randn(16, 32, dtype=torch.float32).cuda()
        output_tensor = torch.empty_like(input_tensor)

        # Run kernel
        launch_normalize(input_tensor, output_tensor)

        # Verify
        expected = input_tensor.float()
        assert torch.allclose(output_tensor, expected, atol=1e-4)

    @requires_cuda
    @requires_triton
    def test_relu_op(self):
        """Test ReLU operation."""
        from devproc.backend.triton.templates.elementwise import launch_relu

        # Create test tensors
        input_tensor = torch.randn(16, 32, dtype=torch.float32).cuda()
        output_tensor = torch.empty_like(input_tensor)

        # Run kernel
        launch_relu(input_tensor, output_tensor)

        # Verify
        expected = torch.relu(input_tensor)
        assert torch.allclose(output_tensor, expected, atol=1e-4)

    @requires_cuda
    @requires_triton
    def test_sigmoid_op(self):
        """Test sigmoid operation."""
        from devproc.backend.triton.templates.elementwise import launch_sigmoid

        # Create test tensors
        input_tensor = torch.randn(16, 32, dtype=torch.float32).cuda()
        output_tensor = torch.empty_like(input_tensor)

        # Run kernel
        launch_sigmoid(input_tensor, output_tensor)

        # Verify
        expected = torch.sigmoid(input_tensor)
        assert torch.allclose(output_tensor, expected, atol=1e-3)

    @requires_cuda
    @requires_triton
    def test_matmul_op(self):
        """Test matrix multiplication."""
        from devproc.backend.triton.templates.matmul import launch_matmul

        # Create test tensors
        a = torch.randn(16, 32, dtype=torch.float32).cuda()
        b = torch.randn(32, 16, dtype=torch.float32).cuda()
        c = torch.empty(16, 16, dtype=torch.float32).cuda()

        # Run kernel
        launch_matmul(a, b, c)

        # Verify (use larger tolerance due to autotune)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, atol=1e-1)

    @pytest.mark.skip(reason="linear kernel has stride issue")
    @requires_cuda
    @requires_triton
    def test_linear_op(self):
        """Test linear (fully connected) operation."""
        from devproc.backend.triton.templates.matmul import launch_linear

        # Create test tensors
        input_tensor = torch.randn(4, 128, dtype=torch.float32).cuda()
        weight = torch.randn(256, 128, dtype=torch.float32).cuda()
        bias = torch.randn(256, dtype=torch.float32).cuda()
        output = torch.empty(4, 256, dtype=torch.float32).cuda()

        # Run kernel with bias
        launch_linear(input_tensor, weight, bias, output)

        # Verify (use larger tolerance due to autotune)
        expected = torch.matmul(input_tensor, weight.t()) + bias
        assert torch.allclose(output, expected, atol=1e-1)

    @requires_cuda
    @requires_triton
    @pytest.mark.skip(reason="linear kernel has stride issue")
    def test_linear_op_no_bias(self):
        """Test linear operation without bias."""
        from devproc.backend.triton.templates.matmul import launch_linear

        # Create test tensors
        input_tensor = torch.randn(4, 128, dtype=torch.float32).cuda()
        weight = torch.randn(256, 128, dtype=torch.float32).cuda()
        output = torch.empty(4, 256, dtype=torch.float32).cuda()

        # Run kernel without bias
        launch_linear(input_tensor, weight, None, output)

        # Verify (use larger tolerance due to autotune)
        expected = torch.matmul(input_tensor, weight.t())
        assert torch.allclose(output, expected, atol=1e-1)

    @requires_cuda
    @requires_triton
    def test_add_op(self):
        """Test element-wise addition."""
        from devproc.backend.triton.templates.elementwise import launch_add

        # Create test tensors
        a = torch.randn(16, 32, dtype=torch.float32).cuda()
        b = torch.randn(16, 32, dtype=torch.float32).cuda()
        output = torch.empty_like(a)

        # Run kernel
        launch_add(a, b, output)

        # Verify
        expected = a + b
        assert torch.allclose(output, expected, atol=1e-4)

    @requires_cuda
    @requires_triton
    def test_softmax_op(self):
        """Test softmax operation."""
        from devproc.backend.triton.templates.reduce import launch_softmax

        # Create test tensors
        input_tensor = torch.randn(4, 128, dtype=torch.float32).cuda()
        output_tensor = torch.empty_like(input_tensor)

        # Run kernel
        launch_softmax(input_tensor, output_tensor)

        # Verify
        expected = torch.softmax(input_tensor, dim=-1)
        assert torch.allclose(output_tensor, expected, atol=1e-3)

    @requires_cuda
    @requires_triton
    def test_argmax_op(self):
        """Test argmax operation."""
        from devproc.backend.triton.templates.reduce import launch_argmax

        # Create test tensors
        input_tensor = torch.randn(4, 128, dtype=torch.float32).cuda()
        output_tensor = torch.empty(4, dtype=torch.int64).cuda()

        # Run kernel
        launch_argmax(input_tensor, output_tensor)

        # Verify
        expected = torch.argmax(input_tensor, dim=-1)
        assert torch.equal(output_tensor, expected)

    @requires_cuda
    @requires_triton
    # @pytest.mark.skip(reason="depends on linear kernel which has issue")
    def test_mlp_pipeline(self):
        """Test complete MLP pipeline."""

        # Define kernel
        @devproc.kernel
        def mlp(x, w1, b1, w2, b2):
            h = devproc.linear(x, w1, b1)
            h = devproc.relu(h)
            out = devproc.linear(h, w2, b2)
            return out

        # AOT compile
        compiled = compile(
            mlp,
            torch.randn(1, 128),
            torch.randn(256, 128),
            torch.randn(256),
            torch.randn(10, 256),
            torch.randn(10),
            backend="triton",
            device_id=0,
        )

        # Create runtime
        rt = Runtime(compiled, device_id=0)

        # Run on GPU
        x = torch.randn(1, 128).cuda()
        w1 = torch.randn(256, 128).cuda()
        b1 = torch.randn(256).cuda()
        w2 = torch.randn(10, 256).cuda()
        b2 = torch.randn(10).cuda()

        result = rt(x=x, w1=w1, b1=b1, w2=w2, b2=b2)

        # Verify with PyTorch
        expected = (
            torch.matmul(
                torch.relu(torch.matmul(x.float(), w1.float().T) + b1.float()),
                w2.float().T,
            )
            + b2.float()
        )

        assert torch.allclose(result[0], expected, atol=1e-1)

    @requires_cuda
    @requires_triton
    def test_compiler_available(self):
        """Test that compiler reports availability."""
        from devproc.backend.triton import TritonCompiler

        compiler = TritonCompiler()
        assert compiler.is_available()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
