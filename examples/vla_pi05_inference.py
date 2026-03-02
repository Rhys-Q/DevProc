"""
π₀.₅ VLA Model Inference Example

This example demonstrates:
1. Running inference with native PyTorch (LeRobot API)
2. Running inference with DevProc
3. Accuracy comparison between both versions

Usage:
    PYTHONPATH=/home/qinzhiqi/tw/global/DevProc/src:$PYTHONPATH python examples/vla_pi05_inference.py

Requirements:
    pip install torch torchvision
    pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def create_sample_inputs():
    """Create sample inputs for π₀.₅ model."""
    # π₀.₅ expects:
    # - 3 RGB images (224x224): base, left_wrist, right_wrist
    # - Robot state (32-dim)
    # - Language instruction (string)

    batch_size = 1

    # Create sample images (3 cameras)
    image_base = torch.randn(batch_size, 3, 224, 224)
    image_left = torch.randn(batch_size, 3, 224, 224)
    image_right = torch.randn(batch_size, 3, 224, 224)

    # Create robot state
    state = torch.randn(batch_size, 32)

    # Language instruction
    language = "pick up the cup"

    return {
        "image_base": image_base,
        "image_left": image_left,
        "image_right": image_right,
        "state": state,
        "language": language,
    }


def run_torch_inference(inputs, model_path="lerobot/pi05_base"):
    """Run inference using native PyTorch (LeRobot API)."""
    print("=" * 60)
    print("Running PyTorch version (LeRobot API)...")
    print("=" * 60)

    try:
        from lerobot.policies.pi05 import PI05Policy
    except ImportError:
        print("ERROR: LeRobot not installed.")
        print("Please install with:")
        print("  pip install \"lerobot[pi]@git+https://github.com/huggingface/lerobot.git\"")
        return None

    # Load policy
    print(f"Loading model from {model_path}...")
    policy = PI05Policy.from_pretrained(model_path)
    policy.eval()

    # Prepare observation
    obs = {
        "observation.images.base_0_rgb": inputs["image_base"],
        "observation.images.left_wrist_0_rgb": inputs["image_left"],
        "observation.images.right_wrist_0_rgb": inputs["image_right"],
        "observation.state": inputs["state"],
        "observation.language": inputs["language"],
    }

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        action = policy.select_action(obs)

    print(f"Torch output shape: {action.shape}")
    print(f"Torch output dtype: {action.dtype}")
    print(f"Torch output (first 5): {action[0, :5]}")

    return action


def run_devproc_inference(inputs, model_path="lerobot/pi05_base"):
    """Run inference using DevProc."""
    print("\n" + "=" * 60)
    print("Running DevProc version...")
    print("=" * 60)

    import devproc

    # Step 1: Load VLA module
    print(f"Loading VLA module from {model_path}...")
    try:
        vla_module = devproc.load_vla_module(model_path)
    except ImportError as e:
        print(f"ERROR: {e}")
        print("DevProc version requires LeRobot to be installed.")
        return None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    # Step 2: Prepare example inputs for tracing
    # π₀.₅ expects images as a list/tuple
    images = torch.cat([
        inputs["image_base"],
        inputs["image_left"],
        inputs["image_right"],
    ], dim=0)  # Shape: (3, 3, 224, 224)

    state = inputs["state"]
    language = inputs["language"]

    example_inputs = (images, state, language)

    # Step 3: Convert to DevProc IR
    print("Converting to DevProc IR...")
    try:
        vla_ir = devproc.from_torch(vla_module, example_inputs, backend="ir")
    except Exception as e:
        print(f"ERROR converting to IR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Verify IR
    print("Verifying IR...")
    verifier = devproc.IRVerifier(vla_ir)
    if not verifier.verify():
        print(f"IR verification failed: {verifier.get_errors()}")
        return None

    print("IR verified successfully!")
    print(f"\nIR Function:")
    print(vla_ir)

    # Step 4: Compile and run (using Triton backend)
    print("\nCompiling with Triton backend...")
    try:
        compiled = devproc.from_torch(vla_module, example_inputs, backend="devproc")
    except Exception as e:
        print(f"ERROR compiling: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Run inference
    print("Running DevProc inference...")
    try:
        action_devproc = compiled(images, state, language)
    except Exception as e:
        print(f"ERROR running inference: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"DevProc output shape: {action_devproc.shape}")
    print(f"DevProc output dtype: {action_devproc.dtype}")
    print(f"DevProc output (first 5): {action_devproc[0, :5]}")

    return action_devproc


def compare_results(action_torch, action_devproc):
    """Compare results between PyTorch and DevProc versions."""
    print("\n" + "=" * 60)
    print("Accuracy Comparison")
    print("=" * 60)

    if action_torch is None or action_devproc is None:
        print("Cannot compare - one or both results are missing")
        return

    # Ensure same device
    if action_torch.device != action_devproc.device:
        action_devproc = action_devproc.to(action_torch.device)

    # Calculate differences
    diff = action_devproc - action_torch
    abs_diff = torch.abs(diff)

    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    std_diff = torch.std(abs_diff).item()

    # Cosine similarity
    flat_torch = action_torch.flatten()
    flat_devproc = action_devproc.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_torch.unsqueeze(0),
        flat_devproc.unsqueeze(0)
    ).item()

    # Relative error
    rel_error = torch.mean(abs_diff / (torch.abs(flat_torch) + 1e-8)).item()

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Std absolute difference: {std_diff:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    # Check if results are close (within tolerance)
    tolerance = 1e-4
    is_close = max_diff < tolerance
    print(f"\nResults match (tolerance={tolerance}): {is_close}")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_similarity": cos_sim,
        "matches": is_close,
    }


def main():
    print("π₀.₅ VLA Model Inference Example")
    print("=" * 60)

    # Create sample inputs
    print("Creating sample inputs...")
    inputs = create_sample_inputs()
    print(f"  image_base: {inputs['image_base'].shape}")
    print(f"  image_left: {inputs['image_left'].shape}")
    print(f"  image_right: {inputs['image_right'].shape}")
    print(f"  state: {inputs['state'].shape}")
    print(f"  language: {inputs['language']}")

    # Run PyTorch version
    action_torch = run_torch_inference(inputs)

    # Run DevProc version
    action_devproc = run_devproc_inference(inputs)

    # Compare results
    compare_results(action_torch, action_devproc)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
