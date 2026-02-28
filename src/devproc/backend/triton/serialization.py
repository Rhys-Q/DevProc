"""
Serialization and Deserialization for AOT-compiled kernels.

Handles saving/loading compiled kernels to/from disk.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from devproc.backend.triton.aot import AOTCompiledKernel

logger = logging.getLogger(__name__)


# Magic number for DevProc AOT format
AOT_MAGIC = b"DPAT"
AOT_VERSION = 1


class KernelMetadata:
    """Metadata for a single kernel."""

    def __init__(
        self,
        name: str,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        num_warps: int,
        num_stages: int,
        shared_memory: int,
        signature: List[Tuple[str, str]],
        input_names: List[str],
        output_names: List[str],
        output_shapes: Dict[str, Tuple[int, ...]],
        output_dtypes: Dict[str, str],
        has_cubin: bool = False,
        cubin_offset: int = 0,
        cubin_size: int = 0,
        constants: Dict[str, Any] = None,
    ):
        self.name = name
        self.grid = grid
        self.block = block
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.shared_memory = shared_memory
        self.signature = signature
        self.input_names = input_names
        self.output_names = output_names
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes
        self.has_cubin = has_cubin
        self.cubin_offset = cubin_offset
        self.cubin_size = cubin_size
        self.constants = constants or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "grid": list(self.grid),
            "block": list(self.block),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "shared_memory": self.shared_memory,
            "signature": [(name, dtype) for name, dtype in self.signature],
            "input_names": self.input_names,
            "output_names": self.output_names,
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "output_dtypes": self.output_dtypes,
            "has_cubin": self.has_cubin,
            "cubin_offset": self.cubin_offset,
            "cubin_size": self.cubin_size,
            "constants": self.constants,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            grid=tuple(data["grid"]),
            block=tuple(data["block"]),
            num_warps=data["num_warps"],
            num_stages=data["num_stages"],
            shared_memory=data.get("shared_memory", 0),
            signature=[(s[0], s[1]) for s in data["signature"]],
            input_names=data["input_names"],
            output_names=data["output_names"],
            output_shapes={k: tuple(v) for k, v in data["output_shapes"].items()},
            output_dtypes=data["output_dtypes"],
            has_cubin=data.get("has_cubin", False),
            cubin_offset=data.get("cubin_offset", 0),
            cubin_size=data.get("cubin_size", 0),
            constants=data.get("constants", {}),
        )


class AOTProgramMetadata:
    """Metadata for an entire AOT program."""

    def __init__(
        self,
        version: int = AOT_VERSION,
        device_id: int = 0,
        kernels: List[KernelMetadata] = None,
        tensor_allocations: Dict[str, Any] = None,
        ir_function_json: Optional[str] = None,
    ):
        self.version = version
        self.device_id = device_id
        self.kernels = kernels or []
        self.tensor_allocations = tensor_allocations or {}
        self.ir_function_json = ir_function_json

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "device_id": self.device_id,
            "kernels": [k.to_dict() for k in self.kernels],
            "tensor_allocations": self.tensor_allocations,
            "ir_function_json": self.ir_function_json,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOTProgramMetadata":
        """Create from dictionary."""
        return cls(
            version=data.get("version", AOT_VERSION),
            device_id=data.get("device_id", 0),
            kernels=[KernelMetadata.from_dict(k) for k in data.get("kernels", [])],
            tensor_allocations=data.get("tensor_allocations", {}),
            ir_function_json=data.get("ir_function_json"),
        )


class SerializationManager:
    """Manages serialization and deserialization of AOT programs."""

    @staticmethod
    def export(
        kernels: List[AOTCompiledKernel],
        output_path: str,
        tensor_allocations: Optional[Dict[str, Tuple[Tuple[int, ...], str]]] = None,
        ir_function_json: Optional[str] = None,
        device_id: int = 0,
    ) -> None:
        """Export AOT kernels to files.

        Creates:
        - output_path: Binary file containing cubin data
        - output_path + ".meta.json": Metadata JSON file

        Args:
            kernels: List of AOT compiled kernels
            output_path: Base output path (without extension)
            tensor_allocations: Tensor allocation info
            ir_function_json: JSON serialized IR function
            device_id: CUDA device ID
        """
        output_path = Path(output_path)
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

        # Build metadata
        metadata = AOTProgramMetadata(
            version=AOT_VERSION,
            device_id=device_id,
            kernels=[],
            tensor_allocations={},
            ir_function_json=ir_function_json,
        )

        # Process kernels
        cubin_data_list = []
        current_offset = 0

        for kernel in kernels:
            # Check if we have cubin
            has_cubin = kernel.cubin is not None
            cubin_size = len(kernel.cubin) if kernel.cubin else 0

            # Create kernel metadata
            kernel_meta = KernelMetadata(
                name=kernel.name,
                grid=kernel.grid,
                block=kernel.block,
                num_warps=kernel.num_warps,
                num_stages=kernel.num_stages,
                shared_memory=kernel.shared_memory,
                signature=kernel.signature,
                input_names=kernel.input_names,
                output_names=kernel.output_names,
                output_shapes=kernel.output_shapes,
                output_dtypes=kernel.output_dtypes,
                has_cubin=has_cubin,
                cubin_offset=current_offset if has_cubin else 0,
                cubin_size=cubin_size,
                constants=kernel.constants,
            )

            metadata.kernels.append(kernel_meta)

            if has_cubin:
                cubin_data_list.append(kernel.cubin)
                current_offset += cubin_size

        # Add tensor allocations
        if tensor_allocations:
            for name, (shape, dtype) in tensor_allocations.items():
                metadata.tensor_allocations[name] = {
                    "shape": list(shape),
                    "dtype": dtype,
                }

        # Write metadata JSON
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Saved metadata to {meta_path}")

        # Write cubin binary
        if cubin_data_list:
            with open(output_path, "wb") as f:
                for cubin in cubin_data_list:
                    f.write(cubin)

            logger.info(f"Saved {len(cubin_data_list)} kernels to {output_path}")
        else:
            logger.warning("No cubin data to save")

    @staticmethod
    def load(
        so_path: str,
        metadata_path: Optional[str] = None,
        device_id: int = 0,
    ) -> Tuple[List[AOTCompiledKernel], AOTProgramMetadata]:
        """Load AOT kernels from files.

        Args:
            so_path: Path to the binary file
            metadata_path: Path to metadata JSON (optional, defaults to so_path + ".meta.json")
            device_id: CUDA device ID

        Returns:
            Tuple of (list of AOT kernels, metadata)
        """
        so_path = Path(so_path)

        if metadata_path is None:
            metadata_path = so_path.with_suffix(so_path.suffix + ".meta.json")
        else:
            metadata_path = Path(metadata_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        metadata = AOTProgramMetadata.from_dict(metadata_dict)

        # Load cubin data if available
        cubin_data = b""
        if so_path.exists():
            with open(so_path, "rb") as f:
                cubin_data = f.read()

        # Reconstruct kernels
        kernels = []
        for kernel_meta in metadata.kernels:
            kernel = AOTCompiledKernel(
                name=kernel_meta.name,
                grid=kernel_meta.grid,
                block=kernel_meta.block,
                num_warps=kernel_meta.num_warps,
                num_stages=kernel_meta.num_stages,
                shared_memory=kernel_meta.shared_memory,
                signature=kernel_meta.signature,
                input_names=kernel_meta.input_names,
                output_names=kernel_meta.output_names,
                output_shapes=kernel_meta.output_shapes,
                output_dtypes=kernel_meta.output_dtypes,
                constants=kernel_meta.constants,
            )

            # Extract cubin if available
            if kernel_meta.has_cubin and cubin_data:
                start = kernel_meta.cubin_offset
                end = start + kernel_meta.cubin_size
                if start < len(cubin_data):
                    kernel.cubin = cubin_data[start:end]

            kernels.append(kernel)

        logger.info(f"Loaded {len(kernels)} kernels from {so_path}")

        return kernels, metadata


def export_program(
    kernels: List[AOTCompiledKernel],
    path: str,
    **kwargs
) -> None:
    """Convenience function to export AOT program."""
    SerializationManager.export(kernels, path, **kwargs)


def load_program(
    path: str,
    device_id: int = 0,
) -> Tuple[List[AOTCompiledKernel], AOTProgramMetadata]:
    """Convenience function to load AOT program."""
    return SerializationManager.load(path, device_id=device_id)
