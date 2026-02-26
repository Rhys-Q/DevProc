# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DevProc is a DSL + Compiler for edge deployment, covering preprocessing + torch model + postprocessing. It unifies these into a single IR and performs cross-device compilation to generate executable files for target hardware.

**Key architectural decision**: DevProc does NOT support raw PyTorch directly - it only accepts `torch.export` output as the model entry point. This is a fundamental constraint.

## Project Structure

```
src/devproc/
├── ir/                    # Intermediate Representation (IR)
│   ├── base.py           # Value and Op classes (SSA semantics)
│   ├── types.py          # Type system (TensorType, ScalarType)
│   ├── function.py       # Function and Block classes
│   ├── ops.py            # OpBuilder for creating IR operations
│   └── verifier.py       # IRVerifier for validation
└── tests/
    └── test_ir.py        # IR tests
```

## Running Tests

```bash
# Run all tests
pytest src/devproc/tests/ -v

# Run a single test file
pytest src/devproc/tests/test_ir.py -v
```

## Code Architecture

### IR Design (SSA-based)

The IR follows Static Single Assignment (SSA) semantics:
- **Value**: Represents a typed value in IR. Each value can only be defined once.
- **Op**: Represents an operation with inputs (Values) and outputs (Values).
- **Function**: Contains input parameters, a single Block (MVP), and an output.
- **Block**: Basic block containing operations.

### Type System

- **TensorType**: Explicit shape, dtype, and device (e.g., `uint8[224,224,3]`)
- **ScalarType**: Scalar values (e.g., `scalar(int64)`)

Supported dtypes: `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`, `float32`, `float64`, `bool`
Supported devices: `cpu`, `cuda`

### Supported Operations (MVP)

- `input`: Pipeline input
- `normalize`: Normalization (converts to float32)
- `argmax`: Argmax operation (outputs scalar int64)
- `matmul`: Matrix multiplication
- `linear`: Fully connected layer
- `relu`: ReLU activation
- `add`: Element-wise addition
- `return`: Return operation

### IR Verification

The `IRVerifier` validates:
- SSA property: each Value defined exactly once
- Def-use integrity: all used Values are defined
- Type consistency: shape/dtype match
- Device validity: device must be cpu or cuda

## MVP Constraints

The MVP explicitly does NOT support:
- Dynamic shapes (only static shapes)
- Control flow (only dataflow)
- Training
- Autograd
- Raw PyTorch (only torch.export)

## Development Notes

- The codebase is in early MVP phase - primarily focused on IR implementation
- The test file at [test_ir.py](src/devproc/tests/test_ir.py) demonstrates the expected IR usage pattern
- When adding new ops to [ops.py](src/devproc/ir/ops.py), also add corresponding verification logic to [verifier.py](src/devproc/ir/verifier.py)
