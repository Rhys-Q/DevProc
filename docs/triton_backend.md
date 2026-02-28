# Triton Backend 编译流程详解

本文档详细解释 DevProc 的 Triton GPU 后端编译流程。

## 整体架构

Triton Backend 负责将 DevProc IR 编译为可在 NVIDIA GPU 上执行的 Triton kernel。整个流程分为三个主要阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                     用户代码                                  │
│  @devproc.kernel def vision_preproc(img):                  │
│      img = devproc.resize(img, (224, 224))                 │
│      img = devproc.to(img, devproc.Float32)                │
│      return img                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   IR Function                                │
│  func @vision_preproc(                                     │
│    inputs: img : float32[256,256,3]                       │
│    {                                                       │
│      %v1 = resize[size=(224, 224)](img)                   │
│      %v2 = to[dtype=float32](%v1)                        │
│    }                                                       │
│    return %v2                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TritonCompiler.compile()                       │
│  1. 遍历 IR 中的每个 op                                     │
│  2. 为每个 op 生成 kernel spec                              │
│  3. 记录 tensor 分配信息                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TritonCompiledProgram                          │
│  - kernels: Kernel 列表                                     │
│  - tensor_allocations: 输出 tensor 信息                     │
│  - memory_pool: GPU 内存池                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GPU 执行                                   │
│  1. 分配输入/输出 tensor                                    │
│  2. 按顺序执行每个 kernel                                   │
│  3. 返回结果到 CPU                                          │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 文件结构

```
src/devproc/backend/triton/
├── __init__.py           # 导出接口
├── base.py               # Backend 基类定义
├── compiler.py           # 编译器和运行时
├── codegen.py            # Kernel 生成器
├── ops.py                # Op 处理器
├── memory.py             # GPU 内存管理
└── templates/            # Triton kernel 模板
    ├── elementwise.py    # Element-wise kernel
    ├── reduce.py         # Reduce kernel (argmax, softmax)
    └── matmul.py         # Matmul kernel
```

### 2. 关键类

| 类名 | 职责 |
|------|------|
| `TritonCompiler` | 编译入口，将 IR 转换为可执行程序 |
| `TritonCompiledProgram` | 编译后的程序，包含 kernel 列表和内存管理 |
| `TritonRuntime` | 运行时封装，方便直接执行 |
| `KernelGenerator` | 根据 op 类型选择对应的 handler |
| `TritonLoweringContext` | 降低过程中的上下文，存储 tensor 映射 |
| `GPUMemoryPool` | GPU 内存池 |
| `TensorManager` | CPU-GPU tensor 传输管理 |

## 编译流程详解

### 步骤 1: 调用入口

用户通过以下方式触发编译：

```python
import devproc
import torch

@devproc.kernel
def vision_preproc(img):
    img = devproc.resize(img, (224, 224))
    img = devproc.to(img, devproc.Float32)
    return img

# 编译
compiled = devproc.compile(vision_preproc, torch.randn(256, 256, 3))
```

内部调用链：
1. `devproc.compile()` → 调用 `TritonCompiler.compile()`
2. `TritonCompiler.compile(ir_function)` → 返回 `TritonCompiledProgram`

### 步骤 2: TritonCompiler.compile()

位置：[compiler.py:147-200](src/devproc/backend/triton/compiler.py#L147-L200)

```python
def compile(self, ir_function: Function) -> TritonCompiledProgram:
    kernels = []
    tensor_allocations = {}

    # 遍历 IR 中的每个 op
    for op in ir_function.block.ops:
        ctx = TritonLoweringContext(self.device_id)

        # 记录输入 tensor 占位符
        for input_val in op.inputs:
            if input_val.name not in ctx.tensor_map:
                ctx.tensor_map[input_val.name] = None

        # 记录输出 tensor 分配信息
        for output_val in op.outputs:
            if isinstance(output_val.type, TensorType):
                shape = output_val.type.shape
                dtype = output_val.type.dtype
                tensor_allocations[output_val.name] = (shape, dtype)

        # 生成 kernel spec (当前是 placeholder)
        kernel_spec = TritonKernelSpec(...)
        kernels.append(kernel_spec)

    return TritonCompiledProgram(...)
```

**关键点**：
- 编译阶段只记录信息，不真正生成 kernel
- `tensor_allocations` 保存了每个输出的 shape 和 dtype，用于运行时分配内存

### 步骤 3: 运行时执行

位置：[compiler.py:43-99](src/devproc/backend/triton/compiler.py#L43-L99)

```python
def run(self, **kwargs) -> List[torch.Tensor]:
    # 1. 准备输入
    inputs = {}
    for name, tensor in kwargs.items():
        if tensor.is_cuda:
            inputs[name] = tensor
        else:
            inputs[name] = tensor.to(f"cuda:{self.device_id}")

    # 2. 分配 GPU tensor
    ctx = TritonLoweringContext(self.device_id)

    # 分配输入 tensor
    for input_param in self.ir_function.inputs:
        gpu_tensor = self.tensor_manager.allocate_output(...)
        gpu_tensor.copy_(tensor)
        ctx.set_tensor(param_name, gpu_tensor)

    # 分配输出 tensor
    for name, (shape, dtype) in self.tensor_allocations.items():
        output_tensor = self.tensor_manager.allocate_output(name, shape, dtype)
        ctx.set_tensor(name, output_tensor)

    # 3. 执行 kernels
    for kernel in self.kernels:
        kernel.kernel_fn()  # 调用 Triton kernel

    # 4. 收集输出
    outputs = []
    for name in output_names:
        tensor = ctx.get_tensor(name)
        outputs.append(tensor.cpu())

    return outputs
```

## Op 处理器映射

每个 IR op 都有对应的 handler 函数，定义在 [ops.py](src/devproc/backend/triton/ops.py)：

| IR Op | Handler 函数 | Kernel 模板 |
|-------|-------------|-------------|
| `normalize` | `handle_normalize` | `elementwise.normalize_kernel` |
| `relu` | `handle_relu` | `elementwise.relu_kernel` |
| `sigmoid` | `handle_sigmoid` | `elementwise.sigmoid_kernel` |
| `softmax` | `handle_softmax` | `reduce.softmax_kernel` |
| `argmax` | `handle_argmax` | `reduce.argmax_kernel` |
| `matmul` | `handle_matmul` | `matmul.matmul_kernel` |
| `linear` | `handle_linear` | `matmul.linear_kernel` |
| `add` | `handle_add` | `elementwise.add_kernel` |
| `to` | `handle_to` | `elementwise.to_dtype_kernel` |
| `resize` | `handle_resize` | torch.nn.functional.interpolate |
| `transpose` | `handle_transpose` | tensor.permute |

## Kernel 模板示例

### Element-wise Kernel

位置：[templates/elementwise.py](src/devproc/backend/triton/templates/elementwise.py)

```python
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU kernel - max(0, x)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(input_ptr + block_start + offsets, mask=mask, other=0.0)

    # Compute (ReLU)
    output = tl.maximum(x, 0.0)

    # Store
    tl.store(output_ptr + block_start + offsets, output, mask=mask)


def launch_relu(input_tensor, output_tensor, BLOCK_SIZE: int = 128):
    """Launch helper"""
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
```

### Launch Config 策略

位置：[codegen.py:98-151](src/devproc/backend/triton/codegen.py#L98-L151)

不同操作使用不同的 grid 配置：

```python
def generate_launch_config(op, tensor_type):
    if op.name in ("matmul", "linear"):
        # Matmul: 2D grid
        M, N = tensor_type.shape
        BLOCK_M, BLOCK_N = 128, 256
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        num_warps = 8

    elif op.name in ("argmax", "softmax"):
        # Reduce: 1D grid over rows
        M = tensor_type.shape[0]
        grid = (M,)
        num_warps = 4

    else:
        # Element-wise: 1D grid
        total_elements = prod(tensor_type.shape)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        num_warps = 4
```

## 内存管理

### GPUMemoryPool

位置：[memory.py:11-54](src/devproc/backend/triton/memory.py#L11-L54)

```python
class GPUMemoryPool:
    def __init__(self, device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.allocated_tensors: Dict[str, torch.Tensor] = {}

    def allocate(self, name, shape, dtype) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.allocated_tensors[name] = tensor
        return tensor
```

### TensorManager

位置：[memory.py:57-167](src/devproc/backend/triton/memory.py#L57-L167)

负责：
1. CPU → GPU 数据传输 (`upload`)
2. GPU → CPU 数据传输 (`download`)
3. 输出 tensor 分配 (`allocate_output`)
4. dtype 转换 (`_convert_dtype`, `convertTorchToStr`)

## 使用示例

### 方式 1: 通过 devproc.compile

```python
import devproc
import torch

@devproc.kernel
def preprocess(img):
    img = devproc.resize(img, (224, 224))
    img = devproc.normalize(img)
    return img

# 编译
compiled = devproc.compile(preprocess, torch.randn(256, 256, 3))

# 执行
result = compiled.run(img=torch.randn(256, 256, 3))
```

### 方式 2: 通过 TritonCompiler

```python
from devproc.backend.triton import TritonCompiler
from devproc.ir.ops import OpBuilder
from devproc.ir.types import TensorType

# 直接使用编译器
compiler = TritonCompiler(device_id=0)

# 构建 IR
func = Function("test")
input_type = TensorType((256, 256, 3), "float32", "cuda")
input_op = OpBuilder.input("img", input_type)
...

# 编译
compiled = compiler.compile(func)

# 执行
result = compiled.run(img=torch.randn(256, 256, 3))
```

### 方式 3: 通过 TritonRuntime

```python
from devproc.backend.triton import TritonRuntime

runtime = TritonRuntime(device_id=0)
runtime.build(ir_function)

result = runtime(img=torch.randn(256, 256, 3))
```

## 当前限制 (MVP)

1. **Kernel 生成是 placeholder**：当前编译阶段只创建空的 kernel spec，真正的 kernel 在运行时才生成
2. **无 AOT 导出**：暂不支持导出为 .so 文件
3. **Resize/Transpose 使用 torch**：这两个操作使用 torch 实现而非 Triton kernel
4. **无 kernel fusion**：每个 op 单独一个 kernel，未进行融合优化

## 扩展开发

### 添加新的 Op 支持

1. 在 `ops.py` 添加 handler 函数
2. 在 `templates/` 添加对应的 Triton kernel
3. 在 `KernelGenerator._register_default_handlers()` 注册 handler

示例：

```python
# 1. ops.py
def handle_myop(op, ctx, device_id):
    input_tensor = ctx.get_tensor(op.inputs[0].name)
    output_tensor = ctx.get_tensor(op.outputs[0].name)

    def kernel_fn():
        from devproc.backend.triton.templates.elementwise import launch_myop
        launch_myop(input_tensor, output_tensor)

    spec = TritonKernelSpec(
        name="myop",
        grid=(output_tensor.numel() // 128,),
        num_warps=4,
        kernel_fn=kernel_fn,
    )
    ctx.add_kernel(spec)
    return spec

# 2. templates/elementwise.py
@triton.jit
def myop_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE):
    ...

def launch_myop(input, output, BLOCK_SIZE=128):
    ...

# 3. codegen.py
self.op_handlers["myop"] = handle_myop
```
