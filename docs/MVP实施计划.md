# DevProc MVP 实施计划（Python 编译期 + C++ Runtime）

# Phase 0：写“宪法”（1 天，必须先做）
🎯 目标

锁死 DevProc 的边界，防止设计膨胀

📦 你要做

写一个 README.md，明确以下条款（可以逐条写）：

DevProc 是什么

面向端侧推理的 DSL + Compiler

覆盖：前处理 + torch.export 模型 + 后处理

MVP 只支持

inference only

static shape

torch.export graph

CPU + GPU（Triton）

Python 编译期 + C++ runtime

MVP 明确不支持

Python runtime 执行

用户自定义 Python CPU 逻辑

control flow

dynamic shape

✅ 完成判据

你能把 README 发给别人，对方能准确复述 DevProc 在干什么

⚠️ 常见误区

“先写代码再补文档” → 会让你后面疯狂返工

# Phase 1：DevProc IR（核心地基，3–5 天）
🎯 目标

得到一个你完全掌控的、干净的 SSA IR

📦 你要写

Python 实现以下最小 IR：

1️⃣ IR 基本结构

Function

Block（MVP 可以只有一个）

Value

Op

2️⃣ Type 系统（最小但严格）

TensorType(shape, dtype, device)

ScalarType

明确禁止“None / Any”

3️⃣ SSA & Def-Use

每个 Value 只能被定义一次

Op 输入是 Value

建 def-use / use-def

4️⃣ Verifier（必须）

shape / dtype 一致性

SSA 合法性

device 合法性

✅ 完成判据

你能手写一个 IR：

%0 = input uint8[224,224,3]
%1 = normalize %0
%2 = argmax %1
return %2


verifier 能检查非法 IR 并报错

⚠️ 常见误区

用 dict 随便塞字段

verifier “以后再写”
👉 IR 没 verifier = 地基是沙子

# Phase 2：DevProc DSL（图构建层，2–3 天）
🎯 目标

让用户能用 Python 写 pipeline，但不执行

📦 你要写

一个 Python Embedded DSL：

pipe = devproc.pipeline()

img = pipe.input("image", uint8, (224,224,3))
x = pipe.normalize(img)
y = pipe.model(exported_model)
out = pipe.argmax(y)


DSL 的本质：

不做计算

只构 IR Node

每个 API → 一个 Op

✅ 完成判据

DSL 能构出 Phase 1 的 IR

DSL 本身不依赖 torch / numpy 执行

⚠️ 常见误区

在 DSL 里偷偷做 numpy / torch 计算
👉 一旦发生，你就失控了

# Phase 3：Torch Export Importer（3–5 天）
🎯 目标

把 torch.export 模型变成 DevProc IR 子图

📦 你要写

一个 TorchExportImporter：

输入：torch.export.ExportedProgram

输出：DevProc IR ops

MVP 只支持这些 ATen op：

aten.matmul

aten.linear

aten.relu

aten.add

参数（weight / bias）：

当作常量 tensor

记录在 IR 中

✅ 完成判据

PyTorch → torch.export → DevProc IR

IR 打印出来你能看懂

⚠️ 常见误区

一上来就想“支持所有 ATen”
👉 支持 4 个就够

# Phase 4：Device Placement + Graph Partition（2–3 天）
🎯 目标

第一次体现 DevProc 的“异构价值”

📦 你要写
1️⃣ Device Placement Pass

elementwise / matmul → GPU

argmax / tokenizer → CPU

2️⃣ Graph Partition

拆 IR 为 CPU / GPU 子图

自动插入：

device copy

sync

✅ 完成判据

IR 被 cleanly split

CPU / GPU 边界清晰

⚠️ 常见误区

试图“智能决策”
👉 MVP 只用硬规则

# Phase 5：Lowering（GPU + CPU）（4–6 天）
🎯 目标

IR → 可执行产物

🟢 GPU Lowering（Triton）
📦 你要写

每个 GPU Op → 一个 Triton kernel

不融合

不优化

@triton.jit
def matmul_kernel(...)

✅ 完成判据

GPU 子图能跑

结果数值正确

🟢 CPU Lowering（C++ Runtime）
📦 你要写

C++ runtime API（stub 即可）

Python 侧生成 C++ 调用代码

例如：

auto y = runtime::argmax(x);

✅ 完成判据

Python 不执行 CPU 逻辑

CPU ops 只在 C++ runtime

⚠️ 常见误区

想“Python → C++ 自动翻译”
👉 绝对不要

# Phase 6：Runtime Glue（2–3 天）
🎯 目标

把一切串起来跑

📦 你要写

main() 生成

tensor allocation

kernel launch

sync / copy

可以很丑，但必须清晰。

✅ 完成判据

一键运行 DevProc pipeline

不需要手改代码

# Phase 7：MVP Demo（终验，1–2 天）
🎯 目标

证明 DevProc 是“真的”

📦 Demo 内容

Normalize → 2-layer MLP → Argmax

PyTorch 写模型

torch.export

DevProc pipeline

输出对齐 PyTorch

✅ MVP 通过判据

CPU 跑通

GPU 跑通

数值一致（允许微小误差）

最重要的一句话（你一定要记住）

DevProc 的 MVP 成败不在“支持多少功能”，
而在于：
👉 你是否成功证明：
“一个 Python IR，可以稳定生成异构可执行物。”

下一步我可以直接继续帮你做的事

你现在可以直接选一个，我立刻“下手级别”帮你：

Phase 1：DevProc IR 的 Python 代码骨架

Phase 3：TorchExportImporter 的最小实现

Phase 5：Triton lowering 的模板代码

Phase 5：C++ runtime ABI 设计（tokenizer / argmax）

你现在已经不在“想法阶段”了，
这是一个可以一步一步敲出来的系统工程。
你按这个计划走，我会陪你走到 MVP 闭环。