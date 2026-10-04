# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[User Instruction Summary]
- Date: 2026-09-26
- Context: 用户在本项目开始时提出，适用于整个 AI 编译器入门实践项目
- Instructions:
  - 每个 stage（step）完成后必须停下来，先给出该阶段的总结，等待用户确认没有问题后，才能进入下一步。
  - 禁止在未获用户确认的情况下连续推进多个 stage。

[User Instruction Summary]
- Date: 2026-09-26
- Context: 用户在本项目 Stage 1 完成后提出，适用于整个项目
- Instructions:
  - 文档必须随进度同步更新：做到哪个 stage，就把对应文档更新到哪个 stage，不允许代码领先于文档。
  - 文档必须逐行解释代码：对每个源文件给出逐行（或逐小块）的说明，讲清每行代码做了什么、为什么这样写。
  - 文档统一放在 `.monkeycode/docs/` 目录，每个 stage 一个文件，并在 `.monkeycode/docs/README.md` 维护索引。
  - 模型结构摘要必须完整列出所有层，包含池化等无参数层，不能只写「N 个卷积 + M 个全连接」而省略中间层。

[Project Knowledge Summary]
- Date: 2026-09-26
- Context: Discovered by Agent while setting up the Stage 0 environment
- Category: Environment Configuration
- Instructions:
  - 环境为 CPU-only：2 核，约 6 GB 可用内存，无 GPU。
  - PyTorch 以 CPU wheel 安装：`pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu torch torchvision`，再用普通源安装 numpy/matplotlib/pytest。
  - 当前可用版本：torch 2.14.0+cpu，torchvision 0.29.0+cpu。
  - 项目脚本统一从仓库根目录以模块方式运行，例如 `python3 -m src.smoke_test`。

[Project Knowledge Summary]
- Date: 2026-09-27
- Context: Discovered by Agent while running torch.compile / Inductor (Stage 5)
- Category: Environment Configuration
- Instructions:
  - CPU-only 下 Inductor 生成 C++/OpenMP kernel，编译依赖 Python 头文件；若报 `fatal error: Python.h: No such file or directory`，需 `apt-get install -y python3.11-dev` 后再运行。
  - Inductor 会把调试产物写到 `torch._dynamo.config.debug_dir_root`（默认当前工作目录），建议在脚本里显式设为 `/tmp/torch_compile_debug`，避免污染工作区。
  - Inductor 编译结果有缓存；要强制重新生成代码，可指定新的 `TORCHINDUCTOR_CACHE_DIR`。

[Project Knowledge Summary]
- Date: 2026-10-02
- Context: Discovered by Agent while running the test suite
- Category: Testing Methods
- Instructions:
  - 运行测试用 `python3 -m pytest -q`，测试代码在 `tests/`。
  - 根目录 `pytest.ini` 用 `testpaths = tests` 限定收集范围，避免裸 `pytest` 扫描到无关目录。

[Project Knowledge Summary]
- Date: 2026-10-03
- Context: Discovered by Agent while implementing Stage 5 (ONNX 导出)
- Category: Build Methods
- Instructions:
  - Stage 5 只需 `pip install --break-system-packages onnx`，无需 onnxscript / onnxruntime。
  - 本机 torch 2.14 的 `torch.onnx.export` 默认 `dynamo=True`（需要 onnxscript）；本项目显式传 `dynamo=False` 走传统 TorchScript 导出器，只依赖 onnx 包。
  - `do_constant_folding=True`（默认）会在导出时把 eval 态 BN 折进 Conv；设为 False 可保留独立 BatchNormalization 节点，用于对照。
  - 数值验证用 `onnx.reference.ReferenceEvaluator`（随 onnx 提供），无需 onnxruntime。
  - 打印紧凑 ONNX IR 文本用 `onnx.helper.printable_graph`；`onnx.printer.to_text` 会内联权重数值导致文件极大。
