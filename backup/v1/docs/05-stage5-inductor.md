# Stage 5：torch.compile / Inductor 观察

本阶段对 Stage 1 的 LeNet 调用 `torch.compile`，观察 Inductor 生成的算子代码，回答两个问题：它把哪些算子融合进了同一个 kernel，以及融合为什么能减少访存、带来加速。

这是「图变换 -> 降级 -> 代码生成」链条中最靠后端的一步：前面 Stage 3 看到的是 fx 层的图，这里看到的是真正被编译成 C++ 循环的计算。

> 环境说明：本机 CPU-only，Inductor 生成的是 C++ / OpenMP 代码，而非 GPU 上的 Triton kernel。观察目标一致——融合模式、访存减少、内存复用。GPU 环境下同名概念的产物是 `triton_xxx` kernel。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| eager 与 compiled 延迟对比 | 给出中位数与加速比 | eager 2.205ms -> compiled 1.195ms，1.85x |
| 导出生成的算子代码 | 落地为 artifact 文件 | `artifacts/inductor_generated_code.py`（341 行） |
| 指出融合了哪些算子 | 能列出融合 kernel 名 | 4 个 `cpp_fused_*` kernel，见第 5 节 |
| 数值一致性 | compiled 与 eager 输出一致 | 最大绝对误差 0.0 |

## 2. torch.compile 的关键概念

| 概念 | 含义 |
|------|------|
| `torch.compile(model)` | 返回一个编译后的 callable，首次带真实输入调用时才追踪（lazy） |
| TorchDynamo | 前端图捕获器，在字节码层记录算子，产出 fx 图 |
| AOTAutograd | 把前向图拆成前向 + 反向图（推理场景只有前向） |
| Inductor | 后端，负责调度、融合、内存规划，并生成 kernel 代码 |
| `extern_kernels` | 生成代码里调用的外部 ATen 算子（如 convolution、addmm），Inductor 不重写它们，直接复用高度优化的库实现 |
| 融合（fusion） | 把多个逐元素 / 归约算子合并进同一个循环，避免中间张量落回内存再读回 |
| `TORCH_COMPILE_DEBUG=1` | 让 Inductor 把每个编译产物的中间文件（含 output_code.py）dump 到磁盘 |
| `torch._dynamo.config.debug_dir_root` | 指定 debug dump 根目录 |

## 3. src/stage5_inductor/compile_demo.py 逐行说明

代码位置：`src/stage5_inductor/compile_demo.py`（共 100 行）

### 3.1 文件头与导入（L1-L31）

- L1-L5 文档字符串：说明脚本职责，并点明 CPU-only 下生成 C++ 而非 Triton。
- L7 `import os`：用于设置环境变量。
- L9 `os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")`：打开 Inductor 调试 dump。用 `setdefault` 是为了不覆盖用户显式设置的值。必须在导入 torch 之前设置。
- L11 `import argparse`：解析命令行参数。
- L12 `import re`：从生成代码里正则提取 kernel 名字。
- L13 `import shutil`：复制 output_code.py 到 artifacts。
- L14 `import time`：测量首次编译耗时。
- L15 `from pathlib import Path`：路径操作。
- L17 `import torch`：主库。
- L18 `import torch._dynamo`：需要它的 config 来指定 debug 根目录。
- L20 `torch._dynamo.config.debug_dir_root = "/tmp/torch_compile_debug"`：把 dump 目录放到 `/tmp`，避免污染工作区、避免被 git 误收集。
- L22-L24：导入公共工具（固定种子、基准计时）与 Stage 1 的 LeNet。
- L26 `ROOT = Path(__file__).resolve().parents[2]`：`__file__` 是该脚本；`parents[0]` 是 `stage5_inductor/`，`parents[1]` 是 `src/`，`parents[2]` 是仓库根。用下标而不是多次 `.parent` 是为了路径一目了然。
- L27 `ARTIFACTS = ROOT / "artifacts"`：产物目录。
- L29-L31 `KERNEL_DEF_RE`：匹配 Inductor 生成代码里定义 kernel 的那一行，形如
  `cpp_fused_convolution_max_pool2d_with_indices_relu_0 = async_compile.cpp_pybinding([...], r'''`。
  - `^(\w+)`：捕获等号左边的 kernel 变量名。
  - `async_compile\.(?:cpp_pybinding|triton)`：匹配 CPU 的 `cpp_pybinding` 或 GPU 的 `triton` 两种注册方式，兼容两种后端。
  - `re.MULTILINE`：让 `^` 匹配每一行行首。

### 3.2 定位最新产物（L34-L38）

- L34 `def newest_output_code(root)`：在 dump 目录里找最近一次编译的 `output_code.py`。
- L35 `files = list(root.rglob("output_code.py"))`：递归搜索。每次编译会新建一个带时间戳的目录，所以可能命中多个。
- L36-L37 无命中返回 `None`，避免后续崩溃。
- L38 `return max(files, key=lambda path: path.stat().st_mtime)`：按修改时间取最新那个，保证分析的是本次编译的代码。

### 3.3 主流程（L41-L96）

- L42-L45：定义三个命令行参数 `--batch-size`、`--iters`，`description` 供 `--help` 显示。
- L47 `set_seed()`：固定随机种子，让权重和输入可复现。
- L48 `device = torch.device("cpu")`：显式声明设备，仅用于计时器内部判断是否需要 CUDA 同步。
- L49 `ARTIFACTS.mkdir(parents=True, exist_ok=True)`：确保产物目录存在。
- L51 `lines: list[str] = []`：收集所有要写入日志的文本。
- L53-L55 `def log(msg)`：既打印到终端，也追加到 `lines`，保证终端输出与日志文件一致。
- L57 `model = LeNet().eval()`：`.eval()` 关键——只做推理，且 BN/Dropout 进入推理态，行为确定。LeNet 本身无 BN，但这是良好习惯。
- L58 `x = torch.randn(args.batch_size, 1, 28, 28)`：构造 MNIST 形状的假输入 `(N, C, H, W)`。
- L60-L63：在 `torch.no_grad()` 下先算一次参考输出 `reference`，再对 eager 模型计时。`benchmark` 内部做 warmup 再取中位数。
- L65 `compiled = torch.compile(model)`：包装模型。注意此调用本身很快，真正编译发生在第一次前向。
- L67-L72：`start = time.time()` 后触发第一次前向，这一步会执行 TorchDynamo 捕获、AOTAutograd 拆图、Inductor 调度与 C++ 代码生成、g++ 编译。结束后算 `max_err = (out - reference).abs().max()`，验证编译后数值等价。
- L74-L76：对 `compiled` 调用同样的计时函数，得到编译后延迟。
- L77 `speedup = eager.median_ms / compiled_result.median_ms`：以中位数算加速比，比用单次更稳。
- L79 `debug_root = Path(getattr(torch._dynamo.config, "debug_dir_root", "/tmp/torch_compile_debug"))`：读回 dump 根目录，`getattr` 带默认值以防未来版本改属性名。
- L80 `code_path = newest_output_code(debug_root)`：定位本次生成的 `output_code.py`。
- L81-L92 命中时：
  - L82-L83：复制到 `artifacts/inductor_generated_code.py`，留下可提交的产物。
  - L84 `code = code_path.read_text()`：读全文。
  - L85 `kernels = sorted(set(KERNEL_DEF_RE.findall(code)))`：`findall` 返回所有 kernel 变量名；`set` 去重（一个名字可能在正文多处出现）；`sorted` 让输出稳定可比。
  - L87-L92：记录产物路径、源路径、总行数、kernel 数量与每个名字。
- L93-L94 未命中时报错提示，避免静默失败。
- L96 `(ARTIFACTS / "stage5_inductor_log.txt").write_text(...)`：把本次所有输出落盘，便于与文档、提交记录对齐。

## 4. 运行方式与实测结果

从仓库根目录以模块方式运行：

```bash
python3 -m src.stage5_inductor.compile_demo --batch-size 64 --iters 20
```

实测输出（fresh 运行，force 重新编译）：

```
eager   : median=2.205ms min=2.149ms max=2.295ms
compile wall time: 12.4s, max abs err vs eager: 0.000e+00
compiled: median=1.195ms min=1.167ms max=1.265ms
speedup (median): 1.85x
generated lines: 341
defined kernels (4):
  cpp_fused_convolution_max_pool2d_with_indices_relu_0
  cpp_fused_max_pool2d_with_indices_relu_1
  cpp_fused_relu_2
  cpp_fused_relu_3
```

补充说明：

- 首次编译 wall time 约 12s，其中大头是调用 g++ 编译生成的 C++。之后相同代码命中 Inductor 缓存，编译耗时降到秒级以内。
- 加速比在 1.45x ~ 1.85x 之间波动，原因是本机只有 2 核且存在内存气球，抖动大。因此结论应表述为「有稳定的、约 1.4x-1.9x 的加速」，而非锁定一个精确数字。
- `max abs err` 为 0.0：融合是保语义重写，不是近似。

## 5. 生成的代码分析（artifacts/inductor_generated_code.py）

生成文件 341 行，分三部分：头部导入（L1-L32）、4 个 kernel 定义（L35-L234）、驱动类 `Runner`（L240-L313）。

### 5.1 头部（L1-L32）

- L1 `# AOT ID: ['0_inference']`：表示这是推理（inference）图。
- L2-L32 是固定样板：从 ctypes、torch._inductor 引入工具。
  - L14 `AsyncCompile`：异步编译管理器，`async_compile.cpp_pybinding(...)` 把 C++ 源码注册为可编译 kernel，`async_compile.wait(globals())`（L237）阻塞等待全部编译完成后把符号注入模块命名空间。
  - L15 `extern_kernels`：外部 ATen 算子集合，是「Inductor 不重写、直接调用优化库」的入口。
  - L21-L24 `assert_size_stride` / `empty_strided_cpu`：运行时 guard 与内存分配，用于校验输入元数据并复用内存规划。

### 5.2 Kernel 0：融合 relu + maxpool，并顺带重排 conv2 权重（L35-L119）

名字 `cpp_fused_convolution_max_pool2d_with_indices_relu_0` 已说明融合内容。它有两个循环块：

- 第一块（L48-L79）：处理 conv1 的输出（`in_ptr0`，shape `64x6x28x28`）。
  - L49 外层按 batch `x0` 并行（`#pragma omp for`）。
  - L62-L65 一次性读出 2x2 窗口的四个元素（步长 2、行距 28）。
  - L66-L72 对每个元素先 `std::max(tmp, 0)` 做 ReLU，再用 `max_propagate_nan` 取窗口最大值——ReLU 和 maxpool 在同一段代码里完成，中间结果不落内存。
  - L73 写回 `out_ptr0`（shape `64x6x14x14`）。
- 第二块（L82-L110）：把 conv2 的权重 `in_ptr1` 从 `(16,6,5,5)` 转置成利于下一次卷积的访存布局，写入 `out_ptr1`（即 buf2）。
  - L92-L98 用 `Vectorized<float>` 向量化加载再 `transpose_mxn` 转置。
  - 这是「图级常量/布局变换」被编译期处理掉的例子：权重重排只做一次，不计入逐样本开销。

要点：ReLU 与 MaxPool 被合并为一个 kernel，一次读 conv1 输出、一次写池化结果，省掉了「先写 ReLU 输出、再读回来做池化」的往返。

### 5.3 Kernel 1：融合 relu + maxpool（L122-L174）

名字 `cpp_fused_max_pool2d_with_indices_relu_1`，处理 conv2 的输出（`in_ptr0`，shape `64x16x10x10`）：

- L149-L152 一次读 2x2 窗口（行距 32），
- L153-L159 每个元素 `clamp_min(..., 0)` 做 ReLU，再 `maximum` 两两合并，
- L160-L162 向量化写回 `out_ptr0`（shape `64x16x5x5`）。
- 输入输出用 16 宽 SIMD，L134 外层按 batch 并行。

与 Kernel 0 同构：conv 本身是外部 kernel，紧随其后的 ReLU+MaxPool 被融合。

### 5.4 Kernel 2 / 3：逐元素 ReLU（L177-L234）

- `cpp_fused_relu_2`（L177-L204）：对 `7680 = 64*120` 个元素做 `clamp_min(x, 0)`，原地写回（签名是 `float* in_out_ptr0`，输入即输出）。
- `cpp_fused_relu_3`（L207-L234）：对 `5376 = 64*84` 个元素同样处理。
- 这两个 kernel 只包含一个 ReLU，看似「没融合到东西」。它们的价值在于**原地**更新 addmm 的输出缓冲，省去一次分配与一次拷贝。这说明 Inductor 的收益不只来自算子融合，也来自内存复用。

### 5.5 驱动类 Runner（L240-L313）

`call()` 方法（L250-L309）是编译后的前向调度序列，逐行读它就能看到「图变换 + 降级」的结果：

- L253 `assert_size_stride_grouped(...)`：运行时校验输入 shape/stride，guard 不通过则触发重编译。
- L256 `buf0 = extern_kernels.convolution(arg2_1, arg0_1, arg1_1, stride=(1,1), padding=(2,2), ...)`：conv1 直接调用 ATen 卷积，未融合、未重写。
- L261-L266：分配两个输出缓冲，调用 Kernel 0：`cpp_fused_convolution_max_pool2d_with_indices_relu_0(buf0, arg3_1, buf1, buf2)`——`buf1` 是 relu+maxpool 后的 conv1 结果，`buf2` 是重排后的 conv2 权重。
- L271 `buf3 = extern_kernels.convolution(buf1, buf2, arg4_1, ...)`：conv2，同样走外部 kernel。
- L276-L279：调用 Kernel 1，把 conv2 输出 + ReLU + MaxPool 一步完成，写入 `buf4`。
- L284 `extern_kernels.addmm(...)`：第一个全连接层。这里能看到 Inductor 的转置优化：`reinterpret_tensor` 以零拷贝方式给权重换一个 stride 视图，避免真的转置。
- L288 `buf6 = buf5; del buf5  # reuse`：内存复用——`buf6` 复用 `buf5` 的显存，把前一个缓冲释放。随后 L291 `cpp_fused_relu_2(buf6)` 原地做 ReLU。
- L295 / L306：第二、第三个全连接层同样走 `addmm`，中间夹 Kernel 3 的原地 ReLU。
- L309 `return (buf9, )`：返回最终 logits。

## 6. 结论：融合为什么减少访存

对 LeNet 的推理，Inductor 做了三类优化，每类都可从生成代码中直接指认：

1. **逐元素 + 归约融合（Kernel 0、1）**：ReLU 后接 MaxPool。若不融合，需要先写一遍 ReLU 结果张量、再读回来做池化，共 2 次额外访存；融合后一次读、一次写。这是最直观的访存节省。
2. **布局变换折叠到编译期（Kernel 0 第二块）**：conv2 权重的重排只做一次，而不是每样本都做。
3. **内存复用与原地更新（Kernel 2、3 及 `buf5->buf6`）**：ReLU 就地改写 addmm 的输出，避免额外分配和拷贝。

同时要注意边界：`convolution` 与 `addmm` 始终以 `extern_kernels` 形式出现，卷积和矩阵乘没有被融合进自定义 kernel——在 CPU 上它们已经由高度优化的库实现，重写通常不划算。这正好解释了 Stage 4 手写 pass 的价值：`Conv+BN+ReLU` 这类融合需要先在图层面把 BN 折叠进卷积权重，才能让后端把整块工作交给一个更优的 kernel；而单纯的 fx 图重写无法让外部卷积 kernel 与后续算子融合。

## 7. 产物清单

- `artifacts/inductor_generated_code.py`：Inductor 本次生成的完整代码（341 行）。
- `artifacts/stage5_inductor_log.txt`：本次运行的延迟、加速比、kernel 列表。
- `/tmp/torch_compile_debug/`：Inductor 完整调试 dump（含 `output_code.py`、调度记录），刻意放在工作区外。
