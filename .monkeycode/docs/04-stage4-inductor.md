# Stage 4：torch.compile / Inductor 观察

本阶段对 Stage 1 的 LeNet-BN 调用 `torch.compile`，观察 Inductor 生成的算子代码，回答两个问题：它把哪些算子融合进了同一个 kernel，以及融合为什么能减少访存、带来加速。

这是「图变换 -> 降级 -> 代码生成」链条中最靠后端的一步：Stage 3 看到的是 fx 层的图，这里看到的是真正被编译成 C++ 循环的计算。

> 环境说明：本机 CPU-only，Inductor 生成的是 C++ / OpenMP 代码，而非 GPU 上的 Triton kernel。观察目标一致——融合模式、访存减少、内存复用。GPU 环境下同名概念的产物是 `triton_xxx` kernel。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| eager 与 compiled 延迟对比 | 给出中位数与加速比 | eager 1.883ms -> compiled 0.935ms，2.01x |
| 导出生成的算子代码 | 落地为 artifact 文件 | `artifacts/inductor_generated_code.py`（426 行） |
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

## 3. src/stage4_inductor/compile_demo.py 逐行说明

代码位置：`src/stage4_inductor/compile_demo.py`（共 100 行）

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
- L22-L24：导入公共工具（固定种子、基准计时）与 Stage 1 的 LeNet-BN。
- L26 `ROOT = Path(__file__).resolve().parents[2]`：`parents[0]` 是 `stage4_inductor/`，`parents[1]` 是 `src/`，`parents[2]` 是仓库根。用下标而不是多次 `.parent` 是为了路径一目了然。
- L27 `ARTIFACTS = ROOT / "artifacts"`：产物目录。
- L29-L31 `KERNEL_DEF_RE`：匹配 Inductor 生成代码里定义 kernel 的那一行，形如
  `cpp_fused_... = async_compile.cpp_pybinding([...], r'''`。
  - `^(\w+)`：捕获等号左边的 kernel 变量名。
  - `async_compile\.(?:cpp_pybinding|triton)`：匹配 CPU 的 `cpp_pybinding` 或 GPU 的 `triton` 两种注册方式。
  - `re.MULTILINE`：让 `^` 匹配每一行行首。

### 3.2 定位最新产物（L34-L38）

- L34 `def newest_output_code(root)`：在 dump 目录里找最近一次编译的 `output_code.py`。
- L35 `files = list(root.rglob("output_code.py"))`：递归搜索。每次编译会新建一个带时间戳的目录，可能命中多个。
- L36-L37 无命中返回 `None`。
- L38 `return max(files, key=lambda path: path.stat().st_mtime)`：按修改时间取最新那个。

### 3.3 主流程（L41-L96）

- L42-L45：定义命令行参数 `--batch-size`、`--iters`。
- L47 `set_seed()`：固定随机种子。
- L48 `device = torch.device("cpu")`：显式声明设备，供计时器判断是否需要 CUDA 同步。
- L49 `ARTIFACTS.mkdir(parents=True, exist_ok=True)`。
- L51-L55：`lines` 缓冲与 `log`（同时打印与收集）。
- L57 `model = LeNetBN().eval()`：`.eval()` 关键——只做推理，BN 走 running 统计、行为确定。
- L58 `x = torch.randn(args.batch_size, 1, 28, 28)`：构造 MNIST 形状的假输入。
- L60-L63：`no_grad` 下先算参考输出 `reference`，再对 eager 模型计时。`benchmark` 内部做 warmup 再取中位数。
- L65 `compiled = torch.compile(model)`：包装模型。此调用本身很快，真正编译发生在第一次前向。
- L67-L72：`start = time.time()` 后触发第一次前向，这一步执行 Dynamo 捕获、AOTAutograd 拆图、Inductor 调度与 C++ 代码生成、g++ 编译；结束后算 `max_err` 验证数值等价。
- L74-L77：对 `compiled` 计时并算加速比。
- L79 `debug_root = Path(getattr(torch._dynamo.config, "debug_dir_root", ...))`。
- L80 `code_path = newest_output_code(debug_root)`。
- L81-L92 命中时：复制到 `artifacts/inductor_generated_code.py`；读全文；`kernels = sorted(set(KERNEL_DEF_RE.findall(code)))` 去重排序；记录产物/源路径、行数、kernel 数量与名字。
- L93-L94 未命中时报错提示。
- L96 `(ARTIFACTS / "stage4_inductor_log.txt").write_text(...)`：把本次输出落盘。

## 4. 运行方式与实测结果

从仓库根目录以模块方式运行：

```bash
python3 -m src.stage4_inductor.compile_demo --batch-size 64 --iters 20
```

实测输出：

```
eager   : median=1.883ms min=1.852ms max=2.136ms
compiling (first call triggers codegen) ...
compile wall time: 46.5s, max abs err vs eager: 0.000e+00
compiled: median=0.935ms min=0.908ms max=1.018ms
speedup (median): 2.01x

generated lines: 426
defined kernels (4):
  cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_0
  cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1
  cpp_fused_relu_2
  cpp_fused_relu_3
```

补充说明：

- 首次编译 wall time 46.5s，包含 g++ 编译生成的 C++。之后相同代码命中 Inductor 缓存，耗时降到秒级以内。
- 加速比 2.01x；本机 2 核抖动大，结论表述为「有约 2x 量级的稳定加速」，不锁定精确数字。
- `max abs err` 为 0.0：融合是保语义重写，不是近似。

## 5. 生成的代码分析（artifacts/inductor_generated_code.py）

生成文件 426 行，分三部分：头部导入（L1-L33）、4 个 kernel 定义（L35-L299）、驱动类 `Runner`（L305-L390），末尾还有 `get_args`（L393-L414）。

### 5.1 头部（L1-L33）

- L1 `# AOT ID: ['0_inference']`：表示这是推理图。
- L14 `AsyncCompile`：`async_compile.cpp_pybinding(...)` 把 C++ 源码注册为可编译 kernel，`async_compile.wait(globals())`（L302-L303）阻塞等待全部编译完成。
- L15 `extern_kernels`：外部 ATen 算子集合，是「Inductor 不重写、直接调用优化库」的入口。
- L21-L33 `assert_size_stride` / `empty_strided_cpu` 等运行时 guard 与内存分配工具。

### 5.2 Kernel 0：BN + ReLU + MaxPool，并顺带重排 conv2 权重（L35-L155）

名字 `cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_0` 点明了融合内容：推理态 BN（`_native_batch_norm_legit_no_training`）、ReLU、MaxPool。它有三个循环块：

- 第一块（L52-L86）：处理 conv1 的输出 `in_out_ptr0`（`64x6x28x28`）。
  - L64-L67 读入 `mean`、`var`、`gamma`、`beta`（即 `in_ptr0..in_ptr3`，逐通道）。
  - L69 `x - mean`，L71-L72 `1/sqrt(var + 1e-5)`，L74-L78 乘 gamma 加 beta，L79 `clamp_min(..., 0)` 做 ReLU，L80 原地写回。
  - 也就是说，BN 被当作 conv 输出上的一趟逐元素 epilogue，与 ReLU 融在一起，卷积本身不在这个 kernel 里。
- 第二块（L87-L116）：读回刚写好的 `in_out_ptr0`，对 2x2 窗口做最大池化，写出 `out_ptr0`（`64x6x14x14`）。
- 第三块（L117-L150）：把 conv2 的权重 `in_ptr4` 从 `(16,6,5,5)` 转置成利于下一次卷积的布局，写入 `out_ptr1`（即 buf3）。这是布局变换被编译期处理掉的例子，只做一次，不计入逐样本开销。

要点：BN 仍是真实计算，没有被折进卷积权重；但 BN、ReLU、MaxPool 被合并进同一个 kernel，中间结果只在寄存器/SIMD 内流转，避免了「BN 输出落内存 -> 读回做 ReLU -> 再落内存 -> 读回做池化」的多趟往返。

### 5.3 Kernel 1：BN + ReLU + MaxPool（L158-L239）

名字 `cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1`，处理 conv2 的输出（`in_ptr0`，`64x16x10x10`）：

- L181-L195 一次向量化读入，做 `(x - mean) * rsqrt(var + eps) * gamma + beta` 再 `clamp_min`，原地写回；
- L201-L234 读回后对 2x2 窗口取最大，并转置写入 `out_ptr0`（`64x16x5x5`，即 conv2 的池化输出）。

与 Kernel 0 同构：卷积是外部 kernel，紧随其后的 BN + ReLU + MaxPool 融为一趟。

### 5.4 Kernel 2 / 3：逐元素 ReLU（L242-L299）

- `cpp_fused_relu_2`（L242-L269）：对 `7680 = 64*120` 个元素做 `clamp_min(x, 0)`，原地写回。
- `cpp_fused_relu_3`（L272-L299）：对 `5376 = 64*84` 个元素同样处理。
- 这两个 kernel 只含一个 ReLU，价值在于**原地**更新 addmm 的输出缓冲，省去一次分配与一次拷贝。Inductor 的收益不只来自算子融合，也来自内存复用。

### 5.5 驱动类 Runner / call（L305-L390）

`call()`（L315-L386）是编译后前向的调度序列：

- L316-L318 解包 19 个输入并做 shape/stride guard。
- L321 `buf0 = extern_kernels.convolution(arg2_1, arg0_1, arg1_1, stride=(1,1), padding=(2,2), ...)`：conv1 直接调用 ATen 卷积，未融合、未重写。
- L328-L333：分配 `buf2`（池化输出）与 `buf3`（重排后的 conv2 权重），调用 Kernel 0：`cpp_fused_..._0(buf1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, buf2, buf3)`。注意 `arg3..arg6` 是 bn1 的 `weight/bias/mean/var`，作为 kernel 入参传入。
- L342 `buf4 = extern_kernels.convolution(buf2, buf3, arg8_1, ...)`：conv2，同样走外部 kernel。
- L352：调用 Kernel 1，把 conv2 输出 + bn2 + ReLU + MaxPool 一步完成，写入 `buf6`（`arg9..arg12` 是 bn2 的参数）。
- L361 `extern_kernels.addmm(...)`：第一个全连接层。`reinterpret_tensor` 以零拷贝方式给权重换一个 stride 视图，避免真的转置。
- L365 `buf8 = buf7; del buf7 # reuse`：内存复用。随后 L368 `cpp_fused_relu_2(buf8)` 原地做 ReLU。
- L372 / L383：第二、第三个全连接层同样走 `addmm`，中间夹 Kernel 3 的原地 ReLU。
- L386 `return (buf11, )`：返回最终 logits。

BN 的 8 个参数（两组 `weight/bias/mean/var`）以输入形式贯穿整张编译图，这也解释了为什么生成的 kernel 需要额外入参：Inductor 没有把它们折叠成常数。

## 6. 结论：融合为什么减少访存

对 LeNet-BN 的推理，Inductor 做了三类优化，每类都可从生成代码中直接指认：

1. **逐元素 + 归约融合（Kernel 0、1）**：把推理态 BN、ReLU、MaxPool 合进同一个循环。若不融合，需要多趟写读中间张量；融合后在同一段代码里完成归一化、激活与池化。这是最直观的访存节省。
2. **布局变换折叠到编译期（Kernel 0 第三块）**：conv2 权重的重排只做一次，而不是每样本都做。
3. **内存复用与原地更新（Kernel 2、3 及 `buf7->buf8`）**：ReLU 就地改写 addmm 的输出，避免额外分配和拷贝。

边界与和 Stage 3 的对照：

- `convolution` 与 `addmm` 始终以 `extern_kernels` 形式出现，CPU 上它们已由高度优化的库实现，Inductor 不重写。
- 默认 CPU 路径下，Inductor **没有**把 BN 代数折叠进卷积；它把 BN 作为逐元素 epilogue 与 ReLU、池化融合，BN 的 `gamma/beta/mean/var` 仍作为 kernel 入参。这与 Stage 3 的手写 pass 形成对照：Stage 3 在**图层面做代数折叠**，BN 的数学从图里消失、参数被吸收进卷积权重；Inductor 在**调度层面做融合**，BN 的数学保留，但省掉了独立访存。两者目标一致（减少算子与访存），层次不同。
- 这也解释了手写 pass 的价值：若希望 BN 被彻底消除（例如让卷积以带最终偏置的单次计算呈现），图层的代数折叠是有效手段；而 Inductor 只承诺保语义的高效调度。

## 7. 产物清单

- `artifacts/inductor_generated_code.py`：Inductor 本次生成的完整代码（426 行）。
- `artifacts/stage4_inductor_log.txt`：本次运行的延迟、加速比、kernel 列表。
- `/tmp/torch_compile_debug/`：Inductor 完整调试 dump（含 `output_code.py`、调度记录），刻意放在工作区外。
