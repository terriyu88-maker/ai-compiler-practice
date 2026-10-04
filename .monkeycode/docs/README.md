# 项目文档索引

本目录记录 AI 编译器入门实践项目的逐行代码说明。文档与代码同步更新：做到哪个阶段，就更新到哪个阶段。

## 阅读顺序

| 文档 | 对应阶段 | 内容 |
|------|----------|------|
| `00-stage0-environment.md` | Stage 0 | 环境安装、公共工具（随机种子、计时器）、冒烟测试的逐行解释 |
| `01-stage1-lenet.md` | Stage 1 | LeNet-BN 模型、eval 模式手写前向/反向、训练脚本、梯度检查的逐行解释 |
| `02-stage2-fx.md` | Stage 2 | torch.fx 图捕获、两种粒度对比、依赖边与 Conv→BN→ReLU 融合候选的逐行解释 |
| `03-stage3-fusion-pass.md` | Stage 3 | BN 折叠公式推导、手写 fx pass、数值/节点/延迟验证的逐行解释 |
| `04-stage4-inductor.md` | Stage 4 | torch.compile 观察、Inductor 生成代码与融合分析、内存复用的逐行解释 |
| `05-stage5-onnx.md` | Stage 5 | ONNX 导出、常量折叠对 BN 的影响、fx 与 ONNX 逐层对照的逐行解释 |
| `06-stage6-review.md` | Stage 6 | 跨抽象层次复盘表、BN 贯穿案例、融合三层次、延伸阅读清单 |

## 维护规则

1. 代码变更后，先更新对应 stage 文档，再进入下一阶段。
2. 新增源文件时，在对应 stage 文档中补充「逐行说明」小节。
3. 文档中的路径以仓库根目录为基准。
