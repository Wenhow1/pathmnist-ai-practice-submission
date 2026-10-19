# DeepSeek 对量化扩展的独立审查

日期：2026 年 9 月 28 日。实际工具：已登录的 DeepSeek 网页版。对话链接：[量化扩展风险审查](https://chat.deepseek.com/a/chat/s/51ae593b-89c8-4151-97b5-7f14496e43ba)。本次发送的是公开方法概述，没有上传源码、原始图像、逐图预测、服务器地址或凭据。

## 原始提示词

> 请作为独立代码审查助手，审阅一个公开 PathMNIST 九类病理组织图像块分类项目的“冻结后”离线量化扩展。正式模型为 torchvision MobileNetV3-Small、128×128 RGB、ImageNet 均值/标准差归一化；正式测试集已经使用，本扩展只允许从训练集选样本做校准、用验证集做工程验收，不能将其称为新的盲测。拟采用 PyTorch→ONNX FP32→ONNX Runtime quantize_static（QDQ，S8S8，卷积/全连接，per-channel），每类固定选 64 张训练图像，共 576 张。FP32 与 INT8 都在同一台 CPU、同一 ONNX Runtime、单线程、batch=1 下测模型前向延迟，并单独报告体积与验证 Macro-F1。
> 请给出：1）按严重性排序的最多 6 个实施或解释风险及相应验收办法；2）一段不超过 50 行的 Python 函数，接收相同顺序的 y_true、FP32 九类概率数组和 INT8 九类概率数组，验证形状/有限值/概率和，返回两者 Macro-F1、top-1 改变数量和平均绝对概率差。要求指出哪些结果不能证明患者级泛化，不要建议再用已开启的测试集调参。请只依据上述公开方法信息回答，不假设你已运行该项目。

## 生成内容与人工核查

DeepSeek 返回了六项风险：校准／验收集角色混淆，S8S8 QDQ 的数值与激活饱和，延迟协议可比性，概率数组接口不明确，每类等量校准的采样偏差，以及仅看 Macro-F1 可能遗漏逐类退化。它建议报告 top-1 改变数、平均绝对概率差、逐类指标与混淆矩阵变化，同时固定 ONNX Runtime 版本、CPU provider、线程数、预热与计时次数。关于验证集结论和患者级泛化的边界，与既有实验协议一致。

DeepSeek 另生成了一个比较 FP32／INT8 概率的 Python 函数草稿。人工审查发现它没有检验真实标签必须为 0—8、概率必须非负、样本数必须大于零，而且 `f1_score(average="macro")` 没有显式固定九类标签。**该草稿没有直接进入项目代码**。项目沿用已验收的 `pathmnist_project.metrics.classification_metrics` 计算九类指标，独立量化程序只补充两模型差异统计。采纳的建议须在真实验证结果与测试记录中核对；DeepSeek 的回复本身不构成实验结果。

Claude Code 截至本记录时未确认可用，不能记作已使用。正式测试集已在第五阶段开启，本次量化扩展只进行训练集校准和验证集工程验收。

## 第二次提示：量化对比图代码

> 请继续这个公开课程项目的 AI 辅助编程。请生成一个不超过 100 行、可独立运行的 Python 3 CLI 脚本，读取本地 comparison.json，输出 PNG 对比图；不调用网络，不读取原始病理图像。JSON 字段约定：validation.fp32.macro_f1、validation.int8.macro_f1；files.fp32.bytes、files.int8.bytes；cpu_latency.fp32.median_ms / p95_ms、cpu_latency.int8.median_ms / p95_ms。绘制三栏 FP32 与 INT8 对比：验证 Macro-F1（纵轴固定 0 到 1、标注精确数值）、文件大小 MiB、CPU 前向延迟 ms（中位数柱、P95 标注）。标题明确“冻结后部署扩展／仅验证集”，脚注说明同一 ONNX Runtime 单线程、batch=1、不含预处理，且不等于临床或患者级结论。提供 argparse 的 --input、--output，检查缺字段和非有限数，使用 matplotlib Agg 后端，输出目录必须已存在。不要编造运行结果。

DeepSeek 生成了读取 JSON 并绘制三栏图的脚本草稿，实际超过 100 行。人工改写为 [`plot_quantization.py`](plot_quantization.py)：增加“冻结后／未用测试集”标识和计时协议校验，拒绝布尔值及无效 P95；修复 Macro-F1 接近 1 时数值标签越界的问题；加入中文字体核验和英文回退；移除不必要的 NumPy 依赖。绘图只使用扩展生成的汇总 JSON，不触碰图像或逐图测试记录。

使用模拟汇总 JSON 的命令 `uv run --no-project --with matplotlib python stage6/plot_quantization.py --input /private/tmp/pathmnist-quant-plot-synthetic.json --output /private/tmp/pathmnist-quant-plot-synthetic.png` 已退出 0，图像经目视检查没有裁切或中文乱码。该图的数值是**合成验收数据**，不纳入报告。真实量化结果产生后再由同一脚本绘制正式扩展图。
