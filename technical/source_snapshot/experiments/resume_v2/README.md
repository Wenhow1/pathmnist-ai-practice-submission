# 独立的续训工程原型

此目录是与原定 A—E 正式实验**隔离的源码副本**。训练协议使用验证损失驱动的 `ReduceLROnPlateau` 调度器，并在每轮结束保存完整状态；它与原项目的 40 轮余弦调度不是同一协议。不要把这里的运行目录放入 `runs/formal`，不要用此程序重写已完成的 A—D 权重或指标，也不要把其分数直接混作原矩阵中的重复种子。

## 保存的内容与恢复规则

- `last.pt` 是权威的每轮完整检查点，包括模型、优化器、调度器、Python／NumPy／PyTorch／CUDA（若适用）随机状态、早停计数、完整历史、最佳模型与验证预测、配置、数据清单标识、源码哈希、环境、来源及扩展事件。采用同目录临时文件、刷盘和原子替换；每次更新前将上一份有效完整检查点保留为 `prev.pt`。两份都带独立 SHA-256 校验文件。若最新文件缺失或损坏，恢复入口从 `prev.pt` 回退，记录事件并隔离损坏文件，最多重跑一个已完成的轮次。
- 每个运行目录旁的 `.lock` 文件由非阻塞系统锁保护，同一运行不能由两个训练进程同时写入；进程终止后系统释放锁。
- `best.pt` 是供推理使用的最佳模型；`history.csv`、`val_predictions.csv`、`val_metrics.json`、`run.json` 和 `status.json` 均可从 `last.pt` 在恢复时修复。训练集每轮重新建立读取器，使用固定的轮次种子；不依赖无法恢复的常驻数据读取进程状态。
- `resume` 仅用于同一运行的中断恢复，要求源码、数据清单、设备、软件环境、工作进程数及 smoke 标志匹配；训练预算与早停状态原样保留。
- `extend` 仅用于**达到轮数上限但未早停**的运行，只能增加轮数预算，调度器与早停状态连续保留，并记录预算变更。原预设 40 轮余弦实验不能因此被称为无缝续训。
- `warm-start` 从一个兼容的 `best.pt` 创建**新的运行目录和新优化器**，保存来源文件的 SHA-256；可用于原 A—D 的权重，但属于另一个实验，必须另行报告。

## 运行方式

在这个目录运行下列命令；请把 `<data>` 指向服务器 `/home/wenhao/pathmnist_course/data`，把 `<run>` 指向你自己的 `/home/wenhao/pathmnist_course/experiments/resume_v2/runs/<新名称>`。服务器只允许写 `/home/wenhao` 下的本项目目录。正式 GPU 验收前先核对 `config.example.json`，不要直接启动长训练。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python -m pathmnist_resume.cli start --data <data> --run <run> --config config.example.json --device cuda --workers 4
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python -m pathmnist_resume.cli resume --data <data> --run <run> --device cuda --workers 4
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python -m pathmnist_resume.cli extend --data <data> --run <run> --epochs 80 --device cuda --workers 4
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python -m pathmnist_resume.cli status --run <run>
```

`warm-start` 须指定 `--checkpoint <原best.pt>` 和新的 `--config`／`--run`。低频监控只读取 `status.json`、PID 与更新时间，不导入 PyTorch 或图像；它在状态变化时写一行 `monitor_events.jsonl`。例如 `python -m pathmnist_resume.monitor --run <run> --interval-seconds 600 --stale-seconds 2400`。这不是自动重启器；出现异常时应先检查原因。

测试可用 `PYTHONPATH=. python -m pytest -q tests/test_resume.py`。`--smoke` 仅供合成小样本验收，禁止将其结果作为 PathMNIST 正式指标。

## 服务器 GPU 小样本验收（待原定 E 组结束后执行）

把这个独立目录同步到 `/home/wenhao/pathmnist_course/experiments/resume_v2` 后，在服务器上执行：

```bash
source /home/wenhao/pathmnist_course/.venv/bin/activate
cd /home/wenhao/pathmnist_course/experiments/resume_v2
CUDA_VISIBLE_DEVICES=1 bash gpu_acceptance.sh
```

脚本使用当前 `python`，因此须先激活项目虚拟环境。验收只读取已有训练、验证数据，按每类 2 张图取样，用真实 ResNet-18／128 在单张 GPU 上跑连续 4 轮与“2 轮暂停＋恢复到 4 轮”两条短轨迹；**不会读取测试图像、下载数据或预训练权重**。`comparison.json` 对比模型、优化器、调度器、RNG、预测和逐轮指标是否严格一致。预计最多约 10—15 分钟、额外磁盘预留 2 GiB（含 `last.pt`、`prev.pt`、临时写入与两条轨迹）；这是保守的验收预算，不是正式训练。如果 GPU 算子无法逐位一致，脚本会返回失败并保留两条轨迹供分析，不会自动启动后续实验。
