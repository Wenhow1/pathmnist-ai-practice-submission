"""Create source-backed figures and tables from completed, frozen experiments."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .common import LABELS, read_json, write_json
from .data import PathDataset
from .experiment import validate_freeze
from .metrics import classification_metrics


def explore(data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_json(Path(data) / "manifest.json")
    dataset = PathDataset(data, "train", 128)
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    samples = []
    for c, ax in enumerate(axes.flat):
        matches = dataset.indices[dataset.labels[dataset.indices] == c]
        if not len(matches):
            ax.set_title(f"{c}: no training samples")
            ax.axis("off")
            continue
        index = int(matches[0])
        ax.imshow(dataset.images[index])
        ax.set_title(f"{c}: {LABELS[c]}", fontsize=9)
        ax.axis("off")
        samples.append({"class": c, "sample_id": f"train:{index}"})
    figure.tight_layout()
    figure.savefig(destination / "class_examples.png", dpi=170)
    plt.close(figure)
    rows = [{"class": c, "name": LABELS[c], **{split: manifest["effective"][split]["class_counts"][c]
             for split in ["train", "val", "test"]}} for c in range(9)]
    pd.DataFrame(rows).to_csv(destination / "class_counts.csv", index=False)
    write_json(destination / "sample_provenance.json", samples)
    lines = ["# PathMNIST 数据探索笔记", "", f"数据标识：{manifest['manifest_id']}",
             f"划分协议：{manifest['protocol']}", f"发现并处理的跨集合完全重复图像：{manifest['duplicate_count']}",
             "", "128×128 RGB 数据统一生成两种实验尺寸。图像为 uint8，标签范围为 0—8。",
             "", "## 数据来源和患者信息", "", manifest["patient_audit"],
             "来源研究提供独立外部测试证据；NPZ 不含患者或切片 ID，训练/验证的患者隔离不能自行证明。",
             "不能把图像块数量当作独立患者数量。", "",
             "原始论文：https://doi.org/10.1371/journal.pmed.1002730", "",
             "类别数量见 class_counts.csv，训练集示例见 class_examples.png。"]
    (destination / "data_exploration.md").write_text("\n".join(lines), encoding="utf-8")


def plot_history(folder, output):
    frame = pd.read_csv(Path(folder) / "history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(frame.epoch, frame.train_loss, color="#246477")
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(frame.epoch, frame.val_macro_f1, color="#9c526d")
    axes[1].set(xlabel="Epoch", ylabel="Validation Macro-F1", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def summarize(data, runs, destination):
    runs, destination = Path(runs), Path(destination)
    frozen = validate_freeze(data, runs)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in frozen["matrix"]:
        folder = runs / config["id"]
        provenance = read_json(folder / "test_provenance.json")
        if provenance["freeze_id"] != frozen["freeze_id"]:
            raise ValueError("Test predictions were produced under another freeze")
        frame = pd.read_csv(folder / "test_predictions.csv")
        test = PathDataset(data, "test", config["size"])
        expected_ids = [f"test:{i}" for i in test.indices]
        if frame.sample_id.tolist() != expected_ids or not np.array_equal(frame.true_label, test.labels[test.indices]):
            raise ValueError("Prediction rows do not match the complete audited test split")
        metrics = classification_metrics(frame.true_label.to_numpy(), frame[[f"p_{i}" for i in range(9)]].to_numpy())
        stored = read_json(folder / "test_metrics.json")
        for key in ["accuracy", "macro_f1", "macro_auc_ovr"]:
            if metrics[key] != stored[key] and not (metrics[key] is not None and stored[key] is not None and abs(metrics[key] - stored[key]) < 1e-9):
                raise ValueError(f"Metric mismatch for {config['id']}: {key}")
        benchmark = read_json(folder / "benchmark.json")
        rows.append({"id": config["id"], "group": config["group"], "model": config["model"],
                     "size": config["size"], "seed": config["seed"], "augment": config["augment"],
                     "val_macro_f1": read_json(folder / "val_metrics.json")["macro_f1"],
                     **{k: metrics[k] for k in ["accuracy", "macro_f1", "macro_auc_ovr"]},
                     "cpu_median_ms": benchmark["median_ms"], "cpu_p95_ms": benchmark["p95_ms"],
                     "parameters": benchmark["parameter_count"], "checkpoint_bytes": benchmark["checkpoint_bytes"]})
        plot_history(folder, destination / f"{config['id']}_learning.png")
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "all_runs.csv", index=False)
    metrics = ["accuracy", "macro_f1", "macro_auc_ovr", "cpu_median_ms", "cpu_p95_ms"]
    summary = frame.groupby("group")[metrics].agg(["mean", "std"])
    summary.to_csv(destination / "group_summary.csv")
    figure, ax = plt.subplots(figsize=(7, 4.5))
    for group, values in frame.groupby("group"):
        ax.errorbar(values.cpu_median_ms.mean(), values.macro_f1.mean(), yerr=values.macro_f1.std(), fmt="o", capsize=4, label=group)
    ax.set(xlabel="CPU latency (ms, batch=1, model only)", ylabel="Test Macro-F1")
    ax.legend(title="Configuration")
    figure.tight_layout()
    figure.savefig(destination / "accuracy_latency.png", dpi=180)
    plt.close(figure)
    selected = frozen["selection"]["run_id"]
    error_analysis(data, runs / selected, destination)
    lines = ["# 结直肠病理组织分类实验报告", "", "## 任务与方法", "",
             "PathMNIST 九类组织分类。比较 ResNet-18 和 MobileNetV3-Small 在 64 与 128 像素输入下的表现。",
             "所有配置采用 ImageNet 预训练权重，全网络微调。E 为 D 关闭数据增强的消融。", "",
             "## 主要实验结果", "", "|配置|测试 Macro-F1 均值 ± 标准差|准确率|CPU 中位延迟 ms|", "|---|---|---|---|"]
    for group, values in frame.groupby("group"):
        lines.append(f"|{group}|{values.macro_f1.mean():.4f} ± {values.macro_f1.std():.4f}|{values.accuracy.mean():.4f}|{values.cpu_median_ms.mean():.3f}|")
    lines += ["", "## 对照与消融", ""]
    means = frame.groupby("group").macro_f1.mean()
    for name, first, second in [("ResNet 提高分辨率", "B", "A"), ("MobileNet 提高分辨率", "D", "C"),
                                 ("64px 使用轻量模型", "C", "A"), ("128px 使用轻量模型", "D", "B"),
                                 ("128px MobileNet 数据增强", "D", "E")]:
        lines.append(f"- {name}：Macro-F1 差值 {means[first] - means[second]:+.4f}（{first} − {second}）。")
    lines += ["", "## 模型选择与数据边界", "", f"演示模型：{selected}。选择完全基于验证集和 CPU 延迟，规则见 frozen.json。",
              "标准差来自三个随机种子，不是患者层面的置信区间。", "官方 NPZ 缺少患者标识，不能自行证明训练/验证患者独立。",
              "CPU 测量不含预处理。图像块组织分类结果不能直接推导患者诊断或生存结局。", "",
              "## 结果追溯", "", "逐次结果见 all_runs.csv，分组统计见 group_summary.csv。",
              "错误样例及混淆矩阵见 error_pairs.csv、error_examples.png、confusion_matrix.png。",
              f"冻结标识：{frozen['freeze_id']}", "", "## 参考资料", "",
              "https://medmnist.com/", "https://doi.org/10.1371/journal.pmed.1002730"]
    (destination / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(destination / "summary.json", {"rows": rows, "selected_run": selected, "freeze_id": frozen["freeze_id"]})


def error_analysis(data, folder, destination):
    frame = pd.read_csv(folder / "test_predictions.csv")
    y = frame.true_label.to_numpy()
    p = frame[[f"p_{i}" for i in range(9)]].to_numpy()
    metrics = classification_metrics(y, p)
    cm = np.asarray(metrics["confusion_matrix"])
    figure, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="white" if value > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class", xticks=range(9), yticks=range(9))
    figure.colorbar(im, ax=ax)
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(7, 5))
    for c in range(9):
        if 0 < (y == c).sum() < len(y):
            fpr, tpr, _ = roc_curve(y == c, p[:, c])
            ax.plot(fpr, tpr, label=f"{c}: {metrics['per_class'][c]['auc_ovr']:.3f}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(title="Class and AUC", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination / "roc_ovr.png", dpi=180)
    plt.close(figure)
    pairs = sorted([(int(cm[i, j] + cm[j, i]), i, j) for i in range(9) for j in range(i + 1, 9)], reverse=True)[:3]
    pd.DataFrame(pairs, columns=["confusion_count", "class_a", "class_b"]).to_csv(destination / "error_pairs.csv", index=False)
    images = np.load(Path(data) / "test_images.npy", mmap_mode="r")
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    provenance = []
    for row, (count, a, b) in enumerate(pairs):
        wrong = frame[((frame.true_label == a) & (frame.predicted_label == b)) | ((frame.true_label == b) & (frame.predicted_label == a))].head(2)
        correct = frame[frame.true_label.isin([a, b]) & (frame.true_label == frame.predicted_label)].head(2)
        examples = pd.concat([wrong, correct])
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col >= len(examples):
                continue
            case = examples.iloc[col]
            index = int(case.sample_id.split(":")[1])
            ax.imshow(images[index])
            ax.set_title(f"{case.sample_id}\nTrue {case.true_label} / Pred {case.predicted_label}\np={case[f'p_{case.predicted_label}']:.3f}", fontsize=9)
            provenance.append({"sample_id": case.sample_id, "pair": [a, b], "confusion_count": count})
    figure.tight_layout()
    figure.savefig(destination / "error_examples.png", dpi=180)
    plt.close(figure)
    write_json(destination / "error_examples_provenance.json", provenance)
