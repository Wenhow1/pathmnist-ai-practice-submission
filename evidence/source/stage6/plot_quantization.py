"""Plot the post-freeze FP32/INT8 engineering comparison from measured JSON.

Adapted after human review of a DeepSeek draft. This script reads only summary
statistics; it never reads PathMNIST images or formal test predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font


def measured(summary: dict, *keys: str) -> float:
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"缺少字段：{'.'.join(keys)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"字段必须为数值：{'.'.join(keys)}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"字段不是有限数：{'.'.join(keys)}")
    return number


def read_comparison(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("scope") != "post-freeze deployment extension; validation only":
        raise ValueError("只接受冻结后且仅验证集的量化扩展记录")
    if data.get("test_set_used") is not False:
        raise ValueError("绘图输入不能包含本扩展的测试集评估")
    method = data.get("export", {}).get("quantization", "")
    if "U8U8" in method:
        quant_label = "U8U8"
    elif "S8S8" in method:
        quant_label = "S8S8"
    else:
        raise ValueError("缺少可核验的 8-bit 数据类型标记")
    values = {}
    for variant in ("fp32", "int8"):
        f1 = measured(data, "validation", variant, "macro_f1")
        size = measured(data, "files", variant, "bytes") / 2**20
        median = measured(data, "cpu_latency", variant, "median_ms")
        p95 = measured(data, "cpu_latency", variant, "p95_ms")
        if not 0 <= f1 <= 1 or size <= 0 or median <= 0 or p95 < median:
            raise ValueError(f"{variant} 的指标范围或 P95/中位数关系错误")
        timing = data["cpu_latency"][variant]
        if (timing.get("threads") != 1 or timing.get("batch_size") != 1
                or timing.get("preprocessing_included") is not False
                or timing.get("runtime") != "ONNX Runtime"
                or timing.get("provider") != "CPUExecutionProvider"):
            raise ValueError(f"{variant} 的计时协议与图注不一致")
        values[variant] = {"f1": f1, "size": size, "median": median, "p95": p95}
    values["quant_label"] = quant_label
    return values


def labels_and_font() -> tuple[dict, str]:
    zh = {"title": "冻结后部署扩展／仅验证集", "f1": "验证 Macro-F1",
          "size": "ONNX 文件大小（MiB）", "latency": "CPU 前向延迟（ms）",
          "note": "同一 ONNX Runtime、CPU 单线程、batch=1；前向计时不含预处理。",
          "boundary": "训练集校准、验证集工程验收；不代表患者级泛化或临床可用性。",
          "median": "中位", "p95": "P95"}
    glyphs = {ord(c) for s in zh.values() for c in s if ord(c) > 127}
    for family in ("Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                   "Microsoft YaHei", "WenQuanYi Micro Hei", "SimHei", "Heiti SC"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)
            if glyphs.issubset(ft2font.FT2Font(path).get_charmap()):
                font_manager.fontManager.addfont(path)
                return zh, font_manager.FontProperties(fname=path).get_name()
        except (OSError, RuntimeError, ValueError):
            pass
    en = {"title": "Post-freeze deployment extension / validation only",
          "f1": "Validation Macro-F1", "size": "ONNX file size (MiB)",
          "latency": "CPU forward latency (ms)",
          "note": "Same ONNX Runtime, one CPU thread, batch=1; timing excludes preprocessing.",
          "boundary": "Train-only calibration and validation acceptance; no patient-level or clinical claim.",
          "median": "median", "p95": "P95"}
    return en, "DejaVu Sans"


def draw(values: dict, output: Path) -> None:
    labels, family = labels_and_font()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    keys, names = ("fp32", "int8"), ("FP32", values["quant_label"])
    colors = ["#32648B", "#DE8A3E"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7))
    for axis, key, title in zip(axes, ("f1", "size", "median"),
                                (labels["f1"], labels["size"], labels["latency"])):
        y = [values[variant][key] for variant in keys]
        bars = axis.bar(names, y, color=colors, width=0.58)
        axis.set_title(title, fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DFE6EC", linewidth=0.6)
        axis.set_axisbelow(True)
        if key == "f1":
            axis.set_ylim(0, 1)
            for bar, value in zip(bars, y):
                inside = value > 0.12
                axis.text(bar.get_x() + bar.get_width()/2,
                          value - 0.055 if inside else value + 0.025,
                          f"{value:.4f}", ha="center", va="center",
                          color="white" if inside else "#203141", fontsize=9, fontweight="bold")
        else:
            top = max(values[variant]["p95"] for variant in keys) if key == "median" else max(y)
            axis.set_ylim(0, top * 1.27)
            for bar, value, variant in zip(bars, y, keys):
                caption = (f"{value:.2f} MiB" if key == "size" else
                           f"{labels['median']} {value:.3f}\n{labels['p95']} {values[variant]['p95']:.3f}")
                axis.text(bar.get_x() + bar.get_width()/2, value + top*0.03,
                          caption, ha="center", va="bottom", fontsize=8.5)
    fig.suptitle(labels["title"], fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.07, labels["note"], ha="center", fontsize=8.5, color="#405464")
    fig.text(0.5, 0.035, labels["boundary"], ha="center", fontsize=8.5, color="#405464")
    fig.tight_layout(rect=(0.02, 0.11, 0.98, 0.91))
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or not args.output.parent.is_dir():
        parser.error("输出必须是已存在目录下的 PNG 文件")
    draw(read_comparison(args.input), args.output)


if __name__ == "__main__":
    main()
