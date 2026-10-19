"""One explicit command runs data preparation, development, freeze and final evaluation."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--runs", default="runs/formal")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--workers", type=int, default=0, help="0 is portable on Windows; increase after validation")
    parser.add_argument("--prepare-data", action="store_true", help="Explicitly allow a 4.26 GB dataset download")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。先检查 NVIDIA 驱动与 CUDA 版 PyTorch，再启动正式实验。")
    def run(command, *extra):
        subprocess.run([sys.executable, "-m", "pathmnist_project.cli", command, "--data", args.data,
                        "--runs", args.runs, "--output", args.output, "--device", "cuda",
                        "--workers", str(args.workers), *extra], check=True)
    if not (Path(args.data) / "manifest.json").exists():
        if not args.prepare_data:
            raise SystemExit("数据尚未准备。增加 --prepare-data 可明确允许下载；或先运行 prepare。")
        run("prepare")
    run("explore")
    frozen = Path(args.runs) / "frozen.json"
    if not frozen.exists():
        if not (Path(args.runs) / "batch_probe.json").exists():
            run("probe")
        run("run-matrix", *(["--restart-incomplete"] if args.restart_incomplete else []))
        run("freeze")
    run("evaluate")
    run("summarize")
    print("正式实验和结果汇总完成。查看", args.output)


if __name__ == "__main__":
    main()
