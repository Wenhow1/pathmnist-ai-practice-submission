"""Command-line entry point for isolated resume-prototype runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Isolated PathMNIST resume prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "resume", "extend", "warm-start"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--smoke", action="store_true", help="Accept synthetic data and skip pretrained weights")
        p.add_argument("--stop-after-epoch", type=int, help=argparse.SUPPRESS)
        if name in ("start", "warm-start"):
            p.add_argument("--config", required=True)
        if name == "warm-start":
            p.add_argument("--checkpoint", required=True)
        if name == "extend":
            p.add_argument("--epochs", type=int, required=True)
    p = sub.add_parser("status")
    p.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "status":
        status = Path(args.run) / "status.json"
        print(status.read_text(encoding="utf-8"))
        return
    from .runner import train
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if hasattr(args, "config") else None
    result = train(args.data, args.run, args.command.replace("-", "_"), config=config,
                   source_checkpoint=getattr(args, "checkpoint", None), epochs=getattr(args, "epochs", None),
                   device_name=args.device, workers=args.workers, smoke=args.smoke,
                   stop_after_epoch=args.stop_after_epoch)
    print(json.dumps({"id": result["id"], "status": result["status"],
                      "last_completed_epoch": result["last_completed_epoch"],
                      "best_val_macro_f1": result["best_val_macro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
