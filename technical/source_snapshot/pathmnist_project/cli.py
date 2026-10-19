import argparse
import json
from pathlib import Path

from .common import environment, write_json


def main():
    parser = argparse.ArgumentParser(description="PathMNIST reproducible experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print device and package information")
    for name in ["prepare", "explore", "probe", "smoke", "run-one", "run-matrix", "freeze", "evaluate", "summarize"]:
        p = sub.add_parser(name)
        p.add_argument("--data", default="data")
        p.add_argument("--runs", default="runs/formal")
        p.add_argument("--output", default="artifacts")
        p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
        p.add_argument("--workers", type=int, default=0)
        if name in ["run-one", "run-matrix", "smoke"]:
            p.add_argument("--restart-incomplete", action="store_true")
        if name == "run-one":
            p.add_argument("--id", required=True, help="Run ID from the fixed matrix, such as A_seed42")
    p = sub.add_parser("predict")
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        from .data import prepare
        print(json.dumps(prepare(args.data), ensure_ascii=False, indent=2))
    elif args.command == "explore":
        from .analysis import explore
        explore(args.data, args.output)
    elif args.command == "probe":
        from .engine import probe
        print(json.dumps(probe(args.data, args.runs, args.device), indent=2))
    elif args.command == "smoke":
        from .engine import train
        from .experiment import make_matrix
        config = make_matrix(16)[0]
        train(config, args.data, Path(args.runs) / "smoke", args.device, args.workers,
              smoke=True, restart=args.restart_incomplete)
    elif args.command == "run-one":
        from .experiment import run_matrix
        run_matrix(args.data, args.runs, args.device, args.workers, args.restart_incomplete,
                   only_ids=[args.id])
    elif args.command == "run-matrix":
        from .experiment import run_matrix
        run_matrix(args.data, args.runs, args.device, args.workers, args.restart_incomplete)
    elif args.command == "freeze":
        from .experiment import freeze
        print(json.dumps(freeze(args.data, args.runs), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        from .experiment import evaluate
        evaluate(args.data, args.runs, args.device, args.workers)
    elif args.command == "summarize":
        from .analysis import summarize
        summarize(args.data, args.runs, args.output)
    elif args.command == "predict":
        from .predict import predict_file
        result = predict_file(args.image, args.checkpoint)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
