"""Compare uninterrupted and resumed trajectories without reading test data."""

import argparse
import json
from pathlib import Path

import torch


def _same(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _history_without_runtime(rows):
    return [{key: value for key, value in row.items() if key != "seconds"} for row in rows]


def compare(full, resumed):
    full = torch.load(Path(full) / "last.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(Path(resumed) / "last.pt", map_location="cpu", weights_only=True)
    fields = ["format_version", "protocol", "config", "manifest_id", "source_id", "environment",
              "device", "workers", "smoke", "epoch", "best_epoch", "best_f1", "stale", "status",
              "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng",
              "best_state_dict", "best_indices", "best_labels", "best_probabilities"]
    results = {field: _same(full[field], resumed[field]) for field in fields}
    results["history_without_runtime"] = _same(
        _history_without_runtime(full["history"]), _history_without_runtime(resumed["history"]))
    results["four_epochs_complete"] = (full["epoch"] == resumed["epoch"] == 4 and
                                       full["status"] == resumed["status"] == "budget_complete")
    return {"passed": all(results.values()), "checks": results, "epoch": full["epoch"],
            "manifest_id": full["manifest_id"], "source_id": full["source_id"],
            "device": full["device"], "environment": full["environment"]}


def main():
    parser = argparse.ArgumentParser(description="Exact checkpoint comparison for resume acceptance")
    parser.add_argument("--full", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.full, args.resumed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failed_checks": [k for k, v in result["checks"].items() if not v]},
                     ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
