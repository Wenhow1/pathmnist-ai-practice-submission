from pathlib import Path

import numpy as np

from .common import GROUPS, SEEDS, digest, get_device, object_digest, read_json, write_json
from .data import PathDataset, verify_data
from .engine import cpu_benchmark, infer, loader, source_id, train
from .metrics import save_predictions
from .model import load_checkpoint


def make_matrix(batch_size):
    return [{"id": f"{group}_seed{seed}", "group": group, "model": model, "size": size,
             "augment": augment, "seed": seed, "batch_size": batch_size, "epochs": 40,
             "patience": 8, "lr": .0003, "weight_decay": .0001}
            for group, (model, size, augment) in GROUPS.items() for seed in SEEDS]


def run_matrix(data, runs, device, workers, restart=False, only_ids=None):
    runs = Path(runs)
    if (runs / "frozen.json").exists():
        raise RuntimeError("Frozen experiment cannot be trained again.")
    probe = read_json(runs / "batch_probe.json")
    manifest = verify_data(data)
    if probe["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Probe and training dataset differ")
    if probe["device"] != str(get_device(device)):
        raise ValueError("Probe must be repeated on the selected training device")
    matrix = make_matrix(probe["batch_size"])
    selected = set(only_ids) if only_ids is not None else {config["id"] for config in matrix}
    unknown = selected - {config["id"] for config in matrix}
    if unknown or not selected:
        raise ValueError(f"Unknown or empty run selection: {sorted(unknown)}")
    if (runs / "matrix.json").exists() and read_json(runs / "matrix.json") != matrix:
        raise ValueError("Batch size or experiment matrix changed; use a new run root")
    write_json(runs / "matrix.json", matrix)
    for config in matrix:
        if config["id"] not in selected:
            continue
        directory = runs / config["id"]
        if (directory / "run.json").exists():
            previous = read_json(directory / "run.json")
            if previous["status"] == "complete":
                if (previous["config"] != config or previous["manifest_id"] != manifest["manifest_id"]
                        or previous["source_id"] != source_id() or previous.get("workers") != workers):
                    raise ValueError("Completed run configuration, code, data, or worker count changed")
                print(f"Skipping completed run {config['id']}", flush=True)
                continue
        train(config, data, directory, device, workers, restart=restart)


def select_configuration(rows):
    groups = {}
    for row in rows:
        if row["group"] not in "ABCD":
            continue
        groups.setdefault(row["group"], []).append(row)
    if set(groups) != set("ABCD") or any(sorted(r["seed"] for r in group) != SEEDS for group in groups.values()):
        raise ValueError("Selection requires all three seeds for each of A, B, C, D")
    candidates = [{"group": key, "mean_val_f1": float(np.mean([r["val_macro_f1"] for r in values])),
                   "median_cpu_ms": float(np.median([r["cpu_median_ms"] for r in values]))}
                  for key, values in groups.items()]
    best = max(c["mean_val_f1"] for c in candidates)
    eligible = [c for c in candidates if c["mean_val_f1"] >= best - .005 - 1e-12]
    chosen = min(eligible, key=lambda c: (c["median_cpu_ms"], -c["mean_val_f1"], c["group"]))
    run = min(groups[chosen["group"]], key=lambda r: (-r["val_macro_f1"], r["seed"]))
    return {"group": chosen["group"], "run_id": run["id"], "candidates": candidates,
            "selection_basis": "validation only; within 0.005 choose lower CPU latency; seed by validation F1"}


def freeze(data, runs):
    runs = Path(runs)
    if (runs / "frozen.json").exists():
        raise FileExistsError("Freeze already exists; use evaluate to continue")
    matrix = read_json(runs / "matrix.json")
    probe = read_json(runs / "batch_probe.json")
    if matrix != make_matrix(probe["batch_size"]):
        raise ValueError("Formal freeze requires the complete 15-run matrix")
    manifest = verify_data(data)
    if manifest.get("data_mode") != "full":
        raise ValueError("Cannot freeze a partial dataset as a formal experiment")
    rows, locks = [], {}
    benchmark_host = None
    for config in matrix:
        folder = runs / config["id"]
        meta = read_json(folder / "run.json")
        if meta["status"] != "complete" or meta["smoke"] or meta["config"] != config:
            raise ValueError(f"Incomplete or mismatched formal run: {folder}")
        if meta["manifest_id"] != manifest["manifest_id"] or meta["source_id"] != source_id():
            raise ValueError("Mixed data manifests or source revisions")
        # Rerun every CPU benchmark now, on one host, before selecting by latency.
        benchmark = cpu_benchmark(folder / "best.pt")
        write_json(folder / "benchmark.json", benchmark)
        benchmark_host = benchmark["environment"]
        metric = read_json(folder / "val_metrics.json")
        rows.append({"id": config["id"], "group": config["group"], "seed": config["seed"],
                     "val_macro_f1": metric["macro_f1"], "cpu_median_ms": benchmark["median_ms"]})
        locks[config["id"]] = {filename: digest(folder / filename) for filename in
                               ["best.pt", "run.json", "val_metrics.json", "val_predictions.csv", "benchmark.json"]}
    result = {"manifest_id": manifest["manifest_id"], "source_id": source_id(),
              "matrix": matrix, "locks": locks, "selection": select_configuration(rows),
              "benchmark_environment": benchmark_host}
    result["freeze_id"] = object_digest(result)
    write_json(runs / "frozen.json", result)
    return result


def validate_freeze(data, runs):
    runs = Path(runs)
    frozen = read_json(runs / "frozen.json")
    identity = {k: v for k, v in frozen.items() if k != "freeze_id"}
    if frozen["freeze_id"] != object_digest(identity):
        raise ValueError("Freeze manifest was modified")
    if frozen["manifest_id"] != read_json(Path(data) / "manifest.json")["manifest_id"]:
        raise ValueError("Dataset changed since freeze")
    if frozen["source_id"] != source_id():
        raise ValueError("Code changed since freeze")
    for run_id, files in frozen["locks"].items():
        for filename, expected in files.items():
            if digest(runs / run_id / filename) != expected:
                raise ValueError(f"Frozen artifact changed: {run_id}/{filename}")
    return frozen


def evaluate(data, runs, device="auto", workers=0):
    verify_data(data)
    frozen = validate_freeze(data, runs)
    runs, device = Path(runs), get_device(device)
    for config in frozen["matrix"]:
        folder = runs / config["id"]
        model, saved = load_checkpoint(folder / "best.pt", device)
        dataset = PathDataset(data, "test", config["size"])
        if saved["manifest_id"] != dataset.manifest_id:
            raise ValueError("Model and evaluation data manifests differ")
        index, y, p = infer(model, loader(dataset, config["batch_size"], workers), device)
        metrics = save_predictions(folder, "test", index, y, p)
        write_json(folder / "test_provenance.json", {"freeze_id": frozen["freeze_id"], "checkpoint_sha256": digest(folder / "best.pt")})
        print(f"{config['id']}: test macro-F1={metrics['macro_f1']:.4f}", flush=True)
    train_data = PathDataset(data, "train", 64)
    majority = int(np.bincount(train_data.labels[train_data.indices], minlength=9).argmax())
    test = PathDataset(data, "test", 64)
    p = np.zeros((len(test), 9), dtype=np.float32)
    p[:, majority] = 1
    save_predictions(runs / "majority_baseline", "test", test.indices, test.labels[test.indices], p)
    write_json(runs / "majority_baseline" / "definition.json", {"class": majority, "chosen_from": "training labels only"})
    return frozen
