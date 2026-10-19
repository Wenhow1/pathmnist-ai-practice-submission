import csv
import gc
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import LABELS, environment, get_device, object_digest, read_json, seed_all, write_json
from .data import PathDataset
from .metrics import classification_metrics, save_predictions
from .model import build_model, load_checkpoint


def worker_seed(worker_id):
    import random
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)


def loader(dataset, batch_size, workers=0, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=worker_seed, generator=generator, persistent_workers=workers > 0,
                      drop_last=shuffle and len(dataset) % batch_size == 1)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def infer(model, batches, device):
    model.eval()
    labels, probabilities, indices = [], [], []
    for x, y, index in batches:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite model output")
        probabilities.append(logits.softmax(1).cpu().numpy())
        labels.append(y.numpy())
        indices.append(index.numpy())
    return np.concatenate(indices), np.concatenate(labels), np.concatenate(probabilities)


def cpu_benchmark(checkpoint, threads=1, warmup=50, repeats=500):
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid benchmark repeat count")
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model, saved = load_checkpoint(checkpoint)
        x = torch.zeros(1, 3, saved["config"]["size"], saved["config"]["size"])
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                timings.append((time.perf_counter() - start) * 1000)
        return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                "threads": threads, "batch_size": 1, "warmup": warmup, "repeats": repeats,
                "preprocessing_included": False, "device": "cpu", "environment": environment(),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "checkpoint_bytes": Path(checkpoint).stat().st_size}
    finally:
        torch.set_num_threads(previous)


def source_id():
    import hashlib
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def train(config, data, directory, device_name="auto", workers=0, smoke=False, restart=False):
    directory = Path(directory)
    if (directory.parent / "frozen.json").exists() and not smoke:
        raise RuntimeError("Experiment is frozen; use a new run root for a new development cycle.")
    if directory.exists():
        if not restart:
            raise FileExistsError(f"Run directory exists: {directory}; completed runs are skipped by run-matrix.")
        backup = directory.with_name(directory.name + ".previous-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S%f"))
        directory.rename(backup)
    directory.mkdir(parents=True)
    manifest = read_json(Path(data) / "manifest.json")
    if not smoke and (manifest.get("data_mode") != "full" or not manifest.get("full_archive_checksum_verified")):
        raise ValueError("Formal training requires the complete, checksum-verified official dataset.")
    config = dict(config)
    if smoke:
        config.update(epochs=1, batch_size=16)
    seed_all(config["seed"])
    device = get_device(device_name)
    torch.set_num_threads(4)
    training = PathDataset(data, "train", config["size"], config["augment"], 16 if smoke else None)
    validation = PathDataset(data, "val", config["size"], False, 8 if smoke else None)
    train_loader = loader(training, config["batch_size"], workers, True, config["seed"])
    val_loader = loader(validation, config["batch_size"], workers)
    model = build_model(config["model"], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    meta = {"config": config, "config_id": object_digest(config), "manifest_id": manifest["manifest_id"],
            "source_id": source_id(), "environment": environment(), "device": str(device), "smoke": smoke,
            "workers": workers,
            "started_utc": datetime.now(UTC).isoformat(), "train_n": len(training), "val_n": len(validation),
            "pretrained_weights": "IMAGENET1K_V1", "status": "running"}
    write_json(directory / "run.json", meta)
    best, stale, history = -1., 0, []
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, n = 0., 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
            n += len(y)
        index, y, p = infer(model, val_loader, device)
        metrics = classification_metrics(y, p)
        synchronize(device)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_macro_f1": metrics["macro_f1"],
               "val_accuracy": metrics["accuracy"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started}
        history.append(row)
        with open(directory / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            checkpoint = {"config": config, "labels": LABELS, "manifest_id": manifest["manifest_id"],
                          "source_id": meta["source_id"], "epoch": epoch, "smoke": smoke,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            temp = directory / "best.tmp.pt"
            torch.save(checkpoint, temp)
            temp.replace(directory / "best.pt")
            save_predictions(directory, "val", index, y, p)
        else:
            stale += 1
        print(f"{directory.name} epoch={epoch} loss={row['train_loss']:.4f} val_f1={metrics['macro_f1']:.4f} seconds={row['seconds']:.1f}", flush=True)
        scheduler.step()
        if stale >= config["patience"]:
            break
    original = load_checkpoint(directory / "best.pt", device)[0]
    index, y, p = infer(original, val_loader, device)
    stored = read_json(directory / "val_metrics.json")
    if abs(classification_metrics(y, p)["macro_f1"] - stored["macro_f1"]) > 1e-9:
        raise AssertionError("Reloaded checkpoint does not reproduce saved validation metrics")
    benchmark = cpu_benchmark(directory / "best.pt", warmup=5 if smoke else 50, repeats=20 if smoke else 500)
    write_json(directory / "benchmark.json", benchmark)
    meta.update(status="complete", best_val_macro_f1=best, epochs_completed=len(history),
                finished_utc=datetime.now(UTC).isoformat())
    write_json(directory / "run.json", meta)
    return meta


def probe(data, runs, device_name="auto"):
    device = get_device(device_name)
    torch.set_num_threads(4)
    measurements = []
    # Check both actual architectures at the largest resolution, including optimizer state.
    for batch_size in [128, 64, 32]:
        success = True
        for name in ["resnet18", "mobilenet_v3_small"]:
            model = optimizer = x = y = loss = None
            try:
                seed_all(42)
                model = build_model(name, pretrained=False).to(device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
                x = torch.randn(batch_size, 3, 128, 128, device=device)
                y = torch.arange(batch_size, device=device) % 9
                start = time.perf_counter()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()
                synchronize(device)
                measurements.append({"model": name, "batch_size": batch_size,
                                     "step_seconds": (time.perf_counter() - start) / 2})
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                success = False
            finally:
                del model, optimizer, x, y, loss
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
            if not success:
                break
        if success:
            result = {"batch_size": batch_size, "device": str(device), "environment": environment(),
                      "measurements": measurements, "manifest_id": read_json(Path(data) / "manifest.json")["manifest_id"]}
            write_json(Path(runs) / "batch_probe.json", result)
            return result
    raise RuntimeError("Neither model pair fits batch size 32. Change hardware before formal runs.")
