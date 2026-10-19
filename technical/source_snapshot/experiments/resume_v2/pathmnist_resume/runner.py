"""Isolated, epoch-boundary-resumable PathMNIST training prototype.

This package never writes to the original ``runs/formal`` tree.  Its results
have a different training protocol and must not be mixed into the A--E matrix.
"""

import csv
import fcntl
import hashlib
import os
import random
import shutil
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import LABELS, digest, environment, object_digest, read_json, seed_all, write_json
from .data import PathDataset
from .metrics import classification_metrics, save_predictions
from .model import build_model

FORMAT_VERSION = 2
SCHEDULER = {"name": "plateau_val_loss", "factor": 0.5, "patience": 4,
             "threshold": 0.0001, "min_lr": 1e-7}
REQUIRED_CONFIG = {"id", "model", "size", "augment", "seed", "batch_size", "epochs",
                   "patience", "lr", "weight_decay"}


def utc_now():
    return datetime.now(UTC).isoformat()


def code_id():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_config(config):
    if not isinstance(config, dict) or not REQUIRED_CONFIG <= set(config):
        raise ValueError(f"Config requires {sorted(REQUIRED_CONFIG)}")
    config = dict(config)
    config.setdefault("scheduler", SCHEDULER.copy())
    if config["scheduler"] != SCHEDULER:
        raise ValueError("This prototype only supports the fixed, extension-safe plateau scheduler")
    if config["model"] not in ("resnet18", "mobilenet_v3_small") or config["size"] not in (64, 128):
        raise ValueError("Unsupported model or input size")
    if type(config["augment"]) is not bool:
        raise ValueError("augment must be boolean")
    if not isinstance(config["id"], str) or not config["id"].strip():
        raise ValueError("Config id must be a nonempty string")
    for name in ("seed", "batch_size", "epochs", "patience"):
        if type(config[name]) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    for name in ("lr", "weight_decay"):
        if not isinstance(config[name], (int, float)) or not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"Invalid {name}")
    if config["lr"] == 0:
        raise ValueError("Learning rate must be positive")
    return config


def validate_development_data(root, smoke):
    """Verify train/validation provenance, without opening test predictions."""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    claimed = manifest["manifest_id"]
    if object_digest({k: v for k, v in manifest.items() if k != "manifest_id"}) != claimed:
        raise ValueError("Modified data manifest")
    if not smoke and (manifest.get("data_mode") != "full" or not manifest.get("full_archive_checksum_verified")):
        raise ValueError("Formal training requires the audited full archive")
    for split in ("train", "val"):
        expected = manifest["effective"][split]["indices_sha256"]
        if digest(root / f"{split}_indices.npy") != expected:
            raise ValueError(f"Changed {split} indices")
        for kind in ("images", "labels"):
            expected = manifest.get("files", {}).get(split, {}).get(f"{kind}_sha256")
            if expected and digest(root / f"{split}_{kind}.npy") != expected:
                raise ValueError(f"Changed {split} {kind}")
    return manifest


def _numpy_rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"name": name, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached": cached}


def capture_rng():
    state = {"python": random.getstate(), "numpy": _numpy_rng_state(),
             "torch_cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device inventory differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device differs from checkpoint")
        torch.mps.set_rng_state(state["mps"])


def _atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        file.write(value)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _hash_path(path):
    return path.with_name(path.name + ".sha256")


def _verified_hash(path):
    expected = _hash_path(path).read_text(encoding="ascii").strip()
    actual = digest(path)
    if expected != actual:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return actual


def _save_full_checkpoint(directory, state):
    """Keep a verified prior full checkpoint while atomically replacing latest."""
    last, previous = directory / "last.pt", directory / "prev.pt"
    if last.is_file():
        prior_hash = _verified_hash(last)
        temporary = previous.with_name(previous.name + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            os.link(last, temporary)
        except OSError:
            with open(last, "rb") as source, open(temporary, "wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        os.replace(temporary, previous)
        _atomic_text(_hash_path(previous), prior_hash + "\n")
    _atomic_torch_save(last, state)
    _atomic_text(_hash_path(last), digest(last) + "\n")


@contextmanager
def run_lock(directory):
    """Nonblocking, OS-released lock for every mutation of one run."""
    directory = Path(directory).resolve()
    lock_path = directory.with_name(directory.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run is already active: {directory}") from exc
        try:
            file.seek(0)
            file.truncate()
            file.write(f"pid={os.getpid()} utc={utc_now()}\n")
            file.flush()
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _atomic_history(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_macro_f1",
                                                    "val_accuracy", "lr", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _best_payload(state):
    return {"config": state["config"], "labels": LABELS, "manifest_id": state["manifest_id"],
            "source_id": state["source_id"], "epoch": state["best_epoch"], "smoke": state["smoke"],
            "protocol": "resume_v2_plateau", "state_dict": state["best_state_dict"]}


def _publish(directory, state):
    """Full checkpoint is authoritative; derived files can be repaired on resume."""
    _save_full_checkpoint(directory, state)
    if state["best_epoch"]:
        _atomic_torch_save(directory / "best.pt", _best_payload(state))
        save_predictions(directory, "val", state["best_indices"].numpy(),
                         state["best_labels"].numpy(), state["best_probabilities"].numpy())
    _atomic_history(directory / "history.csv", state["history"])
    summary = {"format_version": FORMAT_VERSION, "id": state["config"]["id"],
               "status": state["status"], "last_completed_epoch": state["epoch"],
               "epoch_budget": state["config"]["epochs"], "best_epoch": state["best_epoch"],
               "best_val_macro_f1": state["best_f1"] if state["best_epoch"] else None,
               "stale_epochs": state["stale"], "manifest_id": state["manifest_id"],
               "source_id": state["source_id"], "updated_utc": utc_now(),
               "pid": os.getpid(), "checkpoint": "last.pt"}
    write_json(directory / "status.json", summary)
    write_json(directory / "run.json", {
        **summary, "config": state["config"], "environment": state["environment"],
        "device": state["device"], "workers": state["workers"], "smoke": state["smoke"],
        "origin": state["origin"], "extension_events": state["extension_events"],
        "recovery_events": state["recovery_events"]})


def _load_authoritative(directory):
    directory = Path(directory)
    failed = []
    for name in ("last.pt", "prev.pt"):
        path = directory / name
        try:
            _verified_hash(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("format_version") != FORMAT_VERSION or state.get("protocol") != "resume_v2_plateau":
                raise ValueError("Not a compatible full checkpoint")
            if state["epoch"] != len(state["history"]) or state["best_epoch"] > state["epoch"]:
                raise ValueError("Inconsistent checkpoint epoch history")
            if name == "prev.pt":
                corrupt = directory / f"last.invalid-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}.pt"
                if (directory / "last.pt").exists():
                    os.replace(directory / "last.pt", corrupt)
                state["recovery_events"].append({"utc": utc_now(), "from": "prev.pt",
                                                 "failed_latest": failed, "quarantined_latest": corrupt.name if corrupt.exists() else None})
                _atomic_torch_save(directory / "last.pt", state)
                _atomic_text(_hash_path(directory / "last.pt"), digest(directory / "last.pt") + "\n")
            return state
        except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No valid full checkpoint; " + "; ".join(failed))


def _worker_seed(worker_id):
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def _loader(dataset, batch_size, workers, shuffle, epoch_seed):
    generator = torch.Generator().manual_seed(epoch_seed % (2**63 - 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=_worker_seed, generator=generator, persistent_workers=False)


@torch.inference_mode()
def _validate(model, batches, device):
    model.eval()
    total_loss, total = 0., 0
    indices, labels, probabilities = [], [], []
    for x, y, index in batches:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite validation logits")
        total_loss += float(torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum").cpu())
        total += len(y)
        indices.append(index.numpy())
        labels.append(y.numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return np.concatenate(indices), y, p, total_loss / total, classification_metrics(y, p)


def _new_state(config, manifest, workers, device, smoke, origin, model, optimizer, scheduler):
    return {"format_version": FORMAT_VERSION, "protocol": "resume_v2_plateau", "config": config,
            "manifest_id": manifest["manifest_id"], "source_id": code_id(),
            "environment": environment(), "device": str(device), "workers": workers, "smoke": smoke,
            "origin": origin, "extension_events": [], "epoch": 0, "best_epoch": 0,
            "best_f1": -1., "best_state_dict": {}, "best_indices": torch.empty(0, dtype=torch.long),
            "best_labels": torch.empty(0, dtype=torch.long),
            "best_probabilities": torch.empty(0, 9), "stale": 0, "history": [], "status": "running",
            "model_state_dict": _cpu_state(model), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "rng": capture_rng(), "recovery_events": []}


def _check_resume(state, manifest, device, workers, smoke):
    if state["source_id"] != code_id():
        raise ValueError("Training source changed; cannot claim exact resume")
    if state["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Data manifest changed; cannot resume")
    if state["environment"] != environment() or state["device"] != str(device):
        raise ValueError("Software or device environment changed; cannot claim exact resume")
    if state["workers"] != workers or state["smoke"] != smoke:
        raise ValueError("Worker count or smoke mode changed; cannot resume")
    validate_config(state["config"])


def train(data, directory, mode, *, config=None, source_checkpoint=None, epochs=None,
          device_name="cpu", workers=0, smoke=False, stop_after_epoch=None):
    """Start, resume, extend, or warm-start an isolated run.

    ``stop_after_epoch`` is a controlled interruption hook for acceptance tests.
    A real process interruption resumes from the most recent completed epoch.
    """
    directory = Path(directory).resolve()
    if any(part == "formal" for part in directory.parts):
        raise ValueError("The resume prototype cannot write inside a formal run path")
    isolated_runs = (Path(__file__).resolve().parents[1] / "runs").resolve()
    if not smoke and not directory.is_relative_to(isolated_runs):
        raise ValueError(f"Non-smoke runs must stay under {isolated_runs}")
    with run_lock(directory):
        return _train_locked(data, directory, mode, config=config, source_checkpoint=source_checkpoint,
                             epochs=epochs, device_name=device_name, workers=workers,
                             smoke=smoke, stop_after_epoch=stop_after_epoch)


def _train_locked(data, directory, mode, *, config, source_checkpoint, epochs,
                  device_name, workers, smoke, stop_after_epoch):
    if mode not in ("start", "resume", "extend", "warm_start"):
        raise ValueError("Choose start, resume, extend, or warm_start")
    directory = Path(directory).resolve()
    if any(part == "formal" for part in directory.parts):
        raise ValueError("The resume prototype cannot write inside a formal run path")
    if workers < 0 or (stop_after_epoch is not None and stop_after_epoch < 1):
        raise ValueError("Invalid worker count or stop epoch")
    manifest = validate_development_data(data, smoke)
    seed_all((config or {}).get("seed", 0))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    torch.set_num_threads(4)
    state = None
    if mode in ("start", "warm_start"):
        if directory.exists():
            raise FileExistsError(f"Run directory exists: {directory}")
        config = validate_config(config)
        if mode == "warm_start" and not source_checkpoint:
            raise ValueError("warm_start requires source_checkpoint")
        source = None
        if mode == "warm_start":
            source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
            if source.get("labels") != LABELS or source.get("config", {}).get("model") != config["model"]:
                raise ValueError("Warm-start checkpoint label order or model differs")
            if source.get("manifest_id") != manifest["manifest_id"]:
                raise ValueError("Warm-start checkpoint data manifest differs")
        directory.mkdir(parents=True)
        seed_all(config["seed"])
        model = build_model(config["model"], pretrained=(mode == "start" and not smoke)).to(device)
        if source is not None:
            model.load_state_dict(source["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER["factor"], patience=SCHEDULER["patience"],
            threshold=SCHEDULER["threshold"], min_lr=SCHEDULER["min_lr"])
        origin = {"kind": mode, "checkpoint": str(Path(source_checkpoint).resolve()),
                  "checkpoint_sha256": digest(source_checkpoint)} if source is not None else {"kind": "new_pretrained" if not smoke else "synthetic_smoke"}
        state = _new_state(config, manifest, workers, device, smoke, origin, model, optimizer, scheduler)
        _publish(directory, state)
    else:
        if config is not None or source_checkpoint is not None:
            raise ValueError("resume/extend load their original config; do not supply a new one")
        state = _load_authoritative(directory)
        _check_resume(state, manifest, device, workers, smoke)
        if mode == "resume":
            if state["status"] not in ("running", "paused"):
                raise ValueError("Only interrupted or paused runs can resume")
        else:
            if state["status"] != "budget_complete":
                raise ValueError("Only budget-complete runs can extend; early-stopped runs need warm_start")
            if type(epochs) is not int or epochs <= state["config"]["epochs"]:
                raise ValueError("extend requires a larger epoch budget")
            previous_budget = state["config"]["epochs"]
            state["config"] = {**state["config"], "epochs": epochs}
            state["extension_events"].append({"utc": utc_now(), "from_epochs": previous_budget,
                                               "to_epochs": epochs, "checkpoint_epoch": state["epoch"]})
        config = state["config"]
        model = build_model(config["model"], pretrained=False).to(device)
        model.load_state_dict(state["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER["factor"], patience=SCHEDULER["patience"],
            threshold=SCHEDULER["threshold"], min_lr=SCHEDULER["min_lr"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng(state["rng"])
        state["status"] = "running"
        _publish(directory, state)  # Repair derived files if the last process stopped between atomic writes.

    training = PathDataset(data, "train", config["size"], config["augment"], 2 if smoke else None)
    validation = PathDataset(data, "val", config["size"], False, 2 if smoke else None)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
        start_time = time.perf_counter()
        epoch_seed = config["seed"] * 1_000_003 + epoch
        training_batches = _loader(training, config["batch_size"], workers, True, epoch_seed)
        validation_batches = _loader(validation, config["batch_size"], workers, False, config["seed"])
        model.train()
        total_loss, n = 0., 0
        for x, y, _ in training_batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            n += len(y)
        indices, labels, p, val_loss, metrics = _validate(model, validation_batches, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": total_loss / n, "val_loss": val_loss,
               "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"],
               "lr": lr, "seconds": time.perf_counter() - start_time}
        state["history"].append(row)
        improved = metrics["macro_f1"] > state["best_f1"]
        if improved:
            state["best_f1"] = metrics["macro_f1"]
            state["best_epoch"] = epoch
            state["best_state_dict"] = _cpu_state(model)
            state["best_indices"] = torch.from_numpy(indices.copy())
            state["best_labels"] = torch.from_numpy(labels.copy())
            state["best_probabilities"] = torch.from_numpy(p.copy())
            state["stale"] = 0
        else:
            state["stale"] += 1
        state["epoch"] = epoch
        state["model_state_dict"] = _cpu_state(model)
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()
        state["rng"] = capture_rng()
        state["status"] = ("early_stopped" if state["stale"] >= config["patience"] else
                           "budget_complete" if epoch == config["epochs"] else
                           "paused" if stop_after_epoch == epoch else "running")
        _publish(directory, state)
        print(f"{config['id']} epoch={epoch} val_f1={metrics['macro_f1']:.6f} status={state['status']}", flush=True)
        if state["status"] != "running":
            break
    return read_json(directory / "run.json")
