import json

import numpy as np
import pytest
import torch
from pathmnist_resume.common import digest, object_digest, read_json, write_json
from pathmnist_resume.compare import compare
from pathmnist_resume.monitor import check_once
from pathmnist_resume.runner import run_lock, train


def synthetic_data(root):
    root.mkdir()
    rng = np.random.default_rng(3407)
    effective, files = {}, {}
    for split in ("train", "val"):
        y = np.repeat(np.arange(9), 2).astype(np.int64)
        x = rng.integers(0, 256, (len(y), 128, 128, 3), dtype=np.uint8)
        np.save(root / f"{split}_images.npy", x)
        np.save(root / f"{split}_labels.npy", y[:, None])
        np.save(root / f"{split}_indices.npy", np.arange(len(y)))
        effective[split] = {"indices_sha256": digest(root / f"{split}_indices.npy")}
        files[split] = {f"{kind}_sha256": digest(root / f"{split}_{kind}.npy")
                        for kind in ("images", "labels")}
    manifest = {"data_mode": "synthetic_test", "effective": effective, "files": files}
    manifest["manifest_id"] = object_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return root


def tiny_model(name, pretrained=False):
    assert name == "resnet18"
    assert not pretrained
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=2), torch.nn.ReLU(),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                               torch.nn.Dropout(0.2), torch.nn.Linear(4, 9))


def configuration(epochs):
    return {"id": "resume_acceptance", "model": "resnet18", "size": 64, "augment": True,
            "seed": 42, "batch_size": 6, "epochs": epochs, "patience": 8,
            "lr": 0.0003, "weight_decay": 0.0001}


def state(path):
    return torch.load(path / "last.pt", map_location="cpu", weights_only=True)


def compare_state(a, b):
    assert a["epoch"] == b["epoch"] == 4
    assert a["best_epoch"] == b["best_epoch"]
    assert a["best_f1"] == b["best_f1"]
    assert a["stale"] == b["stale"]
    assert a["scheduler_state_dict"] == b["scheduler_state_dict"]
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
    assert a["optimizer_state_dict"]["param_groups"] == b["optimizer_state_dict"]["param_groups"]
    for parameter, left in a["optimizer_state_dict"]["state"].items():
        right = b["optimizer_state_dict"]["state"][parameter]
        for key, value in left.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right[key], rtol=0, atol=0)
            else:
                assert value == right[key]
    for left, right in zip(a["history"], b["history"], strict=True):
        assert {k: v for k, v in left.items() if k != "seconds"} == {
            k: v for k, v in right.items() if k != "seconds"}
    assert a["rng"]["python"] == b["rng"]["python"]
    assert a["rng"]["numpy"] == b["rng"]["numpy"]
    torch.testing.assert_close(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"], rtol=0, atol=0)


def test_epoch_resume_and_budget_extension_match_uninterrupted(tmp_path, monkeypatch):
    import pathmnist_resume.runner as runner

    monkeypatch.setattr(runner, "build_model", tiny_model)
    data = synthetic_data(tmp_path / "data")
    full, paused, extended, recovered = (tmp_path / name for name in ("full", "paused", "extended", "recovered"))
    train(data, full, "start", config=configuration(4), smoke=True)
    train(data, paused, "start", config=configuration(4), smoke=True, stop_after_epoch=2)
    assert read_json(paused / "status.json")["status"] == "paused"
    # Simulate a crash after last.pt was committed but derived outputs were lost.
    for filename in ("best.pt", "val_predictions.csv", "val_metrics.json", "history.csv", "status.json", "run.json"):
        (paused / filename).unlink()
    train(data, paused, "resume", smoke=True)
    assert (paused / "best.pt").is_file()
    assert read_json(paused / "val_metrics.json")["macro_f1"] == state(paused)["best_f1"]
    train(data, extended, "start", config=configuration(2), smoke=True)
    train(data, extended, "extend", epochs=4, smoke=True)
    train(data, recovered, "start", config=configuration(4), smoke=True, stop_after_epoch=2)
    with open(recovered / "last.pt", "r+b") as file:
        file.write(b"BROKEN LATEST CHECKPOINT")
    train(data, recovered, "resume", smoke=True)
    assert read_json(recovered / "run.json")["recovery_events"][0]["from"] == "prev.pt"
    assert list(recovered.glob("last.invalid-*.pt"))
    compare_state(state(full), state(paused))
    compare_state(state(full), state(extended))
    compare_state(state(full), state(recovered))
    assert compare(full, paused)["passed"]
    assert compare(full, extended)["passed"]
    assert read_json(extended / "run.json")["extension_events"][0]["from_epochs"] == 2
    with pytest.raises(ValueError, match="Only interrupted or paused"):
        train(data, full, "resume", smoke=True)
    with pytest.raises(ValueError, match="larger epoch budget"):
        train(data, extended, "extend", epochs=4, smoke=True)
    with pytest.raises(ValueError, match="cannot write inside"):
        train(data, tmp_path / "formal" / "mistake", "start", config=configuration(1), smoke=True)
    with pytest.raises(ValueError, match="must stay under"):
        train(data, tmp_path / "outside_branch", "start", config=configuration(1), smoke=False)
    with run_lock(tmp_path / "locked"):
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(tmp_path / "locked"):
                pass


def test_warm_start_is_new_trajectory_and_monitor_is_low_frequency(tmp_path, monkeypatch):
    import pathmnist_resume.runner as runner

    monkeypatch.setattr(runner, "build_model", tiny_model)
    data = synthetic_data(tmp_path / "data")
    first, warm = tmp_path / "first", tmp_path / "warm"
    train(data, first, "start", config=configuration(1), smoke=True)
    config = {**configuration(1), "id": "warm_new_seed", "seed": 2026}
    train(data, warm, "warm_start", config=config, source_checkpoint=first / "best.pt", smoke=True)
    meta = read_json(warm / "run.json")
    assert meta["origin"]["kind"] == "warm_start"
    assert meta["origin"]["checkpoint_sha256"] == digest(first / "best.pt")
    assert meta["last_completed_epoch"] == 1
    assert check_once(warm, 60) == "budget_complete"
    assert check_once(warm, 60) == "budget_complete"
    assert len((warm / "monitor_events.jsonl").read_text().splitlines()) == 1
    assert json.loads((warm / "monitor_events.jsonl").read_text().splitlines()[0])["event"] == "budget_complete"
