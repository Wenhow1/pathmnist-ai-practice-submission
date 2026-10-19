import numpy as np
import pytest
import torch

from pathmnist_project.common import LABELS, SEEDS, object_digest, read_json, write_json
from pathmnist_project.engine import infer, loader
from pathmnist_project.experiment import make_matrix, run_matrix, select_configuration, validate_freeze
from pathmnist_project.model import build_model, load_checkpoint


def test_matrix_exact_design():
    configs = make_matrix(32)
    assert len(configs) == len({c["id"] for c in configs}) == 15
    assert {c["seed"] for c in configs} == set(SEEDS)
    assert sum(not c["augment"] for c in configs) == 3
    assert all(c["batch_size"] == 32 for c in configs)


def test_run_one_uses_fixed_matrix_without_starting_other_runs(tmp_path, monkeypatch):
    import pathmnist_project.experiment as experiment

    manifest = {"manifest_id": "fixed-data"}
    runs = tmp_path / "runs"
    write_json(runs / "batch_probe.json", {"batch_size": 32, "device": "cpu", "manifest_id": "fixed-data"})
    monkeypatch.setattr(experiment, "verify_data", lambda data: manifest)
    called = []
    monkeypatch.setattr(experiment, "train", lambda config, data, directory, device, workers, restart:
                        called.append(config["id"]))

    run_matrix(tmp_path / "data", runs, "cpu", 0, only_ids=["A_seed42"])
    assert called == ["A_seed42"]
    assert len(read_json(runs / "matrix.json")) == 15
    with pytest.raises(ValueError, match="Unknown or empty"):
        run_matrix(tmp_path / "data", runs, "cpu", 0, only_ids=["not-a-run"])
    assert called == ["A_seed42"]
    from pathmnist_project.engine import source_id
    write_json(runs / "A_seed42" / "run.json", {"status": "complete", "config": make_matrix(32)[0],
               "manifest_id": "fixed-data", "source_id": source_id(), "workers": 4})
    with pytest.raises(ValueError, match="configuration, code, data, or worker count changed"):
        run_matrix(tmp_path / "data", runs, "cpu", 0, only_ids=["A_seed42"])


def rows():
    return [{"group": group, "seed": seed, "id": f"{group}_seed{seed}",
             "val_macro_f1": score, "cpu_median_ms": latency}
            for group, score, latency in [("A", .91, 10), ("B", .92, 20), ("C", .90, 2), ("D", .916, 5)]
            for seed in SEEDS]


def test_selection_uses_validation_and_latency_only():
    values = rows()
    for row in values:
        row["test_f1"] = 1 if row["group"] == "A" else 0
    selected = select_configuration(values)
    assert selected["group"] == "D"
    assert selected["run_id"] == "D_seed42"


def test_incomplete_seeds_cannot_select():
    with pytest.raises(ValueError):
        select_configuration(rows()[:-1])


@pytest.mark.parametrize("name,size", [("resnet18",64),("resnet18",128),("mobilenet_v3_small",64),("mobilenet_v3_small",128)])
def test_checkpoint_prediction_roundtrip(tmp_path, name, size):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False).eval()
    image = torch.randn(2, 3, size, size)
    with torch.inference_mode():
        before = model(image)
    path = tmp_path / "best.pt"
    torch.save({"labels": LABELS, "config": {"model": name, "size": size}, "state_dict": model.state_dict()}, path)
    loaded, _ = load_checkpoint(path)
    with torch.inference_mode():
        after = loaded(image)
    assert before.shape == (2, 9)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_wrong_checkpoint_labels_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"labels": list(reversed(LABELS))}, path)
    with pytest.raises(ValueError, match="label order"):
        load_checkpoint(path)


def test_freeze_tampering_detected(tmp_path, tiny_data):
    runs = tmp_path / "runs"
    frozen = {"manifest_id": read_json(tiny_data / "manifest.json")["manifest_id"], "locks": {}}
    frozen["freeze_id"] = object_digest(frozen)
    frozen["manifest_id"] = "changed"
    write_json(runs / "frozen.json", frozen)
    with pytest.raises(ValueError, match="modified"):
        validate_freeze(tiny_data, runs)


def test_training_integration_with_synthetic_fixture(tmp_path, tiny_data, monkeypatch):
    import pathmnist_project.engine as engine
    from pathmnist_project.data import PathDataset
    monkeypatch.setattr(engine, "build_model", lambda name, pretrained: build_model(name, pretrained=False))
    config = make_matrix(16)[6]
    target = tmp_path / "runs" / "smoke"
    result = engine.train(config, tiny_data, target, "cpu", smoke=True)
    assert result["status"] == "complete"
    assert result["smoke"] is True
    model, checkpoint = load_checkpoint(target / "best.pt")
    index, y, p = infer(model, loader(PathDataset(tiny_data, "val", config["size"]), 16), torch.device("cpu"))
    assert len(index) == len(y) == len(p) == 18
    assert np.allclose(p.sum(1), 1)
    assert checkpoint["smoke"] is True
    with pytest.raises(FileExistsError):
        engine.train(config, tiny_data, target, "cpu", smoke=True)


def test_streamlit_empty_state():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert "病理" in app.title[0].value
    assert len(app.info) == 1


def test_formal_training_rejects_partial_data(tmp_path, tiny_data):
    from pathmnist_project.engine import train
    with pytest.raises(ValueError, match="complete, checksum"):
        train(make_matrix(32)[0], tiny_data, tmp_path / "run", "cpu")


def test_frozen_evaluation_and_summary_end_to_end(tmp_path, tiny_data, monkeypatch):
    """Synthetic predictions exercise orchestration; none of these are medical experiment results."""
    import pathmnist_project.experiment as experiment
    import pathmnist_project.model as model_module
    from pathmnist_project.analysis import summarize
    from pathmnist_project.common import digest, environment
    from pathmnist_project.engine import source_id
    from pathmnist_project.metrics import save_predictions

    def tiny_model(name, pretrained=True):
        return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 9))

    monkeypatch.setattr(model_module, "build_model", tiny_model)
    monkeypatch.setattr(experiment, "cpu_benchmark", lambda path: {
        "median_ms": 1., "p95_ms": 2., "environment": environment(),
        "parameter_count": 36, "checkpoint_bytes": 1000})
    manifest = read_json(tiny_data / "manifest.json")
    manifest.update(data_mode="full", full_archive_checksum_verified=True)
    manifest["files"] = {split: {f"{kind}_sha256": digest(tiny_data / f"{split}_{kind}.npy")
                                 for kind in ["images", "labels"]} for split in ["train", "val", "test"]}
    manifest["manifest_id"] = object_digest({k: v for k, v in manifest.items() if k != "manifest_id"})
    write_json(tiny_data / "manifest.json", manifest)
    runs = tmp_path / "formal"
    matrix = make_matrix(32)
    write_json(runs / "matrix.json", matrix)
    write_json(runs / "batch_probe.json", {"batch_size": 32})
    y = np.repeat(np.arange(9), 2)
    probabilities = np.eye(9)[y]
    for config in matrix:
        folder = runs / config["id"]
        folder.mkdir()
        write_json(folder / "run.json", {"status": "complete", "smoke": False, "config": config,
                   "manifest_id": manifest["manifest_id"], "source_id": source_id()})
        torch.save({"labels": LABELS, "config": config, "state_dict": tiny_model("").state_dict(),
                    "smoke": False, "source_id": source_id(), "manifest_id": manifest["manifest_id"]}, folder / "best.pt")
        save_predictions(folder, "val", np.arange(18), y, probabilities)
        (folder / "history.csv").write_text("epoch,train_loss,val_macro_f1\n1,1.0,0.5\n2,0.9,0.6\n")
    frozen = experiment.freeze(tiny_data, runs)
    assert len(frozen["locks"]) == 15
    experiment.evaluate(tiny_data, runs, "cpu")
    summarize(tiny_data, runs, tmp_path / "report")
    summary = read_json(tmp_path / "report" / "summary.json")
    assert len(summary["rows"]) == 15
    assert (tmp_path / "report" / "experiment_report.md").exists()
    assert (tmp_path / "report" / "error_examples.png").exists()
    (runs / matrix[0]["id"] / "val_metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        experiment.validate_freeze(tiny_data, runs)


def test_streamlit_loaded_checkpoint(tmp_path, tiny_data, monkeypatch):
    import sys
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    model = build_model("mobilenet_v3_small", pretrained=False)
    path = tmp_path / "demo.pt"
    torch.save({"labels": LABELS, "config": {"model": "mobilenet_v3_small", "size": 64},
                "state_dict": model.state_dict(), "smoke": True,
                "manifest_id": read_json(tiny_data / "manifest.json")["manifest_id"]}, path)
    monkeypatch.setattr(sys, "argv", ["app.py", "--checkpoint", str(path), "--data", str(tiny_data)])
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.warning
    assert len(app.dataframe) == 1
    app.selectbox[0].select(4).run()
    assert not app.exception
