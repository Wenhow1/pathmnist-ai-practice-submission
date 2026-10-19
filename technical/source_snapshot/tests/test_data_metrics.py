import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pathmnist_project.common import read_json
from pathmnist_project.data import PathDataset, resolve_duplicates, transform
from pathmnist_project.metrics import classification_metrics, save_predictions
from pathmnist_project.predict import read_image


def test_duplicate_priority_and_conflicts():
    hashes = {"train": ["a", "b", "c", "d"], "val": ["b", "e"], "test": ["c", "f"]}
    labels = {"train": [0, 1, 2, 3], "val": [4, 5], "test": [2, 6]}
    keep, removed = resolve_duplicates(hashes, labels)
    assert keep["train"].tolist() == [0, 3]
    assert keep["test"].tolist() == [0, 1]
    assert sum(row["conflicting_labels"] for row in removed) == 1


def test_test_duplicates_retained_and_development_removed():
    keep, removed = resolve_duplicates({"train": ["a"], "val": ["a"], "test": ["a", "a"]},
                                       {"train": [0], "val": [0], "test": [0, 0]})
    assert keep["test"].tolist() == [0, 1]
    assert len(keep["train"]) == len(keep["val"]) == 0
    assert len(removed) == 2


@pytest.mark.parametrize("size", [64, 128])
def test_dataset_preserves_samples_across_resolution(tiny_data, size):
    dataset = PathDataset(tiny_data, "val", size)
    tensor, y, index = dataset[0]
    assert tensor.shape == (3, size, size)
    assert torch.isfinite(tensor).all()
    assert y == index == 0
    assert torch.equal(tensor, dataset[0][0])
    assert np.array_equal(dataset.indices, PathDataset(tiny_data, "val", 128).indices)


def test_no_test_augmentation(tiny_data):
    with pytest.raises(ValueError, match="only allowed"):
        PathDataset(tiny_data, "test", 64, augment=True)


def test_changed_indices_rejected(tiny_data):
    np.save(tiny_data / "train_indices.npy", np.array([0]))
    with pytest.raises(ValueError, match="checksum"):
        PathDataset(tiny_data, "train", 64)


def test_transform_known_normalization():
    result = transform(64)(Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)))
    assert result[0, 0, 0].item() == pytest.approx(-.485 / .229)


def test_metrics_perfect_and_roundtrip(tmp_path):
    y = np.tile(np.arange(9), 3)
    p = np.eye(9)[y] * .9 + .1 / 9
    metrics = save_predictions(tmp_path, "test", np.arange(len(y)), y, p)
    assert metrics["macro_f1"] == metrics["macro_auc_ovr"] == metrics["accuracy"] == 1
    frame = pd.read_csv(tmp_path / "test_predictions.csv")
    recomputed = classification_metrics(frame.true_label.to_numpy(), frame[[f"p_{i}" for i in range(9)]].to_numpy())
    assert recomputed == read_json(tmp_path / "test_metrics.json")
    assert np.asarray(metrics["confusion_matrix"]).sum() == len(y)


def test_majority_predictions_do_not_hide_poor_macro_f1():
    y = np.arange(9)
    p = np.zeros((9, 9))
    p[:, 0] = 1
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(1 / 9)
    assert m["macro_f1"] == pytest.approx(.2 / 9)
    assert m["per_class"][0]["specificity"] == 0
    assert m["per_class"][1]["sensitivity"] == 0


def test_auc_missing_class_is_explicitly_unavailable():
    m = classification_metrics(np.array([0, 0]), np.eye(9)[[0, 0]])
    assert m["macro_auc_ovr"] is None
    assert m["per_class"][1]["sensitivity"] is None


@pytest.mark.parametrize("p", [np.full((2, 9), np.nan), np.ones((2, 9)), np.zeros((2, 8))])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), p)


def test_image_valid_and_corrupted():
    buffer = io.BytesIO()
    Image.new("L", (64, 64)).save(buffer, format="PNG")
    assert read_image(buffer.getvalue()).mode == "RGB"
    with pytest.raises(ValueError, match="无法读取"):
        read_image(b"not an image")
