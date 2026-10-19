import numpy as np
import pytest

from pathmnist_project.common import digest, object_digest, write_json


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(42)
    effective = {}
    for split in ["train", "val", "test"]:
        labels = np.repeat(np.arange(9), 2).astype(np.int64)
        images = rng.integers(0, 256, (len(labels), 128, 128, 3), dtype=np.uint8)
        np.save(root / f"{split}_images.npy", images)
        np.save(root / f"{split}_labels.npy", labels[:, None])
        np.save(root / f"{split}_indices.npy", np.arange(len(labels)))
        effective[split] = {"n": len(labels), "class_counts": [2] * 9,
                            "indices_sha256": digest(root / f"{split}_indices.npy")}
    manifest = {"effective": effective, "protocol": "test_fixture", "duplicate_count": 0,
                "patient_audit": "synthetic test fixture"}
    manifest["manifest_id"] = object_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return root
