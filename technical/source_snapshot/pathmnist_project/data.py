"""Official data preparation with streaming extraction and memory-mapped training."""
import csv
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .common import LABELS, digest, object_digest, read_json, write_json

DATA_URL = "https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1"
DATA_MD5 = "ac42d08fb904d92c244187169d1fd1d9"
EXPECTED_COUNTS = {"train": 89996, "val": 10004, "test": 7180}


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "pathmnist_128.npz"
    if destination.exists():
        if digest(destination, "md5") != DATA_MD5:
            raise ValueError("Existing NPZ checksum mismatch. Inspect/remove the incomplete file before retrying.")
        return destination
    if shutil.disk_usage(root).free < 11 * 1024**3:
        raise RuntimeError("Download and extraction require at least 11 GiB free space.")
    temporary = destination.with_suffix(".npz.part")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "PathMNIST-course/0.1"})
    with urllib.request.urlopen(request, timeout=120) as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if digest(temporary, "md5") != DATA_MD5:
        raise ValueError("Downloaded NPZ checksum mismatch; .part retained for diagnosis.")
    temporary.replace(destination)
    return destination


def resolve_duplicates(hashes, labels):
    """Priority test > val > train. Never discard test cases based on their labels."""
    seen = {}
    keep, duplicates = {}, []
    for split in ["test", "val", "train"]:
        retained = []
        for index, value in enumerate(hashes[split]):
            prior = seen.get(value)
            if prior is not None and prior[0] != split:
                duplicates.append({"removed_split": split, "removed_index": index,
                                   "retained_split": prior[0], "retained_index": prior[1],
                                   "conflicting_labels": int(labels[split][index]) != prior[2]})
            else:
                retained.append(index)
                seen.setdefault(value, (split, index, int(labels[split][index])))
        keep[split] = np.asarray(retained, dtype=np.int64)
    return keep, duplicates


def prepare(root):
    root = Path(root)
    archive = download(root)
    if shutil.disk_usage(root).free < 6 * 1024**3 and not (root / "train_images.npy").exists():
        raise RuntimeError("Extraction requires approximately 6 GiB additional free space.")
    with zipfile.ZipFile(archive) as z:
        for split in EXPECTED_COUNTS:
            for kind in ["images", "labels"]:
                filename = f"{split}_{kind}.npy"
                target = root / filename
                if not target.exists() or target.stat().st_size != z.getinfo(filename).file_size:
                    temporary = target.with_suffix(".npy.part")
                    with z.open(filename) as source, open(temporary, "wb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                    temporary.replace(target)
    labels, hashes, counts, files = {}, {}, {}, {}
    for split, expected in EXPECTED_COUNTS.items():
        images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(root / f"{split}_labels.npy", allow_pickle=False).reshape(-1)
        if images.shape != (expected, 128, 128, 3) or images.dtype != np.uint8:
            raise ValueError(f"Unexpected image schema: {split} {images.shape} {images.dtype}")
        if len(y) != expected or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, range(9)).all():
            raise ValueError(f"Invalid labels: {split}")
        labels[split] = y
        hashes[split] = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        counts[split] = np.bincount(y, minlength=9).tolist()
        files[split] = {"images_sha256": digest(root / f"{split}_images.npy"),
                        "labels_sha256": digest(root / f"{split}_labels.npy")}
        print(f"Validated {split}: {len(y)} images", flush=True)
    keep, duplicates = resolve_duplicates(hashes, labels)
    effective = {}
    for split, indices in keep.items():
        if not len(indices):
            raise ValueError(f"Deduplication emptied {split}")
        np.save(root / f"{split}_indices.npy", indices)
        effective[split] = {"n": len(indices), "class_counts": np.bincount(labels[split][indices], minlength=9).tolist(),
                            "indices_sha256": digest(root / f"{split}_indices.npy")}
    with open(root / "cross_split_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["removed_split", "removed_index", "retained_split", "retained_index", "conflicting_labels"])
        writer.writeheader()
        writer.writerows(duplicates)
    manifest = {"dataset": "PathMNIST", "data_mode": "full", "full_archive_checksum_verified": True,
                "release": "Zenodo 10519652 / 128px", "source": DATA_URL,
                "md5": DATA_MD5, "labels": LABELS, "original_counts": EXPECTED_COUNTS,
                "original_class_counts": counts, "files": files, "effective": effective,
                "duplicate_count": len(duplicates), "duplicate_policy": "test > val > train; exact pixels only",
                "protocol": "official" if not duplicates else "official_with_cross_split_deduplication",
                "patient_audit": "NPZ has no patient identifiers; patient disjointness cannot be independently verified.",
                "source_evidence": "https://doi.org/10.1371/journal.pmed.1002730"}
    manifest["manifest_id"] = object_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_data(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["manifest_id"] != object_digest({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("Data manifest has been modified")
    for split in EXPECTED_COUNTS:
        for kind in ["images", "labels"]:
            if digest(root / f"{split}_{kind}.npy") != manifest["files"][split][f"{kind}_sha256"]:
                raise ValueError(f"Data file changed after audit: {split}_{kind}")
        if digest(root / f"{split}_indices.npy") != manifest["effective"][split]["indices_sha256"]:
            raise ValueError(f"Split indices changed after audit: {split}")
    return manifest


class QuarterTurn:
    def __call__(self, image):
        return image.rotate(90 * int(torch.randint(0, 4, ()).item()))


def transform(size, augment=False):
    if size not in (64, 128):
        raise ValueError("Supported input sizes are 64 and 128.")
    operations = [T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if augment:
        operations += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), QuarterTurn()]
    return T.Compose(operations + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class PathDataset(Dataset):
    def __init__(self, root, split, size, augment=False, limit_per_class=None):
        if split not in EXPECTED_COUNTS:
            raise ValueError("Unknown split")
        if augment and split != "train":
            raise ValueError("Augmentation is only allowed for training")
        self.root, self.split = Path(root), split
        manifest = read_json(self.root / "manifest.json")
        self.manifest_id = manifest["manifest_id"]
        if digest(self.root / f"{split}_indices.npy") != manifest["effective"][split]["indices_sha256"]:
            raise ValueError("Split index checksum mismatch; rerun data audit")
        self.indices = np.load(self.root / f"{split}_indices.npy", allow_pickle=False)
        self.labels = np.load(self.root / f"{split}_labels.npy", allow_pickle=False).reshape(-1)
        if limit_per_class:
            self.indices = np.concatenate([self.indices[self.labels[self.indices] == c][:limit_per_class] for c in range(9)])
        self.transform = transform(size, augment)
        self._images = None

    @property
    def images(self):
        if self._images is None:
            self._images = np.load(self.root / f"{self.split}_images.npy", mmap_mode="r", allow_pickle=False)
        return self._images

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return self.transform(Image.fromarray(self.images[index])), int(self.labels[index]), index
