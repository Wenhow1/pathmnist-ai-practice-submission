import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

LABELS = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
          "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]
LABELS_ZH = ["脂肪", "背景", "碎屑", "淋巴细胞", "黏液", "平滑肌", "正常结肠黏膜", "癌相关间质", "结直肠腺癌上皮"]
SEEDS = [42, 2026, 3407]
GROUPS = {"A": ("resnet18", 64, True), "B": ("resnet18", 128, True),
          "C": ("mobilenet_v3_small", 64, True), "D": ("mobilenet_v3_small", 128, True),
          "E": ("mobilenet_v3_small", 128, False)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def environment():
    import importlib.metadata
    packages = {}
    for p in ["torch", "torchvision", "medmnist", "numpy", "scikit-learn", "pillow", "streamlit"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    return {"platform": platform.platform(), "python": platform.python_version(), "packages": packages,
            "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

