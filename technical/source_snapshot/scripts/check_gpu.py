"""Check the actual adapter without downloading data or model weights."""
import json

import torch

from pathmnist_project.common import environment

info = environment()
if torch.cuda.is_available():
    info["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                        "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)}
                       for i in range(torch.cuda.device_count())]
    x = torch.ones(2, 2, device="cuda")
    info["cuda_smoke_passed"] = bool((x @ x == 2).all())
print(json.dumps(info, indent=2, ensure_ascii=False))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: check driver and PyTorch installation.")
