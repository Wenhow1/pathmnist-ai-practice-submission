import io
import time

import torch
from PIL import Image, UnidentifiedImageError

from .common import LABELS, LABELS_ZH
from .data import transform
from .model import load_checkpoint


def read_image(source):
    try:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        image = Image.open(source)
        if image.width * image.height > 25_000_000:
            raise ValueError("图片过大，请上传 2500 万像素以内的图像。")
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请上传有效的 PNG 或 JPEG 文件。") from exc


@torch.inference_mode()
def predict_image(image, model, checkpoint):
    start = time.perf_counter()
    x = transform(checkpoint["config"]["size"])(image).unsqueeze(0)
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise ValueError("Demo prediction uses CPU for portable, comparable timing")
    probabilities = model(x).softmax(1)[0].cpu().numpy()
    result = {"class_id": int(probabilities.argmax()), "probabilities": probabilities.tolist(),
              "labels": LABELS, "labels_zh": LABELS_ZH, "elapsed_ms": (time.perf_counter() - start) * 1000,
              "timing_includes_preprocessing": True, "smoke_checkpoint": checkpoint.get("smoke", False)}
    return result


def predict_file(image_path, checkpoint_path):
    model, checkpoint = load_checkpoint(checkpoint_path)
    return predict_image(read_image(image_path), model, checkpoint)
