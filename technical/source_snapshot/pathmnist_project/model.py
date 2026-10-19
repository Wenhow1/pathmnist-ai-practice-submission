import torch
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

from .common import LABELS


def build_model(name, pretrained=True):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, 9)
    elif name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 9)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["labels"] != LABELS:
        raise ValueError("Checkpoint label order differs from the project schema")
    model = build_model(checkpoint["config"]["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint

