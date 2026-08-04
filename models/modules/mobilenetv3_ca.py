"""MobileNetV3 with Coordinate Attention replacing torchvision SE blocks."""

import torch
from torch import nn
from torchvision.ops.misc import SqueezeExcitation

from ultralytics.nn.modules import TorchVision


class CoordinateAttention(nn.Module):
    """Coordinate Attention from height and width directional pooling."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h_pool = x.mean(dim=3, keepdim=True)
        w_pool = x.mean(dim=2, keepdim=True).transpose(2, 3)
        pooled = torch.cat((h_pool, w_pool), dim=2)
        pooled = self.act(self.bn1(self.conv1(pooled)))
        h_attention, w_attention = torch.split(
            pooled, [x.shape[2], x.shape[3]], dim=2
        )
        w_attention = w_attention.transpose(2, 3)
        h_attention = self.conv_h(h_attention).sigmoid()
        w_attention = self.conv_w(w_attention).sigmoid()
        return identity * h_attention * w_attention


def _replace_se(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SqueezeExcitation):
            channels = child.fc1.in_channels
            setattr(module, name, CoordinateAttention(channels))
            replaced += 1
        else:
            replaced += _replace_se(child)
    return replaced


class MobileNetV3CA(TorchVision):
    """Ultralytics TorchVision wrapper with MobileNetV3 SE blocks replaced by CA."""

    def __init__(
        self,
        model: str,
        weights: str = "DEFAULT",
        unwrap: bool = True,
        truncate: int = 2,
        split: bool = False,
    ):
        super().__init__(model, weights, unwrap, truncate, split)
        self.ca_count = _replace_se(self.m)
