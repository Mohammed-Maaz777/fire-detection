# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Experimental modules."""

import math
import numpy as np
import torch
import torch.nn as nn

from yolov5.utils.downloads import attempt_download
from yolov5.models.yolo import Detect, Model


class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""
    def __init__(self, n, weight=False):
        super().__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        n = len(k)
        if equal_ch:
            i = torch.linspace(0, n - 1e-6, c2).floor()
            c_ = [(i == g).sum() for g in range(n)]
        else:
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.
    """
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()

        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]

    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
