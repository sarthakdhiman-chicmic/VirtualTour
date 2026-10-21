"""The fireplace adapter: a small head on COCO OneFormer's internal features -> fireplace mask.

Decision 2026-10-21 (user): the adapter predicts the FIREPLACE MASK only. Lit fires are found as today:
tour_worker/vision/fire_detect.py (flame_mask -> fire_regions) runs inside the predicted fireplace mask,
exactly as it runs inside ADE20K's mask. Every fireplace photo, lit or not, is a positive.

Inputs (on the 1/8 grid of the model input, un-padded part only):
  mask8  pixel-decoder mask features (256)
  dec16  pixel-decoder 1/16 feature (256, upsampled)
  enc16  Swin stage-3 feature (768, upsampled)
Output: one logit map (fireplace), same grid.
Per-source 1x1 projections (the 1/16 ones before upsampling), then three dilated 3x3 convs.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def grid_size(valid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """valid = (vh, vw, ph, pw): un-padded and padded input size -> (h8, w8), (h16, w16)."""
    vh, vw = int(valid[0]), int(valid[1])
    return (math.ceil(vh / 8), math.ceil(vw / 8)), (math.ceil(vh / 16), math.ceil(vw / 16))


def build_inputs(feats: dict) -> dict[str, torch.Tensor]:
    """Crop the stored features to the un-padded input; tensors with a batch dim of 1."""
    (h8, w8), (h16, w16) = grid_size(feats["valid"])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None]   # noqa: E731
    return {"mask8": t(feats["mask8"][:, :h8, :w8]), "dec16": t(feats["dec16"][:, :h16, :w16]),
            "enc16": t(feats["enc16"][:, :h16, :w16])}


class FireplaceAdapter(nn.Module):
    def __init__(self, width: int = 96, drop: float = 0.1):
        super().__init__()
        self.p_mask = nn.Conv2d(256, 48, 1)
        self.p_dec = nn.Conv2d(256, 24, 1)
        self.p_enc = nn.Conv2d(768, 48, 1)
        self.drop = nn.Dropout2d(drop)
        cin = 48 + 24 + 48
        layers = []
        for d in (1, 2, 4):
            layers += [nn.Conv2d(cin, width, 3, padding=d, dilation=d), nn.GroupNorm(8, width), nn.ReLU(inplace=True)]
            cin = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, mask8, dec16, enc16):
        size = mask8.shape[-2:]
        up = lambda x: F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # noqa: E731
        x = torch.cat([F.relu(self.p_mask(self.drop(mask8))), up(F.relu(self.p_dec(self.drop(dec16)))),
                       up(F.relu(self.p_enc(self.drop(enc16))))], 1)
        return self.out(self.body(x))


def targets(fireplace: np.ndarray, hw8: tuple[int, int]) -> torch.Tensor:
    """Teacher fireplace mask (any size, 0/255) -> (1, 1, h8, w8) soft target (share of each cell)."""
    h8, w8 = hw8
    f = cv2.resize((fireplace > 0).astype(np.float32), (w8, h8), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(f)[None, None].float()
