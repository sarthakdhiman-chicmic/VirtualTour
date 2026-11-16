"""One array namespace for the motion fields: NumPy on the CPU, torch on the GPU (render/frames_gpu.py).

The effect modules (water, fabric, foliage, canopy, fire, noise) compute their fields with
`xp = ns(u)` instead of `np`, so the same reviewed arithmetic runs on either - there is no second
copy of any effect to drift. Only what those modules use is here; NumPy calls pass straight through.
"""
from __future__ import annotations

import numpy as np


def is_torch(a) -> bool:
    return type(a).__module__.startswith("torch")


class _Torch:
    """The NumPy calls the motion fields use, on torch tensors (same names, same semantics)."""

    def __init__(self):
        import torch
        self.torch = torch
        self.float32, self.float64, self.int64, self.int32 = torch.float32, torch.float64, torch.int64, torch.int32
        for name in ("sin", "cos", "exp", "tanh", "abs", "sqrt", "floor", "where", "hypot"):
            setattr(self, name, getattr(torch, name))

    def _t(self, x, like):
        return x if is_torch(x) else self.torch.as_tensor(x, dtype=like.dtype, device=like.device)

    def maximum(self, a, b):
        if not is_torch(a):
            a, b = b, a
        return self.torch.clamp(a, min=b) if not is_torch(b) else self.torch.maximum(a, b)

    def minimum(self, a, b):
        if not is_torch(a):
            a, b = b, a
        return self.torch.clamp(a, max=b) if not is_torch(b) else self.torch.minimum(a, b)

    def clip(self, a, lo, hi):
        return self.torch.clamp(a, min=lo, max=hi)

    def zeros_like(self, a, dtype=None):
        return self.torch.zeros_like(a, dtype=dtype)

    def ones_like(self, a, dtype=None):
        return self.torch.ones_like(a, dtype=dtype)

    def astype(self, a, dtype):
        return a.to(dtype)

    def full_like(self, a, value):
        return self.torch.full_like(a, value)

    def on(self, arr, like, owner=None, key=None):
        """A NumPy array as a tensor on like's device; cached on owner (under key) when given."""
        if is_torch(arr):
            return arr
        if owner is not None:
            cache = owner.__dict__.setdefault("_xp_dev", {})
            k = (key or id(arr), like.device)
            if k not in cache:
                cache[k] = self.torch.as_tensor(np.ascontiguousarray(arr), device=like.device)
            return cache[k]
        return self.torch.as_tensor(np.ascontiguousarray(arr), device=like.device)


_NP = np
_TORCH = None


class _Numpy:
    """NumPy, plus astype() as a function (torch tensors have no .astype)."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def astype(a, dtype):
        return a.astype(dtype)

    @staticmethod
    def on(arr, like, owner=None, key=None):
        return arr


_NPNS = _Numpy()


def ns(a):
    """The namespace for arrays like `a`: torch's when a is a tensor, else NumPy's."""
    global _TORCH
    if is_torch(a):
        if _TORCH is None:
            _TORCH = _Torch()
        return _TORCH
    return _NPNS
