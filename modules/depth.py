"""Monocular depth for the parallax renderer.

Depth Anything V2 Small (Apache-2.0; the Base/Large/Giant checkpoints are CC-BY-NC and must
not be used) via onnxruntime. Each shot image gets a smoothed depth PNG next to it in the
Remotion public dir, plus a focus depth the shader pivots around. Any failure leaves the shot
without depth, and the renderer falls back to plain Ken Burns for it.
"""
import hashlib
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/"
    "4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx"
)
MODEL_SHA256 = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "depth_anything_v2_small.onnx"
)

INPUT_LONG_EDGE = 518  # model's native resolution; dims are rounded up to multiples of 14
DEPTH_MAP_LONG_EDGE = 1024  # stored map size; the shader samples it with linear filtering
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_session = None


def model_path() -> str:
    return os.environ.get("DEPTH_MODEL_PATH", DEFAULT_MODEL_PATH)


def ensure_model(path: str | None = None) -> str:
    """Return the model path, downloading the pinned checkpoint if it is missing."""
    path = path or model_path()
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    with open(tmp, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL_SHA256:
        os.remove(tmp)
        raise RuntimeError(f"depth model checksum mismatch: {digest}")
    os.replace(tmp, path)
    return path


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        _session = ort.InferenceSession(ensure_model(), providers=providers)
    return _session


def estimate_depth(bgr: np.ndarray) -> np.ndarray:
    """Relative inverse depth in [0, 1] (1 = nearest) at the model's working resolution."""
    sess = _get_session()
    h, w = bgr.shape[:2]
    s = INPUT_LONG_EDGE / max(w, h)
    nw = max(14, -(-int(round(w * s)) // 14) * 14)
    nh = max(14, -(-int(round(h * s)) // 14) * 14)
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0]
    d = sess.run(None, {inp.name: x})[0][0].astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    return (d - lo) / (hi - lo + 1e-8)


def prepare_depth_map(depth: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Shape the raw map for the warp shader.

    Near objects are dilated slightly so their silhouettes move as one piece and the
    stretching falls in the background behind them; a light blur keeps the warp smooth.
    Wider dilation/blur hides tearing but bends straight lines next to objects further out.
    """
    d = cv2.dilate(depth, np.ones((5, 5), np.uint8))
    d = cv2.GaussianBlur(d, (0, 0), 1.0)
    return cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def build_depth_maps(images_dir: str, basenames: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Write `<name>.depth.png` for each image. Returns ({basename: {image, focus}}, warnings)."""
    maps: dict[str, dict] = {}
    warnings: list[str] = []
    for name in dict.fromkeys(basenames):
        try:
            bgr = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("unreadable image")
            h, w = bgr.shape[:2]
            s = min(1.0, DEPTH_MAP_LONG_EDGE / max(w, h))
            d = prepare_depth_map(estimate_depth(bgr), max(1, int(w * s)), max(1, int(h * s)))
            depth_name = os.path.splitext(name)[0] + ".depth.png"
            cv2.imwrite(os.path.join(images_dir, depth_name), np.clip(d * 255.0, 0, 255).astype(np.uint8))
            # pivot the parallax around the scene's middle distance
            maps[name] = {"image": depth_name, "focus": round(float(np.median(d)), 4)}
        except Exception as e:
            warnings.append(f"depth_unavailable:{name}:{e}")
    return maps, warnings
