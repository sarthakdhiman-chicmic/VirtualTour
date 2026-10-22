import io

import cv2
import numpy as np
from PIL import ExifTags, Image

from .inputs import PhotoInput


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to BGR uint8 (OpenCV applies EXIF orientation)."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def read_focal_35mm(data: bytes) -> float | None:
    """35mm-equivalent focal length from EXIF, or None. Never raises."""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
        v = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.FocalLengthIn35mmFilm)
        f = float(v) if v is not None else None
        return f if f and f > 0 else None
    except Exception:
        return None


def photo_from_bytes(photo_id: str, data: bytes, room_type: str | None = None,
                     room_label: str | None = None) -> PhotoInput:
    return PhotoInput(photo_id=photo_id, image=decode_image(data), room_type=room_type,
                      room_label=room_label, focal_35mm=read_focal_35mm(data))


def resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale so the long side is at most `long_side`. Never upscales."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    s = long_side / m
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
