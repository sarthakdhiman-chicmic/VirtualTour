from dataclasses import dataclass

import numpy as np


@dataclass
class PhotoInput:
    photo_id: str
    image: np.ndarray                    # BGR uint8
    room_type: str | None = None         # as chosen by the user at upload, e.g. "Living Room"
    room_label: str | None = None        # optional, to tell two rooms of one type apart ("Bedroom 2")
    focal_35mm: float | None = None      # EXIF FocalLengthIn35mmFilm if present
    depth: np.ndarray | None = None      # optional relative depth HxW in [0,1]; improves depth-of-field
