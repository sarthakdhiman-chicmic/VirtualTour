"""3D scenes for plan 1.6 tours: a Gaussian splat model (.ply or .splat) and, optionally, the
operator's scene.json (rooms and camera anchors)."""


class SceneInvalid(ValueError):
    """The scene file or its scene.json can't be used; the backend falls back to the photo tour
    (fallback_reason "scene_invalid")."""
