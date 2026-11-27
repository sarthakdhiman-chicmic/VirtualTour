import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

from tour_worker.errors import WorkerError
from tour_worker.motion.masks import mask_relpath

FETCH_WORKERS = 8          # files fetched at once (one at a time took the 8K listing 34 s a job, item 22)


def prefetch(storage, pairs, workers: int = FETCH_WORKERS) -> dict:
    """Fetch (uri, dest) pairs in parallel, skipping dests already there. Returns {dest: error} for
    the ones that failed - the caller decides what a failure means (it fetches again one by one)."""
    todo = [(u, d) for u, d in dict((d, (u, d)) for u, d in pairs).values() if not os.path.exists(d)]
    errors = {}

    def one(pair):
        u, d = pair
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        try:
            storage.get(u, d)
        except Exception as e:  # noqa: BLE001 - reported, the caller's own fetch decides
            errors[d] = e
            try:                    # never a partial file the caller would take as fetched
                os.remove(d)
            except OSError:
                pass
    if todo:
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            list(ex.map(one, todo))
    return errors


def fetch_and_prepare(plan, storage, workdir, image_urls=None):
    if image_urls is None:
        image_urls = {}
    warnings = []
    images_dir = os.path.join(workdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    keep = prepared_scales(plan)
    # every photo and motion mask fetched at once first; the loop below finds them there (and still
    # fetches, and fails or warns, one by one for any that did not come)
    pairs = []
    for shot in plan.shots:
        if shot.image_uri is not None:
            pairs.append((image_urls.get(shot.image_uri, shot.image_uri), os.path.join(images_dir, shot.image_uri.split("/")[-1])))
        for region in shot.motion_regions or []:
            pairs.append((image_urls.get(region.mask_uri, region.mask_uri), os.path.join(images_dir, mask_relpath(shot.shot_id, region.id))))
    prefetch(storage, pairs)
    prepared = set()
    for shot in plan.shots:
        if shot.image_uri is None:              # a 3D shot (plan 1.6): rendered from the scene
            continue
        basename = shot.image_uri.split("/")[-1]
        dest_path = os.path.join(images_dir, basename)

        if not os.path.exists(dest_path):
            try:
                uri = image_urls.get(shot.image_uri, shot.image_uri)
                storage.get(uri, dest_path)
            except Exception as e:
                raise WorkerError("image_fetch_failed", f"failed to fetch shot {shot.shot_id} ({e})")

        # Prepare: once per photo, down to what its tightest shot needs (never below, so no frame
        # enlarges the photo more than the plan's own size would)
        if basename not in prepared:
            prepared.add(basename)
            try:
                with Image.open(dest_path) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size
                    pw, ph = shot.source.width, shot.source.height
                    if (w, h) != (pw, ph):
                        warnings.append(f"Image {basename} dimensions {w}x{h} do not match plan {pw}x{ph}. Clamping camera.")
                    new_w = min(w, max(1, round(pw * keep[basename])))
                    if new_w < w - 1:
                        new_h = max(1, round(h * new_w / w))
                        img.resize((new_w, new_h), Image.Resampling.LANCZOS).save(dest_path, "JPEG", quality=95)
            except Exception as e:
                raise RuntimeError(f"image preparation failed: {e}")

        # plan 1.4 motion region masks: a mask that cannot be fetched drops its region (the
        # renderer never fails on it; tour_worker.motion.loader.load_regions skips it)
        for region in shot.motion_regions or []:
            mask_dest = os.path.join(images_dir, mask_relpath(shot.shot_id, region.id))
            os.makedirs(os.path.dirname(mask_dest), exist_ok=True)
            if os.path.exists(mask_dest):
                continue
            try:
                storage.get(image_urls.get(region.mask_uri, region.mask_uri), mask_dest)
            except Exception as e:
                warnings.append(f"motion_mask_fetch_failed:{shot.shot_id}:{region.id} ({e})")

    # drone video clips (video_clip shots, plan 1.6+): fetch once; a missing clip fails the render
    for shot in plan.shots:
        cam = shot.camera
        if cam.type == "video_clip" and cam.clip_uri:
            dest = clip_dest(images_dir, cam.clip_uri)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                try:
                    uri = image_urls.get(cam.clip_uri, cam.clip_uri)
                    storage.get(uri, dest)
                except Exception as e:
                    raise WorkerError("image_fetch_failed", f"failed to fetch drone clip {shot.shot_id} ({e})")

    # branding images (the agent's photo, the brokerage logo): a failed fetch is a warning and the
    # card is drawn without that image (tour_worker/render/overlays.py)
    branding = getattr(plan.output, "branding", None)
    for uri in (branding.image_uris() if branding is not None else []):
        dest = branding_path(images_dir, uri)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            continue
        try:
            storage.get(image_urls.get(uri, uri), dest)
        except Exception as e:
            warnings.append(f"branding_image_fetch_failed:{type(e).__name__}")

    return warnings


PREPARE_HEADROOM = 1.05       # sampled at most ~1:1 at the tightest frame


def prepared_scales(plan) -> dict[str, float]:
    """Per photo (basename): the share of the plan's source size to keep - its shots' largest
    enlargement (render/camera_path.shot_enlargement) with a little headroom, at most 1. A photo
    much larger than the output is drawn from a copy just big enough for its tightest frame."""
    from tour_worker.render.camera_path import shot_enlargement
    out: dict[str, float] = {}
    for shot in plan.shots:
        if shot.image_uri is None or shot.source is None:
            continue
        e = shot_enlargement(shot.camera, shot.source.width, shot.source.height,
                             plan.output.width, plan.output.height)
        k = 1.0 if e is None else min(1.0, e * PREPARE_HEADROOM)
        name = shot.image_uri.split("/")[-1]
        out[name] = max(out.get(name, 0.0), k)
    return out


def branding_path(images_dir: str, uri: str) -> str:
    """Where a branding image is fetched to (named by its URI's hash: URIs may be long URLs)."""
    import hashlib
    return os.path.join(images_dir, "branding", hashlib.sha1(uri.encode()).hexdigest()[:16])


def clip_dest(images_dir: str, uri: str) -> str:
    """Where a drone video clip is fetched to (hash of URI + original extension, in clips/ subdir)."""
    import hashlib
    stem = hashlib.sha1(uri.encode()).hexdigest()[:16]
    basename = uri.rsplit("/", 1)[-1]
    ext = basename.rsplit(".", 1)[-1] if "." in basename else "mp4"
    return os.path.join(images_dir, "clips", f"{stem}.{ext}")
