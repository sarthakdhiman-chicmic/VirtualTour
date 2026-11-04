import hashlib
import os

from tour_worker.errors import WorkerError
from tour_worker.motion.masks import mask_relpath

def photo_filename(plan, image_uri: str) -> str:
    """The file a shot's photo is kept in under <workdir>/images: its basename, or, when another
    photo of the plan has the same basename, the basename behind a hash of the URI."""
    base = image_uri.split("/")[-1]
    if all(s.image_uri == image_uri or s.image_uri.split("/")[-1] != base for s in plan.shots):
        return base
    return f"{hashlib.sha256(image_uri.encode()).hexdigest()[:12]}_{base}"


def fetch_and_prepare(plan, storage, workdir, image_urls=None):
    if image_urls is None:
        image_urls = {}
    warnings = []
    images_dir = os.path.join(workdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for shot in plan.shots:
        dest_path = os.path.join(images_dir, photo_filename(plan, shot.image_uri))
        
        if not os.path.exists(dest_path):
            try:
                uri = image_urls.get(shot.image_uri, shot.image_uri)
                storage.get(uri, dest_path)
            except Exception as e:
                raise WorkerError("image_fetch_failed", f"failed to fetch shot {shot.shot_id} ({e})") from e
        # the photo is used as fetched: ShotWarp scales it to the plan's source size (warning
        # image_resized) and refuses one with a different aspect (prewarp)

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

    return warnings
