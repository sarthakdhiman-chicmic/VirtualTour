import os
from PIL import Image, ImageOps

from tour_worker.errors import WorkerError
from tour_worker.motion.masks import mask_relpath

def fetch_and_prepare(plan, storage, workdir, image_urls=None):
    if image_urls is None:
        image_urls = {}
    warnings = []
    images_dir = os.path.join(workdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
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
                
        # Prepare
        try:
            with Image.open(dest_path) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                
                # Check dimensions against plan
                if w != shot.source.width or h != shot.source.height:
                    warnings.append(f"Image {basename} dimensions {w}x{h} do not match plan {shot.source.width}x{shot.source.height}. Clamping camera.")
                
                # the output's longer side: a portrait frame needs the photo's full height
                target_max = int(max(plan.output.width, plan.output.height) * 1.2)
                if max(w, h) > target_max:
                    scale = target_max / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Overwrite
                    img.save(dest_path, "JPEG", quality=95)
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


def branding_path(images_dir: str, uri: str) -> str:
    """Where a branding image is fetched to (named by its URI's hash: URIs may be long URLs)."""
    import hashlib
    return os.path.join(images_dir, "branding", hashlib.sha1(uri.encode()).hexdigest()[:16])
