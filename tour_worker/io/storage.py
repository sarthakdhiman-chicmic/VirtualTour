import os
import shutil
import urllib.request
from typing import Protocol

class Storage(Protocol):
    def get(self, uri: str, dest: str) -> None: ...
    def put(self, path: str, uri: str) -> None: ...

class HttpStorage(Storage):
    def get(self, uri: str, dest: str) -> None:
        if not uri.startswith("https://"):
            raise ValueError("Only https:// is supported")
            
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.url.startswith("http://"):
                    raise ValueError("Redirected to non-https")
                
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > 50 * 1024 * 1024:
                    raise ValueError("File too large")
                    
                with open(dest, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > 50 * 1024 * 1024:
                            raise ValueError("File exceeds 50MB limit")
        except Exception as e:
            clean_uri = uri.split('?')[0]
            clean_e_str = str(e).replace(uri, clean_uri)
            raise RuntimeError(f"HTTP fetch failed for {clean_uri}: {clean_e_str}")

    def put(self, path: str, uri: str) -> None:
        raise NotImplementedError("HttpStorage cannot put")

class S3Storage(Storage):
    def _get_client(self):
        import boto3
        return boto3.client('s3')

    def get(self, uri: str, dest: str) -> None:
        if not uri.startswith("s3://"):
            raise ValueError("Not an S3 URI")
        parts = uri[5:].split("/", 1)
        self._get_client().download_file(parts[0], parts[1], dest)

    def put(self, path: str, uri: str) -> None:
        if not uri.startswith("s3://"):
            raise ValueError("Not an S3 URI")
        parts = uri[5:].split("/", 1)
        self._get_client().upload_file(path, parts[0], parts[1])

class LocalMapStorage(Storage):
    """Dev stand-in for remote storage: every URI maps to <images_dir>/<basename>. A missing file
    is created as a red 1920x1080 placeholder when placeholder is True (the WORKER_STORAGE=local
    behaviour), else it is an error."""

    def __init__(self, images_dir: str, placeholder: bool = True):
        self.images_dir = images_dir
        self.placeholder = placeholder

    def get(self, uri: str, dest: str) -> None:
        basename = uri.split("/")[-1]
        src = os.path.join(self.images_dir, basename)
        if not os.path.exists(src):
            if not self.placeholder:
                raise FileNotFoundError(f"{basename} not in {self.images_dir}")
            from PIL import Image
            img = Image.new('RGB', (1920, 1080), color = 'red')
            img.save(src)
        shutil.copy(src, dest)

    def put(self, path: str, uri: str) -> None:
        basename = uri.split("/")[-1]
        dest = os.path.join(self.images_dir, basename)
        shutil.copy(path, dest)


class FileStorage(Storage):
    """file:// URIs and plain filesystem paths."""

    @staticmethod
    def path(uri: str) -> str:
        return uri[len("file://"):] if uri.startswith("file://") else uri

    def get(self, uri: str, dest: str) -> None:
        shutil.copy(self.path(uri), dest)

    def put(self, path: str, uri: str) -> None:
        dest = self.path(uri)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.copy(path, dest)


def is_file_uri(uri: str) -> bool:
    """file:// or an absolute filesystem path."""
    return uri.startswith("file://") or uri.startswith("/")


class CompositeStorage(Storage):
    """Routes each URI by its scheme: s3:// -> S3, https:// -> HTTPS (get only), file:// or a
    filesystem path -> the filesystem, local:// -> the local store.

    local_for_remote=True (WORKER_STORAGE=local, the local CLI): every URI that is not file:// or
    an absolute path - s3://, https://, bare names - is served by the local store (a
    LocalMapStorage: by basename), so a plan with production URIs renders from a folder.
    Before 2026-10 a local store took EVERY URI, file:// too (a file:// music track was looked
    up by basename in IMAGES_DIR)."""

    def __init__(self, s3: Storage, http: Storage, local: Storage | None = None,
                 local_for_remote: bool = False, files: Storage | None = None):
        self.s3 = s3
        self.http = http
        self.local = local
        self.local_for_remote = local_for_remote and local is not None
        self.files = files or FileStorage()

    def _route(self, uri: str, put: bool = False) -> Storage:
        if is_file_uri(uri):
            return self.files
        if self.local_for_remote or (uri.startswith("local://") and self.local is not None):
            return self.local
        if uri.startswith("s3://"):
            return self.s3
        if uri.startswith("https://") and not put:
            return self.http
        if "://" not in uri:                       # a relative path
            return self.files
        raise ValueError(f"Unsupported {'put ' if put else ''}URI scheme: {uri}")

    def get(self, uri: str, dest: str) -> None:
        self._route(uri).get(uri, dest)

    def put(self, path: str, uri: str) -> None:
        self._route(uri, put=True).put(path, uri)


def build_storage(settings, job_type: str = "tour_render") -> CompositeStorage:
    """The storage a job runs with. tour_segment always reads photos from S3 / HTTPS and writes
    masks to S3 (as it always did); tour_render and tour_plan honour WORKER_STORAGE=local."""
    if job_type in ("tour_render", "tour_plan") and settings.storage == "local":
        return CompositeStorage(s3=S3Storage(), http=HttpStorage(), local=LocalMapStorage(settings.images_dir),
                                local_for_remote=True)
    return CompositeStorage(s3=S3Storage(), http=HttpStorage())
