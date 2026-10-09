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

class CompositeStorage(Storage):
    def __init__(self, s3: Storage, http: Storage, local: Storage = None):
        self.s3 = s3
        self.http = http
        self.local = local

    def get(self, uri: str, dest: str) -> None:
        if self.local:
            self.local.get(uri, dest)
        elif uri.startswith("s3://"):
            self.s3.get(uri, dest)
        elif uri.startswith("https://"):
            self.http.get(uri, dest)
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")

    def put(self, path: str, uri: str) -> None:
        if self.local:
            self.local.put(path, uri)
        elif uri.startswith("s3://"):
            self.s3.put(path, uri)
        else:
            raise ValueError(f"Unsupported put URI scheme: {uri}")

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
    def __init__(self, images_dir: str):
        self.images_dir = images_dir

    def get(self, uri: str, dest: str) -> None:
        basename = uri.split("/")[-1]
        src = os.path.join(self.images_dir, basename)
        if not os.path.exists(src):
            from PIL import Image
            img = Image.new('RGB', (1920, 1080), color = 'red')
            img.save(src)
        shutil.copy(src, dest)

    def put(self, path: str, uri: str) -> None:
        basename = uri.split("/")[-1]
        dest = os.path.join(self.images_dir, basename)
        shutil.copy(path, dest)
