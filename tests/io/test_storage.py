"""tour_worker.io.storage: CompositeStorage routes by URI scheme; a local store takes remote URIs
only when configured to (WORKER_STORAGE=local)."""
import os

import pytest

from tour_worker.config import Settings
from tour_worker.io.storage import CompositeStorage, LocalMapStorage, build_storage


class Rec:
    def __init__(self, name):
        self.name, self.calls = name, []

    def get(self, uri, dest):
        self.calls.append(("get", uri))

    def put(self, path, uri):
        self.calls.append(("put", uri))


def _cs(local=None, local_for_remote=False):
    s3, http, files = Rec("s3"), Rec("http"), Rec("files")
    return CompositeStorage(s3, http, local=local, local_for_remote=local_for_remote, files=files), s3, http, files


def test_routes_by_scheme_even_with_a_local_store():
    local = Rec("local")
    cs, s3, http, files = _cs(local=local)
    cs.get("s3://b/k.jpg", "/x")
    cs.get("https://h/k.jpg", "/x")
    cs.get("file:///tmp/a.mp3", "/x")
    cs.get("/tmp/a.mp3", "/x")
    cs.get("local://k.jpg", "/x")
    cs.put("/x", "s3://b/out/master.mp4")
    assert s3.calls == [("get", "s3://b/k.jpg"), ("put", "s3://b/out/master.mp4")]
    assert http.calls == [("get", "https://h/k.jpg")]
    assert files.calls == [("get", "file:///tmp/a.mp3"), ("get", "/tmp/a.mp3")]
    assert local.calls == [("get", "local://k.jpg")]


def test_local_for_remote_serves_remote_uris_locally_but_not_file_paths():
    local = Rec("local")
    cs, s3, http, files = _cs(local=local, local_for_remote=True)
    cs.get("s3://b/k.jpg", "/x")
    cs.get("https://h/k.jpg", "/x")
    cs.get("k.jpg", "/x")
    cs.put("/x", "s3://b/out/master.mp4")
    cs.get("file:///music/a.mp3", "/x")
    assert local.calls == [("get", "s3://b/k.jpg"), ("get", "https://h/k.jpg"), ("get", "k.jpg"),
                           ("put", "s3://b/out/master.mp4")]
    assert s3.calls == http.calls == [] and files.calls == [("get", "file:///music/a.mp3")]


def test_unsupported_schemes_are_rejected():
    cs, *_ = _cs()
    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        cs.get("http://insecure/k.jpg", "/x")
    with pytest.raises(ValueError, match="Unsupported put URI scheme"):
        cs.put("/x", "https://h/out.mp4")


def test_file_storage_and_local_map(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    cs = CompositeStorage(Rec("s3"), Rec("http"))
    cs.put(str(src), str(tmp_path / "out" / "b.bin"))
    assert (tmp_path / "out" / "b.bin").read_bytes() == b"abc"
    strict = LocalMapStorage(str(tmp_path), placeholder=False)
    with pytest.raises(FileNotFoundError):
        strict.get("s3://b/missing.jpg", str(tmp_path / "dest.jpg"))
    LocalMapStorage(str(tmp_path)).get("s3://b/missing.jpg", str(tmp_path / "dest.jpg"))   # red placeholder
    assert os.path.exists(tmp_path / "missing.jpg") and os.path.exists(tmp_path / "dest.jpg")


def test_build_storage_follows_the_settings():
    local = build_storage(Settings(storage="local", images_dir="/tmp/x"), "tour_render")
    assert local.local_for_remote and local.local.images_dir == "/tmp/x"
    assert build_storage(Settings(storage="local"), "tour_segment").local is None   # segment: always remote
    assert build_storage(Settings(), "tour_render").local is None


class FakeS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key, ExtraArgs))


@pytest.mark.parametrize("key, content_type", [
    ("tours/j/master.mp4", "video/mp4"),
    ("masks/s01/r1.png", "image/png"),
    ("tours/j/plan.json", "application/json"),
    ("tours/j/blob", None),
])
def test_s3_put_sets_the_content_type_from_the_key(monkeypatch, key, content_type):
    from tour_worker.io.storage import S3Storage
    client = FakeS3Client()
    s3 = S3Storage()
    monkeypatch.setattr(s3, "_get_client", lambda: client)
    s3.put("/tmp/x", f"s3://bucket/{key}")
    extra = client.uploads[0][2]
    assert (extra or {}).get("ContentType") == content_type
