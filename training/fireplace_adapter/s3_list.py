"""READ-ONLY: list the assets bucket (list_objects_v2 only) into out/fireplace_adapter/s3_index.jsonl.

Run with the BACKEND venv (it has the S3 client + settings):
  <backend>/.venv/bin/python training/fireplace_adapter/s3_list.py
Writes key, size, etag, last_modified per object. No URLs, no credentials are printed or stored.
"""
import json
import os
import sys

BACKEND = "/home/sarthak/Desktop/Articulait/articulait_backend"
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)                       # settings read the backend .env
from app.core.config import settings  # noqa: E402
from app.core.s3_client import S3Client  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "out", "fireplace_adapter", "s3_index.jsonl")


def main() -> None:
    cli = S3Client()._client
    n = 0
    with open(OUT, "w") as fh:
        for page in cli.get_paginator("list_objects_v2").paginate(Bucket=settings.S3_BUCKET_ASSETS):
            for o in page.get("Contents", []):
                fh.write(json.dumps({"key": o["Key"], "size": o["Size"], "etag": o["ETag"].strip('"'),
                                     "mtime": o["LastModified"].isoformat()}) + "\n")
                n += 1
    print("objects:", n)


if __name__ == "__main__":
    main()
