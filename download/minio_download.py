"""Download files from a MinIO bucket.

Credentials are read from environment variables (e.g. from a .env file, loaded
with `set -a; source .env; set +a` before running):
  MINIO_ENDPOINT     bare host, e.g. "play.min.io" or "s3.example.com" (no scheme, no path)
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_SECURE       optional, "true"/"false" (default "true")
  MINIO_BUCKET        optional default for --bucket

Usage:
  python minio_download.py --object path/to/file.bin --dest ./data/file.bin
  python minio_download.py --prefix models/af_ --dest ./models/
"""

import argparse
import os
import sys
import time
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import ProtocolError

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5


def get_client() -> Minio:
    endpoint = os.environ.get("MINIO_ENDPOINT")
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")
    secure = os.environ.get("MINIO_SECURE", "true").lower() == "true"

    missing = [
        name
        for name, val in (
            ("MINIO_ENDPOINT", endpoint),
            ("MINIO_ACCESS_KEY", access_key),
            ("MINIO_SECRET_KEY", secret_key),
        )
        if not val
    ]
    if missing:
        sys.exit(f"Missing required environment variable(s): {', '.join(missing)}")

    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def download_object(client: Minio, bucket: str, object_name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == client.stat_object(bucket, object_name).size:
        print(f"{dest} already downloaded, skipping.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.fget_object(bucket, object_name, str(dest))
            break
        except (ProtocolError, ConnectionError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  connection dropped ({e}), retrying in {wait}s ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    print(f"Downloaded {bucket}/{object_name} -> {dest}")


def download_prefix(client: Minio, bucket: str, prefix: str, dest_dir: Path) -> None:
    """
    Downloads every object matching a prefix, writing each by its basename
    into dest_dir (flat, not mirroring the bucket's key structure). A prefix
    is a name-filter here, not necessarily a real directory boundary -- e.g.
    "models/af_" matches "models/af_50_1_cbow_wxd.csv.bz2" -- so os.path.relpath
    against the prefix isn't safe: when the prefix ends mid-filename it can
    treat part of the filename as a directory segment and produce a "../"
    escaped path pointing outside dest_dir.
    """
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        sys.exit(f"No objects found under {bucket}/{prefix}")

    failed = []
    for obj in objects:
        if obj.is_dir:
            continue
        dest = dest_dir / os.path.basename(obj.object_name)
        try:
            download_object(client, bucket, obj.object_name, dest)
        except (S3Error, ProtocolError, ConnectionError) as e:
            print(f"  giving up on {obj.object_name} after {MAX_RETRIES} attempts: {e}")
            failed.append(obj.object_name)

    if failed:
        sys.exit(f"{len(failed)} object(s) failed to download: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default=os.environ.get("MINIO_BUCKET"), help="MinIO bucket name (defaults to MINIO_BUCKET env var)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object", help="Single object key to download")
    group.add_argument("--prefix", help="Download all objects under this prefix")
    parser.add_argument("--dest", required=True, help="Local file path (--object) or directory (--prefix)")
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("Missing --bucket (and MINIO_BUCKET is not set)")

    client = get_client()

    try:
        if args.object:
            download_object(client, args.bucket, args.object, Path(args.dest))
        else:
            download_prefix(client, args.bucket, args.prefix, Path(args.dest))
    except (S3Error, ProtocolError, ConnectionError) as e:
        sys.exit(f"MinIO error: {e}")


if __name__ == "__main__":
    main()
