"""Uploads trained model files to Zenodo, mirroring zenodo_download.py's
conventions so both scripts agree on file/record layout.

Two upload paths, chosen automatically per language+version:
- Already has a published DOI for this language+version (see
  zenodo_dois.csv)? Uploads as a new Zenodo *version* of that same record
  (same concept DOI lineage) -- the right move for retrained/corrected 2018
  languages, since it's a correction of the same dataset, not a new one.
- No prior DOI? Creates a brand-new record -- for 2024-corpus uploads
  (new languages, or a 2024 supplement to an existing one) and for any
  language's first-ever upload. Its metadata is the language-specific
  title/description plus boilerplate (creators, license, communities,
  related_identifiers, version) copied from REFERENCE_RECORD_ID, an
  existing published record -- so new records carry the same
  authorship/license/community/manuscript-link info as the project's
  existing ones instead of just a bare title. (A new-*version* upload
  needs none of this -- Zenodo's newversion draft already inherits the
  prior version's full metadata automatically.)

Handles what zenodo_download.py's docstring describes from the other side:
oversized model files get split into "{base}_part_aa"/"_part_ab"/... chunks
(matching zenodo_common.CHUNK_PATTERN) before upload, and files get grouped
into separate Zenodo records ("parts") to stay under Zenodo's per-record
limits (100 files, 50GB, per https://developers.zenodo.org/).

In practice, uploads have been unreliable above roughly a few GB per
request (cause unclear -- could be a server-side timeout, a dropped
connection, or both). Zenodo's API has no resumable/chunked upload support
(confirmed against their docs: it's a single all-or-nothing PUT per file),
so this chunks aggressively below that observed trouble threshold, retries
each request with backoff, and verifies each uploaded file's MD5 checksum
against the local file before trusting it actually landed intact.

Requires a Zenodo personal access token (scopes: deposit:write,
deposit:actions), from https://zenodo.org/account/settings/applications/tokens/new/,
in the environment as ZENODO_TOKEN -- add it to .env at the repo root:
    ZENODO_TOKEN=...
then `set -a; source .env; set +a` before running this.

Usage:
    python zenodo_upload.py --language af --version 2018 --models-dir ../models/
    python zenodo_upload.py --language az --version 2024 --models-dir ../models/
    python zenodo_upload.py --language af --version 2018 --models-dir ../models/ --dry-run
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from collections import namedtuple
from pathlib import Path

import requests

import zenodo_common as zc

HERE = os.path.dirname(os.path.abspath(__file__))
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")
REPO_URL = "https://github.com/SemanticPriming/word2manylanguages"

API = "https://zenodo.org/api"

# Stay well under Zenodo's documented limits (100 files/record, 50GB/record)
# -- and well under the rough few-GB-per-request point where uploads have
# been unreliable in practice, not a precisely diagnosed threshold.
CHUNK_BYTES = int(1.5 * 1000**3)          # 1.5GB (decimal) max per uploaded file/chunk
MAX_FILES_PER_RECORD = 90                 # Zenodo caps at 100; leave headroom
MAX_BYTES_PER_RECORD = int(45 * 1000**3)  # Zenodo caps at 50GB (decimal); leave headroom

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5  # doubles each retry: 5, 10, 20, 40, 80s

# A known-good already-published record (af, 2018, part 1) whose boilerplate
# metadata -- creators, license, communities, related_identifiers, version
# -- is reused verbatim for every brand-new record this script creates, so
# newly published languages carry the same authorship/license/community/
# manuscript-link info as the project's existing records instead of just a
# bare title/description. Override with the ZENODO_REFERENCE_RECORD env var
# if that record ever gets superseded.
REFERENCE_RECORD_ID = "17328169"


def _token():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN not set -- add it to .env and `set -a; source .env; set +a` first.")
    return token


def _headers(json=False):
    h = {"Authorization": f"Bearer {_token()}"}
    if json:
        h["Content-Type"] = "application/json"
    return h


_reference_metadata_cache = None


def _reference_metadata():
    """
    Fetches the boilerplate deposit metadata (creators, license,
    communities, related_identifiers, version) from REFERENCE_RECORD_ID and
    translates it from the public /records/ display shape (what GET
    returns) back into the shape /deposit/depositions/ needs on create --
    e.g. license as a bare id string rather than {"id": ...}, communities
    keyed by "identifier" rather than "id". Cached for the life of the
    process: this is called once per brand-new record, and the reference
    record's metadata doesn't change mid-run.
    """
    global _reference_metadata_cache
    if _reference_metadata_cache is not None:
        return _reference_metadata_cache

    record_id = os.environ.get("ZENODO_REFERENCE_RECORD", REFERENCE_RECORD_ID)
    r = _retry(lambda: requests.get(f"{API}/records/{record_id}"), "fetch reference record metadata")
    m = r.json()["metadata"]

    boilerplate = {}
    if m.get("creators"):
        boilerplate["creators"] = [
            {k: v for k, v in c.items() if k in ("name", "affiliation", "orcid") and v}
            for c in m["creators"]
        ]
    if m.get("license", {}).get("id"):
        boilerplate["license"] = m["license"]["id"]
    if m.get("communities"):
        boilerplate["communities"] = [{"identifier": c["id"]} for c in m["communities"]]
    if m.get("related_identifiers"):
        boilerplate["related_identifiers"] = m["related_identifiers"]
    if m.get("version"):
        boilerplate["version"] = m["version"]

    _reference_metadata_cache = boilerplate
    return boilerplate


def _retry(fn, description):
    """
    Calls fn() (a zero-arg callable making one HTTP request), retrying on
    any exception or non-2xx response with exponential backoff. Re-raises
    the last error if every attempt fails -- callers should treat that as
    "did not land," never silently swallow it.
    """
    delay = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = fn()
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  {description} failed (attempt {attempt}/{MAX_RETRIES}): {e} -- retrying in {delay}s")
            time.sleep(delay)
            delay *= 2


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _split_suffix(i):
    """0->'aa', 1->'ab', ..., 25->'az', 26->'ba', ... (classic `split -a 2` order)."""
    first, second = divmod(i, 26)
    if first >= 26:
        raise ValueError("too many chunks for a 2-letter suffix -- shouldn't happen at these file sizes")
    return chr(ord("a") + first) + chr(ord("a") + second)


ChunkSpec = namedtuple("ChunkSpec", ["name", "size", "source_path", "offset"])


def plan_chunks(paths):
    """
    Pure, fast, no-I/O: computes how each path would be split into
    uploadable pieces purely from file sizes on disk (a stat() call per
    file, not a read -- no chunk files are actually written here). Whole
    files under CHUNK_BYTES get one ChunkSpec with offset=None; oversized
    files get one ChunkSpec per CHUNK_BYTES-sized slice, named
    "{stem}_part_aa"/"_part_ab"/... to match zenodo_common.CHUNK_PATTERN.
    Use materialize_chunk() to actually write a spec's bytes to disk, only
    needed right before a real upload.
    """
    specs = []
    for path in paths:
        size = path.stat().st_size
        if size <= CHUNK_BYTES:
            specs.append(ChunkSpec(path.name, size, path, None))
            continue

        stem = path.name[: -len(".csv.bz2")]
        n_chunks = -(-size // CHUNK_BYTES)
        for i in range(n_chunks):
            offset = i * CHUNK_BYTES
            chunk_size = min(CHUNK_BYTES, size - offset)
            specs.append(ChunkSpec(f"{stem}_part_{_split_suffix(i)}", chunk_size, path, offset))
    return specs


def materialize_chunk(spec, chunk_dir):
    """
    Writes one ChunkSpec's actual bytes to chunk_dir and returns the real
    Path to upload -- only copies for oversized-file slices (offset is not
    None); a whole file is returned as-is, no copy needed. Call this one
    spec at a time right before uploading it, not for a whole batch up
    front, so at most one chunk's worth of temp data ever sits on disk
    even for the largest languages.
    """
    if spec.offset is None:
        return spec.source_path
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / spec.name
    with open(spec.source_path, "rb") as f:
        f.seek(spec.offset)
        block = f.read(spec.size)
    with open(chunk_path, "wb") as out:
        out.write(block)
    return chunk_path


def batch_for_records(specs):
    """
    Groups ChunkSpecs into batches that each fit under MAX_FILES_PER_RECORD
    and MAX_BYTES_PER_RECORD -- one batch per Zenodo record needed. Pure,
    no I/O.
    """
    batches = []
    current, current_bytes = [], 0
    for spec in specs:
        if current and (len(current) >= MAX_FILES_PER_RECORD or current_bytes + spec.size > MAX_BYTES_PER_RECORD):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(spec)
        current_bytes += spec.size
    if current:
        batches.append(current)
    return batches


def _create_record(metadata):
    r = _retry(
        lambda: requests.post(f"{API}/deposit/depositions", json={"metadata": metadata}, headers=_headers(json=True)),
        "create deposit",
    )
    return r.json()


def _new_version(existing_record_id):
    r = _retry(
        lambda: requests.post(f"{API}/deposit/depositions/{existing_record_id}/actions/newversion", headers=_headers()),
        "create new version",
    )
    latest_draft_url = r.json()["links"]["latest_draft"]
    r2 = _retry(lambda: requests.get(latest_draft_url, headers=_headers()), "fetch new draft")
    return r2.json()


def _clear_files(deposit):
    """New-version drafts inherit the old published files by default (Zenodo's
    own documented behavior) -- must delete before uploading replacements."""
    r = _retry(lambda: requests.get(f"{API}/deposit/depositions/{deposit['id']}/files", headers=_headers()), "list draft files")
    for f in r.json():
        _retry(
            lambda f=f: requests.delete(f"{API}/deposit/depositions/{deposit['id']}/files/{f['id']}", headers=_headers()),
            f"delete old file {f['key']}",
        )


def _upload_spec(bucket_url, spec, chunk_dir):
    """
    Materializes a ChunkSpec to disk only if needed (a whole file needs no
    copy), uploads it, verifies its checksum, then immediately deletes the
    materialized chunk (not the original source file) -- so at most one
    chunk's worth of temp data exists on disk at a time, regardless of how
    large the language's full model set is.
    """
    path = materialize_chunk(spec, chunk_dir)
    local_md5 = _md5(path)

    def do_put():
        with open(path, "rb") as fp:
            return requests.put(f"{bucket_url}/{spec.name}", data=fp, headers=_headers())

    try:
        r = _retry(do_put, f"upload {spec.name}")
        remote_md5 = r.json().get("checksum", "")
        if remote_md5.startswith("md5:"):
            remote_md5 = remote_md5[len("md5:"):]
        if remote_md5 != local_md5:
            raise RuntimeError(f"checksum mismatch for {spec.name}: local={local_md5} remote={remote_md5} -- upload landed corrupted")
        print(f"  uploaded + verified {spec.name} ({spec.size/1e6:.0f}MB)")
    finally:
        if spec.offset is not None:  # only clean up materialized chunks, never the original source file
            path.unlink(missing_ok=True)


def _publish(deposit_id):
    r = _retry(lambda: requests.post(f"{API}/deposit/depositions/{deposit_id}/actions/publish", headers=_headers()), "publish")
    return r.json()


def _append_dois_csv(rows):
    file_exists = os.path.exists(DOIS_CSV)
    with open(DOIS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "version", "part", "file", "doi"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} row(s) to {DOIS_CSV}")


def upload_batch_as_new_record(language, version, part, batch, chunk_dir, metadata):
    deposit = _create_record(metadata)
    bucket_url = deposit["links"]["bucket"]
    for spec in batch:
        _upload_spec(bucket_url, spec, chunk_dir)
    published = _publish(deposit["id"])
    doi = published["doi"]
    print(f"Published {language} {version} part {part}: {doi}")
    return doi


def upload_batch_as_new_version(language, version, part, batch, chunk_dir, existing_record_id):
    deposit = _new_version(existing_record_id)
    _clear_files(deposit)
    bucket_url = deposit["links"]["bucket"]
    for spec in batch:
        _upload_spec(bucket_url, spec, chunk_dir)
    published = _publish(deposit["id"])
    doi = published["doi"]
    print(f"Published new version of {language} {version} part {part}: {doi}")
    return doi


def logical_rows_for_batch(language, version, part, batch, doi):
    """Maps a batch's ChunkSpecs (whole files or chunks) back to the
    logical model filenames they represent, for zenodo_dois.csv -- which
    always stores logical names; zenodo_common.group_logical_files() does
    the reverse mapping on the download side."""
    logical_names = set()
    for spec in batch:
        m = zc.CHUNK_PATTERN.match(spec.name)
        logical_names.add(f"{m.group('base')}.csv.bz2" if m else spec.name)
    return [
        {"language": language, "version": version, "part": part, "file": name, "doi": doi}
        for name in sorted(logical_names)
    ]


def existing_records_for(language, version):
    """{part: record_id} for whatever's already published under this
    language+version, so retraining can target the right existing DOIs."""
    record_id_by_part = {}
    if not os.path.exists(DOIS_CSV):
        return record_id_by_part
    with open(DOIS_CSV) as f:
        for row in csv.DictReader(f):
            if row["language"] == language and row.get("version", "2018") == version:
                record_id_by_part[int(row["part"])] = row["doi"].rsplit(".", 1)[-1]
    return record_id_by_part


def sync_language(language, version, models_dir, dry_run=False):
    """
    Uploads every {language}_*_wxd.csv.bz2 in models_dir to Zenodo, split
    and batched into as many records as needed. If this language+version
    already has published records (existing_records_for), reuses those
    DOIs via the new-version flow, batch-for-batch in part order -- the
    "publish a correction" path for retrained 2018 languages. Any batches
    beyond however many parts already existed become brand-new records
    (logged clearly, since that changes the language's part count going
    forward). Languages/versions with no prior DOI at all (new 2024
    languages, or a first-ever 2024 supplement) always create new records.
    """
    models_dir = Path(models_dir)
    paths = sorted(models_dir.glob(f"{language}_*_wxd.csv.bz2"))
    if not paths:
        sys.exit(f"No model files found matching {language}_*_wxd.csv.bz2 in {models_dir}")
    print(f"{language} ({version}): {len(paths)} model files, {sum(p.stat().st_size for p in paths)/1e9:.1f}GB total")

    # plan_chunks/batch_for_records are pure stat()-based planning, no I/O --
    # safe and fast to run even for a dry-run preview of a huge language.
    specs = plan_chunks(paths)
    batches = batch_for_records(specs)
    print(f"  -> {len(batches)} record(s) needed ({sum(len(b) for b in batches)} chunk(s)/file(s))")

    if dry_run:
        for i, batch in enumerate(batches, start=1):
            total = sum(s.size for s in batch)
            print(f"  [dry run] part {i}: {len(batch)} files, {total/1e9:.2f}GB")
        return

    chunk_dir = models_dir / f".zenodo_upload_chunks_{language}_{version}"
    existing = existing_records_for(language, version)

    # Only fetched (one network call) if at least one batch actually needs a
    # brand-new record -- a language whose parts are all already published
    # never touches this, same as before.
    needs_new_record = any(i not in existing for i in range(1, len(batches) + 1))
    reference_metadata = _reference_metadata() if needs_new_record else {}

    new_rows = []
    for i, batch in enumerate(batches, start=1):
        part = i
        if part in existing:
            doi = upload_batch_as_new_version(language, version, part, batch, chunk_dir, existing[part])
        else:
            title = f"word2manylanguages: {language} FastText embeddings ({version} corpus)"
            if len(batches) > 1:
                title += f" Part {part}"
            metadata = {
                "upload_type": "dataset",
                "title": title,
                "description": (
                    f"FastText word embeddings for '{language}', trained on the {version} "
                    f"OpenSubtitles + Wikipedia corpus. See {REPO_URL} for the full pipeline."
                ),
                "access_right": "open",
                **reference_metadata,
                # Marks corpus vintage as a queryable field, not just prose in the
                # title/description above -- 2018 and 2024 records are otherwise
                # identical (same creators/license/communities/related_identifiers).
                "keywords": [*reference_metadata.get("keywords", []), f"opensubtitles-{version}"],
            }
            doi = upload_batch_as_new_record(language, version, part, batch, chunk_dir, metadata)
        new_rows.extend(logical_rows_for_batch(language, version, part, batch, doi))

    extra_old_parts = set(existing) - set(range(1, len(batches) + 1))
    if extra_old_parts:
        print(
            f"WARNING: {language} ({version}) previously had part(s) {sorted(extra_old_parts)} that "
            f"aren't used by this upload -- those old records are now stale (not deleted; Zenodo "
            f"doesn't support deleting published records). Decide by hand whether to leave them or "
            f"note the change in their description."
        )

    _append_dois_csv(new_rows)

    if chunk_dir.exists():
        chunk_dir.rmdir()  # _upload_spec cleans up each chunk right after its upload; this just removes the now-empty dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, help="two-letter language code, e.g. 'af'")
    parser.add_argument("--version", default="2018", help="corpus vintage: '2018' (default) or '2024'")
    parser.add_argument("--models-dir", required=True, help="local directory containing {language}_*_wxd.csv.bz2 files")
    parser.add_argument("--dry-run", action="store_true", help="only print the chunking/batching plan, no network calls")
    args = parser.parse_args()

    sync_language(args.language, args.version, args.models_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
