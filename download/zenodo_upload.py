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

Frequency counts (see eval_inputs/build_counts_tokenized.py) are a separate,
much smaller dataset -- every language's counts are bundled into one shared
Zenodo record via sync_all_counts(), run once manually after however many
languages you want are ready, not per language:
    python zenodo_upload.py --sync-counts --counts-dir ../eval_inputs/counts/
    python zenodo_upload.py --sync-counts --counts-dir ../eval_inputs/counts/ --dry-run
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import namedtuple
from pathlib import Path

import requests

import zenodo_common as zc

HERE = os.path.dirname(os.path.abspath(__file__))
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")
COUNTS_DOIS_CSV = os.path.join(HERE, "zenodo_counts_dois.csv")
REPO_URL = "https://github.com/SemanticPriming/word2manylanguages"

# Uploaded alongside any record that ends up containing split "_part_aa"/...
# chunks, so downloaders who grab the record straight from Zenodo's web UI
# (not via zenodo_download.py, which reassembles chunks itself) still have
# instructions + a tool to recombine them.
CHUNKER_ASSETS_DIR = Path(HERE) / "zenodo"
CHUNKER_ASSET_NAMES = ("README.md", "FileChunker.ps1")


LOG_PATH = os.path.join(HERE, "zenodo_upload.log")


def _log(msg):
    """Timestamped, always-flushed progress print -- large model files can
    take minutes per upload, so every progress line is stamped with a
    wall-clock time (not just relative order) to make it obvious whether
    the process is still moving or has stalled. Also appended to LOG_PATH
    (opened fresh each call, not held open) so a notebook kernel reset or
    crash doesn't lose the run's history -- needed to diagnose recurring
    502s/stalls across multiple resumed attempts, not just the current
    notebook output."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

API = "https://zenodo.org/api"

# Stay well under Zenodo's documented limits (100 files/record, 50GB/record)
# -- and well under the rough few-GB-per-request point where uploads have
# been unreliable in practice, not a precisely diagnosed threshold.
CHUNK_BYTES = int(0.5 * 1000**3)          # 500MB (decimal) max per uploaded file/chunk -- lowered from
                                           # 1.5GB after observing 502s/stalls around the ~1GB mark on a
                                           # slow/unstable uplink (~20Mbps, high latency under sustained load)
MAX_FILES_PER_RECORD = 100                # Zenodo's hard cap; no headroom left
MAX_BYTES_PER_RECORD = int(45 * 1000**3)  # Zenodo caps at 50GB (decimal); leave headroom

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5  # doubles each retry: 5, 10, 20, 40, 80s

# (connect timeout, read timeout) applied to every request. Without this,
# requests has NO default timeout -- a stalled connection (server or
# network just stops responding mid-transfer, sends no more bytes either
# way) hangs indefinitely instead of raising, which is why a single upload
# attempt was observed sitting silent for 48 minutes before finally
# surfacing as a 502. 120s read timeout means a genuine stall gets caught
# and retried within ~2 minutes instead of the better part of an hour;
# it's a per-socket-read/write timeout, not a cap on total upload
# duration, so a slow-but-still-moving transfer isn't affected by it.
REQUEST_TIMEOUT = (15, 120)

# How often (seconds) an in-progress upload logs bytes-sent-so-far -- lets
# a running upload's progress be read off stdout/LOG_PATH instead of just
# seeing "uploading..." and then silence until it finishes or fails.
UPLOAD_PROGRESS_INTERVAL_SECONDS = 20

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
    r = _retry(lambda: requests.get(f"{API}/records/{record_id}", timeout=REQUEST_TIMEOUT), "fetch reference record metadata")
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


def _describe_error(e):
    """Appends the HTTP response body (truncated) to an exception's message
    when available -- requests' default str(e) for an HTTPError gives only
    the status line (e.g. "502 Server Error: Bad Gateway"), which isn't
    enough to tell a real Zenodo-side 502 apart from, say, a proxy/CDN
    timeout page or a connection reset -- useful when diagnosing a pattern
    of failures after the fact from LOG_PATH."""
    resp = getattr(e, "response", None)
    if resp is None:
        return str(e)
    try:
        body = resp.text[:300]
    except Exception:
        body = "<unreadable body>"
    return f"{e} | body: {body!r}"


def _retry(fn, description):
    """
    Calls fn() (a zero-arg callable making one HTTP request), retrying on
    any exception or non-2xx response with exponential backoff. Re-raises
    the last error if every attempt fails -- callers should treat that as
    "did not land," never silently swallow it. Every attempt, including the
    final failure, is logged (via _log, so it's in LOG_PATH too) -- so a
    run that dies can still be diagnosed afterward instead of just showing
    a bare traceback in the notebook.
    """
    delay = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = fn()
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == MAX_RETRIES:
                _log(f"  {description} failed (attempt {attempt}/{MAX_RETRIES}): {_describe_error(e)} -- giving up")
                raise
            _log(f"  {description} failed (attempt {attempt}/{MAX_RETRIES}): {_describe_error(e)} -- retrying in {delay}s")
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

    An oversized source file's chunks (consecutive specs sharing the same
    source_path, from plan_chunks) are kept together in one batch rather
    than allowed to fall across a batch boundary -- a model file shouldn't
    end up split across two different Zenodo records.
    """
    groups = []
    for spec in specs:
        if groups and groups[-1][0].source_path == spec.source_path:
            groups[-1].append(spec)
        else:
            groups.append([spec])

    batches = []
    current, current_bytes = [], 0
    for group in groups:
        group_bytes = sum(s.size for s in group)
        if current and (len(current) + len(group) > MAX_FILES_PER_RECORD or current_bytes + group_bytes > MAX_BYTES_PER_RECORD):
            batches.append(current)
            current, current_bytes = [], 0
        current.extend(group)
        current_bytes += group_bytes
    if current:
        batches.append(current)
    return batches


def _batch_has_chunks(batch):
    return any(spec.offset is not None for spec in batch)


def _chunker_asset_specs():
    """Whole-file ChunkSpecs for the recombine README + FileChunker.ps1 --
    same shape as a plan_chunks() output so they can go through
    _upload_spec unchanged."""
    return [
        ChunkSpec(name, (CHUNKER_ASSETS_DIR / name).stat().st_size, CHUNKER_ASSETS_DIR / name, None)
        for name in CHUNKER_ASSET_NAMES
    ]


def _create_record(metadata):
    r = _retry(
        lambda: requests.post(f"{API}/deposit/depositions", json={"metadata": metadata}, headers=_headers(json=True), timeout=REQUEST_TIMEOUT),
        "create deposit",
    )
    return r.json()


def _new_version(existing_record_id):
    r = _retry(
        lambda: requests.post(f"{API}/deposit/depositions/{existing_record_id}/actions/newversion", headers=_headers(), timeout=REQUEST_TIMEOUT),
        "create new version",
    )
    latest_draft_url = r.json()["links"]["latest_draft"]
    r2 = _retry(lambda: requests.get(latest_draft_url, headers=_headers(), timeout=REQUEST_TIMEOUT), "fetch new draft")
    return r2.json()


def _list_draft_depositions():
    r = _retry(
        lambda: requests.get(f"{API}/deposit/depositions", headers=_headers(), params={"size": 100, "status": "draft"}, timeout=REQUEST_TIMEOUT),
        "list draft depositions",
    )
    return r.json()


def _fetch_deposit(deposit_id):
    """The list endpoint (_list_draft_depositions) returns abbreviated
    records missing several fields the rest of this module relies on --
    notably links.bucket -- present only when fetching a single deposit by
    id. Always re-fetch a match from the list before using it as a
    deposit."""
    r = _retry(lambda: requests.get(f"{API}/deposit/depositions/{deposit_id}", headers=_headers(), timeout=REQUEST_TIMEOUT), "fetch deposit")
    return r.json()


def _find_pending_version_draft(conceptrecid):
    """
    An unpublished newversion draft that already exists for this concept --
    from a prior attempt that got interrupted (timeout, Ctrl+C, crash)
    before publishing. Large files fail partway through often enough that
    just discarding and starting over every time wastes real upload time
    (tens of GB re-sent for nothing); finding this lets the caller resume
    into the same draft and skip whatever already landed instead.
    """
    for d in _list_draft_depositions():
        # Zenodo's status=draft filter on the list endpoint isn't reliable --
        # it can include a concept's already-published record (state='done',
        # submitted=True) alongside genuine unpublished drafts. A published
        # record's bucket is read-only, so treating it as resumable causes a
        # 403 on every upload into it -- state must be checked explicitly
        # rather than trusting the query param alone.
        if d.get("conceptrecid") == conceptrecid and d.get("state") == "unsubmitted":
            return _fetch_deposit(d["id"])
    return None


def _find_pending_new_record_draft(title):
    """Same idea as _find_pending_version_draft, but for a brand-new record
    that has no prior published version (and so no conceptrecid) to key on
    yet -- title is the only stable anchor available before the first
    publish. conceptdoi only appears once a concept has been published at
    least once, so its absence confirms this is a first-version draft."""
    for d in _list_draft_depositions():
        # Same defensive state check as _find_pending_version_draft -- see
        # its comment.
        if d.get("state") == "unsubmitted" and not d.get("conceptdoi") and d["metadata"].get("title") == title:
            return _fetch_deposit(d["id"])
    return None


def _remove_stale_draft_files(deposit, keep_names):
    """
    Deletes any file currently in the draft whose name isn't part of the
    batch being uploaded -- covers both a new-version draft's inherited old
    published files (Zenodo copies these into every fresh draft) and any
    leftover chunk from a previous attempt that used different chunk
    boundaries. Returns {filename: md5} for everything that's left (i.e.
    already legitimately present), so the caller can skip re-uploading
    whatever already matches.
    """
    r = _retry(lambda: requests.get(f"{API}/deposit/depositions/{deposit['id']}/files", headers=_headers(), timeout=REQUEST_TIMEOUT), "list draft files")
    files = r.json()
    kept = {}
    for f in files:
        if f["filename"] in keep_names:
            checksum = f.get("checksum", "")
            kept[f["filename"]] = checksum[len("md5:"):] if checksum.startswith("md5:") else checksum
        else:
            _retry(
                lambda f=f: requests.delete(f"{API}/deposit/depositions/{deposit['id']}/files/{f['id']}", headers=_headers(), timeout=REQUEST_TIMEOUT),
                f"delete stale file {f['filename']}",
            )
            _log(f"  deleted stale file {f['filename']}")
    return kept


class _ProgressReader:
    """
    Wraps an open file object so requests streams it through .read() as
    usual, but every call to read() also checks whether
    UPLOAD_PROGRESS_INTERVAL_SECONDS has elapsed since the last progress
    log and, if so, logs bytes-sent-so-far and current speed. This is the
    only way to see movement *during* a single PUT -- previously a chunk
    logged once before uploading and once after, so a stalled request
    (observed once hanging 48 minutes before finally 502ing) produced no
    output at all in between, making it impossible to tell whether it was
    still moving or already dead. Retried automatically on failure (a new
    _ProgressReader/file handle is created per attempt by the caller), so
    this only ever reports one attempt's progress, not a cumulative total
    across retries.
    """

    def __init__(self, fp, name, total_size):
        self.fp = fp
        self.name = name
        self.total_size = total_size
        self.bytes_read = 0
        self.start = time.monotonic()
        self.last_log = self.start

    def read(self, size=-1):
        chunk = self.fp.read(size)
        self.bytes_read += len(chunk)
        now = time.monotonic()
        if now - self.last_log >= UPLOAD_PROGRESS_INTERVAL_SECONDS and self.bytes_read < self.total_size:
            elapsed = now - self.start
            speed = (self.bytes_read / 1e6) / elapsed if elapsed > 0 else 0
            pct = 100 * self.bytes_read / self.total_size if self.total_size else 100
            _log(f"    ...{self.name}: {self.bytes_read/1e6:.0f}/{self.total_size/1e6:.0f}MB ({pct:.0f}%) at {speed:.1f}MB/s")
            self.last_log = now
        return chunk

    def __len__(self):
        # Read once by requests' super_len() before any reads happen, to
        # set the Content-Length header -- must be the file's full size,
        # not remaining bytes.
        return self.total_size

    def __getattr__(self, attr):
        return getattr(self.fp, attr)


def _upload_spec(bucket_url, spec, chunk_dir, already_uploaded=None, progress=None):
    """
    Materializes a ChunkSpec to disk only if needed (a whole file needs no
    copy), uploads it, verifies its checksum, then immediately deletes the
    materialized chunk (not the original source file) -- so at most one
    chunk's worth of temp data exists on disk at a time, regardless of how
    large the language's full model set is.

    `already_uploaded` ({filename: md5}, from _remove_stale_draft_files) is
    checked before uploading -- if this exact file already landed correctly
    in a prior attempt at this same draft, skip re-sending it entirely.

    `progress`, if given, is a (index, total, bytes_done, bytes_total)
    tuple -- purely for the log lines below (how far into the current
    record's batch this chunk is, by count and by bytes), so a run's speed
    and remaining work can be read off LOG_PATH/stdout without guessing
    from file names alone.
    """
    tag = f" [{progress[0]}/{progress[1]}, {progress[2]/1e9:.2f}/{progress[3]/1e9:.2f}GB]" if progress else ""
    path = materialize_chunk(spec, chunk_dir)
    try:
        local_md5 = _md5(path)
        if (already_uploaded or {}).get(spec.name) == local_md5:
            _log(f"  already uploaded + verified {spec.name} ({spec.size/1e6:.0f}MB){tag} -- skipping")
            return

        _log(f"  uploading {spec.name} ({spec.size/1e6:.0f}MB){tag}...")

        def do_put():
            with open(path, "rb") as fp:
                reader = _ProgressReader(fp, spec.name, spec.size)
                return requests.put(f"{bucket_url}/{spec.name}", data=reader, headers=_headers(), timeout=REQUEST_TIMEOUT)

        start = time.monotonic()
        r = _retry(do_put, f"upload {spec.name}")
        elapsed = max(time.monotonic() - start, 1e-9)
        speed_mbps = (spec.size / 1e6) / elapsed
        remote_md5 = r.json().get("checksum", "")
        if remote_md5.startswith("md5:"):
            remote_md5 = remote_md5[len("md5:"):]
        if remote_md5 != local_md5:
            raise RuntimeError(f"checksum mismatch for {spec.name}: local={local_md5} remote={remote_md5} -- upload landed corrupted")
        _log(f"  uploaded + verified {spec.name} ({spec.size/1e6:.0f}MB){tag} in {elapsed:.0f}s ({speed_mbps:.1f}MB/s)")
    finally:
        if spec.offset is not None:  # only clean up materialized chunks, never the original source file
            path.unlink(missing_ok=True)


def _publish(deposit_id):
    r = _retry(lambda: requests.post(f"{API}/deposit/depositions/{deposit_id}/actions/publish", headers=_headers(), timeout=REQUEST_TIMEOUT), "publish")
    return r.json()


def _append_dois_csv(csv_path, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "version", "part", "file", "doi", "zenodo_version"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    _log(f"Appended {len(rows)} row(s) to {csv_path}")


def _deposit_html_link(deposit):
    """The draft's editable-in-browser URL, so a long-running upload's
    progress can be checked/watched on Zenodo's site while it's still going."""
    return deposit["links"].get("html", f"{API}/deposit/{deposit['id']}")


def _upload_all(bucket_url, specs, chunk_dir, already_uploaded):
    """Uploads every spec in order, passing each one its (index, total,
    bytes_done, bytes_total) position within this full list (batch +
    chunker assets combined) so _upload_spec's log lines show overall
    progress through the current record, not just a per-file status."""
    total = len(specs)
    total_bytes = sum(s.size for s in specs)
    bytes_done = 0
    for i, spec in enumerate(specs, start=1):
        _upload_spec(bucket_url, spec, chunk_dir, already_uploaded, progress=(i, total, bytes_done, total_bytes))
        bytes_done += spec.size


def upload_batch_as_new_record(language, version, part, batch, chunk_dir, metadata):
    chunker_specs = _chunker_asset_specs() if _batch_has_chunks(batch) else []
    target_names = {s.name for s in batch} | {s.name for s in chunker_specs}
    pending = _find_pending_new_record_draft(metadata["title"])
    if pending:
        _log(f"  found pending draft {pending['id']} for {language} part {part} -- resuming it (large uploads can time out partway through)")
        deposit = pending
        already_uploaded = _remove_stale_draft_files(deposit, target_names)
    else:
        deposit = _fetch_deposit(_create_record(metadata)["id"])
        already_uploaded = {}

    _log(f"  draft: {_deposit_html_link(deposit)}")
    bucket_url = deposit["links"]["bucket"]
    _upload_all(bucket_url, batch + chunker_specs, chunk_dir, already_uploaded)
    published = _publish(deposit["id"])
    doi = published["doi"]
    _log(f"Published {language} {version} part {part}: {doi}")
    return doi


def upload_batch_as_new_version(language, version, part, batch, chunk_dir, existing_record_id):
    chunker_specs = _chunker_asset_specs() if _batch_has_chunks(batch) else []
    target_names = {s.name for s in batch} | {s.name for s in chunker_specs}
    conceptrecid = _retry(
        lambda: requests.get(f"{API}/deposit/depositions/{existing_record_id}", headers=_headers(), timeout=REQUEST_TIMEOUT),
        "fetch record for conceptrecid",
    ).json()["conceptrecid"]
    pending = _find_pending_version_draft(conceptrecid)
    if pending:
        _log(f"  found pending draft {pending['id']} for {language} part {part} -- resuming it (large uploads can time out partway through)")
        deposit = pending
        already_uploaded = _remove_stale_draft_files(deposit, target_names)
    else:
        deposit = _fetch_deposit(_new_version(existing_record_id)["id"])
        already_uploaded = _remove_stale_draft_files(deposit, target_names)  # clears Zenodo's inherited-old-files copy

    _log(f"  draft: {_deposit_html_link(deposit)}")
    bucket_url = deposit["links"]["bucket"]
    _upload_all(bucket_url, batch + chunker_specs, chunk_dir, already_uploaded)
    published = _publish(deposit["id"])
    doi = published["doi"]
    _log(f"Published new version of {language} {version} part {part}: {doi}")
    return doi


def logical_rows_for_batch(language, version, part, batch, doi, zenodo_version):
    """Maps a batch's ChunkSpecs (whole files or chunks) back to the
    logical model filenames they represent, for zenodo_dois.csv -- which
    always stores logical names; zenodo_common.group_logical_files() does
    the reverse mapping on the download side."""
    logical_names = set()
    for spec in batch:
        m = zc.CHUNK_PATTERN.match(spec.name)
        logical_names.add(f"{m.group('base')}.csv.bz2" if m else spec.name)
    return [
        {"language": language, "version": version, "part": part, "file": name, "doi": doi, "zenodo_version": zenodo_version}
        for name in sorted(logical_names)
    ]


def _existing_records_for(csv_path, language, version):
    """{part: record_id} for whatever's already published under this
    language+version in csv_path, so retraining/rebuilding can target the
    right existing DOIs."""
    record_id_by_part = {}
    if not os.path.exists(csv_path):
        return record_id_by_part
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["language"] == language and row.get("version", "2018") == version:
                record_id_by_part[int(row["part"])] = row["doi"].rsplit(".", 1)[-1]
    return record_id_by_part


def existing_records_for(language, version):
    return _existing_records_for(DOIS_CSV, language, version)


def _next_zenodo_version_for(csv_path, language, version, part):
    """How many distinct DOIs a part already has in csv_path, plus one --
    'v1' for a brand-new record, 'v2'+ for each Zenodo new-version
    republish, so the CSV makes it obvious why a language can have more
    than one DOI (a correction/retrain) rather than looking like a
    duplicate or a mistake."""
    if not os.path.exists(csv_path):
        return "v1"
    dois = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["language"] == language and row.get("version", "2018") == version and int(row["part"]) == part:
                dois.add(row["doi"])
    return f"v{len(dois) + 1}"


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

    for i, batch in enumerate(batches, start=1):
        part = i
        if len(batches) > 1:
            total_bytes = sum(s.size for s in batch)
            _log(f"--- Part {part}/{len(batches)}: {len(batch)} file(s), {total_bytes/1e9:.2f}GB ---")
        zenodo_version = _next_zenodo_version_for(DOIS_CSV, language, version, part)
        if part in existing:
            doi = upload_batch_as_new_version(language, version, part, batch, chunk_dir, existing[part])
        else:
            title = f"word2manylanguages: {language} Word2Vec embeddings ({version} corpus)"
            if len(batches) > 1:
                title += f" Part {part}"
            description = (
                f"Word2Vec word embeddings for '{language}', trained on the {version} "
                f"OpenSubtitles + Wikipedia corpus. See {REPO_URL} for the full pipeline."
            )
            if _batch_has_chunks(batch):
                description += (
                    " Some files here exceeded Zenodo's reliable single-upload size and were "
                    "split into '<file>_part_aa', '_part_ab', ... pieces -- see the included "
                    "README.md for how to recombine them (macOS/Linux: `cat <file>_part_* > "
                    "<file>`; Windows: the included FileChunker.ps1)."
                )
            metadata = {
                "upload_type": "dataset",
                "title": title,
                "description": description,
                "access_right": "open",
                **reference_metadata,
                # Marks corpus vintage as a queryable field, not just prose in the
                # title/description above -- 2018 and 2024 records are otherwise
                # identical (same creators/license/communities/related_identifiers).
                "keywords": [*reference_metadata.get("keywords", []), f"opensubtitles-{version}"],
            }
            doi = upload_batch_as_new_record(language, version, part, batch, chunk_dir, metadata)
        # Written immediately, part-by-part, rather than accumulated and
        # appended once at the end -- so if a later part fails (network
        # error, crash, Zenodo 500...), this part's row is already on disk.
        # A rerun's existing_records_for() then correctly sees this part as
        # already published instead of re-creating a duplicate record for
        # it (see sync_language's docstring / this module's docstring on
        # the "publish a correction" resume flow).
        _append_dois_csv(DOIS_CSV, logical_rows_for_batch(language, version, part, batch, doi, zenodo_version))

    extra_old_parts = set(existing) - set(range(1, len(batches) + 1))
    if extra_old_parts:
        print(
            f"WARNING: {language} ({version}) previously had part(s) {sorted(extra_old_parts)} that "
            f"aren't used by this upload -- those old records are now stale (not deleted; Zenodo "
            f"doesn't support deleting published records). Decide by hand whether to leave them or "
            f"note the change in their description."
        )

    if chunk_dir.exists():
        chunk_dir.rmdir()  # _upload_spec cleans up each chunk right after its upload; this just removes the now-empty dir


# sync_all_counts bundles every language's frequency-count files into one
# shared Zenodo record (or a few, if MAX_FILES_PER_RECORD is exceeded) --
# unlike sync_language, there's no single (language, version) to key the
# "already published?" lookup on, so bookkeeping in COUNTS_DOIS_CSV uses
# this fixed sentinel instead; each row's own language/version columns
# still reflect that row's real file, parsed from its filename.
_COUNTS_BOOKKEEPING_KEY = ("frequency-counts", "all-languages")

_COUNTS_SUBS_RE = re.compile(r"^(?P<lang>[a-z]{2,3})\.subs\.(?P<version>\d{4})\.tsv\.zip$")
_COUNTS_WIKI_RE = re.compile(r"^(?P<lang>[a-z]{2,3})\.wiki\.2018\.tsv\.zip$")


def _parse_counts_filename(name):
    """(language, version) for a counts filename produced by
    eval_inputs/build_counts_tokenized.py, e.g. 'af.subs.2018.tsv.zip' ->
    ('af', '2018'), 'af.wiki.2018.tsv.zip' -> ('af', '2018')."""
    m = _COUNTS_WIKI_RE.match(name)
    if m:
        return m.group("lang"), "2018"
    m = _COUNTS_SUBS_RE.match(name)
    if m:
        return m.group("lang"), m.group("version")
    raise ValueError(f"unrecognized counts filename (doesn't match {{lang}}.subs.{{version}}.tsv.zip or {{lang}}.wiki.2018.tsv.zip): {name}")


def sync_all_counts(counts_dir, dry_run=False):
    """
    Uploads every frequency-count file in counts_dir (all languages, both
    subtitles and wikipedia sides -- see eval_inputs/build_counts_tokenized.py)
    to Zenodo as one bundled dataset, batched into as many records as
    MAX_FILES_PER_RECORD/MAX_BYTES_PER_RECORD require (reuses sync_language's
    chunking infra, though these files are far too small for per-file
    chunking to ever trigger -- only the file-count cap realistically will,
    once enough languages are in).

    Meant to be run once, manually, after however many languages you want
    are done -- not per language. Re-running after more languages are added
    republishes a new version of the same record(s) (tracked in
    zenodo_counts_dois.csv), same "correction" semantics as sync_language's
    new-version path.
    """
    counts_dir = Path(counts_dir)
    paths = sorted(counts_dir.glob("*.tsv.zip"))
    if not paths:
        sys.exit(f"No count files found in {counts_dir}")
    print(f"counts: {len(paths)} file(s) across all languages, {sum(p.stat().st_size for p in paths)/1e6:.1f}MB total")

    specs = plan_chunks(paths)
    batches = batch_for_records(specs)
    print(f"  -> {len(batches)} record(s) needed ({sum(len(b) for b in batches)} file(s))")

    if dry_run:
        for i, batch in enumerate(batches, start=1):
            total = sum(s.size for s in batch)
            print(f"  [dry run] part {i}: {len(batch)} files, {total/1e6:.2f}MB")
        return

    chunk_dir = counts_dir / ".zenodo_upload_chunks_counts"
    bk_language, bk_version = _COUNTS_BOOKKEEPING_KEY
    existing = _existing_records_for(COUNTS_DOIS_CSV, bk_language, bk_version)

    needs_new_record = any(i not in existing for i in range(1, len(batches) + 1))
    reference_metadata = _reference_metadata() if needs_new_record else {}

    for i, batch in enumerate(batches, start=1):
        part = i
        if len(batches) > 1:
            total_bytes = sum(s.size for s in batch)
            _log(f"--- Part {part}/{len(batches)}: {len(batch)} file(s), {total_bytes/1e6:.2f}MB ---")
        zenodo_version = _next_zenodo_version_for(COUNTS_DOIS_CSV, bk_language, bk_version, part)
        if part in existing:
            doi = upload_batch_as_new_version(bk_language, bk_version, part, batch, chunk_dir, existing[part])
        else:
            title = "word2manylanguages: unigram frequency counts (all languages)"
            if len(batches) > 1:
                title += f" Part {part}"
            metadata = {
                "upload_type": "dataset",
                "title": title,
                "description": (
                    f"Unigram frequency count files for every language in this project -- the "
                    f"frequency baseline used to evaluate this project's Word2Vec word embedding "
                    f"models (see the project's other Zenodo records/DOIs), built from this "
                    f"project's own cleaned/deduplicated OpenSubtitles + Wikipedia corpus (see "
                    f"eval_inputs/build_counts_tokenized.py) rather than an external mirror. "
                    f"See {REPO_URL} for the full pipeline."
                ),
                "access_right": "open",
                **reference_metadata,
                "keywords": [*reference_metadata.get("keywords", []), "frequency-counts"],
            }
            doi = upload_batch_as_new_record(bk_language, bk_version, part, batch, chunk_dir, metadata)
        # Written immediately, part-by-part -- see the matching comment in
        # sync_language for why (resume correctness after a mid-run failure).
        part_rows = []
        for spec in batch:
            lang, version = _parse_counts_filename(spec.name)
            part_rows.append({"language": lang, "version": version, "part": part, "file": spec.name, "doi": doi, "zenodo_version": zenodo_version})
        _append_dois_csv(COUNTS_DOIS_CSV, part_rows)

    if chunk_dir.exists():
        chunk_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", help="two-letter language code, e.g. 'af' -- required unless --sync-counts")
    parser.add_argument("--version", default="2018", help="corpus vintage: '2018' (default) or '2024'")
    parser.add_argument("--models-dir", help="local directory containing {language}_*_wxd.csv.bz2 files -- required unless --sync-counts")
    parser.add_argument("--sync-counts", action="store_true", help="upload every language's frequency counts as one bundled record instead of a single language's models -- see sync_all_counts()")
    parser.add_argument("--counts-dir", help="local directory containing {language}.subs.{version}.tsv.zip / {language}.wiki.2018.tsv.zip files -- required with --sync-counts")
    parser.add_argument("--dry-run", action="store_true", help="only print the chunking/batching plan, no network calls")
    args = parser.parse_args()

    if args.sync_counts:
        if not args.counts_dir:
            parser.error("--sync-counts requires --counts-dir")
        sync_all_counts(args.counts_dir, dry_run=args.dry_run)
    else:
        if not args.language or not args.models_dir:
            parser.error("--language and --models-dir are required unless --sync-counts")
        sync_language(args.language, args.version, args.models_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
