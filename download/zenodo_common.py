"""Shared helpers for talking to the project's Zenodo records.

Used by both build_dataset_list.py (regenerating the DOI/file tables) and
zenodo_download.py (downloading model files) so the two agree on what counts
as a model file and how split files are grouped back together.
"""

import re

import requests

# Model files that exceeded Zenodo's per-file size limit were split with the
# classic `split -a N` convention: "{logical_base}_part_aa", "_ab", ... (no
# file extension on the chunks -- see the README.md/FileChunker.ps1 shipped
# alongside them in the affected deposits). Reassembly is just concatenating
# the chunks in filename order.
CHUNK_PATTERN = re.compile(r"^(?P<base>.+)_part_[a-z]{2,}$")

# Only files matching this are trained models (or chunks of one) -- anything
# else in a deposit (README.md, FileChunker.ps1, ...) is packaging, not data.
MODEL_PATTERN = re.compile(r"^[a-z]{2}_\d+_\d+_(cbow|sg)_wxd(\.csv\.bz2|_part_[a-z]{2,})$")


def list_files(record_id):
    """Queries the Zenodo API for a record's file list."""
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}")
    resp.raise_for_status()
    return resp.json().get("files", [])


def group_logical_files(file_entries):
    """
    Groups a Zenodo record's raw file entries into logical model files:
    split parts are reassembled under their shared base name, and non-model
    housekeeping files (README.md, FileChunker.ps1, ...) are dropped.

    Returns {logical_filename: [file_entry, ...]} -- a single-entry list for
    a whole file, or multiple chunk entries in reassembly order (sorted by
    their "_part_aa"/"_part_ab"/... suffix) for a split one.
    """
    groups = {}
    for entry in file_entries:
        key = entry["key"]
        if not MODEL_PATTERN.match(key):
            continue
        chunk_match = CHUNK_PATTERN.match(key)
        logical_name = f"{chunk_match.group('base')}.csv.bz2" if chunk_match else key
        groups.setdefault(logical_name, []).append(entry)

    for entries in groups.values():
        entries.sort(key=lambda e: e["key"])  # chunk suffixes sort in reassembly order

    return groups
