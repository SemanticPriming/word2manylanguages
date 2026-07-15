"""Download trained model files from the project's Zenodo archives.

Reads zenodo_dois.csv (built by build_dataset_list.py from
source/dataset_list_source.pdf) to find which Zenodo record(s) hold a given
language's files. Languages whose files didn't fit in a single upload are
split across multiple records ("parts"); this downloads from all of them
transparently. No authentication is needed -- these are public records.

Some large models (e.g. higher-dimension English embeddings) also exceeded
Zenodo's per-file size limit and were split into "_part_aa", "_part_ab", ...
chunks within their record. This downloads every chunk and reassembles them
back into the real, usable {lang}_{dim}_{window}_{algo}_wxd.csv.bz2 -- you
never see the raw chunks.

Usage:
  python zenodo_download.py --language af --dest ../models/
  python zenodo_download.py --language af --dest ../models/ --pattern af_50_
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import requests

import zenodo_common as zc

HERE = os.path.dirname(os.path.abspath(__file__))
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")


def load_records_for_language(language):
    """Returns the list of Zenodo record IDs, in part order, for a language."""
    record_id_by_part = {}
    with open(DOIS_CSV) as f:
        for row in csv.DictReader(f):
            if row["language"] == language:
                record_id_by_part[int(row["part"])] = row["doi"].rsplit(".", 1)[-1]
    if not record_id_by_part:
        sys.exit(f"No Zenodo DOI found for language '{language}' in {DOIS_CSV}")
    return [record_id_by_part[part] for part in sorted(record_id_by_part)]


def _fetch(file_entry, dest):
    url = file_entry["links"]["self"]
    print(f"  fetching {file_entry['key']} ({file_entry['size'] / 1e6:.1f} MB)")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def download_logical_file(logical_name, chunk_entries, dest_dir):
    """
    Downloads one logical model file. If it's stored as multiple Zenodo
    chunks, downloads each to a temp path and concatenates them in order
    into the final file, then removes the temp chunks.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / logical_name
    total_size = sum(e["size"] for e in chunk_entries)
    if dest.exists() and dest.stat().st_size == total_size:
        print(f"{dest} already downloaded, skipping.")
        return

    if len(chunk_entries) == 1:
        _fetch(chunk_entries[0], dest)
        return

    print(f"Downloading {logical_name} ({len(chunk_entries)} parts)...")
    tmp_paths = [dest_dir / entry["key"] for entry in chunk_entries]
    for entry, tmp_path in zip(chunk_entries, tmp_paths):
        _fetch(entry, tmp_path)

    with open(dest, "wb") as out:
        for tmp_path in tmp_paths:
            with open(tmp_path, "rb") as chunk_file:
                out.write(chunk_file.read())
    for tmp_path in tmp_paths:
        tmp_path.unlink()
    print(f"Reassembled {logical_name} from {len(chunk_entries)} parts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, help="two-letter language code, e.g. 'af'")
    parser.add_argument("--dest", required=True, help="local directory to download files into")
    parser.add_argument("--pattern", help="only download models whose filename contains this substring, e.g. 'af_50_'")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    record_ids = load_records_for_language(args.language)
    print(f"{args.language}: {len(record_ids)} Zenodo record(s): {', '.join(record_ids)}")

    for record_id in record_ids:
        groups = zc.group_logical_files(zc.list_files(record_id))
        for logical_name in sorted(groups):
            if args.pattern and args.pattern not in logical_name:
                continue
            download_logical_file(logical_name, groups[logical_name], dest_dir)


if __name__ == "__main__":
    main()
