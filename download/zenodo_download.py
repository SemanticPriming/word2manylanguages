"""Download trained model files from the project's Zenodo archives.

Reads zenodo_dois.csv (built by build_dataset_list.py from
source/dataset_list_source.pdf) to find which Zenodo record(s) hold a given
language's files. Languages whose files didn't fit in a single upload are
split across multiple records ("parts"); this downloads from all of them
transparently. No authentication is needed -- these are public records.

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

HERE = os.path.dirname(os.path.abspath(__file__))
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")


def load_record_ids(language):
    """Returns the list of Zenodo record IDs, in part order, for a language."""
    records = []
    with open(DOIS_CSV) as f:
        for row in csv.DictReader(f):
            if row["language"] == language:
                record_id = row["doi"].rsplit(".", 1)[-1]
                records.append((int(row["part"]), record_id))
    if not records:
        sys.exit(f"No Zenodo DOI found for language '{language}' in {DOIS_CSV}")
    records.sort()
    return [record_id for _, record_id in records]


def list_files(record_id):
    """Queries the Zenodo API for a record's file list."""
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}")
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_file(file_entry, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_entry["key"]
    if dest.exists() and dest.stat().st_size == file_entry["size"]:
        print(f"{dest} already downloaded, skipping.")
        return

    url = file_entry["links"]["self"]
    print(f"Downloading {file_entry['key']} ({file_entry['size'] / 1e6:.1f} MB)")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, help="two-letter language code, e.g. 'af'")
    parser.add_argument("--dest", required=True, help="local directory to download files into")
    parser.add_argument("--pattern", help="only download files whose name contains this substring, e.g. 'af_50_'")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    record_ids = load_record_ids(args.language)
    print(f"{args.language}: {len(record_ids)} Zenodo record(s): {', '.join(record_ids)}")

    for record_id in record_ids:
        for file_entry in list_files(record_id):
            if args.pattern and args.pattern not in file_entry["key"]:
                continue
            download_file(file_entry, dest_dir)


if __name__ == "__main__":
    main()
