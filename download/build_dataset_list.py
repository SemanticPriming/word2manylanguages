"""Parse the Zenodo DOI source table into a clean, machine-readable list.

The source (source/dataset_list_source.pdf) is a multi-page table, one row
per language, where languages split across several Zenodo uploads have their
DOIs crammed into a single cell as free text ("Part 1: ...\\nPart 2: ...", or
for some languages just multiple bare DOIs with no label at all). That format
is what caused dataset_list.xlsx to get manually mangled -- this script
re-derives both outputs directly from the PDF instead, so there's a single
source of truth and no more hand-editing.

Part numbers are assigned by the ORDER a DOI appears in the cell, not by the
"Part N:" label text -- the source table has at least one mislabeled entry
(two chunks both labeled "Part 4:" for language "en"), so trusting position
over the printed label sidesteps that.

Outputs:
  zenodo_dois.csv           one row per (language, part, file, doi) -- one
                             row per actual model file (chunked large files
                             collapsed back to their logical name), the
                             detailed/machine-readable version
  ../05_manuscript/dataset_list.xlsx   one row per (language, part), with a
                             short "included" summary instead of every
                             filename -- the manuscript appendix table

Building the file-level detail means querying the Zenodo API once per
(language, part) record in addition to parsing the PDF, so this takes a
while and makes ~100 HTTP requests.

Usage:
  python build_dataset_list.py
"""

import csv
import os
import re
import time

import pdfplumber
import requests

import zenodo_common as zc

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_PDF = os.path.join(HERE, "source", "dataset_list_source.pdf")
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")
DATASET_LIST_XLSX = os.path.join(HERE, "..", "05_manuscript", "dataset_list.xlsx")

MODEL_FILENAME_PATTERN = re.compile(r"^[a-z]{2}_(\d+)_(\d+)_(cbow|sg)_wxd\.csv\.bz2$")

# Zenodo record IDs in this table are consistently 8 digits, wrapped across
# PDF lines with stray whitespace/newlines in between. Bounding to exactly 8
# digits (rather than a greedy [\d\s]+) matters: some cells (e.g. language
# "sv") have several bare DOIs back-to-back with no "Part N:" separator, and
# a greedy match would swallow the next DOI's leading digits too.
#
# pdfplumber's text extraction can also inject a stray space *inside* the
# constant "10.5281/zenodo." prefix itself (e.g. language "si"/"sq" extract
# as "1 0.5281/zenodo...") -- tolerate whitespace between every character of
# the prefix too, not just between the trailing record-id digits.
_DOI_PREFIX = "10.5281/zenodo."
_DOI_PREFIX_PATTERN = r"\s*".join(re.escape(ch) for ch in _DOI_PREFIX)
DOI_PATTERN = re.compile(_DOI_PREFIX_PATTERN + r"\s*\d(?:\s*\d){7}")
LANG_PATTERN = re.compile(r"^[a-z]{2}$")

# Known-bad cells in the source PDF. The PDF itself isn't edited (kept as the
# literal historical record) -- these specific (language, part) DOIs are
# corrected here instead, with the actual source of the correction noted.
CORRECTIONS = {
    # source PDF duplicated Part 1's DOI for Part 2; corrected 2026-07-15
    ("es", 2): "10.5281/zenodo.17459793",
}


def extract_rows(pdf_path):
    """
    Reads every table row across all pages, and merges rows that are
    continuations of the previous language's DOI cell (page breaks split a
    single language's cell across two table extractions -- those rows come
    back with every column empty except the DOI one).
    """
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    lang_cell = (row[0] or "").strip().replace("\n", "")
                    doi_cell = row[-1] or ""
                    if LANG_PATTERN.match(lang_cell):
                        rows.append([lang_cell, doi_cell])
                    elif rows and doi_cell.strip():
                        # continuation of the previous language's DOI cell
                        rows[-1][1] += "\n" + doi_cell
    return rows


def parse_dois(doi_cell_text):
    """Returns an ordered list of clean DOI strings found in a cell's raw text."""
    dois = []
    for match in DOI_PATTERN.finditer(doi_cell_text):
        clean = re.sub(r"\s+", "", match.group(0))
        dois.append(clean)
    return dois


def build_doi_table(pdf_path):
    """Returns a list of dicts: {language, part, doi}."""
    entries = []
    for lang, doi_cell in extract_rows(pdf_path):
        dois = parse_dois(doi_cell)
        if not dois:
            print(f"WARNING: no DOI found for language '{lang}', cell: {doi_cell!r}")
            continue
        for part, doi in enumerate(dois, start=1):
            doi = CORRECTIONS.get((lang, part), doi)
            entries.append({"language": lang, "part": part, "doi": doi})
    return entries


def build_file_table(entries):
    """
    Queries Zenodo for each (language, part) record's actual file list and
    returns a list of dicts: {language, part, file, doi} -- one row per
    logical model file (chunked large files collapsed to their single
    logical name via zenodo_common.group_logical_files).
    """
    file_rows = []
    broken = []
    for e in entries:
        record_id = e["doi"].rsplit(".", 1)[-1]
        print(f"Querying Zenodo record {record_id} ({e['language']} part {e['part']})...")
        try:
            groups = zc.group_logical_files(zc.list_files(record_id))
        except requests.exceptions.HTTPError as ex:
            print(f"WARNING: record {record_id} ({e['language']} part {e['part']}) is unreachable: {ex}")
            broken.append((e["language"], e["part"], e["doi"]))
            continue
        if not groups:
            print(f"WARNING: no model files found in record {record_id} ({e['language']} part {e['part']})")
        for logical_name in sorted(groups):
            file_rows.append({"language": e["language"], "part": e["part"], "file": logical_name, "doi": e["doi"]})
        time.sleep(0.25)  # be polite to the Zenodo API across ~100 records

    if broken:
        print(f"\n{len(broken)} record(s) failed and were skipped -- these DOIs need investigating, not silently trusting:")
        for lang, part, doi in broken:
            print(f"  {lang} part {part}: {doi}")

    return file_rows


def write_dois_csv(file_rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "part", "file", "doi"])
        writer.writeheader()
        writer.writerows(file_rows)
    print(f"Wrote {len(file_rows)} rows to {path}")


def summarize_files(filenames):
    """
    Returns a short human-readable description of a set of model filenames,
    e.g. "12 files (dim 200; window 3-6; cbow,sg)". Parts aren't split along
    clean dim/window/algo boundaries (just upload size limits), so this lists
    what's actually present rather than implying full rectangular coverage.
    """
    dims, windows, algos = set(), set(), set()
    for name in filenames:
        m = MODEL_FILENAME_PATTERN.match(name)
        if m:
            dims.add(int(m.group(1)))
            windows.add(int(m.group(2)))
            algos.add(m.group(3))
    dim_str = ",".join(str(d) for d in sorted(dims))
    win_str = ",".join(str(w) for w in sorted(windows))
    algo_str = ",".join(sorted(algos))
    return f"{len(filenames)} files (dim {dim_str}; window {win_str}; {algo_str})"


def write_dataset_list_xlsx(file_rows, path):
    import openpyxl

    by_part = {}
    for row in file_rows:
        by_part.setdefault((row["language"], row["part"]), []).append(row)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Language", "Part", "Included", "DOI"])
    for (lang, part) in sorted(by_part):
        rows = by_part[(lang, part)]
        ws.append([lang, part, summarize_files([r["file"] for r in rows]), rows[0]["doi"]])
    wb.save(path)
    print(f"Wrote {len(by_part)} language/part rows to {path}")


def report_anomalies(entries):
    by_language = {}
    for e in entries:
        by_language.setdefault(e["language"], []).append(e["doi"])

    for lang, dois in by_language.items():
        if len(dois) != len(set(dois)):
            print(f"NOTE: language '{lang}' has duplicate DOIs across its parts: {dois}")


if __name__ == "__main__":
    entries = build_doi_table(SOURCE_PDF)
    report_anomalies(entries)
    file_rows = build_file_table(entries)
    write_dois_csv(file_rows, DOIS_CSV)
    write_dataset_list_xlsx(file_rows, DATASET_LIST_XLSX)
