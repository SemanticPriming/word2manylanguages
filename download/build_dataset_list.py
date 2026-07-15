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
  zenodo_dois.csv           one row per (language, part, doi) -- what
                             zenodo_download.py reads
  ../05_manuscript/dataset_list.xlsx   one row per language, DOIs joined
                             consistently, for the manuscript appendix table

Usage:
  python build_dataset_list.py
"""

import csv
import os
import re

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_PDF = os.path.join(HERE, "source", "dataset_list_source.pdf")
DOIS_CSV = os.path.join(HERE, "zenodo_dois.csv")
DATASET_LIST_XLSX = os.path.join(HERE, "..", "05_manuscript", "dataset_list.xlsx")

# Zenodo record IDs in this table are consistently 8 digits, wrapped across
# PDF lines with stray whitespace/newlines in between. Bounding to exactly 8
# digits (rather than a greedy [\d\s]+) matters: some cells (e.g. language
# "sv") have several bare DOIs back-to-back with no "Part N:" separator, and
# a greedy match would swallow the next DOI's leading digits too.
DOI_PATTERN = re.compile(r"10\.5281/zenodo\.\s*\d(?:\s*\d){7}")
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


def write_dois_csv(entries, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "part", "doi"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Wrote {len(entries)} rows to {path}")


def write_dataset_list_xlsx(entries, path):
    import openpyxl

    by_language = {}
    for e in entries:
        by_language.setdefault(e["language"], []).append(e)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Language", "Parts", "DOI"])
    for lang, parts in by_language.items():
        if len(parts) == 1:
            doi_display = parts[0]["doi"]
        else:
            doi_display = "; ".join(f"Part {p['part']}: {p['doi']}" for p in parts)
        ws.append([lang, len(parts), doi_display])
    wb.save(path)
    print(f"Wrote {len(by_language)} languages to {path}")


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
    write_dois_csv(entries, DOIS_CSV)
    write_dataset_list_xlsx(entries, DATASET_LIST_XLSX)
