# Downloading Trained Models and Frequency Data

Trained models (`{lang}_{dim}_{window}_{algo}_wxd.csv.bz2`) are too large to check into this repo — the full `models/` output across all 59 languages is ~2.9 TiB. Two ways to get them locally, into `../models/`:

## MinIO (lab-internal)

`minio_download.py` downloads from the lab's MinIO bucket. Requires credentials in `.env` at the repo root:

```
MINIO_ENDPOINT=<bare host, no scheme/path -- e.g. s3.example.cloud>
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_BUCKET=<bucket name>
```

> The MinIO web console and the S3 API it's proxying are usually different hosts even when they share credentials — copy the *console* URL's bucket name into `MINIO_BUCKET`, but don't use the console URL itself as `MINIO_ENDPOINT`.

```bash
set -a; source .env; set +a   # load credentials into the environment
pip install minio
python download/minio_download.py --object models/af_50_1_cbow_wxd.csv.bz2 --dest models/af_50_1_cbow_wxd.csv.bz2
python download/minio_download.py --prefix models/af_ --dest models/   # every af_ model in one go
```

## Zenodo (public archive)

Trained models are also permanently archived on Zenodo, one DOI per language (or several "part" DOIs for languages whose files didn't fit a single upload's size limit). `zenodo_download.py` resolves a language to its DOI(s) via `zenodo_dois.csv` and downloads directly — no credentials needed, these are public records.

```bash
pip install requests
python download/zenodo_download.py --language af --dest models/
python download/zenodo_download.py --language af --dest models/ --pattern af_50_   # just the dim=50 configs
```

### Where `zenodo_dois.csv` comes from

`source/dataset_list_source.pdf` is the source-of-truth DOI table (one row per language, multi-part languages listed as free text like "Part 1: ...\nPart 2: ..."). Hand-editing that into a spreadsheet is exactly what corrupted `../05_manuscript/dataset_list.xlsx` before — instead, regenerate both derived files straight from the PDF:

```bash
pip install pdfplumber openpyxl
python download/build_dataset_list.py
```

This writes:
- `zenodo_dois.csv` — one row per `(language, part, doi)`, what `zenodo_download.py` reads
- `../05_manuscript/dataset_list.xlsx` — the manuscript appendix's "DOIs for Word Embeddings" table, one row per language, DOIs joined consistently

Known data issues in the source PDF:
- **`es`** Part 2: the PDF lists the same DOI as Part 1 (`17450685`, a copy-paste error) — corrected in code via the `CORRECTIONS` dict in `build_dataset_list.py` (confirmed correct value: `17459793`)
- **`de`**: total file count shown as `60*` in the source (unexplained footnote) — left as-is, unresolved
- **`eo`**: dimension list shown as `...,500x` in the source (stray character) — left as-is, unresolved

If the source table changes (new language, more parts, another wrong DOI discovered), edit `source/dataset_list_source.pdf` and re-run `build_dataset_list.py`, adding a new `CORRECTIONS` entry if it's a specific-cell fix rather than a PDF update — don't hand-edit `zenodo_dois.csv` or `dataset_list.xlsx` directly.

## Frequency count data (eval_inputs/counts/)

`eval_inputs/counts/` ships with just the Afrikaans example (`dedup.af.words.unigrams.tsv` and `dedup.afwiki-meta.words.unigrams.tsv`, ~6MB total — small enough to keep tracked). Other languages' frequency data lives in the same MinIO bucket under `frequency_source/`, zip-compressed; `evaluation.py`'s `load_count_freqs` reads `.tsv.zip` transparently, so there's no need to unzip after downloading:

```bash
set -a; source .env; set +a
pip install minio
python download/minio_download.py --prefix frequency_source/dedup.bg --dest eval_inputs/counts/   # both dedup.bg. and dedup.bgwiki-meta. files
```

(There's no Zenodo path for this one — the `frequency_source/` data isn't part of the DOI-archived deposits, only the trained models are.)
