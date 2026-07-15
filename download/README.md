# Downloading Data and Trained Models

Everything the pipeline reads that's too large (or too numerous) to check into this repo, and where it goes:

1. [Raw corpus text](#1-raw-corpus-text-raw) → `raw/`
2. [Prebuilt trained models](#2-prebuilt-trained-models-models) → `models/`
3. [Frequency counts](#3-frequency-count-data-eval_inputscounts) → `eval_inputs/counts/`
4. [Extended psycholinguistic norms](#4-extended-norms-eval_inputsnorms) → `eval_inputs/norms/`
5. [Replication norms](#5-replication-norms-eval_inputsreplication) → `eval_inputs/replication/`

Each destination folder is checked in with just an Afrikaans (`af`)-scale worked example — real data for 1–3, a synthetic `af-fake-2025` file for 4–5 (there's no real Afrikaans data in the LAB or subs2vec's replication set) — see [`raw/README.md`](../raw/README.md) for the full rundown of what's checked in vs. gitignored across all these working folders.

## 1. Raw corpus text (`raw/`)

The OpenSubtitles and Wikipedia dumps that [`01_corpus_preprocessing/`](../01_corpus_preprocessing/) cleans are fetched by that stage's own `download()` function, not by anything in this folder — see [`01_corpus_preprocessing/README.md`](../01_corpus_preprocessing/README.md) for the full worked example. In short:

```python
import sys
sys.path.insert(0, '01_corpus_preprocessing')
import corpus_preprocessing as cp

cp.basedir = '.'
cp.download('wikipedia', 'af')   # -> raw/wikipedia-af.bz2
cp.download('subtitles', 'af')   # -> raw/subtitles-af.zip
```

## 2. Prebuilt trained models (`models/`)

Trained models (`{lang}_{dim}_{window}_{algo}_wxd.csv.bz2`) are too large to check into this repo — the full `models/` output across all 59 languages is ~2.9 TiB. Two ways to get them locally, into `../models/`:

### MinIO (lab-internal)

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

### Zenodo (public archive)

Trained models are also permanently archived on Zenodo, one DOI per language (or several "part" DOIs for languages whose files didn't fit a single upload's size limit). `zenodo_download.py` resolves a language to its DOI(s) via `zenodo_dois.csv` and downloads directly — no credentials needed, these are public records.

```bash
pip install requests
python download/zenodo_download.py --language af --dest models/
python download/zenodo_download.py --language af --dest models/ --pattern af_50_   # just the dim=50 configs
```

#### Where `zenodo_dois.csv` comes from

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

## 3. Frequency count data (`eval_inputs/counts/`)

Word-frequency counts computed from van Paridon & Thompson's (2021) subtitle/Wikipedia corpora — used by `evaluation.py`'s `predict_counts` (Research Question 2, frequency prediction). `eval_inputs/counts/` ships with just the Afrikaans example (`dedup.af.words.unigrams.tsv.zip` and `dedup.afwiki-meta.words.unigrams.tsv.zip`, small enough to keep tracked). Other languages' frequency data lives in the lab's MinIO bucket under `frequency_source/`, zip-compressed; `evaluation.py`'s `load_count_freqs` reads `.tsv.zip` transparently, so there's no need to unzip after downloading:

```bash
set -a; source .env; set +a
pip install minio
python download/minio_download.py --prefix frequency_source/dedup.bg --dest eval_inputs/counts/   # both dedup.bg. and dedup.bgwiki-meta. files
```

(There's no Zenodo path for this one — the `frequency_source/` data isn't part of the DOI-archived deposits, only the trained models are. The original public source is van Paridon's own [subs2vec](https://github.com/jvparidon/subs2vec) project; the lab's MinIO bucket is a mirror of those counts, kept alongside the trained models for convenience.)

## 4. Extended norms (`eval_inputs/norms/`)

Psycholinguistic norm datasets from Buchanan, Valentine, & Maxwell's (2019) [**Linguistic Annotated Bibliography (LAB)**](https://doi.org/10.3758/s13428-018-1130-8) — used by `evaluation.py`'s `load_extended_norms`/`predict_norms` (Research Question 3, extended norm prediction beyond the original subs2vec replication set). `eval_inputs/norms/` ships with only a synthetic `af-fake-2025.csv` for exercising this code path — the real per-dataset files (`Riegel2015.csv`, `Torrance2018.csv`, `Alario1999.csv`, etc., ~300 files, one per cited study) are published as release assets on the [`SemanticPriming/semanticprimeR`](https://github.com/SemanticPriming/semanticprimeR) repo, release [`v0.0.1` ("LAB-data")](https://github.com/SemanticPriming/semanticprimeR/releases/tag/v0.0.1).

`eval_inputs/datasets.csv` maps each language to the pipe-separated list of filenames `load_extended_norms` expects for it (`eval_inputs/datasets_original.csv` is the same mapping before some renaming/consolidation). Download with the [GitHub CLI](https://cli.github.com/):

```bash
# every dataset (~300 files, all languages)
gh release download v0.0.1 --repo SemanticPriming/semanticprimeR --dir eval_inputs/norms/

# just the files needed for one language -- look them up in datasets.csv first
gh release download v0.0.1 --repo SemanticPriming/semanticprimeR --dir eval_inputs/norms/ \
  --pattern 'Riegel2015.csv' --pattern 'Russell1970.csv' --pattern 'Schauenburg2015.csv'
```

`--skip-existing` avoids re-downloading files you already have.

## 5. Replication norms (`eval_inputs/replication/`)

The lexical norm datasets van Paridon & Thompson (2021) used to evaluate subs2vec — used by `evaluation.py`'s `load_replication_norms`/`predict_norms` (Research Question 1, the direct replication of their evaluation). `eval_inputs/replication/` ships with only a synthetic `af-fake-2025.tsv` for exercising this code path — the real files live in the [`jvparidon/subs2vec`](https://github.com/jvparidon/subs2vec) repo, under `subs2vec/datasets/norms/`, one file per `{language}-{author}-{year}.tsv` (e.g. `en-kuperman-2012.tsv`) — the same naming convention the synthetic example follows.

GitHub doesn't offer per-folder downloads, so pull just that subfolder with a sparse, blobless clone rather than the whole repo:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/jvparidon/subs2vec.git /tmp/subs2vec
git -C /tmp/subs2vec sparse-checkout set subs2vec/datasets/norms

cp /tmp/subs2vec/subs2vec/datasets/norms/en-*.tsv eval_inputs/replication/   # just English
cp /tmp/subs2vec/subs2vec/datasets/norms/*.tsv eval_inputs/replication/      # every language subs2vec covers
rm -rf /tmp/subs2vec
```

`load_replication_norms(lang)` picks up any file in `eval_inputs/replication/` whose name starts with `lang`, so there's nothing further to wire up beyond copying the files in.
