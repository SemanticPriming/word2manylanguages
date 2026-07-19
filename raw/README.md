# Pipeline Data Folders

`raw/`, `preprocessed/`, `corpora/`, `models/`, `eval_inputs/`, and `eval_results/` are the working directories the Python pipeline (`01_corpus_preprocessing/` → `02_model_training/` → `03_evaluation/`) reads from and writes to. `word2manylanguages_workflow.ipynb` sets `basedir = '.'`, so these folder names are exactly the constants (`datadir`, `processdir`, `corpusdir`, `modeldir`, `datasetsdir`, `evaldir`) defined at the top of each stage's script.

## 📁 Folder Descriptions

- `raw/`
  Raw corpus files downloaded from OpenSubtitles and Wikipedia, before cleaning or tokenizing.

- `preprocessed/`
  Cleaned and normalized versions of the raw corpora — XML, markup, and duplicate text removed.

- `corpora/`
  Final concatenated corpus for each language, one sentence per line — input to model training.

- `models/`
  Output of the fastText training step — `.csv` word-by-dimension matrices per parameter combination.

- `eval_inputs/`
  Ground-truth norms and lexical property datasets used to *evaluate* the trained embeddings (e.g., valence, AoA, concreteness, word frequency counts).

- `eval_results/`
  Raw output from the ridge regression evaluation — predictive accuracy of one model configuration against one `eval_inputs/` dataset.

## 📦 Why most of these are (nearly) empty

The full corpora, trained models, and per-language evaluation outputs for all 59 languages are too large to host in this GitHub repository. What's checked in here is a **single-language (Afrikaans) worked example** — enough to run the pipeline end-to-end and see real output at every stage:

- `raw/` — the real downloaded `wikipedia-af.bz2` and `subtitles-af.zip`
- `eval_inputs/` — `datasets_norms.csv` and `datasets_replication.csv`, full catalogs of every norms/replication dataset, language, and predicted variable (see [`eval_inputs/build_datasets_norms.py`](../eval_inputs/build_datasets_norms.py) / [`build_datasets_replication.py`](../eval_inputs/build_datasets_replication.py)), plus example data: `norms/Luniewska2016.csv` (the one real norms dataset that covers Afrikaans), `counts/dedup.af*.tsv.zip`, and a README standing in for `replication/` (no dataset there covers Afrikaans, real or synthetic)
- `models/` — one real trained model (`af_50_1_cbow_wxd.csv.bz2`) kept as a worked example; the other 59 dim/window/algo combinations for `af`, and every other language, are download-on-demand (see below)
- `preprocessed/`, `corpora/` — gitignored; running the pipeline against the `raw/` files above regenerates them (see their own READMEs for what to expect: [preprocessed/README.md](../preprocessed/README.md), [corpora/README.md](../corpora/README.md))
- `eval_results/` — empty except for a `.gitkeep` placeholder; running the pipeline against the `eval_inputs/` files above will populate it

For the real, full-scale raw corpus text, trained models, frequency counts, and evaluation norm datasets for all 59 languages — from the lab's MinIO storage, the public Zenodo archives, or upstream sources like the Linguistic Annotated Bibliography and subs2vec — see [`download/README.md`](../download/README.md).

## 📝 Notes

- This structure is expected automatically by the Python pipeline — edit it in `01_corpus_preprocessing/`, `02_model_training/`, and `03_evaluation/`.
- If you're running the pipeline manually or modifying it, keep this folder layout for compatibility with the existing scripts.
