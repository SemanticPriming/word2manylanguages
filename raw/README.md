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
- `eval_inputs/` — real reference files (`datasets.csv`, `datasets_original.csv`, and example norms/counts data, including a synthetic `af-fake-2025` dataset for testing the norm-prediction code path)
- `models/` — one real trained model (`af_50_1_cbow_wxd.csv.bz2`) kept as a worked example; the other 59 dim/window/algo combinations for `af`, and every other language, are download-on-demand (see below)
- `preprocessed/`, `corpora/` — gitignored; running the pipeline against the `raw/` files above regenerates them (see their own READMEs for what to expect: [preprocessed/README.md](../preprocessed/README.md), [corpora/README.md](../corpora/README.md))
- `eval_results/` — empty except for a `.gitkeep` placeholder; running the pipeline against the `eval_inputs/` files above will populate it

For the real, full-scale corpora, trained models, and frequency-count data for all 59 languages — either from the lab's MinIO storage or the public Zenodo archives — see [`download/README.md`](../download/README.md).

## 📝 Notes

- This structure is expected automatically by the Python pipeline — edit it in `01_corpus_preprocessing/`, `02_model_training/`, and `03_evaluation/`.
- If you're running the pipeline manually or modifying it, keep this folder layout for compatibility with the existing scripts.
