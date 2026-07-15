# Pipeline Data Folders

`data/`, `preprocessed/`, `corpora/`, `models/`, `datasets/`, `evals/`, and `scores/` are the working directories the Python pipeline (`01_corpus_preprocessing/` → `02_model_training/` → `03_evaluation/`) reads from and writes to. `word2manylanguages_workflow.ipynb` sets `basedir = '.'`, so these folder names are exactly the constants (`datadir`, `processdir`, `corpusdir`, `modeldir`, `datasetsdir`, `evaldir`, `scoredir`) defined at the top of each stage's script.

## 📁 Folder Descriptions

- `data/`
  Raw corpus files downloaded from OpenSubtitles and Wikipedia, before cleaning or tokenizing.

- `preprocessed/`
  Cleaned and normalized versions of the raw corpora — XML, markup, and duplicate text removed.

- `corpora/`
  Final concatenated corpus for each language, one sentence per line — input to model training.

- `models/`
  Output of the fastText training step — `.csv` word-by-dimension matrices per parameter combination.

- `datasets/`
  Norms and lexical property datasets used for evaluating the trained embeddings (e.g., valence, AoA, concreteness).

- `evals/`
  Output from the ridge regression evaluation — predictive accuracy of a model configuration against one dataset.

- `scores/`
  Summary tables combining evaluation results across models and datasets for each language — used to identify top-performing settings.

## 📦 Why most of these are (nearly) empty

The full corpora, trained models, and per-language evaluation outputs for all 59 languages are too large to host in this GitHub repository. What's checked in here is a **single-language (Afrikaans) worked example** — enough to run the pipeline end-to-end and see real output at every stage:

- `data/` — the real downloaded `wikipedia-af.bz2` and `subtitles-af.zip`
- `datasets/` — real reference files (`datasets.csv`, `datasets_original.csv`, and example norms/counts data, including a synthetic `af-fake-2025` dataset for testing the norm-prediction code path)
- `preprocessed/`, `corpora/`, `models/`, `evals/`, `scores/` — empty except for `.gitkeep` placeholders; running the notebook against the `data/` and `datasets/` files above will populate them

For the real, full-scale corpora, trained models, and results for all 59 languages, see the linked Zenodo repositories referenced in the top-level [README.md](../README.md) and [06_manuscript/appendix.Rmd](../06_manuscript/appendix.Rmd) — the finished evaluation summaries (not the intermediate corpora/models) are also available pre-aggregated in [results/](../results/).

## 📝 Notes

- This structure is expected automatically by the Python pipeline — edit it in `01_corpus_preprocessing/`, `02_model_training/`, and `03_evaluation/`.
- If you're running the pipeline manually or modifying it, keep this folder layout for compatibility with the existing scripts.
