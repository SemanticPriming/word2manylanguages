# Code Overview

Code is organized by pipeline stage, in the order the workflow actually runs. Each numbered folder is self-contained (its own script, its own copy of the small set of shared path/parameter constants) so it can be handed off and run on its own.

`word2manylanguages_workflow.ipynb` at this level orchestrates all three Python stages end-to-end; `word2manylanguages_process.png` is the accompanying diagram.

## 📁 `01_corpus_preprocessing/`

Downloads and cleans the raw Wikipedia and OpenSubtitles corpora, deduplicates (optional), and concatenates them into one sentence-per-line corpus file per language.

## 📁 `02_model_training/`

Trains fastText word embedding models (via Gensim) across the swept dimensions (50–500), window sizes (1–6), and algorithms (CBOW / skip-gram).

## 📁 `03_evaluation/`

Evaluates trained models against replication norms, extended norms datasets, and word frequency counts using ridge regression, then summarizes/ranks results.

> 🔁 `01`–`03` together replicate the full pipeline for any of the 59 languages in the study.

## 📁 `04_postprocessing/`

R code used during the **manuscript preparation phase** to combine and summarize evaluation datasets across multiple runs (output of `03_evaluation/`) into the format used by the Shiny app.

## 📁 `05_visualization/`

The Shiny app providing an **interactive interface** for exploring model performance across languages, corpora, and hyperparameters.

You can launch the app locally using R:

```r
shiny::runApp("code/05_visualization") # assuming you are in the main Rproject folder
```

## 📁 `06_manuscript/`

Contains the manuscript and supporting materials for the paper — compiled PDF, source files, figures, and tables.

📦 Reproducibility Notes

- The Python pipeline was tested with Python 3.10 and Gensim 3.8.3
- R code was written for R 4.4.2
- All datasets and model outputs referenced are available via the linked Zenodo repositories
