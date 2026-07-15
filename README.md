# word2manylanguages

**Modeling Languages with Their Own Parameters**  

This repository contains all code, models, and documentation associated with the paper: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17243814.svg)](https://doi.org/10.5281/zenodo.17243814)

## 📖 Overview

This project builds multilingual word embedding models using corpora from OpenSubtitles and Wikipedia. Unlike prior work, we optimize model parameters — including embedding dimension and window size — *separately for each language*, improving prediction of psycholinguistic norms.

We provide:

- Optimized fastText embeddings
- Tools for training and evaluating embeddings
- A reproducible pipeline for multilingual modeling
- A [Shiny app](link-to-app) for interactive exploration COMING SOON!

## 🔍 Key Findings

- Default fastText settings (300d, window=5) are **not optimal** across languages.
- Best-performing settings vary widely by task, corpus type, and language.
- Small models (e.g., 50d, window=1) often outperform larger ones on some tasks.

## 📦 Repository Contents

Code is organized by pipeline stage, in the order the workflow actually runs:

- `01_corpus_preprocessing/` → `02_model_training/` → `03_evaluation/`: the Python pipeline that builds corpora, trains embedding models, and evaluates them against psycholinguistic norms (see `word2manylanguages_workflow.ipynb` for a walk-through)
- `04_visualization/`: the Shiny app for interactive exploration
- `05_manuscript/`: the manuscript and supporting materials
- `06_presentations/`: presentations from conferences on this project
- `raw/`, `preprocessed/`, `corpora/`, `models/`, `eval_inputs/`, `eval_results/`: working folders for the Python pipeline, checked in with a single-language (Afrikaans) worked example — see [raw/README.md](raw/README.md)
- `download/`: scripts for pulling trained models from either lab-internal storage (MinIO) or the public Zenodo archives — see [download/README.md](download/README.md)

## 🚀 Get Started

### Requirements

- Python 3.10+
- fastText (via Gensim 3.8.3)
- R 4.4.2 for reproducible manuscript analysis