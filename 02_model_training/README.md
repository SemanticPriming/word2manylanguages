# Stage 2: Model Training

Trains fastText word embedding models (via Gensim) on the corpus produced by [`01_corpus_preprocessing/`](../01_corpus_preprocessing/), sweeping embedding dimension, window size, and algorithm. Output word-by-dimension matrices feed into [`03_evaluation/`](../03_evaluation/).

## 📂 Contents

- `model_training.py` — `build_models` loops over dimensions (50–500), window sizes (1–6), and algorithms (`cbow`, `sg`), calling `vectorize_stream` to train each Gensim `FastText` model and writing a `{language}_{dim}_{window}_{algo}_wxd.csv` word-by-dimension matrix per configuration.

See [`code/word2manylanguages_workflow.ipynb`](../word2manylanguages_workflow.ipynb) for a walk-through that drives this module alongside the other two stages.

## 📦 Requirements

- Python 3.10+
- `gensim==3.8.3`
- `numpy`, `pandas`
