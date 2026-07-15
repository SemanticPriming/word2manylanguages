# Stage 2: Model Training

Trains fastText word embedding models (via Gensim) on the corpus produced by [`01_corpus_preprocessing/`](../01_corpus_preprocessing/), sweeping embedding dimension, window size, and algorithm. Output word-by-dimension matrices feed into [`03_evaluation/`](../03_evaluation/).

## 📂 Contents

- `model_training.py` — `build_models` loops over dimensions (50–500), window sizes (1–6), and algorithms (`cbow`, `sg`), calling `vectorize_stream` to train each Gensim `FastText` model and writing a bz2-compressed `{language}_{dim}_{window}_{algo}_wxd.csv.bz2` word-by-dimension matrix per configuration.

## 📦 Requirements

- Python 3.10+
- `gensim>=4.0` 
- `numpy`, `pandas`

## 🧠 What training does

`build_models(language, overwrite=False)` reads `corpora/corpus-{language}.txt` (one sentence per line, produced by [`01_corpus_preprocessing/`](../01_corpus_preprocessing/)) and trains one Gensim `FastText` model per combination of:

- **dimension** (`dimension_list`): 50, 100, 200, 300, 500
- **window size** (`window_list`): 1–6
- **algorithm** (`algo_list`): `cbow` (Continuous Bag of Words) or `sg` (Skip-Gram)

— 5 × 6 × 2 = 60 models per language. For each combination:

1. `sentences(language)` streams `corpus-{language}.txt` line by line as a generator, splitting each line on whitespace into a list of tokens — Gensim trains directly off this iterator instead of loading the whole corpus into memory.
2. `vectorize_stream(language, min_freq, dim, win, alg)` builds `FastText(vector_size=dim, window=win, min_count=min_freq, sg=(1 if alg == "sg" else 0), sample=1e-2, negative=10, alpha=0.05, min_n=3, max_n=6)`, then calls `build_vocab` and `train` (10 epochs) on the `sentences` generator. `build_models` always calls this with `min_freq=5`, so words appearing fewer than 5 times in the corpus are dropped from the vocabulary; `min_n`/`max_n` set the character n-gram range fastText uses to build subword representations.
3. The trained vectors for every in-vocabulary word (`model.wv`) are assembled into a word-by-dimension `pandas.DataFrame` (one row per word, one column per embedding dimension) and written directly as a bz2-compressed CSV to `models/{language}_{dim}_{window}_{algo}_wxd.csv.bz2` (`pandas.DataFrame.to_csv(..., compression='bz2')` — a full sweep's 60 files add up fast otherwise). [`03_evaluation/`](../03_evaluation/)'s `load_model` reads this transparently, falling back to a plain `.csv` if one exists instead.

Like the stage 1 scripts, each configuration is skipped (with a printed message) if its output file already exists; pass `overwrite=True` to force a re-run.

## ▶️ Running it (Afrikaans example)

`model_training.py` has no `__main__` block either — same pattern as stage 1: import it and set `basedir` to the repo root, where `corpora/` and `models/` live.

```python
import sys
sys.path.insert(0, '02_model_training')
import model_training as mt

mt.basedir = '.'  # repo root, where corpora/ and models/ live

mt.build_models('af')   # corpora/corpus-af.txt -> models/af_{dim}_{window}_{algo}_wxd.csv.bz2, x60
```

The full sweep trains 60 models and can take a long time on a full-size corpus. To try just one configuration — e.g. to reproduce the checked-in `models/af_50_1_cbow_wxd.csv.bz2` worked example — shrink the sweep lists before calling `build_models`:

```python
mt.dimension_list = [50]
mt.window_list = [1]
mt.algo_list = ['cbow']

mt.build_models('af')   # -> models/af_50_1_cbow_wxd.csv.bz2 only
```

Or skip `build_models`'s file-writing/skip-if-exists logic entirely and train a single model directly:

```python
model = mt.vectorize_stream('af', min_freq=5, dim=50, win=1, alg='cbow')
```
