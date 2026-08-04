# Stage 2: Model Training

Trains fastText word embedding models (via Gensim) on the corpus produced by [`01_corpus_preprocessing/`](../01_corpus_preprocessing/), sweeping embedding dimension, window size, and algorithm. Output word-by-dimension matrices feed into [`03_evaluation/`](../03_evaluation/).

## 📂 Contents

- `model_training.py` — `build_models` loads `corpora/corpus-{language}.txt` into memory once (`load_corpus`), then loops over dimensions (50–500), window sizes (1–6), and algorithms (`cbow`, `sg`), calling `vectorize_stream` to train each Gensim `FastText` model off that same in-memory corpus and writing a bz2-compressed `{language}_{dim}_{window}_{algo}_wxd.csv.bz2` word-by-dimension matrix per configuration.

## 📦 Requirements

- Python 3.10+
- `gensim>=4.0` 
- `numpy`, `pandas`

## 🧠 What training does

`build_models(language, overwrite=False)` reads `corpora/corpus-{language}.txt` (one sentence per line, produced by [`01_corpus_preprocessing/`](../01_corpus_preprocessing/)) and trains one Gensim `FastText` model per combination of:

- **dimension** (`dimension_list`): 50, 100, 200, 300, 500
- **window size** (`window_list`): 1–6
- **algorithm** (`algo_list`): `cbow` (Continuous Bag of Words) or `sg` (Skip-Gram)

— 5 × 6 × 2 = 60 models per language. `build_models` first checks whether any config's output file is still missing; if every one already exists (and `overwrite` isn't set) it skips the language without touching the corpus at all. Otherwise:

1. `load_corpus(language)` reads `corpus-{language}.txt` once, splitting each line on whitespace into a list of tokens, and returns the whole thing as an in-memory list of token lists — reused across every remaining config, rather than re-reading the corpus file from disk for each one.
2. `vectorize_stream(corpus, min_freq, dim, win, alg)` builds `FastText(vector_size=dim, window=win, min_count=min_freq, sg=(1 if alg == "sg" else 0), sample=1e-2, negative=10, alpha=0.05, min_n=3, max_n=6, workers=mt.workers)`, then calls `build_vocab` and `train` (10 epochs, matching the subs2vec parameters this project replicates -- see `05_manuscript/manuscript.Rmd`) on that same in-memory `corpus` list. `build_models` always calls this with `min_freq=5`, so words appearing fewer than 5 times in the corpus are dropped from the vocabulary; `min_n`/`max_n` set the character n-gram range fastText uses to build subword representations.

`workers` (module-level, like `dimension_list`/`window_list`/`algo_list`) defaults to `os.cpu_count() - 1` -- gensim's own default is a hardcoded 3, which badly underuses a large machine. Override it directly, e.g. `mt.workers = 32` on a bigger server, before calling `build_models`.

Each of the 10 requested epochs is a genuine full pass over the corpus -- `corpus` (from `load_corpus`) is a plain, repeatedly-iterable Python list, which gensim's `train()` requires: it calls `iter()` on the corpus fresh once per epoch (`for cur_epoch in range(self.epochs): ...` in gensim's own `word2vec.py`), and needs each call to yield the data again from the start. A single-use generator (the old streaming approach here, before `load_corpus`) breaks silently under this: once epoch 1 exhausts it, `iter()` on later epochs returns the same spent object, which immediately raises `StopIteration` -- so epochs 2 through 10 processed zero sentences, with no warning. If you're timing a run against an old benchmark from before this fix, expect roughly 10x longer per model now that all 10 epochs actually run.
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
corpus = mt.load_corpus('af')
model = mt.vectorize_stream(corpus, min_freq=5, dim=50, win=1, alg='cbow')
```
