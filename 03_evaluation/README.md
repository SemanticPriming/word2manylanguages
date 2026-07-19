# Stage 3: Evaluation

Evaluates the word-by-dimension models produced by [`02_model_training/`](../02_model_training/) against replication norms, extended norms datasets, and word frequency counts, producing predictive accuracy scores per model configuration. This is the last stage of the Word2ManyLanguages pipeline; its output (`eval_results/`) is what [`05_manuscript/`](../05_manuscript/) reads.

## 📂 Contents

- `evaluation.py`:
  - `load_model` — loads a single trained model's word-by-dimension matrix; returns `(None, None)` and logs a skip if that dim/window/algorithm combination was never trained; words are casefolded so they join cleanly against ground-truth words below
  - `normalize_vectors` — L2-normalizes a loaded model's vectors, for the with/without-normalization comparison
  - `load_replication_norms` / `load_extended_norms` / `load_count_freqs` — load and clean each ground-truth dataset once per language (not once per model); all three casefold their word index to match `load_model`
    - `load_extended_norms` reads `eval_inputs/datasets_norms.csv` (see [`eval_inputs/build_datasets_norms.py`](../eval_inputs/build_datasets_norms.py)) to know which file/column pairs exist for a language — the catalog is already filtered to mean-valued columns for the constructs this project predicts (valence, arousal, dominance, concreteness, familiarity, imageability, aoa, emotion, sensory), so no column filtering happens here. `lang_aliases` covers the handful of `code2lang` names that don't literally match the language token in the catalog (e.g. `farsi` vs. `persian`); `norms_encoding_overrides` covers a few source files that predate UTF-8 and need a specific 8-bit encoding to read correctly.
    - `load_replication_norms` predicts **every** non-`word` column in its files, unlike `load_extended_norms` — see [`eval_inputs/replication/README.md`](../eval_inputs/replication/README.md) and `eval_inputs/datasets_replication.csv`.
  - `predict` — the shared ridge regression helper: joins vectors to targets on the (casefolded) word index, then for each target column runs `RepeatedKFold` cross-validated `Ridge` regression, penalizing the score for words the model has no vector for; `predict_norms`/`predict_counts` just fix `label_col` for their respective output shape
  - `evaluate_replication` / `evaluate_norms` / `evaluate_counts` — score one already-loaded model against one already-loaded ground-truth set via `predict`
  - `append_scores` — appends a scores `DataFrame` to a per-language output file, writing the header only the first time
  - `evaluate_language` — the driver: loops over every model configuration for a language, loads each model once, runs all three evaluation types with both raw and L2-normalized vectors, and appends results as it goes; set `overwrite=True` to force re-running a language that already has output

## 📦 Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `unidecode`

## 🎯 What evaluation does

For each `(dim, window, algorithm)` model configuration for a language, `evaluate_language`:

1. Loads the model's word-by-dimension matrix (`load_model`), skipping configurations that were never trained.
2. Runs it twice — once as-is, once L2-normalized (`normalize_vectors`) — through all three evaluation types:
   - **Replication** (`evaluate_replication`): predicts every column of every `eval_inputs/replication/{lang}-*.tsv` file, replicating van Paridon & Thompson's original subs2vec evaluation.
   - **Extended norms** (`evaluate_norms`): predicts the mean-valued target-construct columns cataloged in `eval_inputs/datasets_norms.csv` for every `eval_inputs/norms/*.csv` file that covers this language.
   - **Counts** (`evaluate_counts`): predicts word frequency from `eval_inputs/counts/dedup.{lang}*.tsv[.zip]`.
3. Appends each type's scores to `eval_results/{replication,norms,counts}/{lang}_eval.csv` (`append_scores`), tagged with the source model config and whether vectors were normalized.

All three ground-truth loaders and `load_model` casefold their word index, so predictions aren't silently dropped by a case mismatch between a model's vocabulary and a dataset's word column.

`eval_results/` is what [`05_manuscript/`](../05_manuscript/) reads; if `manuscript.Rmd`'s import path doesn't match this stage's actual output layout, that's a manuscript-side wiring issue, not something to fix here.

## ▶️ Running it (Afrikaans example)

`evaluation.py` has no `__main__` block either — same pattern as stages 1 and 2: import it and set `basedir` to the repo root, where `models/`, `eval_inputs/`, and `eval_results/` live.

The repo checks in real Afrikaans ground truth for all three evaluation types — `models/af_50_1_cbow_wxd.csv.bz2` (see [`02_model_training/README.md`](../02_model_training/README.md) for training the rest of the sweep), `eval_inputs/norms/Luniewska2016.csv` (the one LAB dataset with an Afrikaans column), and `eval_inputs/counts/dedup.af*.tsv.zip` — except replication norms, since no dataset in subs2vec's replication set covers Afrikaans (`load_replication_norms('af')` just returns an empty list, and `evaluate_language` skips that step gracefully).

```python
import sys
sys.path.insert(0, '03_evaluation')
import evaluation as ev

ev.basedir = '.'  # repo root, where models/, eval_inputs/, and eval_results/ live

ev.evaluate_language('af')
# -> eval_results/norms/af_eval.csv    (Luniewska2016.csv's aoa_mean_afrikaans)
# -> eval_results/counts/af_eval.csv   (dedup.af / dedup.afwiki-meta frequency counts)
# eval_results/replication/af_eval.csv is not created -- no replication data for af
```

Each output file is skipped (with a printed message) if it already exists; pass `overwrite=True` to force a re-run: `ev.evaluate_language('af', overwrite=True)`.

To evaluate a single already-loaded model against a single already-loaded dataset directly, without the full `evaluate_language` sweep:

```python
words, vectors = ev.load_model('af', dim=50, win=1, alg='cbow')
wordsXdims = ev.pd.DataFrame(vectors, index=words)

norms = ev.load_extended_norms('af')
scores = ev.evaluate_norms(wordsXdims, norms)
```
