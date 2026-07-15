# Stage 3: Evaluation

Evaluates the word-by-dimension models produced by [`02_model_training/`](../02_model_training/) against replication norms, extended norms datasets, and word frequency counts, producing predictive accuracy scores per model configuration.

Each model is loaded from disk **exactly once** and evaluated against all three ground-truth types (with both raw and L2-normalized vectors, to compare) before moving on to the next model — models are large files, so this avoids re-reading each one three times. Results are written incrementally, one file per language per evaluation type (`eval_results/replication/{lang}_eval.csv`, `eval_results/norms/{lang}_eval.csv`, `eval_results/counts/{lang}_eval.csv`), so all three follow the same shape instead of the previous mix of one-file-per-model-config, overwrite, and append behaviors.

## 📂 Contents

- `evaluation.py`:
  - `load_model` — loads a single trained model's word-by-dimension matrix; returns `(None, None)` and logs a skip if that dim/window/algorithm combination was never trained
  - `normalize_vectors` — L2-normalizes a loaded model's vectors, for the with/without-normalization comparison
  - `load_replication_norms` / `load_extended_norms` / `load_count_freqs` — load and clean each ground-truth dataset once per language (not once per model)
  - `evaluate_replication` / `evaluate_norms` / `evaluate_counts` — score one already-loaded model against one already-loaded ground-truth set, via the shared `predict` (ridge regression) helper
  - `evaluate_language` — the driver: loops over every model configuration for a language, loads each once, runs all three evaluation types, and appends results as it goes; set `overwrite=True` to force re-running a language that already has output

Output (`eval_results/`) previously fed a `results/` folder used by [`05_manuscript/`](../05_manuscript/), but that folder held output from the old (pre-rewrite) evaluation shape and has been set aside pending this module's planned rewrite to output pre-formatted data directly. See [`word2manylanguages_workflow.ipynb`](../word2manylanguages_workflow.ipynb) for a walk-through that drives this module alongside the other two stages.

> `05_manuscript/manuscript.Rmd` still reads from `../results/formatted_data/...`, which doesn't currently exist — that import will need updating once this rewrite lands.

## 📦 Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `unidecode`
