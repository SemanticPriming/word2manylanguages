# Stage 3: Evaluation

Evaluates the word-by-dimension models produced by [`02_model_training/`](../02_model_training/) against replication norms, extended norms datasets, and word frequency counts, then summarizes/ranks the results by predictive accuracy.

## 📂 Contents

- `evaluation.py`:
  - `loop_norms_vp` / `evaluate_norms_vp` / `predict_norms` — replicate the original van Paridon & Thompson norm predictions using ridge regression
  - `evaluate_norms` — predicts extended psycholinguistic norms datasets (valence, AoA, concreteness, etc.)
  - `evaluate_counts` / `predict_counts` — predicts word frequency counts
  - `score_vp` / `score_norms` / `score_counts` — combine and rank per-language evaluation outputs across all model configurations

Output is used by [`04_postprocessing/`](../04_postprocessing/) to prepare data for the Shiny app, and by [`06_manuscript/`](../06_manuscript/) for reported results. See [`code/word2manylanguages_workflow.ipynb`](../word2manylanguages_workflow.ipynb) for a walk-through that drives this module alongside the other two stages.

## 📦 Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `unidecode`
