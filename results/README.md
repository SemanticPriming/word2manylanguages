# Results Directory

This folder contains the final model evaluation results for the full 59-language run — the outputs of [`03_evaluation/`](../03_evaluation/) and [`04_postprocessing/`](../04_postprocessing/). For a runnable example of the pipeline that *produces* results like these, see [`data/README.md`](../data/README.md) — the `data/`, `preprocessed/`, `corpora/`, `models/`, `datasets/`, `evals/`, and `scores/` folders at the repository root hold a single-language (Afrikaans) worked example.

## 📁 Subfolders

- `count_evals/`  
  Evaluation results for predicting **word frequency** using the trained embedding models.

- `rep_evals/`  
  Evaluation results replicating the **original norms** used in the subs2vec paper (e.g., valence, AoA, concreteness).

- `extension_evals/`  
  Evaluation results for additional **norms datasets** not used in the original paper (e.g., familiarity, imageability).

- `formatted_data/`  
  Cleaned and merged versions of the above three datasets — prepared for use in the Shiny app and summary analyses.

## 📝 Notes

- All evaluations were conducted using ridge regression with consistent preprocessing across languages.
- Cleaned files in `formatted_data/` are the basis for all tables, figures, and interactive visualizations in the project.