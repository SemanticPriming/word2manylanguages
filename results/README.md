# Results Directory

This folder contains the final model evaluation results for the full 59-language run — the outputs of [`03_evaluation/`](../03_evaluation/). For a runnable example of the pipeline that *produces* results like these, see [`raw/README.md`](../raw/README.md) — the `raw/`, `preprocessed/`, `corpora/`, `models/`, `eval_inputs/`, and `eval_results/` folders at the repository root hold a single-language (Afrikaans) worked example.

`formatted_data/` below was produced by an R aggregation step (previously `04_postprocessing/`) that has since been removed pending a rewrite of `03_evaluation/` to output pre-formatted data directly; until that rewrite lands, `count_evals/`, `rep_evals/`, and `extension_evals/` cannot be re-combined into `formatted_data/` from scratch.

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