# Data Directory

This folder contains all model evaluation results and processed data used in the Word2ManyLanguages project.

## 📁 Subfolders

- `count_evals/`  
  Evaluation results for predicting **word frequency** using the trained embedding models.

- `rep_evals/`  
  Evaluation results replicating the **original norms** used in the subs2vec paper (e.g., valence, AoA, concreteness).

- `extension_evals/`  
  Evaluation results for additional **norms datasets** not used in the original paper (e.g., familiarity, imageability).

- `formatted_data/`  
  Cleaned and merged versions of the above three datasets — prepared for use in the Shiny app and summary analyses.

- `processing_example/`  
  Contains example folder structures and scripts for setting up and processing data.  
  *(This folder includes its own README.)*

## 📝 Notes

- All evaluations were conducted using ridge regression with consistent preprocessing across languages.
- Cleaned files in `formatted_data/` are the basis for all tables, figures, and interactive visualizations in the project.