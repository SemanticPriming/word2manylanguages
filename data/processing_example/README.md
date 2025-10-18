# Example Folder Structure for Processing Pipeline

This folder outlines the expected directory structure for running the full Word2ManyLanguages data processing and model evaluation pipeline. Each folder corresponds to a different stage of the workflow.

## 📁 Folder Descriptions

- `data/`  
  Raw corpus files downloaded from OpenSubtitles and Wikipedia.  
  These are not cleaned or tokenized yet.

- `preprocessed/`  
  Cleaned and normalized versions of the raw corpora.  
  XML, markup, and duplicate text removed.

- `corpora/`  
  Final concatenated corpus for each language, used as input to the model training script.  
  Each file has one sentence per line.

- `models/`  
  Output of the fastText training step.  
  Contains `.csv` word-by-dimension matrices for each parameter combination.

- `datasets/`  
  Norms and lexical property datasets used for evaluating the trained embeddings (e.g., valence, AoA, concreteness).

- `evals/`  
  Output from the ridge regression evaluation.  
  Each file stores the predictive accuracy of a model configuration against one dataset.

- `scores/`  
  Summary tables combining evaluation results across models and datasets for each language.  
  Used to identify top-performing model settings.

## 📦 Notes

- This structure is expected automatically by the Python pipeline - you may edit it in the python_scripts folder.
- If you are running the pipeline manually or modifying it, follow this structure for compatibility with existing scripts.