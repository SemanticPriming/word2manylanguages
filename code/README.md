# Code Overview

## 📁 `manuscript/`

- Contains the manuscript and supporting materials for the paper. This includes the compiled PDF, source files (e.g., .Rmd or .qmd), and any figures or tables generated during analysis.

## 📁 `python_scripts/`

This folder includes a **reproducible implementation** of the full workflow:

- Preprocessing corpora from Wikipedia and OpenSubtitles
- Training word embedding models using fastText via Gensim
- Evaluating model performance against psycholinguistic norms
- Outputting results in standardized formats

> 🔁 These scripts can be used to replicate the full pipeline for any of the 59 languages in the study.

## 📁 `r_scripts/`

This folder contains R code used during the **manuscript preparation phase** to:

- Combine and summarize evaluation datasets across multiple runs

## 📁 `shiny/`

The Shiny app in this folder provides an **interactive interface** for exploring:

- Model performance across languages, corpora, and hyperparameters
- The effect of embedding configurations on predictive accuracy

You can launch the app locally using R:

```r
shiny::runApp("code/shiny/") # assuming you are in the main Rproject folder 
```

📦 Reproducibility Notes

- The Python pipeline was tested with Python 3.10 and Gensim 3.8.3
- R code was written for R 4.4.2
- All datasets and model outputs referenced are available via the linked Zenodo repositories
