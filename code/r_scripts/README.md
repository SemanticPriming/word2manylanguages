# R Scripts: Output Cleanup

This folder contains R code for post-processing model evaluation outputs to prepare them for use in the Shiny app.

## 📂 Contents

- `output_cleanup.Rmd`: R Markdown file that:

  - Loads model evaluation outputs (e.g., from Python pipeline)
  - Cleans, reshapes, and merges data
  - Saves a standardized dataset for the Shiny app
- `output_cleanup.html`: Rendered HTML report with summary outputs