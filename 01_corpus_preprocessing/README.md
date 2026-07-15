# Stage 1: Corpus Preprocessing

Downloads raw Wikipedia and OpenSubtitles data, cleans/normalizes it, and concatenates it into a single sentence-per-line corpus file per language. This is the first stage of the Word2ManyLanguages pipeline; its output (`corpora/corpus-{language}.txt`) feeds into [`02_model_training/`](../02_model_training/).

## 📂 Contents

- `corpus_preprocessing.py` — `download`, `clean` (`clean_subtitles`/`clean_wikipedia`), `prune` (optional deduplication), `concatenate_corpus`, and the supporting text-cleaning helpers.

See [`word2manylanguages_workflow.ipynb`](../word2manylanguages_workflow.ipynb) for a walk-through that drives this module alongside the other two stages.

## 📦 Requirements

- Python 3.10+
- `lxml`, `simhash`, `requests`
