# corpora/

Final output of [`01_corpus_preprocessing/`](../01_corpus_preprocessing/)'s `concatenate_corpus()` step — the input to [`02_model_training/`](../02_model_training/). Contents are gitignored (see `.gitignore`) — this README stands in for the data. Regenerate by running the pipeline; see [`01_corpus_preprocessing/README.md`](../01_corpus_preprocessing/README.md) for a worked example using the checked-in `af` (Afrikaans) data.

## 📄 Files

- `corpus-{language}.txt` — one language's cleaned and deduplicated subtitle + Wikipedia text, one sentence per line, concatenated from `preprocessed/{subtitles,wikipedia}-{language}-pruned.zip`.
