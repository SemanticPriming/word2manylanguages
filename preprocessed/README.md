# preprocessed/

Intermediate output of [`01_corpus_preprocessing/`](../01_corpus_preprocessing/)'s `clean()` step, built from `raw/`. Contents are gitignored (see `.gitignore`) — this README stands in for the data. Regenerate by running the pipeline; see [`01_corpus_preprocessing/README.md`](../01_corpus_preprocessing/README.md) for a worked example using the checked-in `af` (Afrikaans) data.

## 📄 Files

For each `{source}` (`subtitles` or `wikipedia`) and `{language}` (ISO 639-1 code, e.g. `af`):

- `{source}-{language}-pre.zip` — one cleaned `.txt` file per source document (subtitle file or Wikipedia article), written by `clean_subtitles`/`clean_wikipedia`.
- `{source}-{language}-pruned.zip` — the same documents with near-duplicates removed by `prune()` (simhash-based document-level deduplication, run automatically as the last step of `clean()`). This is what `concatenate_corpus()` reads to build `corpora/corpus-{language}.txt`.
