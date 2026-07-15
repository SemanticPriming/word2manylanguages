# Stage 1: Corpus Preprocessing

Downloads raw Wikipedia and OpenSubtitles data, cleans/normalizes it, and concatenates it into a single sentence-per-line corpus file per language. This is the first stage of the Word2ManyLanguages pipeline; its output (`corpora/corpus-{language}.txt`) feeds into [`02_model_training/`](../02_model_training/).

## 📂 Contents

- `corpus_preprocessing.py` — `download`, `clean` (`clean_subtitles`/`clean_wikipedia`, followed automatically by `prune`'s document-level deduplication), `concatenate_corpus`, and the supporting text-cleaning helpers.

## 📦 Requirements

- Python 3.10+
- `lxml`, `simhash`, `requests`

## 🧽 What cleaning does

`clean(source, language)` routes to `clean_subtitles` or `clean_wikipedia` depending on `source`, then automatically runs `prune` (see [Deduplication](#-deduplication) below) on the result. Both cleaning functions end by lowercasing the text and dropping every character that isn't alphanumeric or whitespace, but they get there differently:

**`clean_subtitles`** — reads `raw/subtitles-{language}.zip` (an OpenSubtitles OPUS dump), and for every XML subtitle file under it:
1. Parses the XML and drops `<meta>` nodes, keeping just the spoken-line text (`sub_xml_to_text`).
2. Runs a sequence of regex passes (`subs_patterns`): strip any leftover XML tags, strip URLs, strip parenthetical asides (e.g. `(laughs)`), break the text into one line per sentence at `.`/`!`/`?`/`:`/`;`, turn hyphens/apostrophes/slashes into spaces, and collapse blank lines and repeated spaces.
3. Writes the result as one `.txt` file per subtitle file into `preprocessed/subtitles-{language}-pre.zip`.

**`clean_wikipedia`** — streams `raw/wikipedia-{language}.bz2` (a MediaWiki XML dump) one `<page>...</page>` article at a time (`articles`, with HTML entities unescaped), and for each article:
1. Skips it entirely if it looks like a redirect/template stub (starts with `#`) or carries `<noinclude>`/`__NOINDEX__` markup.
2. Otherwise strips deeply nested `{{...}}` template markup, tracking brace depth, and discards the article if its braces are unbalanced (`wiki_strip_circumflex`).
3. Runs a sequence of regex passes (`wiki_patterns`): strip `<ref>`/`<references>`/`<table>`/`<gallery>`/`<kml>` blocks and other tags, strip URLs, collapse `[[target|label]]` wikilinks down to just their label text and drop category/file links, drop lines that are mostly markup or timestamps, strip parentheticals, break into one line per sentence, turn hyphens/apostrophes/slashes into spaces, and collapse blank lines and repeated spaces.
4. Writes each cleaned article as `wiki-{language}-{i}.txt` into `preprocessed/wikipedia-{language}-pre.zip`.

## ▶️ Running it (Afrikaans example)

`corpus_preprocessing.py` has no `__main__` block — it's meant to be imported and driven by a script/shell that sets `basedir` first. `raw/`, `preprocessed/`, and `corpora/` live at the repo root (see [raw/README.md](../raw/README.md)), so `basedir` should point there, not at `01_corpus_preprocessing/`.

The repo already checks in a real `raw/wikipedia-af.bz2` and `raw/subtitles-af.zip` as a worked example, so you can skip `download()` and go straight to cleaning and concatenating. Run this from the repo root:

```python
import sys
sys.path.insert(0, '01_corpus_preprocessing')
import corpus_preprocessing as cp

cp.basedir = '.'  # repo root, where raw/, preprocessed/, corpora/ live

# only needed if raw/wikipedia-af.bz2 / raw/subtitles-af.zip aren't present yet
cp.download('wikipedia', 'af')
cp.download('subtitles', 'af')

cp.clean('wikipedia', 'af')   # raw/wikipedia-af.bz2 -> preprocessed/wikipedia-af-pre.zip -> preprocessed/wikipedia-af-pruned.zip
cp.clean('subtitles', 'af')  # raw/subtitles-af.zip  -> preprocessed/subtitles-af-pre.zip  -> preprocessed/subtitles-af-pruned.zip

cp.concatenate_corpus('af')   # preprocessed/*-af-pruned.zip -> corpora/corpus-af.txt
```

Every step skips and prints a message if its output file already exists; pass `overwrite=True` to force a re-run (e.g. `cp.clean('wikipedia', 'af', overwrite=True)`) — note `clean()` doesn't forward `overwrite` to the `prune()` step it triggers, so to force a re-prune call `cp.prune('wikipedia', 'af', overwrite=True)` directly.

## 🧹 Deduplication

`clean()` finishes by calling `prune(source, language)` on its own output, so deduplication always runs as part of the standard pipeline — `concatenate_corpus()` reads the pruned archives, not the raw `-pre.zip` ones. `prune()` drops near-duplicate documents (e.g. the same subtitle file re-uploaded under a different ID) using [simhash](https://pypi.org/project/simhash/):

1. Each document's text is lowercased, split into words, rejoined, and broken into overlapping 4-character shingles.
2. A document's shingles are combined into one 64-bit simhash fingerprint (`simhash.Simhash`) — a locality-sensitive hash where similar text produces fingerprints that differ in only a few bits, unlike a cryptographic hash where a single changed character scrambles the whole output.
3. All fingerprints go into a `simhash.SimhashIndex` with a Hamming-distance tolerance of `k=2`.
4. Walking documents in archive order, each not-yet-removed document has its near-duplicates (distance ≤ 2, excluding itself) looked up and marked for removal — keeping the first copy of each near-duplicate cluster and dropping the rest.
5. The kept documents are written to `preprocessed/{source}-{language}-pruned.zip`, which is what `concatenate_corpus()` consumes.

To dedupe a single already-cleaned archive without re-running `clean()`, call `prune()` directly, e.g. `cp.prune('subtitles', 'af')`.
