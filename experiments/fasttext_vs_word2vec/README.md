# FastText vs. Word2Vec

Evidence for one methodology decision: is FastText's subword mechanism
(`min_n=3, max_n=6`, matched to subs2vec's Table 1 -- see
[`05_manuscript/manuscript.Rmd:314`](../../05_manuscript/manuscript.Rmd))
worth its measured ~1.4-1.9x slower training, across the 58 languages this
project still has to (re)train? This is a one-off investigation, not a
pipeline stage -- it doesn't touch `models/`, `eval_results/`, or any of the
real numbered pipeline folders.

## Why this exists

`02_model_training/model_training.py` trains `FastText`, not plain
`Word2Vec`. FastText decomposes every word into character n-grams and learns
+ sums their vectors jointly during training -- this isn't an optional extra
feature bolted on top of a normal word vector, it's how FastText computes a
word's vector in the first place (see the
[FastText paper](https://arxiv.org/abs/1607.04606)), which is *why* it's
slower than Word2Vec's one-free-parameter-per-word approach. `af`'s real
measured comparison ([see conversation / `results/af_2018_*.csv`](results/)):

- **Timing**: Word2Vec 1.4-1.9x faster (cbow vs. sg)
- **RSA** (do the two agree on word-similarity structure?): FastText and
  Word2Vec agree with each other (Pearson r ≈ 0.80-0.84) *more* than cbow
  and sg agree with each other within FastText alone (r ≈ 0.67)
- **Predictive power**: no consistent winner on `af` -- Word2Vec-cbow
  actually narrowly beat every FastText variant on the norms-prediction task

But `af` (Afrikaans) is a poor test case: it's unusually morphologically
*simple* for a Germanic language (it shed most of Dutch's inflection), which
is exactly the condition under which FastText's subword generalization
should matter least. The question this experiment answers is whether that
picture changes for languages where subword structure should carry real
signal -- agglutinative languages with heavy, regular morpheme-stacking,
templatic Semitic morphology, and non-Latin scripts the pipeline hasn't been
validated against for this specific question.

## Coverage plan

Chosen from the full 58-language list to span morphological typology and
writing systems while avoiding the languages flagged for schedule
reconsideration (`it`, `pl`, `de`, `fr`, `es`, `en` -- see the training-time
estimates artifact). This pilot only trains a handful of config points per
language (not the full 60-config sweep), so even Tier B/C languages are far
cheaper here than their full-sweep time estimate suggests.

### Core set -- small, run these first (`compare_lib.CORE_LANGUAGES`)

| lang | family / script | morphology |
|---|---|---|
| `af` | Germanic, Latin | unusually simple for its family -- the original pilot language, kept for a like-for-like baseline |
| `eu` | isolate, Latin | agglutinative (extreme -- unrelated to any neighbor) |
| `kk` | Turkic, Cyrillic/Latin | agglutinative |
| `ta` | Dravidian, Tamil script | agglutinative |
| `hi` | Indo-Aryan, Devanagari (abugida) | fusional/inflecting |
| `mk` | Slavic, Cyrillic | fusional |
| `th` | Kra-Dai, Thai abugida, **no whitespace word boundaries** | isolating, needs a real segmenter |
| `vi` | Austroasiatic, Latin + diacritics, tonal | isolating/analytic (near-zero inflection) |
| `ko` | Koreanic, Hangul (featural script) | agglutinative |

Between them (excluding the `af` baseline): an isolate, three agglutinative
families (Turkic, Dravidian, Koreanic), two fusional families (Indo-Aryan,
Slavic), one isolating language, and the one writing system in this project
that needs real word segmentation instead of whitespace splitting -- across
five scripts (Latin, Cyrillic, Devanagari, Thai, Hangul).

### Extended set -- optional, fills a real gap, costs more (`compare_lib.EXTENDED_LANGUAGES`)

| lang | family / script | morphology |
|---|---|---|
| `he` | Semitic, Hebrew abjad | templatic (root-and-pattern) |
| `ar` | Semitic, Arabic abjad | templatic (root-and-pattern) |
| `zh` | Sinitic, Han logographic, no spaces | isolating |

There's no small representative for abjad scripts or templatic morphology
in this project's language list -- `he`/`ar` are the only options and both
are Tier C in the full-sweep estimate. Templatic morphology (root consonants
recurring across very different surface forms) is arguably where FastText's
subword mechanism should matter *most*, so it's worth including if you can
afford the extra time, just not in the first pass.

## Usage

**Overnight (recommended):** open `compare_algorithms.ipynb`, run section 1.
`cl.run_comparison_batch()` builds each language's corpus itself if it
doesn't exist yet (download, clean, prune, concatenate -- same steps
`run_language_pipeline.ipynb` does by hand), trains and scores everything,
keeps going if one language fails (logged to `results/batch_summary.csv`
rather than stopping the run), and writes `REPORT.md` at the end.

```python
import sys
sys.path.insert(0, 'experiments/fasttext_vs_word2vec')
import compare_lib as cl

cl.basedir = '.'  # repo root
summary = cl.run_comparison_batch(cl.CORE_LANGUAGES)  # + cl.EXTENDED_LANGUAGES for full coverage
```

**One language at a time:** section 2 of the notebook, or call
`run_comparison()` directly -- useful for debugging or a quick look before
committing to an overnight run. Requires `corpora/corpus-{language}.txt` to
already exist (steps 1-5 of `../../run_language_pipeline.ipynb`).

```python
result = cl.run_comparison('eu', configs=[(50, 1), (100, 2), (200, 3), (300, 4), (500, 6)])
```

`generate_report()` (called automatically at the end of a batch run) can
also be run standalone any time to refresh `REPORT.md` after adding a
language by hand.

## What gets written

- `models/{language}/*_wxd.csv.bz2` -- trained vectors, same `wxd` shape as
  the real pipeline's models, isolated from `../../models/` (uses `{language}ft`/
  `{language}wv` filename tags, e.g. `euft_50_1_cbow_wxd.csv.bz2`).
  **Gitignored** -- these are throwaway pilot vectors, regenerable by
  rerunning the notebook, not the point of this experiment.
- `results/{language}_{version}_timing.csv` -- one row per (dim, window, alg, family)
- `results/{language}_{version}_rsa.csv` -- one row per (dim, window, alg,
  comparison) -- `comparison` is either `fasttext_vs_word2vec` or a
  same-family `cbow_vs_sg` reference point
- `results/{language}_{version}_predictive.csv` -- long-format scores from
  `03_evaluation/evaluation.py`'s real `evaluate_replication`/
  `evaluate_norms`/`evaluate_counts`, raw and L2-normalized, tagged with
  `dim`/`window`/`alg`/`family`
- `results/batch_summary.csv` -- one row per language run via
  `run_comparison_batch()`: `ok` or `failed`, and the error if it failed
- `REPORT.md` -- the human-readable rollup of everything above, regenerated
  at the end of every batch run

Everything except `models/` is meant to be committed and reviewed once the
coverage set is run -- these CSVs (and the report) are the actual evidence
this experiment exists to produce.
