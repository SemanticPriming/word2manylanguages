# %% [markdown]
# # FastText vs. Word2Vec: evidence report
#
# Trains FastText and plain Word2Vec (cbow + sg) at a handful of (dim, window)
# points, sharing a single corpus load and vocabulary count across every model
# (see `compare_lib.py`'s `run_comparison()`), and compares them three ways:
# 1. **timing** -- wall-clock seconds per config
# 2. **RSA** -- do FastText and Word2Vec agree on which words are similar to
#    which (representational similarity analysis over a shared word sample),
#    independent of the two spaces' unrelated, arbitrarily-rotated axes
# 3. **predictive power** -- scored through `03_evaluation/evaluation.py`'s
#    own real pipeline (replication norms, extended norms, frequency counts),
#    not a bespoke metric
#
# This exists because FastText's subword n-grams (`min_n=3, max_n=6`, matched
# to subs2vec's Table 1 -- see `05_manuscript/manuscript.Rmd:314`) make it
# ~1.4-1.9x slower to train than Word2Vec (measured on `af`), which compounds
# badly across 58 remaining languages x 60 configs. `af` alone isn't a fair
# test of whether that cost is worth it, though -- Afrikaans is unusually
# morphologically *simple* for its family, which is exactly the case where
# FastText's subword generalization should matter least. `README.md`'s
# coverage set spans agglutinative, fusional, isolating, and templatic
# morphology, and Latin, Cyrillic, Devanagari, Thai, and Hangul scripts, to
# see whether the tradeoff looks different for languages where subwords
# should actually help.
#
# **Two ways to run this notebook:**
# - **Overnight, the whole coverage set at once** (section 1 below) --
#   `cl.run_comparison_batch()` builds each language's corpus if needed,
#   trains and scores everything, keeps going if one language fails, and
#   writes `REPORT.md` at the end. This is the one to kick off before bed.
# - **One language at a time** (section 2) -- for debugging or a quick look
#   at a single language before committing to an overnight run. Requires
#   `corpora/corpus-{language}.txt` to already exist (steps 1-5 of
#   `run_language_pipeline.ipynb`).

# %%
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import compare_lib as cl

cl.basedir = REPO_ROOT

version = "2018"    # '2018' or '2024'

# A diagonal sample across the full (dim, window) grid rather than all 30
# combinations -- five points span dim=50..500 and window=1..6 while
# keeping the per-language cost small. Edit freely; more points = more
# confidence, at roughly linear extra cost. Used by both sections below.
configs = [(50, 1), (100, 2), (200, 3), (300, 4), (500, 6)]

# %% [markdown]
# ## 1. Overnight: run the whole coverage set
# in:  nothing extra needed -- builds each language's corpus itself if it
#      doesn't exist yet (download, clean, prune, concatenate; skipped for
#      languages already prepared, same as `run_language_pipeline.ipynb`)
# out: `models/{language}/*_wxd.csv.bz2` per language (gitignored -- see
#      README.md), `results/{language}_{version}_{timing,rsa,predictive}.csv`,
#      `results/batch_summary.csv`, and `REPORT.md`
#
# Runs `cl.CORE_LANGUAGES` by default (`af`, `eu`, `kk`, `ta`, `hi`, `mk`,
# `th`, `vi`, `ko`). Add `cl.EXTENDED_LANGUAGES` (`he`, `ar`, `zh`) too if you
# have the time budget for it -- see README.md's coverage table for why
# they're pricier. One language failing (network hiccup, missing corpus
# dependency, disk space) doesn't stop the rest; check `results/batch_summary.csv`
# or the printed summary at the end for anything that needs a rerun.

# %%
languages = cl.CORE_LANGUAGES  # or cl.CORE_LANGUAGES + cl.EXTENDED_LANGUAGES for full coverage
summary = cl.run_comparison_batch(languages, version=version, configs=configs)

# %% [markdown]
# ### View the generated report

# %%
from IPython.display import Markdown, display

report_path = os.path.join(REPO_ROOT, "experiments", "fasttext_vs_word2vec", "REPORT.md")
with open(report_path) as f:
    display(Markdown(f.read()))

# %% [markdown]
# ## 2. One language at a time (manual / debugging)
# Skip this section for the normal overnight workflow -- it's for testing a
# single language interactively before committing it to a batch run, or for
# rerunning just one language by hand.
#
# Set `language` below first. Requires `corpora/corpus-{language}.txt` to
# already exist.
# in: `corpora/corpus-{language}.txt` (built by `run_language_pipeline.ipynb` steps 1-5)

# %%
language = "eu"     # two-letter code -- see README.md's coverage table

corpus_key = language if version == "2018" else f"{language}-{version}"
corpus_path = os.path.join(REPO_ROOT, "corpora", f"corpus-{corpus_key}.txt")
if not os.path.exists(corpus_path):
    raise FileNotFoundError(
        f"{corpus_path} doesn't exist yet -- run steps 1-5 of run_language_pipeline.ipynb "
        f"for language='{language}' first (through cp.concatenate_corpus(), nothing further needed)."
    )
print(f"Found {corpus_path} ({os.path.getsize(corpus_path) / 1e6:.1f}MB) -- ready.")

# %% [markdown]
# ### 2.1 Run the comparison for this one language
# in:  the corpus confirmed above
# out: `experiments/fasttext_vs_word2vec/models/{language}/*_wxd.csv.bz2`
#      (fasttext + word2vec, cbow + sg, at every config point -- kept fully
#      separate from the real `models/` directory)
#      `experiments/fasttext_vs_word2vec/results/{language}_{version}_{timing,rsa,predictive}.csv`
#
# This is the slow step -- roughly `len(configs) * 4` model trainings
# (FastText + Word2Vec, cbow + sg), each comparable in cost to one config of
# the real pipeline. Five config points is a fraction of the real 60-config
# sweep's time for this language.

# %%
result = cl.run_comparison(language, version=version, configs=configs)

# %% [markdown]
# ### 2.2 Look at this language's results
# Same three angles the overnight report aggregates: timing, RSA
# (vector-geometry agreement), and predictive power.

# %%
import pandas as pd

print("--- timing: mean seconds by family/alg, and word2vec speedup ---")
timing = result["timing"]
print(timing.groupby(["family", "alg"])["seconds"].mean().round(1))
pivot = timing.pivot_table(index=["dim", "window", "alg"], columns="family", values="seconds")
pivot["word2vec_speedup"] = pivot["fasttext"] / pivot["word2vec"]
print()
print(pivot.round(2))

# %%
print("--- RSA: fasttext_vs_word2vec agreement vs. each family's own cbow-vs-sg agreement ---")
rsa = result["rsa"]
print(rsa.groupby(["comparison", "alg"])[["pearson_r", "spearman_r"]].mean().round(3))

# %%
print("--- predictive power: mean r / r-squared by eval_type x family x alg (normalized vectors only) ---")
pred = result["predictive"]
normalized_only = pred[pred["normalized"] == True]
print(normalized_only.groupby(["eval_type", "family", "alg"])[["r", "r-squared"]].mean().round(4))

# %% [markdown]
# ## 3. Aggregate across whatever's been run so far
# Reads every `results/*_timing.csv` / `*_rsa.csv` / `*_predictive.csv` on
# disk (from either section above) into three combined tables -- the same
# data `REPORT.md` is generated from, if you want to work with it directly
# instead of reading the rendered report.

# %%
import glob

results_dir = os.path.join(REPO_ROOT, "experiments", "fasttext_vs_word2vec", "results")

def load_all(kind):
    paths = sorted(glob.glob(os.path.join(results_dir, f"*_{kind}.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

all_timing = load_all("timing")
all_rsa = load_all("rsa")
all_predictive = load_all("predictive")

print(f"{all_timing['language'].nunique() if len(all_timing) else 0} language(s) so far: "
      f"{sorted(all_timing['language'].unique()) if len(all_timing) else []}")

if len(all_timing):
    print()
    print("--- word2vec speedup by language ---")
    p = all_timing.pivot_table(index=["language", "dim", "window", "alg"], columns="family", values="seconds")
    p["word2vec_speedup"] = p["fasttext"] / p["word2vec"]
    print(p.groupby("language")["word2vec_speedup"].mean().round(2))

if len(all_predictive):
    print()
    print("--- predictive power by language x family (normalized vectors, mean r across eval types) ---")
    norm_only = all_predictive[all_predictive["normalized"] == True]
    print(norm_only.groupby(["language", "family"])["r"].mean().round(4))
