# %% [markdown]
# # word2manylanguages: batch pipeline (download + train), multiple languages
#
# Loops `languages` below through steps 1-7 of `run_language_pipeline.py`
# (download, clean, prune, concatenate corpus, build frequency counts, train
# models) for each language in turn. One language failing (network hiccup,
# missing corpus dependency, disk space) doesn't stop the rest -- failures
# are printed and the batch moves on, same pattern as
# experiments/fasttext_vs_word2vec/compare_lib.py's run_comparison_batch().
#
# Deliberately stops before step 8 (Zenodo upload) and step 10 (git push) --
# those stay manual, one language at a time via run_language_pipeline.py,
# so their dry-run/diff review gates never get skipped. Once this batch
# finishes, review models/ and go through steps 8-10 per language by hand.
#
# `mt.build_models()` now defaults to training Word2Vec (see
# experiments/fasttext_vs_word2vec/REPORT.md for why); pass overwrite=True
# below to replace a language's existing FastText files.
#
# Usage (from the repo root, inside tmux for a long run):
#   python3 -u run_language_pipeline_batch.py \
#       2>&1 | tee -a language_pipeline_batch.log

# %%
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "01_corpus_preprocessing"))
sys.path.insert(0, os.path.join(HERE, "02_model_training"))
sys.path.insert(0, os.path.join(HERE, "eval_inputs"))

import corpus_preprocessing as cp
import model_training as mt
import build_counts_tokenized as bc

cp.basedir = mt.basedir = bc.basedir = HERE

# Override mt.workers here for a bigger server, e.g.: mt.workers = 32

languages = ["ka", "lt", "lv", "bg"]   # two-letter codes -- everything with both wikipedia-*.bz2 and subtitles-*.zip already in raw/
version = "2018"      # '2018' or '2024' -- see run_language_pipeline.py's module docstring
overwrite = False      # pass True to replace existing (e.g. FastText) model files


def run_one_language(language, version="2018", overwrite=False):
    """Steps 1-7 of run_language_pipeline.py for a single language."""
    subs_key = language if version == "2018" else f"{language}-{version}"

    # 1. download
    if language != "tw":
        cp.download("wikipedia", language, version=version)
    cp.download("subtitles", language, version=version)

    # 2. clean + prune wikipedia (skipped for tw -- see step 3)
    if language != "tw":
        cp.clean_wikipedia(language)
        cp.prune("wikipedia", language)

    # 3. tw only: materialize wikipedia data from zh instead of downloading
    if language == "tw":
        bc.materialize_tw_wikipedia_pruned()

    # 4. clean + prune subtitles
    cp.clean_subtitles(language, version=version)
    cp.prune("subtitles", subs_key)

    # 5. concatenate into the training corpus
    cp.concatenate_corpus(language, version=version)

    # 6. build frequency counts
    if language == "tw":
        bc.build_tw_wiki_counts()
    else:
        bc.count_unigrams("wikipedia", language, version)
    bc.count_unigrams("subtitles", language, version)

    # 7. train models (word2vec by default -- see mt.build_models's docstring)
    mt.build_models(subs_key, overwrite=overwrite)


# %%
results = {}
for language in languages:
    print(f"\n{'=' * 60}\n{language}\n{'=' * 60}", flush=True)
    try:
        run_one_language(language, version=version, overwrite=overwrite)
        results[language] = "ok"
    except Exception as e:
        print(f"  FAILED: {language}: {e}", flush=True)
        results[language] = f"failed: {e}"
        continue

print("\n\nBatch summary:")
for language, status in results.items():
    print(f"  {language}: {status}")

print(
    "\nDownload+train done for languages listed above. Next: review models/, "
    "then run steps 8-10 of run_language_pipeline.py per language by hand "
    "(Zenodo upload + git push, with their dry-run/diff review gates)."
)
