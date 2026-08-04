# %% [markdown]
# # word2manylanguages: one-language pipeline, step by step
#
# Each `# %%` block below is one step, runnable on its own (VS Code / Jupyter
# "Run Cell") so you can inspect output between steps rather than running the
# whole thing blind. Every step reuses the underlying function's own
# skip-if-exists behavior (pass overwrite=True on any call to force a redo).
#
# Set `language` and `version` in the Setup cell, then run cells top to
# bottom. tw (Traditional Chinese) is a special case handled inline below --
# see its cells' comments.
#
# version: '2018' (default -- matches every already-published DOI; only
# meaningful for the subtitles side, since Wikipedia has no dated-vintage
# concept and is shared/reused either way) or '2024' (the newer OpenSubtitles
# add-on corpus -- the only option for languages with no 2018 data at all).

# %%
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "01_corpus_preprocessing"))
sys.path.insert(0, os.path.join(HERE, "02_model_training"))
sys.path.insert(0, os.path.join(HERE, "eval_inputs"))

import corpus_preprocessing as cp
import model_training as mt
import build_counts_tokenized as bc

cp.basedir = mt.basedir = bc.basedir = HERE

language = "af"     # two-letter code, e.g. 'af'
version = "2018"    # '2018' or '2024' -- see module docstring above

# the (possibly version-suffixed) key used for every subtitles-side and
# downstream (corpus/counts/model) filename -- e.g. 'en' for 2018, 'en-2024'
# for 2024. Wikipedia-side filenames always use the bare `language`, never this.
subs_key = language if version == "2018" else f"{language}-{version}"

# %% [markdown]
# ## 1. Download raw data
# in:  (network) Wikimedia dump, OpenSubtitles
# out: raw/wikipedia-{language}.bz2
#      raw/subtitles-{subs_key}.zip

# %%
if language == "tw":
    print("tw has no real Wikipedia of its own (see cell 3's comment below) -- skipping wikipedia download.")
else:
    cp.download("wikipedia", language, version=version)

# %%
cp.download("subtitles", language, version=version)

# %% [markdown]
# ## 2. Clean + prune wikipedia (skip this pair entirely for tw -- see cell 3)
# in:  raw/wikipedia-{language}.bz2
# out: preprocessed/wikipedia-{language}-pre.zip (clean)
#      preprocessed/wikipedia-{language}-pruned.zip (prune -- document-level
#      dedup only, e.g. the same movie/article re-uploaded under a different
#      ID; never touches sentence/phrase content within or across distinct
#      documents)

# %%
if language != "tw":
    cp.clean_wikipedia(language)

# %%
if language != "tw":
    cp.prune("wikipedia", language)

# %% [markdown]
# ## 3. tw only: materialize wikipedia data from zh instead of downloading
# tw (Traditional Chinese / Taiwan) has no Wikipedia of its own -- ISO 639-1
# "tw" is Twi, an unrelated Ghanaian language. Chinese Wikipedia only exists
# as the single "zh" wiki (mixed simplified/traditional per article as each
# editor wrote it). This converts zh's already-cleaned wiki text to
# Taiwan-standard Traditional Chinese via OpenCC (script AND phrasing, e.g.
# "software" -> 軟體 not 软件/軟件), so every step after this treats tw as if
# it had legitimate wiki data all along.
# in:  preprocessed/wikipedia-zh-pruned.zip (zh's cells 2 must already be done)
# out: preprocessed/wikipedia-tw-pruned.zip

# %%
if language == "tw":
    bc.materialize_tw_wikipedia_pruned()

# %% [markdown]
# ## 4. Clean + prune subtitles
# in:  raw/subtitles-{subs_key}.zip
# out: preprocessed/subtitles-{subs_key}-pre.zip (clean)
#      preprocessed/subtitles-{subs_key}-pruned.zip (prune, document-level only)

# %%
cp.clean_subtitles(language, version=version)

# %%
cp.prune("subtitles", subs_key)

# %% [markdown]
# ## 5. Concatenate into the training corpus
# in:  preprocessed/wikipedia-{language}-pruned.zip
#      preprocessed/subtitles-{subs_key}-pruned.zip
# out: corpora/corpus-{subs_key}.txt  (one sentence per line, what
#      02_model_training actually trains on)

# %%
cp.concatenate_corpus(language, version=version)

# %% [markdown]
# ## 6. Build frequency counts (this project's own corpus, not an external mirror)
# in:  preprocessed/wikipedia-{language}-pruned.zip
#      preprocessed/subtitles-{subs_key}-pruned.zip
# out: eval_inputs/counts/dedup.{language}wiki-meta.words.unigrams.tsv.zip
#      eval_inputs/counts/dedup.{subs_key}.words.unigrams.tsv.zip

# %%
if language == "tw":
    bc.build_tw_wiki_counts()  # derives from zh, same as cell 3
else:
    bc.count_unigrams("wikipedia", language)

# %%
bc.count_unigrams("subtitles", subs_key)

# %% [markdown]
# ## 7. Train models -- 60 configs (dim: 50/100/200/300/500, window: 1-6, algo: cbow/sg)
# in:  corpora/corpus-{subs_key}.txt
# out: models/{subs_key}_{dim}_{window}_{algo}_wxd.csv.bz2 x 60
# This is the slow step. To try just one configuration first, shrink the
# sweep before calling build_models (see 02_model_training/README.md):
#   mt.dimension_list = [50]; mt.window_list = [1]; mt.algo_list = ['cbow']

# %%
mt.build_models(subs_key)

# %% [markdown]
# ## 8. Upload to Zenodo
# in:  models/{subs_key}_*_wxd.csv.bz2 (all configs trained in step 7)
# out: published Zenodo record(s) (new record, or a new version of an
#      existing DOI if this language+version already has one -- see
#      download/zenodo_upload.py's module docstring), plus new rows in
#      download/zenodo_dois.csv
# Requires ZENODO_TOKEN in .env at the repo root (scopes: deposit:write,
# deposit:actions) -- loaded into os.environ below, no shell sourcing needed.
# Split into a dry-run (no network calls, just the chunking/batching plan)
# and the real upload, same review-then-act pattern as step 10's git push --
# check the dry-run output before running the cell after it.

# %%
sys.path.insert(0, os.path.join(HERE, "download"))
import zenodo_upload as zu

env_path = os.path.join(HERE, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

zu.sync_language(subs_key, version, os.path.join(HERE, "models"), dry_run=True)

# %%
# Only run this once the dry-run plan above looks right.
zu.sync_language(subs_key, version, os.path.join(HERE, "models"), dry_run=False)

# %% [markdown]
# ### 8b. Commit the new DOI record
# in:  download/zenodo_dois.csv (appended by sync_language above)
# out: a commit + push, so the DOI history is versioned alongside the code
# that produced it -- this is the single source of truth existing_records_for()
# reads next time to decide new-record vs. new-version, and what
# zenodo_download.py reads to find a language+version's files, so it should
# never be left as an uncommitted local-only change.

# %%
print(subprocess.run(["git", "diff", "--", "download/zenodo_dois.csv"], cwd=HERE, capture_output=True, text=True).stdout)

# %%
# Only run this once the diff above looks right (new rows for subs_key/version, nothing else).
subprocess.run(["git", "add", "download/zenodo_dois.csv"], cwd=HERE, check=True)
subprocess.run(["git", "commit", "-m", f"Add {subs_key} ({version}) Zenodo DOIs"], cwd=HERE, check=True)
subprocess.run(["git", "push"], cwd=HERE, check=True)

# %% [markdown]
# ## 9. Evaluate -- score every trained model against replication norms,
# extended norms, and frequency counts.
# in:  models/{subs_key}_{dim}_{window}_{algo}_wxd.csv.bz2 x up to 60
#      eval_inputs/replication/, eval_inputs/norms/, eval_inputs/counts/
# out: eval_results/replication/{subs_key}_eval.csv
#      eval_results/norms/{subs_key}_eval.csv
#      eval_results/counts/{subs_key}_eval.csv
# `version` is passed through so 2024-vintage languages read models/counts
# from their `-2024`-suffixed files and write to non-colliding output files;
# every output row also carries a `version` column -- see
# 03_evaluation/README.md's "2018 vs. 2024 corpus vintage" section.

# %%
sys.path.insert(0, os.path.join(HERE, "03_evaluation"))
import evaluation as ev
ev.basedir = HERE

ev.evaluate_language(language, version=version)

# %% [markdown]
# ## 10. Push results to GitHub
# in:  eval_results/{replication,norms,counts}/{subs_key}_eval.csv (written above)
# out: a commit on the current branch, pushed to origin
# Not run automatically -- review what changed first (`git status`,
# `git diff --stat`), especially if you're re-running a language and
# `overwrite=True` rewrote existing eval_results rows. Run this cell only
# once you're happy with the diff.

# %%
eval_paths = [
    os.path.join(HERE, "eval_results", "replication", f"{subs_key}_eval.csv"),
    os.path.join(HERE, "eval_results", "norms", f"{subs_key}_eval.csv"),
    os.path.join(HERE, "eval_results", "counts", f"{subs_key}_eval.csv"),
]
eval_paths = [p for p in eval_paths if os.path.exists(p)]

print(subprocess.run(["git", "status", "--", *eval_paths], cwd=HERE, capture_output=True, text=True).stdout)

# %%
# Only run this once the `git status` output above looks right.
subprocess.run(["git", "add", *eval_paths], cwd=HERE, check=True)
subprocess.run(["git", "commit", "-m", f"Add {subs_key} evaluation results ({version})"], cwd=HERE, check=True)
subprocess.run(["git", "push"], cwd=HERE, check=True)
