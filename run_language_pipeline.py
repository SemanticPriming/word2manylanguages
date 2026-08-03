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
# ## 8. Upload to Zenodo (separate script, not run automatically -- inspect
# the models first). See download/zenodo_upload.py's module docstring.
#
#   python download/zenodo_upload.py --language {language} --version {version} \
#       --models-dir models/ --dry-run
