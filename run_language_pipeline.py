"""
Runs the full word2manylanguages pipeline for one (language, version) pair,
one step at a time, so progress survives interruption -- every step reuses
the underlying pipeline function's own skip-if-exists behavior (pass
--overwrite to force every step to redo its work regardless).

Pipeline stages, in order (see each stage's own README for full detail):

  1. download     (network)                          -> raw/wikipedia-{language}.bz2
                                                          raw/subtitles-{language}[-{version}].zip
  2. clean+prune  01_corpus_preprocessing             -> preprocessed/wikipedia-{language}-pruned.zip
                                                          preprocessed/subtitles-{language}[-{version}]-pruned.zip
                                                          (+ intermediate -pre.zip for each)
  3. concatenate  01_corpus_preprocessing             -> corpora/corpus-{language}[-{version}].txt
  4. counts       eval_inputs (build_counts_tokenized) -> eval_inputs/counts/dedup.{language}wiki-meta.words.unigrams.tsv.zip
                                                          eval_inputs/counts/dedup.{language}[-{version}].words.unigrams.tsv.zip
  5. train        02_model_training                   -> models/{language}[-{version}]_{dim}_{window}_{algo}_wxd.csv.bz2 x60

Zenodo upload (download/zenodo_upload.py) is intentionally NOT chained in
here -- run it as a separate, explicit step after a language finishes and
you've spot-checked its output, not automatically.

version: '2018' (default -- the original OpenSubtitles+Wikipedia corpus,
matches every already-published DOI) or '2024' (the newer OpenSubtitles
add-on corpus -- the only option for languages with no 2018 data at all).
Only the *subtitles* side is version-specific; Wikipedia has no dated-vintage
concept (always "latest"), so it's downloaded/cleaned once per language and
shared across both a language's 2018 and 2024 builds -- see
corpus_preprocessing.py's download()/clean_subtitles()/concatenate_corpus()
docstrings for exactly how the version suffix threads through filenames.

tw (Traditional Chinese) is a special case with no real Wikipedia of its own
-- ISO 639-1 "tw" is Twi, an unrelated Ghanaian language (raw/wikipedia-tw.bz2
would actually be the Twi wiki; see build_tw_wiki_counts()'s docstring in
eval_inputs/build_counts_tokenized.py). This script detects language == 'tw'
and materializes its wiki-side data by OpenCC-converting zh's instead of
downloading -- zh's clean step must already be done first.

Usage:
    python run_language_pipeline.py --language af --version 2018
    python run_language_pipeline.py --language az --version 2024
    python run_language_pipeline.py --language tw --version 2018   # needs zh done first
    python run_language_pipeline.py --language af --version 2018 --steps clean,concatenate
    python run_language_pipeline.py --language af --version 2018 --overwrite
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "01_corpus_preprocessing"))
sys.path.insert(0, os.path.join(HERE, "02_model_training"))
sys.path.insert(0, os.path.join(HERE, "eval_inputs"))

import corpus_preprocessing as cp
import model_training as mt
import build_counts_tokenized as bc

cp.basedir = mt.basedir = bc.basedir = HERE


def _step(name, inputs, outputs, fn, *args, **kwargs):
    print(f"\n=== [{name}] start {time.strftime('%H:%M:%S')} ===")
    print(f"    in:  {inputs}")
    print(f"    out: {outputs}")
    t0 = time.time()
    fn(*args, **kwargs)
    print(f"=== [{name}] done in {time.time()-t0:.0f}s ===")


def _subs_key(language, version):
    """The (possibly version-suffixed) string used for every subtitles-side
    filename -- e.g. 'en' for 2018, 'en-2024' for 2024. Wikipedia-side
    filenames always use the bare `language`, never this."""
    return language if version == "2018" else f"{language}-{version}"


def download_step(language, version, overwrite):
    if language == "tw":
        print("tw: skipping wikipedia download (see module docstring) -- only fetching subtitles.")
    else:
        _step(
            "download wikipedia", "(network) Wikimedia dump", f"raw/wikipedia-{language}.bz2",
            cp.download, "wikipedia", language, version=version, overwrite=overwrite,
        )
    subs_key = _subs_key(language, version)
    _step(
        "download subtitles", "(network) OpenSubtitles", f"raw/subtitles-{subs_key}.zip",
        cp.download, "subtitles", language, version=version, overwrite=overwrite,
    )


def clean_step(language, version, overwrite):
    subs_key = _subs_key(language, version)

    if language == "tw":
        _step(
            "materialize tw wikipedia",
            "preprocessed/wikipedia-zh-pruned.zip (must already exist -- run zh's clean step first)",
            "preprocessed/wikipedia-tw-pruned.zip",
            bc.materialize_tw_wikipedia_pruned, overwrite=overwrite,
        )
    else:
        _step(
            "clean wikipedia", f"raw/wikipedia-{language}.bz2", f"preprocessed/wikipedia-{language}-pre.zip",
            cp.clean_wikipedia, language, overwrite=overwrite,
        )
        _step(
            "prune wikipedia", f"preprocessed/wikipedia-{language}-pre.zip", f"preprocessed/wikipedia-{language}-pruned.zip",
            cp.prune, "wikipedia", language, overwrite=overwrite,
        )

    _step(
        "clean subtitles", f"raw/subtitles-{subs_key}.zip", f"preprocessed/subtitles-{subs_key}-pre.zip",
        cp.clean_subtitles, language, version=version, overwrite=overwrite,
    )
    _step(
        "prune subtitles", f"preprocessed/subtitles-{subs_key}-pre.zip", f"preprocessed/subtitles-{subs_key}-pruned.zip",
        cp.prune, "subtitles", subs_key, overwrite=overwrite,
    )


def concatenate_step(language, version, overwrite):
    subs_key = _subs_key(language, version)
    _step(
        "concatenate corpus",
        f"preprocessed/wikipedia-{language}-pruned.zip, preprocessed/subtitles-{subs_key}-pruned.zip",
        f"corpora/corpus-{subs_key}.txt",
        cp.concatenate_corpus, language, version=version, overwrite=overwrite,
    )


def counts_step(language, version, overwrite):
    subs_key = _subs_key(language, version)

    if language == "tw":
        _step(
            "counts: tw wikipedia (derived from zh)", "preprocessed/wikipedia-zh-pruned.zip",
            "eval_inputs/counts/dedup.twwiki-meta.words.unigrams.tsv.zip",
            bc.build_tw_wiki_counts, overwrite=overwrite,
        )
    else:
        _step(
            "counts: wikipedia", f"preprocessed/wikipedia-{language}-pruned.zip",
            f"eval_inputs/counts/dedup.{language}wiki-meta.words.unigrams.tsv.zip",
            bc.count_unigrams, "wikipedia", language, overwrite=overwrite,
        )

    _step(
        "counts: subtitles", f"preprocessed/subtitles-{subs_key}-pruned.zip",
        f"eval_inputs/counts/dedup.{subs_key}.words.unigrams.tsv.zip",
        bc.count_unigrams, "subtitles", subs_key, overwrite=overwrite,
    )


def train_step(language, version, overwrite):
    model_key = _subs_key(language, version)
    _step(
        "train models (60 configs)", f"corpora/corpus-{model_key}.txt",
        f"models/{model_key}_{{dim}}_{{window}}_{{algo}}_wxd.csv.bz2 x 60 (dim: 50/100/200/300/500, window: 1-6, algo: cbow/sg)",
        mt.build_models, model_key, overwrite=overwrite,
    )


STEP_ORDER = ["download", "clean", "concatenate", "counts", "train"]
STEP_FUNCS = {
    "download": download_step,
    "clean": clean_step,
    "concatenate": concatenate_step,
    "counts": counts_step,
    "train": train_step,
}


def run(language, version="2018", steps=None, overwrite=False):
    steps = steps or STEP_ORDER
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        sys.exit(f"Unknown step(s) {sorted(unknown)} -- valid steps are {STEP_ORDER}")

    print(f"##### {language} ({version} corpus) -- steps: {', '.join(steps)} #####")
    for step_name in STEP_ORDER:
        if step_name in steps:
            STEP_FUNCS[step_name](language, version, overwrite)

    print(f"\n##### {language} ({version}) pipeline complete #####")
    if "train" in steps:
        print("Next (not run automatically -- inspect the models first):")
        print(f"  python download/zenodo_upload.py --language {language} --version {version} --models-dir models/ --dry-run")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, help="two-letter language code, e.g. 'af'")
    parser.add_argument("--version", default="2018", help="corpus vintage: '2018' (default) or '2024'")
    parser.add_argument("--steps", help=f"comma-separated subset of {STEP_ORDER} (default: all, in order)")
    parser.add_argument("--overwrite", action="store_true", help="force every requested step to redo its work, even if output already exists")
    args = parser.parse_args()

    steps = args.steps.split(",") if args.steps else None
    run(args.language, args.version, steps, args.overwrite)


if __name__ == "__main__":
    main()
