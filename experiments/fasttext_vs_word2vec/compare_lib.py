"""
Reusable pieces for comparing FastText against plain Word2Vec on one
language: timing, vector-geometry agreement (RSA), and real predictive power
via 03_evaluation/evaluation.py's own pipeline. Built to answer one
question -- is FastText's subword mechanism (min_n=3, max_n=6, matched to
subs2vec's Table 1, see 05_manuscript/manuscript.Rmd:314) worth its ~1.4-1.9x
slower training across this project's 58 remaining languages, or does plain
Word2Vec do just as well for languages whose morphology doesn't need it?

Deliberately reuses 02_model_training/model_training.py's load_corpus() /
count_word_freq() and 03_evaluation/evaluation.py's load_model() /
evaluate_replication() / evaluate_norms() / evaluate_counts() rather than
reimplementing any of it, so this comparison is evaluated exactly the same
way every real model in this project is -- not a bespoke metric that could
disagree with the actual pipeline.

Usage (see compare_algorithms.ipynb for the full walkthrough):
    import sys
    sys.path.insert(0, 'experiments/fasttext_vs_word2vec')
    import compare_lib as cl

    cl.basedir = '.'  # repo root, same convention as every other module here
    result = cl.run_comparison('eu', configs=[(50, 1), (100, 2), (200, 3), (300, 4), (500, 6)])
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from gensim.models import FastText, Word2Vec

basedir = "."
resultsdir = "experiments/fasttext_vs_word2vec/results"
expmodeldir = "experiments/fasttext_vs_word2vec/models"

# Core coverage set from README.md -- an isolate, three agglutinative
# families (Turkic, Dravidian, Koreanic), two fusional families (Indo-Aryan,
# Slavic), one isolating language, and the one script in this project that
# needs real word segmentation instead of whitespace splitting (Thai) --
# across five scripts. 'af' is included too: its corpus already exists and
# it's the language the original ad-hoc pilot was measured on, so keeping
# it in the same 5-config sweep as everything else makes it directly
# comparable instead of a one-off with different settings.
CORE_LANGUAGES = ["af", "eu", "kk", "ta", "hi", "mk", "th", "vi", "ko"]

# Fills the abjad/templatic-morphology gap the core set can't -- no small
# representative exists for either, so these are Tier C in the full-sweep
# time estimate. See README.md.
EXTENDED_LANGUAGES = ["he", "ar", "zh"]

_mt = None
_ev = None


def _modules():
    """Lazily imports model_training/evaluation using this repo's own
    basedir-relative sys.path convention -- deferred so importing this
    module doesn't require the path to already be set up."""
    global _mt, _ev
    if _mt is None:
        sys.path.insert(0, os.path.join(basedir, "02_model_training"))
        sys.path.insert(0, os.path.join(basedir, "03_evaluation"))
        import model_training as mt
        import evaluation as ev
        mt.basedir = basedir
        ev.basedir = basedir
        _mt, _ev = mt, ev
    return _mt, _ev


def train_and_time(tag, alg, cls, sg, corpus, word_freq, corpus_count, dim, win, extra, workers, epochs=10):
    """
    Trains one model (FastText or Word2Vec, cbow or sg) at (dim, win),
    seeding its vocabulary from the precomputed word_freq table (see
    model_training.py's count_word_freq() -- same bit-identical-verified
    shortcut used in the real pipeline) rather than rescanning the corpus.
    Returns (model, elapsed_seconds).
    """
    t0 = time.time()
    kwargs = dict(
        vector_size=dim, window=win, min_count=5, sample=0.0001,
        negative=10, alpha=0.05, workers=workers, sg=sg,
    )
    kwargs.update(extra)
    model = cls(**kwargs)
    model.build_vocab_from_freq(word_freq, corpus_count=corpus_count)
    model.train(corpus_iterable=corpus, total_examples=corpus_count, epochs=epochs)
    elapsed = time.time() - t0
    print(f"  {tag}_{dim}_{win}_{alg}: {elapsed:.1f}s, {len(model.wv)} words")
    return model, elapsed


def cosine_sim_matrix(vectors):
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return norm @ norm.T


def rsa(words_a, vectors_a, words_b, vectors_b, word_freq, n_words=800):
    """
    Representational similarity analysis: two independently-trained
    embedding spaces have unrelated, arbitrarily-rotated axes, so raw
    per-word cosine similarity between them is meaningless. Instead: take a
    shared sample of frequent common words, compute each model's own
    pairwise cosine-similarity matrix over that sample, then correlate the
    two matrices' flattened upper triangles -- this asks "do the two models
    agree on which words are relatively similar to which," independent of
    alignment. Returns (pearson_r, spearman_r, n_words_used).
    """
    idx_a = {w: i for i, w in enumerate(words_a)}
    idx_b = {w: i for i, w in enumerate(words_b)}
    common = [w for w in words_a if w in idx_b]
    common_sorted = sorted(common, key=lambda w: -word_freq.get(w, 0))[:n_words]
    if len(common_sorted) < 10:
        return None, None, len(common_sorted)

    vecs_a = np.array([vectors_a[idx_a[w]] for w in common_sorted])
    vecs_b = np.array([vectors_b[idx_b[w]] for w in common_sorted])
    sim_a = cosine_sim_matrix(vecs_a)
    sim_b = cosine_sim_matrix(vecs_b)
    iu = np.triu_indices(len(common_sorted), k=1)
    pear = pearsonr(sim_a[iu], sim_b[iu])
    spear = spearmanr(sim_a[iu], sim_b[iu])
    return pear.statistic, spear.statistic, len(common_sorted)


def run_predictive_eval(language, version, tag, dim, win, alg, model_dir):
    """
    Scores one already-exported model file (written by run_comparison()
    below in the same wxd.csv.bz2 shape evaluation.py's load_model()
    expects) against replication norms, extended norms, and frequency
    counts -- raw and L2-normalized -- using evaluation.py's own functions
    unmodified. Returns a long-format DataFrame, one row per (dataset,
    normalized) combo, or None if nothing scored.
    """
    mt, ev = _modules()
    prior_modeldir = ev.modeldir
    ev.modeldir = model_dir  # absolute path -> os.path.join(basedir, modeldir, ...) ignores basedir
    try:
        words, raw_vectors = ev.load_model(tag, dim, win, alg)
    finally:
        ev.modeldir = prior_modeldir
    if words is None:
        return None

    replication_norms = ev.load_replication_norms(language)
    extended_norms = ev.load_extended_norms(language)
    count_freqs = ev.load_count_freqs(language, version=version)

    rows = []
    for normalized in (False, True):
        vectors = ev.normalize_vectors(raw_vectors) if normalized else raw_vectors
        wordsXdims = pd.DataFrame(vectors)
        wordsXdims.set_index(words, inplace=True)

        for eval_type, loaded, evaluator in (
            ("replication", replication_norms, ev.evaluate_replication),
            ("norms", extended_norms, ev.evaluate_norms),
            ("counts", count_freqs, ev.evaluate_counts),
        ):
            if not loaded:
                continue
            scores = evaluator(wordsXdims, loaded)
            if scores is None:
                continue
            scores["eval_type"] = eval_type
            scores["normalized"] = normalized
            rows.append(scores)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def run_comparison(language, version="2018", configs=((50, 1), (100, 2), (200, 3), (300, 4), (500, 6)), workers=None, epochs=10, overwrite=False):
    """
    Full pipeline for one language: trains FastText + Word2Vec, cbow + sg,
    at every (dim, window) pair in `configs` (sharing one corpus load and
    one vocabulary count across all of them, same shortcut as
    model_training.py's real build_models()), times each, exports vectors,
    runs RSA between matched FastText/Word2Vec pairs, and scores every
    model through evaluation.py's real predict pipeline.

    Resumable: if a config's model file already exists in
    models/{language}/ (e.g. this language's run got interrupted partway
    through last time), that config is loaded back from disk instead of
    retrained -- its timing is pulled from this language's own prior
    _timing.csv if a recorded value is there, else left blank (it was
    trained in a run whose timing CSV never got written, e.g. the process
    died before reaching that point). Pass overwrite=True to force
    retraining everything regardless of what's already on disk.

    Every row (one model's timing, one RSA comparison, one model's
    predictive scores) is appended to its results CSV and printed as soon
    as it's computed, not batched into memory and written once at the end
    -- if this crashes partway through a language (most likely during the
    slow ridge-regression predictive step), whatever finished before the
    crash is already safely on disk and visible in the notebook, not lost.

    Requires corpora/corpus-{language}.txt to already exist -- run steps
    1-5 of run_language_pipeline.ipynb for this language first if it
    doesn't (this module deliberately doesn't redo preprocessing; see
    compare_algorithms.ipynb).

    Returns a dict with 'timing' (DataFrame), 'rsa' (DataFrame), and
    'predictive' (DataFrame) -- also written to
    experiments/fasttext_vs_word2vec/results/{language}_{version}_*.csv.
    """
    mt, ev = _modules()
    workers = workers or mt.workers
    model_dir = os.path.join(basedir, expmodeldir, language)
    results_dir = os.path.join(basedir, resultsdir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Real recorded seconds for configs that were actually trained in an
    # earlier (possibly interrupted) attempt at this exact language --
    # reused below so resumed configs don't just show up blank in timing.
    prior_timing_path = os.path.join(results_dir, f"{language}_{version}_timing.csv")
    prior_timing = {}
    if os.path.exists(prior_timing_path):
        for _, row in pd.read_csv(prior_timing_path).iterrows():
            prior_timing[(row["dim"], row["window"], row["alg"], row["family"])] = row["seconds"]

    print(f"Loading {language} corpus + vocab (shared across every config below)...")
    corpus = mt.load_corpus(language) if version == "2018" else mt.load_corpus(f"{language}-{version}")
    word_freq = mt.count_word_freq(corpus)
    corpus_count = len(corpus)
    print(f"{corpus_count} sentences, {len(word_freq)} unique tokens, workers={workers}")

    # All three result files below are appended to as each row is computed,
    # not batched into memory and written once at the end -- a crash or
    # interruption partway through a language (training, RSA, or the slow
    # ridge-regression predictive step) still leaves everything computed so
    # far safely on disk instead of losing the whole language's progress.
    timing_path = os.path.join(results_dir, f"{language}_{version}_timing.csv")
    rsa_path = os.path.join(results_dir, f"{language}_{version}_rsa.csv")
    predictive_path = os.path.join(results_dir, f"{language}_{version}_predictive.csv")

    def _append_row(path, row):
        write_header = not os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)

    # Which configs will be (re)trained fresh this call -- computed ahead of
    # the main loop so timing_path can be purged of their stale rows first.
    # Without this, overwrite=True (or retraining a config whose model file
    # got deleted) would append a new row on top of the old one instead of
    # replacing it, since timing_path otherwise accumulates *across* calls
    # (unlike rsa_path/predictive_path, which are always fully recomputed
    # and cleared every call -- timing_path deliberately isn't, so a smaller
    # earlier configs= call's rows survive a later, wider one).
    fresh_keys = set()
    for dim, win in configs:
        for alg in ("cbow", "sg"):
            for family in ("fasttext", "word2vec"):
                tag = f"{language}ft" if family == "fasttext" else f"{language}wv"
                out_path = os.path.join(model_dir, f"{tag}_{dim}_{win}_{alg}_wxd.csv.bz2")
                if overwrite or not os.path.exists(out_path):
                    fresh_keys.add((dim, win, alg, family))
    if fresh_keys and os.path.exists(timing_path):
        existing = pd.read_csv(timing_path)
        keep = existing.apply(lambda r: (r["dim"], r["window"], r["alg"], r["family"]) not in fresh_keys, axis=1)
        existing[keep].to_csv(timing_path, index=False)

    models = {}  # (dim, win, alg, family) -> {"words":..., "vectors":..., "time":...}
    timing_rows = []

    n_configs = len(configs) * 2 * 2
    i = 0
    for dim, win in configs:
        for alg, sg in (("cbow", 0), ("sg", 1)):
            for family, tag, cls, extra in (
                ("fasttext", f"{language}ft", FastText, dict(min_n=3, max_n=6)),
                ("word2vec", f"{language}wv", Word2Vec, {}),
            ):
                i += 1
                out_path = os.path.join(model_dir, f"{tag}_{dim}_{win}_{alg}_wxd.csv.bz2")
                is_fresh = not (os.path.exists(out_path) and not overwrite)
                if not is_fresh:
                    elapsed = prior_timing.get((dim, win, alg, family), float("nan"))
                    note = f" (recorded {elapsed:.1f}s)" if elapsed == elapsed else " (no prior timing recorded)"
                    print(f"  [{i}/{n_configs}] {tag}_{dim}_{win}_{alg}: already exists, skipping training{note}", flush=True)
                else:
                    model, elapsed = train_and_time(
                        tag, alg, cls, sg, corpus, word_freq, corpus_count, dim, win, extra, workers, epochs,
                    )
                    out_words = list(model.wv.key_to_index)
                    out_vectors = model.wv[out_words]
                    pd.DataFrame(out_vectors, index=out_words).to_csv(out_path, index_label="word", compression="bz2")
                    del model  # one language's 4x models per config point can add up in RAM; done with the object itself
                    print(f"  [{i}/{n_configs}] {tag}_{dim}_{win}_{alg}: trained", flush=True)

                # Always reload from the file just written/found, rather than
                # keeping a freshly-trained model's in-memory (non-casefolded)
                # vectors -- ev.load_model() casefolds + dedupes case variants
                # ("Apple"/"apple"), so a resumed config would otherwise carry
                # a slightly different vocab than a freshly-trained one in the
                # same run, silently skewing the RSA comparison between them.
                prior_modeldir = ev.modeldir
                ev.modeldir = model_dir
                try:
                    words_arr, vectors = ev.load_model(tag, dim, win, alg)
                finally:
                    ev.modeldir = prior_modeldir
                words = list(words_arr)

                models[(dim, win, alg, family)] = {"words": words, "vectors": vectors, "time": elapsed, "tag": tag}
                row = {
                    "language": language, "dim": dim, "window": win, "alg": alg,
                    "family": family, "seconds": elapsed, "n_words": len(words),
                }
                timing_rows.append(row)
                if is_fresh:
                    _append_row(timing_path, row)  # resumed rows are already on disk from a prior call -- don't duplicate

    timing_df = pd.DataFrame(timing_rows)
    # The incremental appends above are a crash-safety net (whatever
    # finished before an interruption is already on disk); this final
    # merge-and-rewrite is the correctness guarantee -- it covers edge
    # cases the append-only path can't (timing_path missing/incomplete, a
    # resumed config whose row never actually made it to disk in an earlier
    # interrupted run) without discarding rows for configs *outside*
    # this call's `configs` -- e.g. an earlier wider sweep's rows must
    # survive a later, narrower call, same as a narrower call's rows must
    # survive a later, wider one.
    call_keys = {(r["dim"], r["window"], r["alg"], r["family"]) for r in timing_rows}
    if os.path.exists(timing_path):
        prior_full = pd.read_csv(timing_path)
        outside = prior_full[~prior_full.apply(lambda r: (r["dim"], r["window"], r["alg"], r["family"]) in call_keys, axis=1)]
        pd.concat([outside, timing_df], ignore_index=True).to_csv(timing_path, index=False)
    else:
        timing_df.to_csv(timing_path, index=False)

    print("\nComputing RSA (FastText vs Word2Vec agreement, per config point + cbow-vs-sg sanity checks)...")
    if os.path.exists(rsa_path):
        os.remove(rsa_path)  # RSA is always fully recomputed this call (cheap) -- start clean, then append as we go
    rsa_rows = []
    n_rsa = len(configs) * 4  # 2 fasttext_vs_word2vec + 2 cbow_vs_sg comparisons per config point
    j = 0
    for dim, win in configs:
        for alg in ("cbow", "sg"):
            j += 1
            ft = models[(dim, win, alg, "fasttext")]
            wv = models[(dim, win, alg, "word2vec")]
            pear, spear, n = rsa(ft["words"], ft["vectors"], wv["words"], wv["vectors"], word_freq)
            row = {
                "language": language, "dim": dim, "window": win, "alg": alg,
                "comparison": "fasttext_vs_word2vec", "n_common_words": n,
                "pearson_r": pear, "spearman_r": spear,
            }
            rsa_rows.append(row)
            _append_row(rsa_path, row)
            print(f"  [{j}/{n_rsa}] RSA {dim}_{win}_{alg} fasttext_vs_word2vec: pearson_r={pear:.3f}", flush=True)
        # same-family cbow-vs-sg, as a reference point for how much RSA moves
        # just from the training objective, independent of subwords at all
        for family in ("fasttext", "word2vec"):
            j += 1
            a = models[(dim, win, "cbow", family)]
            b = models[(dim, win, "sg", family)]
            pear, spear, n = rsa(a["words"], a["vectors"], b["words"], b["vectors"], word_freq)
            row = {
                "language": language, "dim": dim, "window": win, "alg": "cbow_vs_sg",
                "comparison": family, "n_common_words": n,
                "pearson_r": pear, "spearman_r": spear,
            }
            rsa_rows.append(row)
            _append_row(rsa_path, row)
            print(f"  [{j}/{n_rsa}] RSA {dim}_{win}_cbow_vs_sg {family}: pearson_r={pear:.3f}", flush=True)
    rsa_df = pd.DataFrame(rsa_rows)

    print("\nScoring every model through evaluation.py's real predictive pipeline (the slow step -- ridge regression per dataset)...")
    if os.path.exists(predictive_path):
        os.remove(predictive_path)  # also always fully recomputed this call -- start clean, then append as we go
    pred_rows = []
    n_models = len(models)
    for k, ((dim, win, alg, family), info) in enumerate(models.items(), start=1):
        scores = run_predictive_eval(language, version, info["tag"], dim, win, alg, model_dir)
        if scores is not None:
            scores["language"] = language
            scores["dim"] = dim
            scores["window"] = win
            scores["alg"] = alg
            scores["family"] = family
            pred_rows.append(scores)
            write_header = not os.path.exists(predictive_path)
            scores.to_csv(predictive_path, mode="a", header=write_header, index=False)
        print(f"  [{k}/{n_models}] {info['tag']}_{dim}_{win}_{alg}: predictive eval done "
              f"({'no local eval data for this language' if scores is None else f'{len(scores)} rows'})", flush=True)
    if not os.path.exists(predictive_path):
        # No model produced any scoreable data (no local replication/norms/counts
        # for this language) -- touch the file anyway so its mere existence marks
        # "the predictive step ran to completion", distinguishing this from a
        # language whose predictive step never ran at all. _results_complete()
        # below relies on this to tell the two cases apart.
        open(predictive_path, "a").close()
    predictive_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    print(f"\nDone -- {results_dir}/{language}_{version}_*.csv are all up to date.")
    return {"timing": timing_df, "rsa": rsa_df, "predictive": predictive_df}


def ensure_corpus(language, version="2018"):
    """
    Builds corpora/corpus-{language}.txt if it doesn't exist yet, running
    the same steps run_language_pipeline.ipynb's cells 1-14 (download,
    clean, prune, concatenate) perform by hand -- see
    01_corpus_preprocessing/corpus_preprocessing.py. Each step's own
    overwrite=False default means an already-partially-built language
    resumes rather than redoing finished work, same as the notebook.

    Doesn't handle tw's zh-derived-wikipedia special case (see
    run_language_pipeline.ipynb cells 3/9) -- build tw's corpus by hand
    first if it's ever added to the coverage set.
    """
    sys.path.insert(0, os.path.join(basedir, "01_corpus_preprocessing"))
    import corpus_preprocessing as cp
    cp.basedir = basedir

    subs_key = language if version == "2018" else f"{language}-{version}"
    corpus_path = os.path.join(basedir, "corpora", f"corpus-{subs_key}.txt")
    if os.path.exists(corpus_path):
        return corpus_path

    print(f"  building corpus for {language} ({version})...")
    cp.download("wikipedia", language, version=version)
    cp.download("subtitles", language, version=version)
    cp.clean_wikipedia(language)
    cp.prune("wikipedia", language)
    cp.clean_subtitles(language, version=version)
    cp.prune("subtitles", subs_key)
    cp.concatenate_corpus(language, version=version)
    return corpus_path


def ensure_counts(language, version="2018"):
    """
    Builds eval_inputs/counts/{language}.subs.{version}.tsv.zip and
    {language}.wiki.2018.tsv.zip if they don't exist yet -- the same step 6
    run_language_pipeline.ipynb performs by hand (see
    eval_inputs/build_counts_tokenized.py) -- needed for
    run_predictive_eval()'s "counts" eval_type. Reuses whatever
    preprocessed/{wikipedia,subtitles}-{language}-pruned.zip ensure_corpus()
    already left on disk, so this doesn't re-download anything; call
    ensure_corpus() first if it hasn't run yet for this language.

    Doesn't handle tw's zh-derived wiki-counts special case (see
    run_language_pipeline.ipynb's build_tw_wiki_counts()) -- build tw's
    counts by hand first if it's ever added to the coverage set.
    """
    sys.path.insert(0, os.path.join(basedir, "eval_inputs"))
    import build_counts_tokenized as bc
    bc.basedir = basedir

    counts_dir = os.path.join(basedir, "eval_inputs", "counts")
    subs_path = os.path.join(counts_dir, f"{language}.subs.{version}.tsv.zip")
    wiki_path = os.path.join(counts_dir, f"{language}.wiki.2018.tsv.zip")

    if not os.path.exists(subs_path):
        print(f"  building subtitles frequency counts for {language} ({version})...")
        bc.count_unigrams("subtitles", language, version)
    if not os.path.exists(wiki_path):
        print(f"  building wikipedia frequency counts for {language}...")
        bc.count_unigrams("wikipedia", language, version)


def _results_complete(language, version, configs, model_dir, results_dir):
    """True only if every (dim, window, alg, family) model file `configs`
    implies already exists in model_dir *and* this language's rsa/predictive
    result files actually contain the corresponding data -- the real source
    of truth for whether a language needs (re)work, rather than a bare "ok"
    flag (can't tell a five-config run apart from a one-config run that
    happened to finish first) or model-files-alone (misses a language whose
    training finished but got interrupted during RSA/predictive scoring, or
    whose result CSVs were separately restored/deleted -- see af's timing.csv
    being restored from git history without touching its models/)."""
    for dim, win in configs:
        for alg in ("cbow", "sg"):
            for tag in (f"{language}ft", f"{language}wv"):
                path = os.path.join(model_dir, f"{tag}_{dim}_{win}_{alg}_wxd.csv.bz2")
                if not os.path.exists(path):
                    return False

    rsa_path = os.path.join(results_dir, f"{language}_{version}_rsa.csv")
    predictive_path = os.path.join(results_dir, f"{language}_{version}_predictive.csv")

    # predictive_path's mere existence (even 0 rows) means the predictive
    # step ran to completion last time -- see run_comparison()'s trailing
    # open(...).close() for the "no local eval data" case. Its absence means
    # that step either never ran or was interrupted before finishing.
    if not os.path.exists(predictive_path):
        return False

    # rsa.csv is written one comparison at a time as it's computed (see
    # run_comparison()) -- a run interrupted mid-RSA leaves fewer rows than
    # len(configs) * 4 (2 fasttext_vs_word2vec + 2 cbow_vs_sg per config
    # point), which is exactly the signal that this language needs a rerun.
    expected_rsa_rows = len(configs) * 4
    if not os.path.exists(rsa_path):
        return False
    try:
        actual_rsa_rows = sum(1 for _ in open(rsa_path)) - 1  # minus header
    except OSError:
        return False
    if actual_rsa_rows < expected_rsa_rows:
        return False

    return True


def run_comparison_batch(
    languages=None, version="2018",
    configs=((50, 1), (100, 2), (200, 3), (300, 4), (500, 6)),
    workers=None, epochs=10, build_corpus=True, build_counts=True, overwrite=False,
):
    """
    Runs run_comparison() for every language in `languages` (defaults to
    CORE_LANGUAGES), one at a time -- meant for an unattended overnight run.
    Builds each language's corpus first if it doesn't exist (build_corpus=True,
    the default) and its frequency-count files (build_counts=True, the
    default -- needed for run_predictive_eval()'s "counts" eval_type, cheap
    once the corpus step above has already left preprocessed/*-pruned.zip on
    disk); set either False to fail loudly on missing input instead of
    silently kicking off a download/preprocess/count-building run you didn't
    expect.

    Resumable at two levels, so rerunning after a crash or an interrupted
    overnight run doesn't redo finished work -- and so that widening
    `configs` on a later call (e.g. you first tried one config point, now
    want the full five-point sweep) correctly trains only what's missing
    rather than either redoing everything or wrongly skipping the whole
    language:
    - a language is skipped entirely (fast -- doesn't even load its corpus)
      only if *every* model file implied by the current `configs` already
      exists on disk for it *and* its rsa.csv/predictive.csv actually
      contain complete results for those configs (see _results_complete())
      -- not just because it was marked "ok" in an earlier call, and not
      just because the model files exist while the RSA/predictive step
      never finished or its result files were separately deleted/restored
    - anything not fully covered still gets a full run_comparison() call,
      which itself skips any individual (dim, window, alg, family) config
      whose model file already exists (see run_comparison()'s docstring) --
      so a language missing only its RSA/predictive data retrains no models
      at all, it just recomputes those two cheap-relative-to-training steps
    Pass overwrite=True to ignore all of the above and redo everything.

    Each language is wrapped in its own try/except so one language's
    problem (network hiccup, a language-specific preprocessing edge case,
    disk space) doesn't take down the whole run -- failures are logged and
    the batch moves on. Writes results/batch_summary.csv (language, status,
    error) and, once every language has been attempted, calls
    generate_report() so a finished overnight run leaves a readable report
    behind, not just raw CSVs to interpret by hand in the morning.
    """
    languages = list(languages) if languages is not None else list(CORE_LANGUAGES)
    results_dir = os.path.join(basedir, resultsdir)
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "batch_summary.csv")

    summary_rows = []
    for i, language in enumerate(languages, start=1):
        model_dir = os.path.join(basedir, expmodeldir, language)
        if not overwrite and _results_complete(language, version, configs, model_dir, results_dir):
            print(f"\n[{i}/{len(languages)}] {language}: models trained and rsa/predictive results complete, "
                  f"skipping (pass overwrite=True, or widen configs, to redo/extend).", flush=True)
            summary_rows.append({"language": language, "status": "ok", "error": "(skipped -- already done)"})
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            continue

        print(f"\n{'=' * 60}\n[{i}/{len(languages)}] {language}\n{'=' * 60}", flush=True)
        try:
            if build_corpus:
                ensure_corpus(language, version)
            if build_counts:
                ensure_counts(language, version)
            run_comparison(language, version=version, configs=configs, workers=workers, epochs=epochs, overwrite=overwrite)
            summary_rows.append({"language": language, "status": "ok", "error": ""})
        except Exception as e:
            print(f"  FAILED: {language}: {e}", flush=True)
            summary_rows.append({"language": language, "status": "failed", "error": str(e)})
            continue

        # write the summary after every language, not just at the end, so a
        # run that gets interrupted overnight still leaves a partial record
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    print("\n\nBatch summary:")
    print(summary_df.to_string(index=False))

    try:
        generate_report()
    except Exception as e:
        print(f"Report generation failed (raw CSVs in results/ are still there): {e}")

    return summary_df


def _markdown_table(df):
    """Minimal DataFrame -> GitHub-flavored-markdown table, no tabulate
    dependency (not installed in this project's environment)."""
    df = df.reset_index()
    df.columns = [
        " ".join(str(p) for p in col if str(p) != "") if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float) and np.isnan(v):
                vals.append("")  # pivot's cross-product cell that never applies (e.g. a
                                  # cbow_vs_sg row has no fasttext_vs_word2vec column value)
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report():
    """
    Reads every results/*_{timing,rsa,predictive}.csv (and batch_summary.csv,
    if present) written so far and writes REPORT.md -- a human-readable
    summary of the FastText-vs-Word2Vec evidence gathered across however
    many languages have been run. Safe to call standalone (e.g. to refresh
    the report after adding one more language by hand), not just from the
    end of run_comparison_batch().
    """
    results_dir = os.path.join(basedir, resultsdir)
    report_path = os.path.join(basedir, "experiments", "fasttext_vs_word2vec", "REPORT.md")

    def load_all(kind):
        paths = sorted(glob.glob(os.path.join(results_dir, f"*_{kind}.csv")))
        dfs = []
        for p in paths:
            try:
                dfs.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                # predictive.csv in particular can be genuinely empty for a
                # language with no local replication/norms/counts data to
                # score against (see run_predictive_eval()) -- not a bug,
                # just nothing to report for that language/kind.
                pass
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    timing = load_all("timing")
    rsa_df = load_all("rsa")
    predictive = load_all("predictive")
    summary_path = os.path.join(results_dir, "batch_summary.csv")
    batch_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    languages = sorted(timing["language"].unique()) if len(timing) else []

    lines = [
        "# FastText vs. Word2Vec -- evidence report",
        "",
        f"Auto-generated by `compare_lib.generate_report()`. Covers {len(languages)} language(s): "
        f"{', '.join(f'`{l}`' for l in languages) if languages else '(none yet)'}.",
        "",
        "See `README.md` for methodology and the full coverage plan.",
        "",
    ]

    if batch_summary is not None:
        lines += ["## Run summary", ""]
        lines.append(_markdown_table(batch_summary.set_index("language")))
        lines.append("")

    if len(timing):
        lines += ["## Overall: timing", ""]
        p = timing.pivot_table(index=["language", "dim", "window", "alg"], columns="family", values="seconds")
        p["word2vec_speedup"] = p["fasttext"] / p["word2vec"]
        by_lang = p.groupby("language")["word2vec_speedup"].mean().round(2)
        lines.append("Mean Word2Vec speedup over FastText, by language (>1 = word2vec faster):")
        lines.append("")
        lines.append(_markdown_table(by_lang.to_frame("word2vec_speedup")))
        lines.append("")
        lines.append(f"**Overall mean speedup across all languages run so far: {by_lang.mean():.2f}x**")
        lines.append("")

    if len(rsa_df):
        lines += ["## Overall: vector geometry (RSA)", ""]
        lines.append(
            "Pearson r between FastText's and Word2Vec's word-similarity structure, "
            "vs. each family's own cbow-vs-sg agreement (a same-family reference point -- "
            "if fasttext_vs_word2vec is *lower* than either cbow_vs_sg column for a language, "
            "that's a language where the algorithm choice reshapes geometry more than the "
            "training objective does, unlike what af showed)."
        )
        lines.append("")
        g = rsa_df.groupby(["language", "comparison"])["pearson_r"].mean().round(3).unstack()
        lines.append(_markdown_table(g))
        lines.append("")

    if len(predictive):
        lines += ["## Overall: predictive power", ""]
        lines.append("Mean r on normalized vectors, by language x eval_type x family:")
        lines.append("")
        norm_only = predictive[predictive["normalized"] == True]
        g = norm_only.groupby(["language", "eval_type", "family"])["r"].mean().round(4).unstack()
        lines.append(_markdown_table(g))
        lines.append("")

    lines += ["## Per-language detail", ""]
    for lang in languages:
        lines.append(f"### `{lang}`")
        lines.append("")
        lt = timing[timing["language"] == lang]
        lr = rsa_df[rsa_df["language"] == lang] if len(rsa_df) else pd.DataFrame()
        lp = predictive[predictive["language"] == lang] if len(predictive) else pd.DataFrame()

        if len(lt):
            lines.append("**Timing (seconds):**")
            lines.append("")
            t = lt.pivot_table(index=["dim", "window", "alg"], columns="family", values="seconds").round(1)
            lines.append(_markdown_table(t))
            lines.append("")
        if len(lr):
            lines.append("**RSA (Pearson r):**")
            lines.append("")
            r = lr.pivot_table(index=["dim", "window", "alg"], columns="comparison", values="pearson_r").round(3)
            lines.append(_markdown_table(r))
            lines.append("")
        if len(lp):
            lines.append("**Predictive power (normalized vectors, mean r by eval_type x family):**")
            lines.append("")
            norm_only = lp[lp["normalized"] == True]
            pr = norm_only.groupby(["eval_type", "family"])["r"].mean().round(4).unstack()
            lines.append(_markdown_table(pr))
            lines.append("")
        else:
            lines.append(
                "**Predictive power:** no local replication norms, extended norms, or frequency "
                "counts for this language yet -- see `03_evaluation/evaluation.py`'s "
                "`load_replication_norms`/`load_extended_norms`/`load_count_freqs`. Timing and RSA "
                "above are still valid; nothing to score predictive power against until that data "
                "exists (e.g. run `eval_inputs/build_counts_tokenized.py` for frequency counts)."
            )
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {report_path}")
    return report_path
