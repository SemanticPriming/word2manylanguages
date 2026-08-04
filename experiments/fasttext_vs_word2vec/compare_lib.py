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


def run_comparison(language, version="2018", configs=((50, 1), (100, 2), (200, 3), (300, 4), (500, 6)), workers=None, epochs=10):
    """
    Full pipeline for one language: trains FastText + Word2Vec, cbow + sg,
    at every (dim, window) pair in `configs` (sharing one corpus load and
    one vocabulary count across all of them, same shortcut as
    model_training.py's real build_models()), times each, exports vectors,
    runs RSA between matched FastText/Word2Vec pairs, and scores every
    model through evaluation.py's real predict pipeline.

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

    print(f"Loading {language} corpus + vocab (shared across every config below)...")
    corpus = mt.load_corpus(language) if version == "2018" else mt.load_corpus(f"{language}-{version}")
    word_freq = mt.count_word_freq(corpus)
    corpus_count = len(corpus)
    print(f"{corpus_count} sentences, {len(word_freq)} unique tokens, workers={workers}")

    models = {}  # (dim, win, alg, family) -> {"words":..., "vectors":..., "time":...}
    timing_rows = []

    for dim, win in configs:
        for alg, sg in (("cbow", 0), ("sg", 1)):
            for family, tag, cls, extra in (
                ("fasttext", f"{language}ft", FastText, dict(min_n=3, max_n=6)),
                ("word2vec", f"{language}wv", Word2Vec, {}),
            ):
                model, elapsed = train_and_time(
                    tag, alg, cls, sg, corpus, word_freq, corpus_count, dim, win, extra, workers, epochs,
                )
                words = list(model.wv.key_to_index)
                vectors = model.wv[words]
                models[(dim, win, alg, family)] = {"words": words, "vectors": vectors, "time": elapsed, "tag": tag}
                out_path = os.path.join(model_dir, f"{tag}_{dim}_{win}_{alg}_wxd.csv.bz2")
                pd.DataFrame(vectors, index=words).to_csv(out_path, index_label="word", compression="bz2")
                timing_rows.append({
                    "language": language, "dim": dim, "window": win, "alg": alg,
                    "family": family, "seconds": elapsed, "n_words": len(words),
                })
                del model  # one language's 4x models per config point can add up in RAM; done with the object itself

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(os.path.join(results_dir, f"{language}_{version}_timing.csv"), index=False)

    print("\nComputing RSA (FastText vs Word2Vec agreement, per config point + cbow-vs-sg sanity checks)...")
    rsa_rows = []
    for dim, win in configs:
        for alg in ("cbow", "sg"):
            ft = models[(dim, win, alg, "fasttext")]
            wv = models[(dim, win, alg, "word2vec")]
            pear, spear, n = rsa(ft["words"], ft["vectors"], wv["words"], wv["vectors"], word_freq)
            rsa_rows.append({
                "language": language, "dim": dim, "window": win, "alg": alg,
                "comparison": "fasttext_vs_word2vec", "n_common_words": n,
                "pearson_r": pear, "spearman_r": spear,
            })
        # same-family cbow-vs-sg, as a reference point for how much RSA moves
        # just from the training objective, independent of subwords at all
        for family in ("fasttext", "word2vec"):
            a = models[(dim, win, "cbow", family)]
            b = models[(dim, win, "sg", family)]
            pear, spear, n = rsa(a["words"], a["vectors"], b["words"], b["vectors"], word_freq)
            rsa_rows.append({
                "language": language, "dim": dim, "window": win, "alg": "cbow_vs_sg",
                "comparison": family, "n_common_words": n,
                "pearson_r": pear, "spearman_r": spear,
            })
    rsa_df = pd.DataFrame(rsa_rows)
    rsa_df.to_csv(os.path.join(results_dir, f"{language}_{version}_rsa.csv"), index=False)

    print("\nScoring every model through evaluation.py's real predictive pipeline...")
    pred_rows = []
    for (dim, win, alg, family), info in models.items():
        scores = run_predictive_eval(language, version, info["tag"], dim, win, alg, model_dir)
        if scores is None:
            continue
        scores["language"] = language
        scores["dim"] = dim
        scores["window"] = win
        scores["alg"] = alg
        scores["family"] = family
        pred_rows.append(scores)
    predictive_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    predictive_df.to_csv(os.path.join(results_dir, f"{language}_{version}_predictive.csv"), index=False)

    print(f"\nDone -- wrote timing/rsa/predictive CSVs to {results_dir}/{language}_{version}_*.csv")
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


def run_comparison_batch(
    languages=None, version="2018",
    configs=((50, 1), (100, 2), (200, 3), (300, 4), (500, 6)),
    workers=None, epochs=10, build_corpus=True,
):
    """
    Runs run_comparison() for every language in `languages` (defaults to
    CORE_LANGUAGES), one at a time -- meant for an unattended overnight run.
    Builds each language's corpus first if it doesn't exist (build_corpus=True,
    the default; set False to fail loudly on a missing corpus instead of
    silently kicking off a download/preprocess run you didn't expect).

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

    summary_rows = []
    for i, language in enumerate(languages, start=1):
        print(f"\n{'=' * 60}\n[{i}/{len(languages)}] {language}\n{'=' * 60}", flush=True)
        try:
            if build_corpus:
                ensure_corpus(language, version)
            run_comparison(language, version=version, configs=configs, workers=workers, epochs=epochs)
            summary_rows.append({"language": language, "status": "ok", "error": ""})
        except Exception as e:
            print(f"  FAILED: {language}: {e}", flush=True)
            summary_rows.append({"language": language, "status": "failed", "error": str(e)})
            continue

        # write the summary after every language, not just at the end, so a
        # run that gets interrupted overnight still leaves a partial record
        pd.DataFrame(summary_rows).to_csv(os.path.join(results_dir, "batch_summary.csv"), index=False)

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
        if not paths:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

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

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {report_path}")
    return report_path
