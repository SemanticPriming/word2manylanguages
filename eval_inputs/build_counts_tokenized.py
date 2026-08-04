"""
Builds unigram frequency counts for every language directly from this
project's own cleaned/deduplicated corpus, rather than relying on
download/README.md's section 3 (eval_inputs/counts/, mirrored from van
Paridon & Thompson's subs2vec frequency_source/) -- so the frequency
baseline is always self-consistent with whatever corpus actually trained
that language's embeddings, and doesn't depend on trusting an external
pipeline's own (unaudited) cleaning history.

Most languages split on whitespace fine (_tokenize_whitespace, the default).
Japanese, Chinese, Traditional Chinese, and Thai don't use whitespace to
separate words at all, so they need a real segmenter -- see TOKENIZE_FUNCS.

Reads preprocessed/{subtitles,wikipedia}-{language}-pruned.zip -- the same
deduplicated per-document archives 01_corpus_preprocessing's
concatenate_corpus() reads, except kept apart by source instead of merged
into corpora/corpus-{language}.txt. Output is named
{language}.subs.{version}.tsv (subtitles) and {language}.wiki.2018.tsv
(wikipedia -- always tagged 2018 regardless of `version`, since Wikipedia
has no dated-vintage concept and is shared/reused across corpus versions;
see corpus_preprocessing.py's version param). Uses a two-column (unigram,
unigram_freq) tsv format, zip-compressed, so evaluation.py's
load_count_freqs reads it with no changes.

Requires per-language segmenters:
    pip install fugashi[unidic-lite] jieba pythainlp opencc

tw (Traditional Chinese / Taiwan) is a special case with no Wikipedia of its
own to download -- see build_tw_wiki_counts()'s docstring.

Usage (basedir set the same way as 01_corpus_preprocessing/corpus_preprocessing.py):
    import sys
    sys.path.insert(0, 'eval_inputs')
    import build_counts_tokenized as bc

    bc.basedir = '.'
    bc.build_counts('ja')   # -> eval_inputs/counts/ja.subs.2018.tsv.zip
                             #    eval_inputs/counts/ja.wiki.2018.tsv.zip
"""
import os
import zipfile
from collections import Counter

basedir = "."
processdir = "preprocessed"
countdir = os.path.join("eval_inputs", "counts")

# Lazily-constructed, cached per language -- segmenter startup (loading a
# dictionary) is expensive enough to matter across many documents.
_taggers = {}


# MeCab's C++ parser segfaults on a single parse call above ~193,357
# characters (bisected empirically -- not controllable via the `-b` input
# buffer flag). Wikipedia articles routinely exceed that as one blob, so
# chunk defensively; comfortably under the observed threshold.
_JA_MAX_CHARS = 100_000

def _tokenize_ja(text):
    import fugashi
    # NOT _taggers.setdefault("ja", fugashi.Tagger()) -- setdefault evaluates
    # its default argument unconditionally, so that would construct a fresh
    # Tagger (reloading the ~260MB MeCab dictionary) on every single call.
    if "ja" not in _taggers:
        _taggers["ja"] = fugashi.Tagger()
    tagger = _taggers["ja"]
    tokens = []
    # split on the corpus's own sentence-per-line breaks first (semantically
    # correct for MeCab either way), then hard-chunk any line still too long
    for line in text.split("\n"):
        for start in range(0, max(len(line), 1), _JA_MAX_CHARS):
            chunk = line[start:start + _JA_MAX_CHARS]
            tokens.extend(word.surface for word in tagger(chunk))
    return tokens


def _tokenize_zh(text):
    import jieba
    return list(jieba.cut(text))


def _tokenize_th(text):
    from pythainlp.tokenize import word_tokenize
    return word_tokenize(text)


def _tokenize_whitespace(text):
    """
    Default for every language whose script already separates words with
    whitespace (i.e. everything except the entries in TOKENIZE_FUNCS below).
    Matches how the rest of the pipeline treats corpus text -- 02_model_training's
    sentences() class splits on whitespace the same way.
    """
    return text.split()


TOKENIZE_FUNCS = {
    "ja": _tokenize_ja,
    "zh": _tokenize_zh,
    "tw": _tokenize_zh,  # Traditional Chinese -- same jieba tokenizer as zh
    "th": _tokenize_th,
}


def count_unigrams(source, language, version="2018", overwrite=False, read_language=None, convert=None):
    """
    Tokenizes preprocessed/{source}-{read_language or subs_key}-pruned.zip
    with the language-appropriate segmenter (or a plain whitespace split for
    languages not in TOKENIZE_FUNCS) and writes unigram counts to
    eval_inputs/counts/{language}.subs.{version}.tsv.zip (subtitles) or
    eval_inputs/counts/{language}.wiki.2018.tsv.zip (wikipedia, always
    tagged 2018 regardless of `version` -- see module docstring) -- built
    from this project's own cleaned corpus rather than relying on
    download/README.md's downloaded frequency_source/ mirror, so the
    frequency baseline always matches whatever corpus actually trained the
    embeddings for that language.

    language is always the bare two-letter code (never version-suffixed).
    version selects which subtitles vintage to read/tag ('2018' or '2024',
    same convention as corpus_preprocessing.py's subs_key); ignored for the
    wikipedia side's naming, which is always the shared 2018 dump.
    read_language: read a *different* language's preprocessed archive than
        the one being labeled/output -- used for tw's Wikipedia counts,
        which have no Wikipedia of their own (see build_tw_wiki_counts()).
    convert: optional text -> text transform applied before tokenizing (e.g.
        OpenCC simplified-to-traditional conversion for tw).
    """
    tokenize = TOKENIZE_FUNCS.get(language, _tokenize_whitespace)

    if source == "subtitles":
        subs_key = language if version == "2018" else f"{language}-{version}"
        read_key = read_language or subs_key
        stem = f"{language}.subs.{version}"
    else:
        read_key = read_language or language
        stem = f"{language}.wiki.2018"

    input_path = os.path.join(basedir, processdir, f"{source}-{read_key}-pruned.zip")
    tsv_name = f"{stem}.tsv"
    output_path = os.path.join(basedir, countdir, tsv_name + ".zip")

    if os.path.exists(output_path) and not overwrite:
        print(f"File {tsv_name}.zip exists, and overwrite not specified. Skipping.")
        return

    counts = Counter()
    with zipfile.ZipFile(input_path, "r") as archive:
        names = archive.namelist()
        print(f"Tokenizing {len(names)} {language} {source} files.")
        for name in names:
            text = archive.read(name).decode("utf-8", errors="replace")
            if convert is not None:
                text = convert(text)
            counts.update(token for token in tokenize(text) if token and not token.isspace())

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as out:
        lines = ["unigram\tunigram_freq"]
        lines.extend(f"{word}\t{freq}" for word, freq in counts.most_common())
        out.writestr(tsv_name, "\n".join(lines) + "\n")

    print(f"Wrote {len(counts)} unigrams to {tsv_name}.zip")


_opencc_converters = {}


def build_tw_wiki_counts(overwrite=False):
    """
    tw (Traditional Chinese / Taiwan) has no Wikipedia of its own to
    download -- ISO 639-1 "tw" is Twi, an unrelated Ghanaian language
    (raw/wikipedia-tw.bz2 in this repo really is the Twi wiki; don't use it
    here). Chinese Wikipedia only exists as the single "zh" wiki, mixing
    simplified/traditional script per article as each editor happened to
    write it. tw's wiki counts are instead built from zh's already-cleaned
    Wikipedia text (preprocessed/wikipedia-zh-pruned.zip), converted to
    Taiwan-standard Traditional Chinese -- script AND phrasing, e.g.
    "software" becomes 軟體 (Taiwan usage) not 软件/軟件 (mainland) -- via
    OpenCC's s2twp config, then tokenized with jieba same as zh.

    Unlike Wikipedia, OpenSubtitles' zh_tw subtitles ARE independently
    authored in Traditional Chinese already (a different fansub community
    than zh_cn's), so tw's subtitles need no conversion -- just call
    count_unigrams('subtitles', 'tw') directly for those.
    """
    import opencc
    if "s2twp" not in _opencc_converters:
        _opencc_converters["s2twp"] = opencc.OpenCC("s2twp")
    converter = _opencc_converters["s2twp"]
    count_unigrams(
        "wikipedia", "tw", overwrite=overwrite,
        read_language="zh", convert=converter.convert,
    )


def materialize_tw_wikipedia_pruned(overwrite=False):
    """
    Writes preprocessed/wikipedia-tw-pruned.zip by OpenCC-converting every
    document in preprocessed/wikipedia-zh-pruned.zip to Traditional Chinese
    (same s2twp config as build_tw_wiki_counts() -- see that docstring for
    why tw has no real Wikipedia of its own). This is the *training corpus*
    counterpart to build_tw_wiki_counts(): once this file exists,
    01_corpus_preprocessing's concatenate_corpus('tw') and 02_model_training's
    build_models('tw') work completely unmodified, same as any language with
    genuine Wikipedia data -- they never need to know tw's wiki side is
    derived rather than downloaded.
    """
    import opencc
    input_path = os.path.join(basedir, processdir, "wikipedia-zh-pruned.zip")
    output_path = os.path.join(basedir, processdir, "wikipedia-tw-pruned.zip")

    if os.path.exists(output_path) and not overwrite:
        print("File wikipedia-tw-pruned.zip exists, and overwrite not specified. Skipping.")
        return

    if "s2twp" not in _opencc_converters:
        _opencc_converters["s2twp"] = opencc.OpenCC("s2twp")
    converter = _opencc_converters["s2twp"]

    with zipfile.ZipFile(input_path, "r") as src, zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as out:
        names = src.namelist()
        print(f"Converting {len(names)} zh wikipedia documents to Traditional Chinese for tw's corpus.")
        for name in names:
            text = src.read(name).decode("utf-8", errors="replace")
            out.writestr(name, converter.convert(text))

    print("Wrote preprocessed/wikipedia-tw-pruned.zip")


def build_counts(language, version="2018", overwrite=False):
    """
    Builds both the subtitle and wikipedia unigram count files for one
    language -- the pair evaluation.py's load_count_freqs expects.
    """
    count_unigrams("subtitles", language, version, overwrite=overwrite)
    count_unigrams("wikipedia", language, version, overwrite=overwrite)
