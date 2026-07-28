"""
Builds unigram frequency counts for languages that download/README.md's
section 3 (eval_inputs/counts/, mirrored from van Paridon & Thompson's
subs2vec frequency_source/) has no data for at all: Japanese, Chinese, Thai.
None of the three use whitespace to separate words, so the counts can't be
produced by splitting on whitespace like the rest of the pipeline assumes --
they need an actual segmenter first. Add one to TOKENIZE_FUNCS to cover a
new language.

Reads preprocessed/{subtitles,wikipedia}-{language}-pruned.zip -- the same
deduplicated per-document archives 01_corpus_preprocessing's
concatenate_corpus() reads, except kept apart by source instead of merged
into corpora/corpus-{language}.txt, matching how the downloaded counts are
themselves split into dedup.{language}.words.unigrams.tsv (subtitles) and
dedup.{language}wiki-meta.words.unigrams.tsv (wikipedia). Output uses the
same two-column (unigram, unigram_freq) tsv format, zip-compressed the same
way, so evaluation.py's load_count_freqs reads it with no changes.

Requires per-language segmenters:
    pip install fugashi[unidic-lite] jieba pythainlp

Usage (basedir set the same way as 01_corpus_preprocessing/corpus_preprocessing.py):
    import sys
    sys.path.insert(0, 'eval_inputs')
    import build_counts_tokenized as bc

    bc.basedir = '.'
    bc.build_counts('ja')   # -> eval_inputs/counts/dedup.ja.words.unigrams.tsv.zip
                             #    eval_inputs/counts/dedup.jawiki-meta.words.unigrams.tsv.zip
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


def _tokenize_ja(text):
    import fugashi
    tagger = _taggers.setdefault("ja", fugashi.Tagger())
    return [word.surface for word in tagger(text)]


def _tokenize_zh(text):
    import jieba
    return list(jieba.cut(text))


def _tokenize_th(text):
    from pythainlp.tokenize import word_tokenize
    return word_tokenize(text)


TOKENIZE_FUNCS = {
    "ja": _tokenize_ja,
    "zh": _tokenize_zh,
    "th": _tokenize_th,
}


def count_unigrams(source, language, overwrite=False):
    """
    Tokenizes preprocessed/{source}-{language}-pruned.zip with the
    language-appropriate segmenter and writes unigram counts to
    eval_inputs/counts/dedup.{language}[wiki-meta].words.unigrams.tsv.zip.
    """
    if language not in TOKENIZE_FUNCS:
        raise ValueError(
            f"No tokenizer registered for '{language}'. "
            f"Add one to TOKENIZE_FUNCS in this file."
        )
    tokenize = TOKENIZE_FUNCS[language]

    input_path = os.path.join(basedir, processdir, f"{source}-{language}-pruned.zip")
    stem = language if source == "subtitles" else f"{language}wiki-meta"
    tsv_name = f"dedup.{stem}.words.unigrams.tsv"
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
            counts.update(token for token in tokenize(text) if token and not token.isspace())

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as out:
        lines = ["unigram\tunigram_freq"]
        lines.extend(f"{word}\t{freq}" for word, freq in counts.most_common())
        out.writestr(tsv_name, "\n".join(lines) + "\n")

    print(f"Wrote {len(counts)} unigrams to {tsv_name}.zip")


def build_counts(language, overwrite=False):
    """
    Builds both the subtitle and wikipedia unigram count files for one
    language -- the pair evaluation.py's load_count_freqs expects.
    """
    count_unigrams("subtitles", language, overwrite=overwrite)
    count_unigrams("wikipedia", language, overwrite=overwrite)
