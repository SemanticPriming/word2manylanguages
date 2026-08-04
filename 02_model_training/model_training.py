# Libraries
import os
from collections import Counter
import pandas as pd
from gensim.models import FastText

# Define locations
corpusdir = "corpora"
modeldir = 'models'

# Read the concatenated corpus for gensim
def load_corpus(language):
    """
    Reads corpora/corpus-{language}.txt into memory once, pre-tokenized the
    same way the old streaming `sentences` generator did (rstrip + split on
    ' ', dropping empty tokens; blank lines still yield an empty list, same
    as before, so corpus_count/training behavior is unchanged).

    build_models() calls this exactly once per language and reuses the
    result across all up-to-60 (dim, window, algo) configs. The old
    `sentences(language)` generator reopened and re-read the corpus file
    from scratch for *every* build_vocab() and every train() call -- two
    full-file passes per config, 120 total for a full sweep -- through a
    slow per-line Python readline()/split() loop. A language's corpus is
    small enough to hold in memory (a list of token lists) relative to the
    300GB+ RAM this pipeline runs on, so read it once and hand the same
    in-memory list to gensim every time instead.
    """
    path_name = os.path.join(basedir, corpusdir, f'corpus-{language}.txt')
    with open(path_name, 'r') as f:
        return [[w for w in line.rstrip().split(' ') if len(w) > 0] for line in f]

# Number of gensim training threads. Default leaves one core free for the OS/
# Jupyter kernel itself; gensim's own default (3) badly underuses a large
# machine -- override this (e.g. `mt.workers = 32`) for a bigger server.
workers = max(1, (os.cpu_count() or 1) - 1)

def count_word_freq(corpus):
    """
    Counts word frequencies across the whole corpus once, for reuse across
    every one of the up-to-60 (dim, window, algo) configs via
    build_vocab_from_freq() below -- vocabulary (which words pass min_count,
    the downsampling table, and FastText's subword-ngram buckets) depends
    only on the corpus text and min_count, never on dim/window/algo, so
    build_vocab() re-scanning the full corpus inside every config was 60
    redundant passes over the same text. Confirmed empirically against
    build_vocab()'s own scan (same vocab, same per-word counts, same
    downsampling ints, and bit-identical trained vectors under a fixed
    seed/workers=1) before switching to this path.
    """
    word_freq = Counter()
    for sentence in corpus:
        word_freq.update(sentence)
    return dict(word_freq)

# Build gensim models
def vectorize_stream(corpus, word_freq, corpus_count, min_freq=5, dim=50, win=3, alg=0):
    """
    Creates the word2vec model using gensim.
    `corpus` is a pre-tokenized list of token lists, as returned by
    load_corpus(); `word_freq`/`corpus_count` are count_word_freq()'s and
    len(corpus)'s one-time results, passed in so every config seeds its
    vocabulary from the precomputed table (build_vocab_from_freq) instead of
    re-scanning the corpus itself (build_vocab) -- see count_word_freq()'s
    docstring.
    """
    algo = 1 if alg == "sg" else 0
    print(f"Training model {dim} {win} {alg}")
    model = FastText(vector_size=dim, window=win, min_count=min_freq, sg=algo, sample=0.0001, negative=10, alpha=0.05, min_n=3, max_n=6, workers=workers)
    model.build_vocab_from_freq(word_freq, corpus_count=corpus_count)
    model.train(corpus_iterable=corpus, total_examples=corpus_count, epochs=10)

    return model

dimension_list = [50,100,200,300,500]
window_list = [1,2,3,4,5,6]
algo_list = ['cbow','sg']

def build_models(language,overwrite=False):
    """
    Loops over model requirements and uses vectorize stream to create gensim models.
    """
    configs = [
        (dim, win, alg)
        for dim in dimension_list
        for win in window_list
        for alg in algo_list
    ]

    remaining = [
        (dim, win, alg) for dim, win, alg in configs
        if overwrite or not os.path.exists(
            os.path.join(basedir, modeldir, f'{language}_{dim}_{win}_{alg}_wxd.csv.bz2')
        )
    ]
    if not remaining:
        print(f'All {len(configs)} configs for {language} already exist, and overwrite not specified. Skipping.')
        return

    print(f"Loading {language} corpus.")
    corpus = load_corpus(language)
    print(f"Counting {language} vocabulary (once, reused across all configs).")
    word_freq = count_word_freq(corpus)
    corpus_count = len(corpus)

    for dim, win, alg in configs:
        base_file_name = f'{language}_{str(dim)}_{str(win)}_{alg}'
        output_path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv.bz2')
        if os.path.exists(output_path) and not overwrite:
            print(f'File {base_file_name}_wxd.csv.bz2 exists, and overwrite not specified. Skipping.');
        else:
            print("Building model " + base_file_name)
            model = vectorize_stream(corpus, word_freq, corpus_count, 5, dim, win, alg)
            #Write down the model?
            words=list(model.wv.key_to_index)
            wordsxdims = pd.DataFrame(model.wv[words],words)
            wordsxdims.to_csv(output_path,index_label='word',compression='bz2')
