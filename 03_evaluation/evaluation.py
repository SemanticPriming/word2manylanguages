# Libraries
import bz2
import numpy as np
import os
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.utils
import unicodedata
from unidecode import unidecode


# Define directories
modeldir = 'models'
evaldir = 'eval_results'
datasetsdir = "eval_inputs"
countdir = "counts"
normsdir = "norms"
replicationdir = "replication"
normscatalog = "datasets_norms.csv"

dimension_list = [50,100,200,300,500]
window_list = [1,2,3,4,5,6]
algo_list = ['cbow','sg']

# Load one trained model at a time 
def load_model(lang, dim, win, alg):
    """
    Loads a single trained word-by-dimension model file. Transparently reads
    either a plain .csv or a bz2-compressed .csv.bz2 (as downloaded from
    storage) without decompressing to disk first.
    Returns (words, vectors) as numpy arrays, or (None, None) if this
    dim/window/algorithm combination was never trained.
    """
    base_file_name = f'{lang}_{dim}_{win}_{alg}'
    plain_path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv')
    compressed_path = plain_path + '.bz2'

    if os.path.exists(plain_path):
        opener, input_path = open, plain_path
    elif os.path.exists(compressed_path):
        opener, input_path = bz2.open, compressed_path
    else:
        print(f'Model {base_file_name} not found, skipping.')
        return None, None

    with opener(input_path, 'rt', encoding='utf-8') as vecfile:
        # skip header
        next(vecfile)
        # initialize arrays
        vectors = np.zeros((10000000, dim))
        words = np.empty(10000000, dtype=object)
        i = -1
        for i, line in enumerate(vecfile):
            # Limit to 10 million, although it looks like 7.5 million is the largest
            if i >= 10000000:
                break
            rowentries = line.rstrip('\n').split(',')
            words[i] = rowentries[0].casefold()
            vectors[i] = rowentries[1:dim + 1]

        # truncate empty part of arrays, if necessary
        vectors = vectors[:i + 1]
        words = words[:i + 1]

    # the training corpus isn't casefolded, so words that only differ in
    # case (e.g. "Apple"/"apple") get separate vectors here; since words[]
    # above is already casefolded, those collapse to duplicate entries,
    # which would otherwise multiply rows on the join in predict(). Average
    # each casefolded word's vectors down to one row.
    if len(words) != len(set(words)):
        deduped = pd.DataFrame(vectors, index=words).groupby(level=0).mean()
        words = deduped.index.to_numpy()
        vectors = deduped.to_numpy()

    return words, vectors

def normalize_vectors(vectors):
    """L2-normalizes each word's vector to unit length."""
    return vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)

# Shared ridge regression prediction
def predict(vectors, targets, alpha=1.0, label_col='norm', fallback_index=None):
    """
    Ridge regression function to use the embeddings to predict target values
    (psycholinguistic norms, replication norms, or frequency counts).
    `fallback_index` (currently only set for counts, see load_count_freqs) is
    an alternate, Unicode-transliterated key tried only for words that don't
    match `vectors` under `targets`'s own index, so a lossy fallback form
    never displaces a real, direct match.
    """
    cols = targets.columns.values
    df = targets.join(vectors, how='inner')

    if fallback_index is not None:
        unmatched = ~targets.index.isin(df.index)
        if unmatched.any():
            fallback_targets = targets[unmatched].copy()
            fallback_targets.index = fallback_index[unmatched]
            fallback_df = fallback_targets.join(vectors, how='inner')
            if len(fallback_df) > 0:
                df = pd.concat([df, fallback_df])

    # compensate for missing ys somehow
    total = len(targets)
    missing = len(targets) - len(df)
    penalty = (total - missing) / total
    print(f'missing vectors for {missing} out of {total} words')
    df = sklearn.utils.shuffle(df)  # shuffle is important for unbiased results on ordered datasets!

    model = sklearn.linear_model.Ridge(alpha=alpha)  # use ridge regression models
    n_splits = 5
    cv = sklearn.model_selection.RepeatedKFold(n_splits=n_splits, n_repeats=10)

    # compute crossvalidated prediction scores
    scores = []
    for col in cols:
        # set dependent variable and calculate 10-fold mean fit/predict scores
        df_subset = df.loc[:, vectors.columns.values]  # use .loc[] so copy is created and no setting with copy warning is issued
        # some norms files mark missing values with non-numeric placeholders
        # (e.g. '.', '…..') that aren't caught by na_values at load time;
        # coerce to numeric here so those become NaN and get dropped below
        df_subset[col] = pd.to_numeric(df[col], errors='coerce')
        df_subset = df_subset.dropna()  # drop NaNs for this specific y
        if len(df_subset) < n_splits:
            # RepeatedKFold needs at least n_splits samples; some norm/replication
            # columns have very few non-missing values once joined to a language's
            # vocabulary, so skip rather than let cross_val_score raise
            print(f'skipping {col}: only {len(df_subset)} words have both a vector and a value, need at least {n_splits}')
            continue
        x = df_subset[vectors.columns.values]
        y = df_subset[col]
        cv_scores = sklearn.model_selection.cross_val_score(model, x, y, cv=cv)
        median_score = np.median(cv_scores)
        penalized_score = median_score * penalty
        ars = np.sqrt(penalized_score) if penalized_score > 0 else 0
        rs = np.sqrt(median_score) if median_score > 0 else 0
        scores.append({
            label_col: col,
            'adjusted r': ars,
            'adjusted r-squared': penalized_score,
            'r-squared': median_score,
            'r': rs
        })
    return pd.DataFrame(scores)

def predict_norms(vectors, norms, alpha=1.0):
    return predict(vectors, norms, alpha, label_col='norm')

def predict_counts(vectors, freqs, alpha=1.0):
    return predict(vectors, freqs, alpha, label_col='var', fallback_index=freqs.attrs.get('fallback_index'))

# Load ground-truth evaluation datasets once per language
def load_replication_norms(lang):
    """Loads every replication norms file for this language once."""
    norms_path = os.path.join(basedir, datasetsdir, replicationdir)
    loaded = []
    for norms_fname in os.listdir(norms_path):
        if norms_fname.startswith(lang):
            norms = pd.read_csv(os.path.join(norms_path, norms_fname), sep='\t', comment='#')
            norms = norms.set_index('word')
            # lowercase to match load_model's casefolded vector words, so the
            # join in predict() doesn't silently drop case-mismatched rows
            norms.index = norms.index.str.casefold()
            loaded.append((norms_fname, norms))
    return loaded

# Map two-letter language code to language name.  Some data files have "word",
# and others have "word_{language}" so we need to try both.
code2lang = {'af':'afrikaans',
             'ar':'arabic',
             'bg':'bulgarian',
             'bn':'bengali',
             'br':'breton',
             'bs':'bosnian',
             'ca':'catalan',
             'cs':'czech',
             'da':'danish',
             'de':'german',
             'el':'greek',
             'en':'english',
             'eo':'esperanto',
             'es':'spanish',
             'et':'estonian',
             'eu':'basque',
             'fa':'farsi',
             'fi':'finnish',
             'fr':'french',
             'gl':'galacian',
             'he':'hebrew',
             'hi':'hindi',
             'hr':'croatian',
             'hu':'hungarian',
             'hy':'armenian',
             'id':'indonesian',
             'is':'icelandic',
             'it':'italian',
             'ja':'japanese',
             'ka':'georgian',
             'kk':'kazakh',
             'ko':'korean',
             'lt':'lithuanian',
             'lv':'latvian',
             'mk':'macedonian',
             'ml':'malayalam',
             'ms':'maylay',
             'nl':'dutch',
             'no':'norwegian',
             'pl':'polish',
             'pt':'portuguese',
             'ro':'romanian',
             'ru':'russian',
             'si':'sinhalese',
             'sk':'slovak',
             'sl':'slovenian',
             'sq':'albanian',
             'sr':'serbian',
             'sv':'swedish',
             'ta':'tamil',
             'te':'telugu',
             'th':'thai',
             'tl':'tagalog',
             'tr':'turkish',
             'tw':'taiwanese',
             'uk':'ukrainian',
             'ur':'urdu',
             'vi':'vietnamese',
             'zh':'chinese'}

# A few code2lang names don't match the literal language tokens found in
# eval_inputs/datasets_norms.csv's `language` column (built from each
# dataset's word_{language} column name, spelled as that paper's authors
# wrote it) -- these are the mismatches that actually occur in the data.
lang_aliases = {
    'farsi': {'farsi', 'persian'},
    'galacian': {'galacian', 'galician'},
    'maylay': {'maylay', 'malay'},
    'chinese': {'chinese', 'chinese_simplified', 'chinese_traditional'},
}

# A handful of LAB source files predate UTF-8 becoming universal and were
# saved in an OS-specific 8-bit encoding; everything else is plain UTF-8,
# optionally with a BOM (utf-8-sig strips that transparently, and reads
# identically to plain utf-8 when there's no BOM). Verified by decoding each
# file with a few candidate encodings and checking the result against known
# words in that language -- see conversation history / git blame for how
# these were identified, in case more turn up as new datasets are added.
norms_encoding_overrides = {
    'Eilola2010.csv': 'mac_roman',
    'Stadthagen-Gonzalez2017.csv': 'latin-1',
    'Stadthagen-Gonzalez2017a.csv': 'latin-1',
    'Kremer2011_de.csv': 'latin-1',
}

def load_extended_norms(lang):
    """
    Loads and resolves the word column for every extended norms file that
    has at least one target-construct mean column for this language, using
    the pre-built catalog at eval_inputs/datasets_norms.csv (see
    eval_inputs/build_datasets_norms.py) to know which file/column pairs to
    read. The catalog is already filtered to mean-valued columns for the
    constructs we predict (valence, arousal, dominance, concreteness,
    familiarity, imageability, aoa, emotion, sensory), so no further column
    filtering is needed here.
    """
    catalogpath = os.path.join(basedir, datasetsdir, normscatalog)
    catalog = pd.read_csv(catalogpath)
    langname = code2lang[lang]
    accepted = lang_aliases.get(langname, {langname})
    match = catalog[catalog['language'].fillna('').apply(
        lambda cell: bool(accepted & set(cell.split('|'))))]
    if len(match) == 0:
        return []

    loaded = []
    for langfile, group in match.groupby('dataset'):
        datapath = os.path.join(basedir, datasetsdir, normsdir, langfile)
        try:
            encoding = norms_encoding_overrides.get(langfile, 'utf-8-sig')
            norms = pd.read_csv(datapath, sep=',', comment='#', na_values=['-', '–'],
                                 encoding=encoding)

            # Get the column that has the words in it.  It might just be word, or
            # it might be word_{language_name}
            if 'word' in norms.columns:
                wordcol = 'word'
            else:
                wordcol = 'word_' + langname

            check = norms.columns
            if wordcol not in check:
                # There are some other special cases
                if wordcol + '_simple' in check:
                    wordcol = wordcol + '_simple'
                elif wordcol + '_uk' in check:
                    wordcol = wordcol + '_uk'
                elif lang == 'fa' and 'word_persian' in check:
                    wordcol = 'word_persian'
                elif lang == 'zh':
                    for candidate in ('word_chinese_simplified', 'word_chinese_traditional'):
                        if candidate in check:
                            wordcol = candidate
                            break

            # Catalog already tells us exactly which mean columns this
            # dataset/language pair has -- just select them.
            cols = [c for c in dict.fromkeys(group['variable_original']) if c in check]
            if wordcol not in check or not cols:
                continue

            norms = norms[[wordcol] + cols]
            norms.set_index(wordcol, inplace=True)
            # lowercase to match load_model's casefolded vector words, so the
            # join in predict() doesn't silently drop case-mismatched rows
            norms.index = norms.index.str.casefold()
            loaded.append((langfile, norms))
        except Exception as ex:
            print("Error loading " + langfile)
            print(f"An exception of type {type(ex).__name__} occurred loading {langfile}. Arguments:\n{ex.args!r}")
    return loaded

def load_count_freqs(lang):
    """
    Loads and cleans both frequency-count datasets for this language once.
    Transparently reads either a plain .tsv or a zip-compressed .tsv.zip (as
    downloaded from storage) without decompressing to disk first.
    """
    datasetspath = os.path.join(basedir, datasetsdir, countdir)
    flist = [f'dedup.{lang}.words.unigrams.tsv', f'dedup.{lang}wiki-meta.words.unigrams.tsv']

    loaded = []
    for langfile in flist:
        plain_path = os.path.join(datasetspath, langfile)
        compressed_path = plain_path + '.zip'
        if os.path.exists(plain_path):
            datapath = plain_path
        elif os.path.exists(compressed_path):
            datapath = compressed_path
        else:
            print(f'Counts file {langfile} not found, skipping.')
            continue
        freqs = pd.read_csv(datapath, sep='\t', comment='#', na_values=['-', '–'])
        freqs.set_index('unigram', inplace=True)

        # lowercase to match load_model's casefolded vector words, so the
        # join in predict() doesn't silently drop case-mismatched rows.
        # Corpus preprocessing never transliterates, so the model vocabulary
        # keeps whatever script/diacritics the source text used -- matching
        # on the untouched word first (rather than an always-transliterated
        # one) is what actually lines up with it.
        freqs_index = list(freqs.index.values)
        for i in range(len(freqs_index)):
            if isinstance(freqs_index[i], str):
                freqs_index[i] = freqs_index[i].casefold()
        freqs.index = freqs_index

        # Unicode-normalize + transliterate as a *fallback* key only, tried
        # in predict() solely for words that don't match directly. Applying
        # this unconditionally used to actively destroy matches for
        # non-Latin-script languages (e.g. Bengali words in native script
        # transliterated to a romanization the model vocabulary never
        # contains), while still occasionally rescuing a genuinely
        # differently-encoded word for other languages.
        fallback_index = []
        for word in freqs_index:
            if isinstance(word, str):
                fallback = unidecode(unicodedata.normalize("NFKD", word)).strip().casefold()
            else:
                fallback = word
            fallback_index.append(fallback)
        freqs.attrs['fallback_index'] = pd.Index(fallback_index)

        loaded.append((langfile, freqs))
    return loaded

##### evaluate one already-loaded model against one already-loaded dataset group #####
def evaluate_replication(wordsXdims, replication_norms, alpha=1.0):
    """Replicates the original van Paridon & Thompson norm predictions using ridge regression."""
    scores = []
    for norms_fname, norms in replication_norms:
        print(f'predicting norms from {norms_fname}')
        score = predict_norms(wordsXdims, norms, alpha)
        score['source'] = norms_fname
        scores.append(score)
    if len(scores) > 0:
        return pd.concat(scores)
    return None

def evaluate_norms(wordsXdims, extended_norms, alpha=1.0):
    """Predicts extended psycholinguistic norms datasets (valence, AoA, concreteness, etc.)."""
    scores = []
    for langfile, norms in extended_norms:
        print('Evaluating ' + langfile)
        score = predict_norms(wordsXdims, norms, alpha)
        score['dataset'] = langfile
        scores.append(score)
    if len(scores) > 0:
        return pd.concat(scores)
    return None

def evaluate_counts(wordsXdims, count_freqs, alpha=1.0):
    """Predicts word frequency counts."""
    scores = []
    for langfile, freqs in count_freqs:
        print('Evaluating ' + langfile)
        score = predict_counts(wordsXdims, freqs, alpha)
        score['dataset'] = langfile
        scores.append(score)
    if len(scores) > 0:
        return pd.concat(scores)
    return None

# Write output incrementally, one file per language per evaluation type 
def append_scores(outfile, scores):
    """Appends a scores dataframe to a per-language eval file, writing the header only once."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    write_header = not os.path.exists(outfile)
    with open(outfile, 'a') as f:
        scores.to_csv(f, mode='a', header=write_header, index=False)

# Main driver: one pass per language, one model load per configuration 
def evaluate_language(lang, alpha=1.0, overwrite=False):
    """
    For a given language, loads each trained model exactly once and runs it
    against replication norms, extended norms, and frequency counts -- with
    both the raw vectors and L2-normalized vectors, to compare the two --
    writing results to one output file per evaluation type as each model
    finishes, before moving on to the next model.
    """
    replication_out = os.path.join(basedir, evaldir, replicationdir, f'{lang}_eval.csv')
    norms_out = os.path.join(basedir, evaldir, normsdir, f'{lang}_eval.csv')
    counts_out = os.path.join(basedir, evaldir, countdir, f'{lang}_eval.csv')
    outfiles = [replication_out, norms_out, counts_out]

    if not overwrite and any(os.path.exists(f) for f in outfiles):
        print(f'Evaluation output for {lang} already exists, and overwrite not specified. Skipping.')
        return

    for outfile in outfiles:
        if os.path.exists(outfile):
            os.remove(outfile)

    replication_norms = load_replication_norms(lang)
    extended_norms = load_extended_norms(lang)
    count_freqs = load_count_freqs(lang)

    for dim in dimension_list:
        for win in window_list:
            for alg in algo_list:
                base_file_name = f'{lang}_{dim}_{win}_{alg}'
                words, raw_vectors = load_model(lang, dim, win, alg)
                if words is None:
                    continue
                print(f'Evaluating model {base_file_name}')

                for normalized in (False, True):
                    vectors = normalize_vectors(raw_vectors) if normalized else raw_vectors
                    wordsXdims = pd.DataFrame(vectors)
                    wordsXdims.set_index(words, inplace=True)

                    if replication_norms:
                        scores = evaluate_replication(wordsXdims, replication_norms, alpha)
                        if scores is not None:
                            scores['source'] = base_file_name
                            scores['normalized'] = normalized
                            append_scores(replication_out, scores)

                    if extended_norms:
                        scores = evaluate_norms(wordsXdims, extended_norms, alpha)
                        if scores is not None:
                            scores['source'] = base_file_name
                            scores['normalized'] = normalized
                            append_scores(norms_out, scores)

                    if count_freqs:
                        scores = evaluate_counts(wordsXdims, count_freqs, alpha)
                        if scores is not None:
                            scores['source'] = base_file_name
                            scores['normalized'] = normalized
                            append_scores(counts_out, scores)
