###### libraries ######
import glob
import numpy as np
import os
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils
import unicodedata
from unidecode import unidecode

###### define directories ######
# basedir defined in main file
modeldir = 'models'
evaldir = 'evals'
datasetsdir = "datasets"
countdir = "counts"
normsdir = "norms"
replicationdir = "replication"
scoredir = "scores"
datasetsindex = "datasets.csv"
dimensions = ["500", "300", "200", "100", "50"]
windows = ["1", "2", "3","4", "5", "6"]

dimension_list = [50,100,200,300,500]
window_list = [1,2,3,4,5,6]
algo_list = ['cbow','sg']

##### evaluate replication #####
def loop_norms_vp(language):
    """
    1. Loops through the saved models
    2. Uses evaluate_norms_vp to compare to the files within the replication folder
    3. Uses predict norms to calculate the ridge regression
    """
    for dim in dimensions:
        for win in windows:
            for algo in ['cbow', 'sg']:
                base_file_name = f'{language}_{str(dim)}_{str(win)}_{algo}'
                print("Evaluating model " + base_file_name)
                path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv')
                wordsXdims = pd.read_csv(path)
                wordsXdims.set_index('word', inplace=True)
                scores = evaluate_norms_vp(language, wordsXdims)

                outpath = os.path.join(basedir, evaldir, replicationdir)
                fname = base_file_name + '_eval.csv'
                outfile = os.path.join(outpath, fname)
                scores.to_csv(outfile)

def evaluate_norms_vp(lang, wordsXdims, alpha=1.0):
    """
    1. Loops through the normed replication data folder
    2. Uses predict norms to calculate the ridge regression
    3. Returns scores to function for saving
    """
    norms_path = os.path.join(basedir, datasetsdir, replicationdir)
    scores = []
    for norms_fname in os.listdir(norms_path):
        if norms_fname.startswith(lang):
            print(f'predicting norms from {norms_fname}')
            norms = pd.read_csv(os.path.join(norms_path, norms_fname), sep='\t', comment='#')
            norms = norms.set_index('word')
            score = predict_norms(wordsXdims, norms, alpha)
            score['source'] = norms_fname
            scores.append(score)

    if len(scores) > 0:
        scores = pd.concat(scores)
        return scores

def predict_norms(vectors, norms, alpha=1.0):
    """
    Ridge regression function to use the embeddings to predict norms (replication or new).
    """
    cols = norms.columns.values
    df = norms.join(vectors, how='inner')

    # compensate for missing ys somehow
    total = len(norms)
    missing = len(norms) - len(df)
    penalty = (total - missing) / total
    print(f'missing vectors for {missing} out of {total} words')
    df = sklearn.utils.shuffle(df)  # shuffle is important for unbiased results on ordered datasets!

    model = sklearn.linear_model.Ridge(alpha=alpha)  # use ridge regression models
    cv = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

    # compute crossvalidated prediction scores
    scores = []
    for col in cols:
        # set dependent variable and calculate 10-fold mean fit/predict scores
        df_subset = df.loc[:, vectors.columns.values]  # use .loc[] so copy is created and no setting with copy warning is issued
        df_subset[col] = df[col]
        df_subset = df_subset.dropna()  # drop NaNs for this specific y
        x = df_subset[vectors.columns.values]
        y = df_subset[col]
        cv_scores = sklearn.model_selection.cross_val_score(model, x, y, cv=cv)
        median_score = np.median(cv_scores)
        penalized_score = median_score * penalty
        ars = np.sqrt(penalized_score) if penalized_score > 0 else 0
        rs = np.sqrt(median_score) if median_score > 0 else 0
        scores.append({
            'norm': col,
            'adjusted r': ars,
            'adjusted r-squared': penalized_score,
            'r-squared': median_score,
            'r': rs
        })
    return pd.DataFrame(scores)

##### evaluate counts ######
def evaluate_counts(lang, alpha=1.0):
    """
    Evaluate word frequency prediction for the given language
    """
    # Load the counts dataset for the language
    datasetspath = os.path.join(basedir,datasetsdir,countdir)

    flist = [f'dedup.{lang}.words.unigrams.tsv', f'dedup.{lang}wiki-meta.words.unigrams.tsv'];

    # Predict for all of the matrices
    # Make this the outer look because they take longer to read
    scores = []
    for dim in dimension_list:
        for win in window_list:
            for alg in algo_list:
                # Load the words by dimensions matrix
                base_file_name = f'{lang}_{str(dim)}_{str(win)}_{alg}'
                input_path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv')
                print("Loading model " + base_file_name)
                with open(input_path, 'r', encoding='utf-8') as vecfile:
                    # skip header
                    next(vecfile)
                    # initialize arrays
                    vectors = np.zeros((10000000, dim))
                    words = np.empty(10000000, dtype=object) # fill arrays
                    for i, line in enumerate(vecfile):
                        # Limit to 10 million, although it looks like 7.5 million is the largest
                        if i >= 10000000:
                            break
                        rowentries = line.rstrip('\n').split(',')
                        words[i] = rowentries[0].casefold()
                        vectors[i] = rowentries[1:dim+1]

                    # truncate empty part of arrays, if necessary
                    vectors = vectors[:i]
                    words = words[:i]

                    # normalize by L1 norm
                    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)

                    wordsXdims = pd.DataFrame(vectors)
                    wordsXdims.set_index(words, inplace=True)

                    # Do it for each language dataset
                    for langfile in flist:
                        print('Loading ' + langfile)
                        datapath = os.path.join(datasetspath,langfile)
                        freqs = pd.read_csv(datapath,sep='\t', comment='#', na_values=['-','–'])
                        freqs.set_index('unigram', inplace=True)

                        # Clean up the data
                        print('Cleaning ' + langfile)
                        freqs_index = list(freqs.index.values)
                        for i in range(len(freqs_index)):
                            if (isinstance(freqs_index[i], str)):
                                freqs_index[i] = unicodedata.normalize("NFKD", freqs_index[i])
                                freqs_index[i] = unidecode(freqs_index[i]).strip()

                        freqs.index = freqs_index

                        print('Evaluating ' + langfile)
                        score = predict_counts(wordsXdims, freqs, alpha)
                        score['source'] = base_file_name
                        score['dataset'] = langfile
                        scores.append(score)

    # Concatenate the results
    if len(scores) > 0:
        scores = pd.concat(scores)
        outpath = os.path.join(basedir, evaldir, countdir, f'{lang}_eval.csv')
        scores.to_csv(outpath)

def predict_counts(vectors, freqs, alpha=1.0):
    cols = freqs.columns.values
    df = freqs.join(vectors, how='inner')

    # compensate for missing ys somehow
    total = len(freqs)
    missing = len(freqs) - len(df)
    penalty = (total - missing) / total
    print(f'vectors: {len(vectors)}  freqs: {total}  matches: {len(df)}')
    print(f'missing vectors for {missing} out of {total} words')
    df = sklearn.utils.shuffle(df)  # shuffle is important for unbiased results on ordered datasets!

    model = sklearn.linear_model.Ridge(alpha=alpha)  # use ridge regression models
    cv = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10)

    # compute crossvalidated prediction scores
    scores = []
    for col in cols:
        # set dependent variable and calculate 10-fold mean fit/predict scores
        df_subset = df.loc[:, vectors.columns.values]  # use .loc[] so copy is created and no setting with copy warning is issued
        df_subset[col] = df[col]
        df_subset = df_subset.dropna()  # drop NaNs for this specific y
        x = df_subset[vectors.columns.values]
        y = df_subset[col]
        cv_scores = sklearn.model_selection.cross_val_score(model, x, y, cv=cv)
        median_score = np.median(cv_scores)
        penalized_score = median_score * penalty
        ars = np.sqrt(penalized_score) if penalized_score > 0 else 0
        rs = np.sqrt(median_score) if median_score > 0 else 0
        scores.append({
            'var': col,
            'adjusted r': ars,
            'adjusted r-squared': penalized_score,
            'r-squared': median_score,
            'r': rs
        })
    return pd.DataFrame(scores)

##### evaluate norms #####
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

# uses predict_norms defined above
def evaluate_norms(lang, alpha=1.0):
    """
    Evaluate norms prediction for the given language
    """
    # Load the norms dataset for the language
    datasetspath = os.path.join(basedir,datasetsdir,datasetsindex)
    datasets = pd.read_csv(datasetspath, sep=',', comment='#')

    langfiles = datasets[datasets['language'] == lang].iat[0,1]
    flist = langfiles.split('|');

    # Predict for all of the matrices
    # Make this the outer look because they take longer to read
    scores = []
    for dim in dimension_list:
        for win in window_list:
            for alg in algo_list:
                # Load the words by dimensions matrix
                base_file_name = f'{lang}_{str(dim)}_{str(win)}_{alg}'
                input_path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv')
                print("Loading model " + base_file_name)
                with open(input_path, 'r', encoding='utf-8') as vecfile:
                    # skip header
                    next(vecfile)
                    # initialize arrays
                    vectors = np.zeros((10000000, dim))
                    words = np.empty(10000000, dtype=object) # fill arrays
                    for i, line in enumerate(vecfile):
                        # Limit to 10 million, although it looks like 7.5 million is the largest
                        if i >= 10000000:
                            break
                        rowentries = line.rstrip('\n').split(',')
                        words[i] = rowentries[0]
                        vectors[i] = rowentries[1:dim+1]

                    # truncate empty part of arrays, if necessary
                    vectors = vectors[:i]
                    words = words[:i]

                    # normalize by L1 norm
                    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)

                    wordsXdims = pd.DataFrame(vectors)
                    wordsXdims.set_index(words, inplace=True)

                    # Do it for each language dataset
                    for langfile in flist:
                        print('Evaluating ' + langfile)
                        datapath = os.path.join(basedir,datasetsdir,normsdir,langfile)
                        try:
                            norms = pd.read_csv(datapath,sep=',', comment='#',na_values=['-','–'])

                            # Get the subset of columns that we need for prediction
                            cols = list_norm_columns(norms)
                            # Get the column that has the words in it.  It might just be word, or
                            # it might be word_{language_name}
                            if 'word' in norms.columns:
                                wordcol = 'word'
                            else:
                                wordcol = 'word_' + code2lang[lang]

                            check = norms.columns
                            if not wordcol in check:
                                # There are some other special cases
                                if wordcol + '_simple' in check:
                                    wordcol = wordcol + '_simple'
                                elif wordcol + '_uk' in check:
                                    wordcol = wordcol + '_uk'
                                elif lang == fa and 'word_persian' in check:
                                    wordcol = 'word_persian'

                            cols = [wordcol] + cols
                            print("Considering columns " + str(cols))
                            norms = norms[cols]
                            norms.set_index(wordcol, inplace=True)

                            score = predict_norms(wordsXdims, norms, alpha)
                            score['source'] = base_file_name
                            score['dataset'] = langfile
                            scores.append(score)
                        except Exception as ex:
                            print("Error processing " + langfile)
                            print(f"An exception of type {type(ex).__name__} occurred processing {langfile}. Arguments:\n{ex.args!r}")

    # Concatenate the results
    if len(scores) > 0:
        scores = pd.concat(scores)
        outpath = os.path.join(basedir, evaldir, normsdir, f'{lang}_eval.csv')
        with open(outpath, 'a') as f:
            scores.to_csv(f, mode='a', header=f.tell()==0)

approved = ['AOA_M', 'FAMILIAR_M', 'CONCRETE_M', 'IMAGINE_M', 'DOMINATE_M', 'VALENCE_M',
            'AROUSAL_M', 'IMAGEABILITY_M','TYPICAL_M', 'ANGRY_M', 'SAD_M']

def list_norm_columns(df):
    """
    Returns a list of columns in the given data frame that end with _M,
    those being the columns that contain mean norms.
    """
    cols_out = []
    cols = df.columns.values
    for col in cols:
        if col.endswith("_M") or "_M_" in col:
            cols_out.append(col)
    return cols_out


##### function to combine all scores together #####
def score_vp(language, typedir):
    rows = []
    base_file_name = f'{language}*_eval.csv'
    evalpath = os.path.join(basedir, evaldir, typedir)
    files = glob.glob(os.path.join(evalpath, base_file_name))

    for file in files:
        print("Loading eval " + file)

        parts = os.path.basename(file).split('_')

        eval_df = pd.read_csv(file, header=0)

        if 'adjusted r-squared' not in eval_df.columns:
            print(f"⚠️ 'adjusted r-squared' column missing in: {file}")
            continue
        if 'norm' not in eval_df.columns:
            print(f"⚠️ 'norm' column missing in: {file}")
            continue

        for index, stuff in eval_df.iterrows():
            row = {
                'Language': language,
                'Dimensions': parts[1],
                'Window': parts[2],
                'Algorithm': parts[3],
                'Norm': stuff['norm'],
                'Score': stuff['adjusted r-squared']
            }
            rows.append(row)

    if not rows:
        print(f"❌ No valid data found for language '{language}' in '{typedir}'")
        return

    dataframe = pd.DataFrame(rows)

    # Ensure all sort columns exist
    expected_cols = {'Score', 'Dimensions', 'Window'}
    if not expected_cols.issubset(dataframe.columns):
        print(f"❌ Missing columns in DataFrame: {expected_cols - set(dataframe.columns)}")
        return

    dataframe.sort_values(by=['Score', 'Dimensions', 'Window'], ascending=[False, True, True], inplace=True)

    outfilename = f'{language}_scores.csv'
    outfile = os.path.join(basedir, scoredir, typedir, outfilename)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    dataframe.to_csv(outfile, index=False)
    print(f"✅ Saved sorted results to {outfile}")

def score_norms(language, typedir):
    rows = []
    base_file_name = f'{language}_eval.csv'
    outfilename = f'{language}_scores.csv'
    print("Loading eval " + base_file_name)
    path = os.path.join(basedir, evaldir, typedir, base_file_name)
    eval = pd.read_csv(path)

    for index, stuff in eval.iterrows():
        row = dict()
        model = stuff['source']
        parts = model.split('_')

        row['Language'] = language
        row['Dimensions'] = parts[1]
        row['Window'] = parts[2]
        row['Algorithm'] = parts[3]
        row['Norm'] = stuff['norm']
        row['Score'] = stuff['adjusted r-squared']
        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.sort_values(by=['Score',"Dimensions","Window"], ascending=[False, True, True], inplace=True)

    print(dataframe)

    outfile = os.path.join(basedir, scoredir, typedir, outfilename)
    dataframe.to_csv(outfile)

def score_counts(language, typedir):
    rows = []
    base_file_name = f'{language}*_eval.csv'
    evalpath = os.path.join(basedir, evaldir, typedir)
    files = glob.glob(os.path.join(evalpath, base_file_name))

    for file in files:
        print("Loading eval " + file)

        parts = os.path.basename(file).split('_')

        eval_df = pd.read_csv(file, header=0)

        if 'adjusted r-squared' not in eval_df.columns:
            print(f"⚠️ 'adjusted r-squared' column missing in: {file}")
            continue
        if 'var' not in eval_df.columns:
            print(f"⚠️ 'var' column missing in: {file}")
            continue

        for _, row_data in eval_df.iterrows():
            parts = row_data['source'].split('_')

            if len(parts) < 4:
                print(f"⚠️ Skipping row with bad source format: {row_data['source']}")
                continue

            row = {
                'Language': language,
                'Dimensions': parts[1],
                'Window': parts[2],
                'Algorithm': parts[3],
                'Var': row_data['var'],
                'Score': row_data['adjusted r-squared']
            }
            rows.append(row)

    if not rows:
        print(f"❌ No valid data found for language '{language}' in '{typedir}'")
        return

    dataframe = pd.DataFrame(rows)

    expected_cols = {'Score', 'Dimensions', 'Window'}
    if not expected_cols.issubset(dataframe.columns):
        print(f"❌ Missing columns in DataFrame: {expected_cols - set(dataframe.columns)}")
        return

    dataframe.sort_values(by=['Score', 'Dimensions', 'Window'], ascending=[False, True, True], inplace=True)

    outfilename = f'{language}_scores.csv'
    outfile = os.path.join(basedir, scoredir, typedir, outfilename)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    dataframe.to_csv(outfile, index=False)
    print(f"✅ Saved sorted results to {outfile}")
