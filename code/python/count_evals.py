import os
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils
import sys
import unicodedata
from unidecode import unidecode

basedir = os.path.join('Y:\\', 'Dissertation')
countdir = "frequencies"
modeldir = 'models'
evaldir = 'count_evals'
scoredir = 'count_scores'

scorepath = os.path.join(basedir, scoredir)
if not os.path.exists(scorepath):
    os.makedirs(scorepath)
    
evalpath = os.path.join(basedir, evaldir)
if not os.path.exists(evalpath):
    os.makedirs(evalpath)
    
dimension_list = [50,100,200,300,500]
window_list = [1,2,3,4,5,6]
algo_list = ['cbow','sg']

# 'en' removed since it was done separately
#langs = ['af','ar','bg','bn','br','bs','ca','cs','da','de','el','eo','es','et','eu','fa','fi','fr',
#         'gl','he','hi','hr','hu','hy','id','is','it','ka','kk','ko','lt','lv','mk','ml','ms','nl',
#         'no','pl','pt','ro','ru','si','sk','sl','sq','sr','sv','ta','te','tl','tr','uk','ur','vi']

langs1 = ['de','el','eo','es','et','eu','fa','fi','fr',
          'gl','he','hi','hr','hu','hy','id','is']

langs2 = ['it','ka','kk','ko','lt','lv','mk','ml','ms','nl',
          'no','pl','pt','ro','ru','si','sk','sl','sq','sr','sv','ta','te','tl','tr','uk','ur','vi']

def evaluate_counts(lang, alpha=1.0):
    """
    Evaluate word frequency prediction for the given language
    """
    # Load the counts dataset for the language
    datasetspath = os.path.join(basedir,countdir)

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
        outpath = os.path.join(basedir, evaldir, f'{lang}_count_eval.csv')
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

if __name__ == '__main__':
    
    if sys.argv[1] == '1':
        for lang in langs1:
            evaluate_counts(lang)
    elif sys.argv[1] == '2':
        for lang in langs2:
            evaluate_counts(lang)