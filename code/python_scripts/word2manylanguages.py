###### libraries ######
import bz2
import html
import numpy as np
import os
import pandas as pd
import re
import requests
import simhash
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils
import zipfile
from lxml import etree
from gensim.models import FastText
import glob


###### define directories ######
# basedir defined in main file
datadir = "data"
processdir = "preprocessed"
corpusdir = "corpora"
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

###### downloading subtitles and/or wikipedia dumps ######
def download(source, language, overwrite=False): 
    """
    Download data by source and language.  
    Source must be one of {'subtitles', 'wikipedia'}.
    Language must be a valid ISO3166 country code (lower case)
    
    Output file will be named in the pattern 'source-language.extension'.  
    Subtitle files use the 'zip' extension, while Wikipedia dumps use 'bz2'.  
    For example, download('subtitles', 'fr') will result in a file called 'subtitles-fr.zip'
    """
    # Special case for OpenSubtitles names for Chinese variants
    language_tran = language
    if language == 'tw' : 
        language_tran = 'zh_tw'
    elif language == 'zh': 
        language_tran = 'zh_cn'

    sources = {
                #'subtitles': f'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw/{language_tran}.zip',
                'subtitles': f'https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/raw/{language}.zip',
                'wikipedia': f'http://dumps.wikimedia.your.org/{language}wiki/latest/{language}wiki-latest-pages-meta-current.xml.bz2'
    }
    extensions = {
        'subtitles': 'zip',
        'wikipedia': 'bz2'
    }
    file_name = f'{source}-{language}.{extensions[source]}'
    print(f'Remote file {sources[source]}, Local file {file_name}')
    path_name = os.path.join(basedir, datadir, file_name)

    if os.path.exists(path_name) and not overwrite:
        print(f'File {file_name} exists, and overwrite not specified. Skipping.');
    else:
        r = requests.get(sources[source], stream=True)
        with open(path_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")

###### clean up files ######
class sentences(object):
    """
    Return lines from a full corpus text file as a sequence
    using the generator pattern (an iterable)
    """
    def __init__(self, language):
        path_name = os.path.join(basedir,corpusdir,f'corpus-{language}.txt')
        self.myfile = open(path_name, 'r')

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        line = self.myfile.readline()
        if line:
            tok = [w for w in line.rstrip().split(' ') if len(w) > 0]
            return tok

        raise StopIteration()

class articles(object):
    """
    Read a wikipedia dump file and return one article at a time
    using the generator pattern (an iterable)
    """
    def __init__(self, language):
        path_name = os.path.join(basedir,datadir, f'wikipedia-{language}.bz2')
        self.myfile = bz2.open(path_name, 'rt', encoding='utf-8')

    def __iter__(self):
        return self
 
    # Python 3 compatibility
    def __next__(self):
        return self.next()
 
    def next(self):
        article = ""
        body = False
        line = self.myfile.readline()
        while line:
            if "<page>" in line:
                body = True
            
            if "</page>" in line:
                return html.unescape(html.unescape(article))    
            
            if body:
                article = article + line
            
            line = self.myfile.readline()
            
        self.myfile.close()
        raise StopIteration()


def clean(source, language):
    """
    Start the cleaning process for a given source and language.  
    Routes to the appropriate file handing functions for the given source.
    """
    
    if ('subtitles' == source):
        clean_subtitles(language)
        #prune(source, language)
    else:
        clean_wikipedia(language)
        #prune(source, language)

        
def sub_xml_to_text(xml, parser):
    """
    Extracts text from xml nodes in subtitle files, removes unused nodes.
    """
    tree = etree.fromstring(xml, parser)
    for node in tree.iter():
        if node.tag == 'meta':
            tree.remove(node)
    return etree.tostring(tree, encoding=str, method='text')


def wiki_strip_circumflex(txt):
    """
    Removes the (deeply nested) circumflex characters from wiki text.
    """
    circumflex = 0
    txt = list(txt)
    for i in range(len(txt)):
        if txt[i] == '{':
            circumflex += 1
        elif txt[i] == '}':
            circumflex -= 1
            txt[i] = ''
        if circumflex > 0:
            txt[i] = ''
        elif circumflex < 0:
            # discard unmatched
            txt = []
            break
    return ''.join(txt)

# Regular expressions for cleaning subtitle text
subs_expressions = [
    (r'<.*?>', ''),  # xml tags
    (r'http.*?(?:[\s\n\]]|$)', ''),  # links
    (r'\s\(.*?\)', ''),  # parentheses
    (r'([^\s]{2,})[\.\!\?\:\;]+?[\s\n]|$', '\\1\n'),  # break sentences at periods
    (r"[-–—/']", ' '),  # hyphens, apostrophes and slashes
    (r'\s*\n\s*', '\n'),  # empty lines
    (r'\s{2,}', ' '),  # excessive spaces
]
subs_patterns = [(re.compile(expression[0], re.IGNORECASE), expression[1]) for expression in subs_expressions]

# Regular expressions for cleaning wikipedia text
wiki_expressions = [
    (r'(?s)<ref.*?</ref>', ''),  # strip reference links
    (r'(?s)<references.*?</references>', ''),  # strip references
    (r'(?s)<table.*?</table>', ''),  # strip tables
    (r'(?s)<gallery.*?</gallery>', ''),  # strip galleries
    (r'(?s)<kml.*?</kml>', ''),  # strip KML tags
    (r'<.*?>', ''),  # strip other xml tags
    (r'http.*?(?:[\s\n\]]|$)', ''),  # strip external http(s) links
    (r'\[\[[^\]]*?:.*\|(.*?)\]\]', '\\1'),  # strip links to files, etc. but keep labels
    (r'\[\[[^\]]*?:(.*?)\]\]', ''),  # strip category links
    (r'\[\[[^\]]*?\|(.*?)\]\]', '\\1'),  # convert labeled links to just labels
    (r'(?m)^[\s]*[!?*;:=+\-|#_].*?$', ''),  # strip lines that do not start with alphanumerics, quotes, or brackets
    (r'(?m)^.*?\(UTC\).*?$', ''),  # strip lines containing a time stamp
    (r'\s\(.*?\)', ''),  # remove everything in parentheses
    (r'([^\s.!?:;]{2})[.!?:;]+?[\s\n]|$', '\\1\n'),  # break sentences at periods
    (r"[-–—/']", ' '),  # replace hyphens, apostrophes and slashes with spaces
    (r'\s*\n\s*', '\n'),  # strip empty lines and lines containing whitespace
    (r'\s{2,}', ' '),  # strip excessive spaces
]
wiki_patterns = [(re.compile(expression[0], re.IGNORECASE), expression[1]) for expression in wiki_expressions]

def clean_text(text, patterns):
    """
    Applies the given patterns to the input text. Ensures lower-casing of all text.
    """
    txt = text
    for pattern in patterns:
        txt = pattern[0].sub(pattern[1], txt)
    txt = ''.join([letter for letter in txt if (letter.isalnum() or letter.isspace())]) 
    return txt.lower() 
    
def clean_subtitles(language,overwrite=False):
    """
    Prepare subtitle files for processing.
    """
    # Special case for OpenSubtitles names for Chinese variants
    language_tran = language
    if language == 'tw' : 
        language_tran = 'zh_tw'
    elif language == 'zh': 
        language_tran = 'zh_cn'
    input_path = os.path.join(basedir,datadir,f'subtitles-{language}.zip')
    output_path = os.path.join(basedir,processdir,f'subtitles-{language}-pre.zip')

    if os.path.exists(output_path) and not overwrite:
        print(f'File subtitles-{language}-pre.zip exists, and overwrite not specified. Skipping.');
    else:
        input_file = zipfile.ZipFile(input_path, 'r')
        output_file = zipfile.ZipFile(output_path, 'a', zipfile.ZIP_DEFLATED)
    
        xmlparser = etree.XMLParser(recover=True, encoding='utf-8')
    
        # Make list of files to process
        files = []
        for f in input_file.namelist():
            if f.endswith('xml'):
                if f.startswith(f'OpenSubtitles/raw/{language_tran}'):
                    files.append(f)
        print(f'Preprocessing {len(files)} {language} subtitle files.')
        for f in sorted(files):
            output_file.writestr(f.replace('xml', 'txt'),
                                 clean_text(sub_xml_to_text(input_file.open(f).read(), xmlparser), subs_patterns))
        print('Complete')
       
def clean_wikipedia(language, overwrite=False):
    """
    Prepare wikipedia files for processing.
    """
    wiki_output_path = os.path.join(basedir,processdir,f'wikipedia-{language}-pre.zip')
    if os.path.exists(wiki_output_path) and not overwrite:
        print(f'File wikipedia-{language}-pre.zip exists, and overwrite not specified. Skipping.');
    else:
        with zipfile.ZipFile(wiki_output_path, 'a', zipfile.ZIP_DEFLATED) as output_archive:
            i = 0
            print(f'Preprocessing {language} Wikipedia dump.')
            for article in articles(language):
                filename = f'wiki-{language}-{str(i)}.txt'
                txt = article.lower()
                txt = wiki_strip_circumflex(article) if ((not txt.startswith('#'))
                                     and ('<noinclude>' not in txt)
                                     and ('__noindex__' not in txt)
                                     ) else ''
                for pattern in wiki_patterns:
                    txt = pattern[0].sub(pattern[1], txt)
        
                output_archive.writestr(filename, ''.join([letter for letter in txt if (letter.isalnum() or letter.isspace())]))
                i += 1
            
            print("Complete")

def get_hash(tokens):
    """
    Creates the simhash for the given list of tokens
    """
    shingles = [''.join(shingle) for shingle in
                    simhash.shingle(''.join(tokens), 4)]
    hashes = [simhash.unsigned_hash(s.encode('utf8')) for s in shingles]
    return simhash.compute(hashes)

def prune(source, language):
    """
    Remove duplicate documents from archive file using simhash.
    """
    input_path = os.path.join(basedir,datadir,f'{source}-{language}-pre.zip')
    output_path = os.path.join(basedir,datadir,f'{source}-{language}-pruned.zip')
    input_file = zipfile.ZipFile(input_path, 'r')

    to_remove = []
    hash_list = []
    hash_dict = dict()
    
    print("Checking for duplicates.")
    for f in input_file.namelist():
        text = str(input_file.open(f).read())
        tokens = re.split(r'\W+', text.lower(), flags=re.UNICODE)
        hash = get_hash(tokens)
        hash_list.append(hash)
        hash_dict[hash] = f
            
    input_file.close()

    blocks = 4
    distance = 2
    matches = simhash.find_all(hash_list, blocks, distance)
    print(f'Got {len(matches)} matches')
    for match in matches:
        print(f'({hash_dict[match[0]]}, {hash_dict[match[1]]})')
        to_remove.append(hash_dict[match[1]])
    
    print(f'Found {len(to_remove)} files to prune.')
    input_file = zipfile.ZipFile(input_path, 'r')
    output_file = zipfile.ZipFile(output_path, 'a', zipfile.ZIP_DEFLATED)
    
    for f in input_file.namelist():
        if f not in to_remove:
            output_file.writestr(f, input_file.open(f).read())

    output_file.close()
    
###### after cleaning, put together the whole corpus into one big text ######
def concatenate_corpus(language,overwrite=False):
    """
    Reads pre-processed subtitle and wikipedia text, and creates a single
    text file containing all of the tokenized sentences.
    """
    subs_input_path = os.path.join(basedir,processdir,f'subtitles-{language}-pre.zip')
    wiki_input_path = os.path.join(basedir,processdir,f'wikipedia-{language}-pre.zip')
    corpus_output_path = os.path.join(basedir,corpusdir,f'corpus-{language}.txt')
    if os.path.exists(corpus_output_path) and not overwrite:
        print(f'File corpus-{language}.txt exists, and overwrite not specified. Skipping.');
    else:
        print(f"Concatenating {language} corpus.")
        with open(corpus_output_path, mode="w") as out:
            subs_input_file = zipfile.ZipFile(subs_input_path, 'r')
            for f in subs_input_file.namelist():
                inf = subs_input_file.open(f)
                line = inf.readline()
                while line:
                    try:
                        out.write(line.decode("utf-8"))
                    except:
                        print("decode error s - skipping: ", line)
                    line = inf.readline()
            subs_input_file.close()
            wiki_input_file = zipfile.ZipFile(wiki_input_path, 'r')
            for f in wiki_input_file.namelist():
                inf = wiki_input_file.open(f)
                line = inf.readline()
                while line:
                    try:
                        out.write(line.decode("utf-8"))
                    except:
                        print("decode error w - skipping: ", line)
                    line = inf.readline()
            wiki_input_file.close()
      

##### build gensim models #####
def vectorize_stream(language, min_freq=5, dim=50, win=3, alg=0): 
    """
    Creates the word2vec model using gensim.
    """
    algo = 1 if alg == "sg" else 0
    print(f"Training model {language} {dim} {win} {alg}") 
    model = FastText(vector_size=dim, window=win, min_count=min_freq, sg=algo, sample=1e-2, negative=10, alpha=0.05, min_n=3, max_n=6)
    model.build_vocab(corpus_iterable=sentences(language))
    total_examples = model.corpus_count
    model.train(corpus_iterable=sentences(language), total_examples=total_examples, epochs=10)

    return model

dimension_list = [50,100,200,300,500]
window_list = [1,2,3,4,5,6]
algo_list = ['cbow','sg']

def build_models(language,overwrite=False):
    """
    Loops over model requirements and uses vectorize stream to create gensim models. 
    """
    for dim in dimension_list:
        for win in window_list:
            for alg in algo_list:
                base_file_name = f'{language}_{str(dim)}_{str(win)}_{alg}'
                output_path = os.path.join(basedir, modeldir, f'{base_file_name}_wxd.csv')
                if os.path.exists(output_path) and not overwrite:
                    print(f'File {base_file_name}_wxd.csv exists, and overwrite not specified. Skipping.');
                else:
                    print("Building model " + base_file_name)
                    model = vectorize_stream(language, 5, dim, win, alg)
                    #Write down the model?
                    words=list(model.wv.key_to_index)
                    wordsxdims = pd.DataFrame(model.wv[words],words)
                    wordsxdims.to_csv(output_path,index_label='word')

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
        outpath = os.path.join(basedir, evaldir, f'{lang}_eval.csv')
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
def score(language, typedir):
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