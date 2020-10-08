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
from gensim.models import Word2Vec

def download(source, language): 
    """
    Download data by source and language.  
    Source must be one of {'subtitles', 'wikipedia'}.
    Language must be a valid ISO3166 country code (lower case)
    
    Output file will be named in the pattern 'source-language.extension'.  
    Subtitle files use the 'zip' extension, while Wikipedia dumps use 'bz2'.  
    For example, download('subtitles', 'fr') will result in a file called 'subtitles-fr.zip'
    """
    sources = {
                'subtitles': f'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw/{language}.zip',
                'wikipedia': f'http://dumps.wikimedia.your.org/{language}wiki/latest/{language}wiki-latest-pages-meta-current.xml.bz2'
    }
    extensions = {
        'subtitles': 'zip',
        'wikipedia': 'bz2'
    }
    file_name = f'{source}-{language}.{extensions[source]}'
    print(f'Remote file {sources[source]}, Local file {file_name}')
    
    r = requests.get(sources[source], stream=True)
    with open(file_name, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Download complete.")

class sentences(object):
    """
    Return lines from a full corpus text file as a sequence
    using the generator pattern (an iterable)
    """
    def __init__(self, language):
        self.myfile = open(f'corpus-{language}.txt', 'r')

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
        self.myfile = bz2.open(f'wikipedia-{language}.bz2', 'rt', encoding='utf-8')

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
        prune(source, language)
    else:
        clean_wikipedia(language)
        prune(source, language)

        
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
    
def clean_subtitles(language):
    """
    Prepare subtitle files for processing.
    """
    input_file = zipfile.ZipFile(f'subtitles-{language}.zip', 'r')
    output_file = zipfile.ZipFile(f'subtitles-{language}-pre.zip', 'a', zipfile.ZIP_DEFLATED)
    
    xmlparser = etree.XMLParser(recover=True, encoding='utf-8')
    
    # Make list of files to process
    files = []
    for f in input_file.namelist():
        if f.endswith('xml'):
            if f.startswith(os.path.join('OpenSubtitles/raw', language)):
                files.append(f)
    print(f'Preprocessing {len(files)} {language} subtitle files.')
    for f in sorted(files):
        output_file.writestr(f.replace('xml', 'txt'),
                             clean_text(sub_xml_to_text(input_file.open(f).read(), xmlparser), subs_patterns))
    print('Complete')
    
    
def token_frequency_check(tokens):
    """
    Checking to see if the 30 most frequent tokens cover 30% of all tokens.
    Probably not doing this.
    """
    s = set(tokens)
    freqs = []
    for t in s:
        freqs.append((t, tokens.count(t)))

    freqs.sort(key = lambda x: x[1])
    
    thresh = 30
    if len(freqs) < 30:
        thresh = len(freqs)
    t30 = 0
    for i in range(thresh):
        t30 += freqs[i][1]
    return t30 >= len(tokens) * 0.3
    
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
    input_file = zipfile.ZipFile(f'{source}-{language}-pre.zip', 'r')
    output_file = zipfile.ZipFile(f'{source}-{language}-pruned.zip', 'a', zipfile.ZIP_DEFLATED)

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
            
    blocks = 4
    distance = 2
    matches = simhash.find_all(hash_list, blocks, distance)
    print(f'Got {len(matches)} matches')
    for match in matches:
        print(f'({hash_dict[match[0]]}, {hash_dict[match[1]]})')
        to_remove.append(hash_dict[match[1]])
    
    print(f'Found {len(to_remove)} files to prune.')
    
    for f in input_file.namelist():
        if f not in to_remove:
            output_file.writestr(f, input_file.open(f).read())

    output_file.close()
    
    
def concatenate_corpus(language):
    """
    Reads pre-processed subtitle and wikipedia text, and creates a single
    text file containing all of the tokenized sentences.
    """
    subs_input_file = zipfile.ZipFile(f'subtitles-{language}-pruned.zip', 'r')
    wiki_input_file = zipfile.ZipFile(f'wikipedia-{language}-pruned.zip', 'r')
    output_corpus = f'corpus-{language}.txt'
    with open(output_corpus, mode="w") as out:
        for f in subs_input_file.namelist():
            out.write(subs_input_file.open(f).read().decode("utf-8"))
        for f in wiki_input_file.namelist():
            out.write(wiki_input_file.open(f).read().decode("utf-8"))

    
    
def clean_wikipedia(language):
    """
    Prepare wikipedia files for processing.
    """
    with zipfile.ZipFile(f'wikipedia-{language}-pre.zip', 'a', zipfile.ZIP_DEFLATED) as output_archive:
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


def vectorize_stream(language, min_freq=5, dim=50, win=3, alg=0): 
    """
    Creates the word2vec model using gensim.
    """
    model = Word2Vec(sentences(language), min_count=min_freq, size=dim, workers=3, window=win, sg=alg)
    return model


def evaluate_norms(lang, wordsXdims, alpha=1.0):
    # Using subs2vec norms data for now
    path = os.path.join('/', 'home', 'pgrim', 'workspace', 'subs2vec', 'subs2vec')
    norms_path = os.path.join(path, 'datasets', 'norms')

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
        scores.append({
            'norm': col,
            'adjusted r': np.sqrt(penalized_score),  # take square root of explained variance to get Pearson r
            'adjusted r-squared': penalized_score,
            'r-squared': median_score,
            'r': np.sqrt(median_score),
        })
    return pd.DataFrame(scores)

dimension_list = [50,100,200,300,500]
window_list = [3,4,5,6,7,8,9,10,11,12,13]
algo_list = [0,1]

def build_models(language):
    for dim in dimension_list:
        for win in window_list:
            for alg in algo_list:
                algo = 'cbow' if alg ==0 else 'sg'
                base_file_name = f'{language}_{str(dim)}_{str(win)}_{algo}'
                print("Building model " + base_file_name)
                model = vectorize_stream(language, dim, win, alg)
                #Write down the model?
                words=list(model.wv.vocab)
                wordsxdims = pd.DataFrame(model[words],words)
                wordsxdims.to_csv(f'{base_file_name}_wxd.csv')

