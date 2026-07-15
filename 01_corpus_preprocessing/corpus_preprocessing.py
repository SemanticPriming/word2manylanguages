###### libraries ######
import bz2
import html
import os
import re
import requests
import simhash
import zipfile
from lxml import etree

###### define directories ######
# basedir defined in main file
datadir = "raw"
processdir = "preprocessed"
corpusdir = "corpora"

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
