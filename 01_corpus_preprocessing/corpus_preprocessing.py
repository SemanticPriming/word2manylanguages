###### libraries ######
import bz2
import html
import os
import re
import requests
import simhash
import unicodedata
import zipfile
from lxml import etree

###### define directories ######
# basedir defined in main file
datadir = "raw"
processdir = "preprocessed"
corpusdir = "corpora"

# Downloading subtitles and/or wikipedia dumps
def download(source, language, version='2018', overwrite=False):
    """
    Download data by source and language.
    Source must be one of {'subtitles', 'wikipedia'}.
    Language must be a valid ISO3166 country code (lower case)
    version is the OpenSubtitles corpus vintage ('2018' or '2024', see
    https://opus.nlpl.eu/OpenSubtitles/) -- only meaningful for source=
    'subtitles'. Wikipedia has no equivalent dated-vintage concept (the URL
    below always serves whatever is currently "latest"), so wikipedia
    downloads ignore version entirely and are shared/reused across a
    language's 2018- and 2024-subtitle-paired corpus builds rather than
    being fetched twice.

    Output file will be named 'source-language.extension' for version=2018
    (unchanged, for backward compatibility with every 2018-era file already
    on disk) or 'source-language-version.extension' for any other version.
    Subtitle files use the 'zip' extension, while Wikipedia dumps use 'bz2'.
    For example, download('subtitles', 'fr') will result in a file called
    'subtitles-fr.zip'; download('subtitles', 'fr', version='2024') will
    result in 'subtitles-fr-2024.zip'.
    """
    # Special case for OpenSubtitles names for Chinese variants
    language_tran = language
    if language == 'tw' :
        language_tran = 'zh_tw'
    elif language == 'zh':
        language_tran = 'zh_cn'

    sources = {
                #'subtitles': f'http://opus.nlpl.eu/download.php?f=OpenSubtitles/{version}/raw/{language_tran}.zip',
                'subtitles': f'https://object.pouta.csc.fi/OPUS-OpenSubtitles/{version}/raw/{language_tran}.zip',
                'wikipedia': f'http://dumps.wikimedia.your.org/{language}wiki/latest/{language}wiki-latest-pages-meta-current.xml.bz2'
    }
    extensions = {
        'subtitles': 'zip',
        'wikipedia': 'bz2'
    }
    suffix = '' if (source == 'wikipedia' or version == '2018') else f'-{version}'
    file_name = f'{source}-{language}{suffix}.{extensions[source]}'
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

# Clean up files and prepare for processing
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


def clean(source, language, version='2018'):
    """
    Start the cleaning process for a given source and language.
    Routes to the appropriate file handing functions for the given source.
    version only affects the subtitles branch (see clean_subtitles()) --
    wikipedia has no dated-vintage concept, always "latest" regardless.
    """

    if ('subtitles' == source):
        subs_key = language if version == '2018' else f'{language}-{version}'
        clean_subtitles(language, version=version)
        prune(source, subs_key)
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


_ns_re = re.compile(r'<ns>(\d+)</ns>')
_text_re = re.compile(r'(?s)<text[^>]*>(.*?)</text>')

def extract_wiki_text(article):
    """
    Pulls just the actual wikitext payload out of a raw <page>...</page>
    block, discarding page/revision metadata (title, timestamps, revision
    ids, contributor names, edit comments, and especially <model>/<format>,
    whose inner text -- e.g. "wikitext" from <model>wikitext</model>, or
    "text/x-wiki" from <format>text/x-wiki</format> -- would otherwise leak
    into the corpus as bogus high-frequency "words" once <.*?> strips just
    the tag brackets around them. Also drops non-main-namespace pages
    (Talk:, User:, File:, Wikipedia: upload logs, etc. -- anything with a
    <ns> other than 0), which aren't article prose either.
    Returns None to signal "skip this page" (wrong namespace, or no <text>
    payload at all, e.g. a deleted/oversighted revision).
    """
    ns_match = _ns_re.search(article)
    if ns_match and ns_match.group(1) != '0':
        return None
    text_match = _text_re.search(article)
    if not text_match:
        return None
    return text_match.group(1)


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

def _is_kept_char(c):
    """
    True for anything that should survive final cleanup: alphanumerics,
    whitespace, and Unicode combining marks (category M*). NOT just
    isalnum()/isspace() -- str.isalnum() only recognizes Letter/Number
    categories, not Mark, so combining tone marks and vowel signs (e.g.
    Thai's MAI THO/MAI EK/etc., category Mn) would otherwise be silently
    stripped even though they're essential, meaning-changing parts of the
    base letter they attach to.
    """
    return c.isalnum() or c.isspace() or unicodedata.category(c).startswith('M')

def clean_text(text, patterns):
    """
    Applies the given patterns to the input text. Ensures lower-casing of all text.
    """
    txt = text
    for pattern in patterns:
        txt = pattern[0].sub(pattern[1], txt)
    txt = ''.join([letter for letter in txt if _is_kept_char(letter)])
    return txt.lower()

def clean_subtitles(language,version='2018',overwrite=False):
    """
    Prepare subtitle files for processing. version picks which vintage of
    raw/subtitles-{language}[-{version}].zip to read (see download()) --
    only affects filenames, not the zh/tw internal-folder translation below.
    """
    # Special case for OpenSubtitles names for Chinese variants
    language_tran = language
    if language == 'tw' :
        language_tran = 'zh_tw'
    elif language == 'zh':
        language_tran = 'zh_cn'
    suffix = '' if version == '2018' else f'-{version}'
    input_path = os.path.join(basedir,datadir,f'subtitles-{language}{suffix}.zip')
    output_name = f'subtitles-{language}{suffix}-pre.zip'
    output_path = os.path.join(basedir,processdir,output_name)

    if os.path.exists(output_path) and not overwrite:
        print(f'File {output_name} exists, and overwrite not specified. Skipping.');
    else:
        input_file = zipfile.ZipFile(input_path, 'r')
        if os.path.exists(output_path):
            os.remove(output_path)  # 'a' mode below would otherwise append to stale content on overwrite=True
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
        if os.path.exists(wiki_output_path):
            os.remove(wiki_output_path)  # 'a' mode below would otherwise append to stale content on overwrite=True
        with zipfile.ZipFile(wiki_output_path, 'a', zipfile.ZIP_DEFLATED) as output_archive:
            i = 0
            print(f'Preprocessing {language} Wikipedia dump.')
            for article in articles(language):
                text = extract_wiki_text(article)
                if text is None:
                    continue
                filename = f'wiki-{language}-{str(i)}.txt'
                txt = text.lower()
                txt = wiki_strip_circumflex(text) if ((not txt.startswith('#'))
                                     and ('<noinclude>' not in txt)
                                     and ('__noindex__' not in txt)
                                     ) else ''
                for pattern in wiki_patterns:
                    txt = pattern[0].sub(pattern[1], txt)

                output_archive.writestr(filename, ''.join([letter for letter in txt if _is_kept_char(letter)]))
                i += 1

            print("Complete")

def get_hash(tokens):
    """
    Creates the simhash for the given list of tokens.
    Builds the same 4-character shingles as the original shingle-then-hash
    implementation, and passes them as an unweighted, ungrouped list:
    Simhash()'s own text-mode groups repeated shingles and weights them by
    count, which overflows numpy's uint8 arithmetic on highly repetitive text.
    """
    text = ''.join(tokens)
    shingles = [text[i:i + 4] for i in range(max(len(text) - 3, 1))]
    return simhash.Simhash(shingles)

def prune(source, language, overwrite=False):
    """
    Remove near-duplicate documents from archive file using simhash.
    Files within `distance` Hamming bits of an earlier (kept) file are pruned.
    """
    input_path = os.path.join(basedir,processdir,f'{source}-{language}-pre.zip')
    output_path = os.path.join(basedir,processdir,f'{source}-{language}-pruned.zip')

    if os.path.exists(output_path) and not overwrite:
        print(f'File {source}-{language}-pruned.zip exists, and overwrite not specified. Skipping.');
        return

    input_file = zipfile.ZipFile(input_path, 'r')

    print("Checking for duplicates.")
    hashes = []
    for f in input_file.namelist():
        # NOT str(bytes) -- that gives the escaped repr ("b'\\xe0\\xb8...'")
        # instead of decoding, so every non-ASCII script's actual characters
        # get replaced by hex-escape fragments before shingling/simhashing.
        # Scripts confined to a narrow Unicode block (e.g. Thai, where nearly
        # every character shares the same UTF-8 lead bytes) then collapse to
        # near-identical fingerprints regardless of real content, causing
        # massive false-positive "duplicate" pruning.
        text = input_file.open(f).read().decode('utf-8', errors='replace')
        tokens = re.split(r'\W+', text.lower(), flags=re.UNICODE)
        hashes.append((f, get_hash(tokens)))

    input_file.close()

    distance = 2
    index = simhash.SimhashIndex(hashes, k=distance)

    to_remove = set()
    for f, h in hashes:
        if f in to_remove:
            continue
        for dup in index.get_near_dups(h):
            if dup != f:
                print(f'({f}, {dup})')
                to_remove.add(dup)

    print(f'Found {len(to_remove)} files to prune.')
    input_file = zipfile.ZipFile(input_path, 'r')
    if os.path.exists(output_path):
        os.remove(output_path)  # 'a' mode below would otherwise append to stale content on overwrite=True
    output_file = zipfile.ZipFile(output_path, 'a', zipfile.ZIP_DEFLATED)

    for f in input_file.namelist():
        if f not in to_remove:
            output_file.writestr(f, input_file.open(f).read())

    output_file.close()

# After cleaning, put together the whole corpus into one big text 
def concatenate_corpus(language,version='2018',overwrite=False):
    """
    Reads pruned (deduplicated) subtitle and wikipedia text, and creates a
    single text file containing all of the tokenized sentences.
    version picks which subtitle vintage to merge in (see clean()/download())
    -- the output filename and subtitles side both carry the version suffix,
    but the wikipedia side always reads the shared, version-agnostic
    wikipedia-{language}-pruned.zip (wikipedia has no dated-vintage concept).
    """
    subs_key = language if version == '2018' else f'{language}-{version}'
    corpus_key = language if version == '2018' else f'{language}-{version}'
    subs_input_path = os.path.join(basedir,processdir,f'subtitles-{subs_key}-pruned.zip')
    wiki_input_path = os.path.join(basedir,processdir,f'wikipedia-{language}-pruned.zip')
    corpus_output_path = os.path.join(basedir,corpusdir,f'corpus-{corpus_key}.txt')
    if os.path.exists(corpus_output_path) and not overwrite:
        print(f'File corpus-{corpus_key}.txt exists, and overwrite not specified. Skipping.');
    else:
        print(f"Concatenating {corpus_key} corpus.")
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
