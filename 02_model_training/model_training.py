###### libraries ######
import os
import pandas as pd
from gensim.models import FastText

###### define directories ######
# basedir defined in main file
corpusdir = "corpora"
modeldir = 'models'

##### read the concatenated corpus for gensim #####
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
