import pandas as pd
import word2manylanguages as w
import os

basedir = './models'
language = 'es'
dim = 300
win = 5
alg = 1

algo = 'cbow' if alg ==0 else 'sg'
base_file_name = f'{language}_{str(dim)}_{str(win)}_{algo}'
print("Evaluating model " + base_file_name) 
path = os.path.join(basedir, f'{base_file_name}_wxd.csv')
wordsXdims = pd.read_csv(path)
cols = wordsXdims.columns.tolist()
cols[0] = 'word'
wordsXdims.columns = cols;
wordsXdims.set_index('word', inplace=True)
print(wordsXdims.head())
scores = w.evaluate_norms(language, wordsXdims)
print(scores)
