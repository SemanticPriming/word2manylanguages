import os
import pandas as pd
import word2manylanguages as w
import sys

language = sys.argv[1]
basedir = '.'
if len(sys.argv) > 2:
    basedir = sys.argv[2]
outdir = '.'
if len(sys.argv) > 3:
    outdir = sys.argv[3]

outdir = os.path.join(outdir, language+'_eval')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for dim in w.dimension_list:
        for win in w.window_list:
            for alg in w.algo_list:
                algo = 'cbow' if alg ==0 else 'sg'
                base_file_name = f'{language}_{str(dim)}_{str(win)}_{algo}'
                print("Evaluating model " + base_file_name) 
                path = os.path.join(basedir, f'{base_file_name}_wxd.csv')
                wordsXdims = pd.read_csv(path)
                scores = w.evaluate_norms(language, wordsXdims)
                if scores == None:
                    print("No evaluation data for model.")
                else:
                    path = os.path.join(outdir, base_file_name)
                    if not os.path.exists(outdir):
                        os.makedirs(path)
                    for score in scores:
                        fname = score['source'] + '_result_' + base_file_name + '.csv'
                        outfile = os.path.join(path, fname)
                        scores.to_csv(outfile)
