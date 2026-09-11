import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "02_model_training"))

import model_training as mt

mt.basedir = HERE

language = "en"   # subs_key -- 'en' for 2018 vintage

# shrink to just the configs you need (comment out / edit as needed)
mt.dimension_list = [50]        # subset of 50/100/200/300/500
mt.window_list = [4]            # subset of 1-6
mt.algo_list = ['cbow', 'sg']   # subset of cbow/sg

# mt.workers = os.cpu_count() - 1  # default; override if you want to cap it

mt.build_models(language)   # -> models/en_{dim}_{window}_{algo}_wxd.csv.bz2
