"""
Plain-script version of compare_algorithms_ubuntu.ipynb, for streaming live
progress output under tmux instead of `jupyter nbconvert --execute`'s
buffered-until-done behavior.

Usage (from the repo root, inside tmux, with the venv activated):
    python3 -u experiments/fasttext_vs_word2vec/run_ar_zh_ubuntu.py \
        2>&1 | tee -a experiments/fasttext_vs_word2vec/ar_zh_run.log

`-u` disables Python's stdout buffering so prints show up immediately rather
than only when the buffer fills; `tee -a` mirrors everything to a log file
(appending, so rerunning after an interruption doesn't overwrite earlier
output) while still showing it live in the terminal.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import compare_lib as cl

cl.basedir = REPO_ROOT

version = "2018"
configs = [(50, 1), (100, 2), (200, 3), (300, 4), (500, 6)]

# This machine has 32 physical cores available -- set explicitly rather than
# relying on model_training.py's auto default (cpu_count() - 1).
workers = 32

print(f"REPO_ROOT = {REPO_ROOT}")
print(f"workers   = {workers}")

languages = ["ar", "zh"]
summary = cl.run_comparison_batch(languages, version=version, configs=configs, workers=workers)
print(summary)
