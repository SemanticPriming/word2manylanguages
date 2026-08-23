import sys, os, re

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "03_evaluation"))
import evaluation as ev
ev.basedir = HERE

# derive the set of languages/subs_keys present in models/
subs_keys = sorted({
    re.sub(r"_\d+_\d+_(cbow|sg)_wxd\.csv\.bz2$", "", f)
    for f in os.listdir(os.path.join(HERE, "models"))
    if f.endswith("_wxd.csv.bz2")
})

for subs_key in subs_keys:
    if "-" in subs_key:
        lang, version = subs_key.rsplit("-", 1)
    else:
        lang, version = subs_key, "2018"
    print(f"\n=== {subs_key} ===", flush=True)
    try:
        ev.evaluate_language(lang, version=version)
    except Exception as e:
        print(f"  FAILED: {subs_key}: {e}", flush=True)
