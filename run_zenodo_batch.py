# %% [markdown]
# # word2manylanguages: overnight Zenodo upload batch
#
# Loops `languages` below through step 8 (Zenodo upload) + 8b (commit/push
# the new DOI rows) of `run_language_pipeline.py`, for every language whose
# models are already fully trained. Meant to run unattended overnight, so
# unlike step 8's normal dry-run/diff review gates, this commits to
# dry_run=False and pushes automatically once the dry-run plan is printed
# to the log.
#
# Zenodo uploads have been unreliable above a few GB/request (see
# download/zenodo_upload.py's module docstring -- sync_language() already
# retries individual HTTP requests internally). This adds a second,
# coarser retry layer on top: if a whole sync_language() call raises (e.g.
# the connection drops entirely, or all its internal retries are
# exhausted), wait 5 minutes and retry the same language up to
# max_retries times before giving up on it and moving to the next one --
# same "one language failing doesn't stop the rest" pattern as
# run_language_pipeline_batch.py.
#
# Usage (from the repo root, inside tmux/nohup for an overnight run):
#   python3 -u run_zenodo_batch.py 2>&1 | tee -a zenodo_batch.log

# %%
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.join(HERE, "download"))
import zenodo_upload as zu

env_path = os.path.join(HERE, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# two-letter codes -- everything below has models fully trained (60/60,
# per progress_tracker.md) but isn't yet at its target Zenodo version.
# Edit freely: drop languages you don't want tonight, or add more once
# their models finish.
languages = ["it", "ja", "ka", "kk", "ko", "lt", "lv",
    "mk", "ml", "ms", "nl", "no", "pl", "pt", "ro", "ru", "si", "sk", "sl",
    "sq", "sv", "ta", "te", "th", "tr", "ur", "vi", "zh",
]
version = "2018"       # '2018' or '2024' -- see run_language_pipeline.py's module docstring
max_retries = 5         # whole-call retries per language before giving up on it
retry_wait_seconds = 300  # 5 minutes


def sync_and_commit_one_language(language, version="2018"):
    """Step 8 + 8b of run_language_pipeline.py for a single language."""
    subs_key = language if version == "2018" else f"{language}-{version}"
    models_dir = os.path.join(HERE, "models")

    zu.sync_language(subs_key, version, models_dir, dry_run=True)
    zu.sync_language(subs_key, version, models_dir, dry_run=False)

    diff = subprocess.run(
        ["git", "diff", "--", "download/zenodo_dois.csv"],
        cwd=HERE, capture_output=True, text=True,
    ).stdout
    if not diff.strip():
        print(f"  no new rows in download/zenodo_dois.csv for {subs_key} -- skipping commit/push", flush=True)
        return
    print(diff, flush=True)

    subprocess.run(["git", "add", "download/zenodo_dois.csv"], cwd=HERE, check=True)
    subprocess.run(["git", "commit", "-m", f"Add {subs_key} ({version}) Zenodo DOIs"], cwd=HERE, check=True)
    subprocess.run(["git", "push"], cwd=HERE, check=True)


# %%
results = {}
for language in languages:
    print(f"\n{'=' * 60}\n{language}\n{'=' * 60}", flush=True)

    for attempt in range(1, max_retries + 1):
        try:
            sync_and_commit_one_language(language, version=version)
            results[language] = "ok"
            break
        except Exception as e:
            print(f"  attempt {attempt}/{max_retries} FAILED for {language}: {e}", flush=True)
            results[language] = f"failed: {e}"
            if attempt < max_retries:
                print(f"  waiting {retry_wait_seconds}s before retrying {language}...", flush=True)
                time.sleep(retry_wait_seconds)
    else:
        print(f"  giving up on {language} after {max_retries} attempts", flush=True)

print("\n\nBatch summary:")
for language, status in results.items():
    print(f"  {language}: {status}")
