"""
Progress tracker for the 59-language word2manylanguages pipeline.

For each language, checks:
  - models/          : how many of the 60 trained model files exist
                        (5 dims x 6 windows x 2 algos)
  - eval_results/counts     : whether the frequency-counts eval file exists,
                        and how many of the present models it has scored
  - eval_results/norms      : same, but only for languages that actually
                        appear in eval_inputs/datasets_norms.csv
  - eval_results/replication: same, but only for languages that have a
                        replication-norms file in eval_inputs/replication

Re-run any time to refresh: `python progress_tracker.py`
Writes progress_tracker.md next to this script.
"""
import os
import pandas as pd

basedir = os.path.dirname(os.path.abspath(__file__))
modeldir = os.path.join(basedir, 'models')
evaldir = os.path.join(basedir, 'eval_results')
datasetsdir = os.path.join(basedir, 'eval_inputs')
zenodo_dois_csv = os.path.join(basedir, 'download', 'zenodo_dois.csv')

dimension_list = [50, 100, 200, 300, 500]
window_list = [1, 2, 3, 4, 5, 6]
algo_list = ['cbow', 'sg']

# code -> language name, matches 03_evaluation/evaluation.py's code2lang
code2lang = {
    'af': 'afrikaans', 'ar': 'arabic', 'bg': 'bulgarian', 'bn': 'bengali',
    'br': 'breton', 'bs': 'bosnian', 'ca': 'catalan', 'cs': 'czech',
    'da': 'danish', 'de': 'german', 'el': 'greek', 'en': 'english',
    'eo': 'esperanto', 'es': 'spanish', 'et': 'estonian', 'eu': 'basque',
    'fa': 'farsi', 'fi': 'finnish', 'fr': 'french', 'gl': 'galacian',
    'he': 'hebrew', 'hi': 'hindi', 'hr': 'croatian', 'hu': 'hungarian',
    'hy': 'armenian', 'id': 'indonesian', 'is': 'icelandic', 'it': 'italian',
    'ja': 'japanese', 'ka': 'georgian', 'kk': 'kazakh', 'ko': 'korean',
    'lt': 'lithuanian', 'lv': 'latvian', 'mk': 'macedonian',
    'ml': 'malayalam', 'ms': 'maylay', 'nl': 'dutch', 'no': 'norwegian',
    'pl': 'polish', 'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian',
    'si': 'sinhalese', 'sk': 'slovak', 'sl': 'slovenian', 'sq': 'albanian',
    'sr': 'serbian', 'sv': 'swedish', 'ta': 'tamil', 'te': 'telugu',
    'th': 'thai', 'tl': 'tagalog', 'tr': 'turkish', 'tw': 'taiwanese',
    'uk': 'ukrainian', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese',
}

# Target Zenodo version for the current upload round -- every language
# needs v2 except af, which already reached v2 previously and now needs
# v3. Update this by hand at the start of the next round (once every
# language below is checked off, i.e. already at its target version).
zenodo_target_version = {'_default': 2, 'af': 3}

lang_aliases = {
    'farsi': {'farsi', 'persian'},
    'galacian': {'galacian', 'galician'},
    'maylay': {'maylay', 'malay'},
    'chinese': {'chinese', 'chinese_simplified', 'chinese_traditional'},
}

display_names = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali',
    'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech',
    'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English',
    'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque',
    'fa': 'Farsi', 'fi': 'Finnish', 'fr': 'French', 'gl': 'Galician',
    'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian',
    'hy': 'Armenian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian',
    'ja': 'Japanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'ko': 'Korean',
    'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian',
    'ml': 'Malayalam', 'ms': 'Malay', 'nl': 'Dutch', 'no': 'Norwegian',
    'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
    'si': 'Sinhalese', 'sk': 'Slovak', 'sl': 'Slovenian', 'sq': 'Albanian',
    'sr': 'Serbian', 'sv': 'Swedish', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish',
    'tw': 'Taiwanese / Traditional Chinese', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'zh': 'Chinese / Mandarin',
}

languages = sorted(code2lang.keys())
assert len(languages) == 59, f'expected 59 languages, got {len(languages)}'


def model_combos():
    return [(dim, win, alg)
            for dim in dimension_list
            for win in window_list
            for alg in algo_list]


def existing_models(lang):
    found = []
    for dim, win, alg in model_combos():
        base = f'{lang}_{dim}_{win}_{alg}_wxd.csv'
        if os.path.exists(os.path.join(modeldir, base)) or \
           os.path.exists(os.path.join(modeldir, base + '.bz2')):
            found.append(base)
    return found


def needs_norms(lang):
    catalogpath = os.path.join(datasetsdir, 'datasets_norms.csv')
    catalog = pd.read_csv(catalogpath)
    langname = code2lang[lang]
    accepted = lang_aliases.get(langname, {langname})
    match = catalog[catalog['language'].fillna('').apply(
        lambda cell: bool(accepted & set(cell.split('|'))))]
    return len(match) > 0


def load_zenodo_versions():
    """
    Maps each language to the highest Zenodo version it's already been
    uploaded as (parsed from the 'vN' strings in download/zenodo_dois.csv),
    so the tracker can show what the *next* upload would be (current max +
    1) without hand-maintaining that number as uploads happen.
    """
    versions = {}
    if not os.path.exists(zenodo_dois_csv):
        return versions
    df = pd.read_csv(zenodo_dois_csv, usecols=['language', 'zenodo_version'])
    for lang, group in df.groupby('language')['zenodo_version']:
        nums = [int(v.lstrip('v')) for v in group.unique()]
        versions[lang] = max(nums)
    return versions


def needs_replication(lang):
    repdir = os.path.join(datasetsdir, 'replication')
    return any(fname.startswith(lang + '-')
               for fname in os.listdir(repdir))


def eval_sources(eval_csv):
    """Distinct model 'source' values already scored in an eval_results file."""
    if not os.path.exists(eval_csv) or os.path.getsize(eval_csv) == 0:
        return set()
    try:
        df = pd.read_csv(eval_csv, usecols=['source'])
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(df['source'].unique())


def check(mark):
    return 'x' if mark else ' '


def build_row(lang, zenodo_versions):
    models = existing_models(lang)
    n_models = len(models)
    model_names = {m[:-len('_wxd.csv')] for m in models}

    counts_csv = os.path.join(evaldir, 'counts', f'{lang}_eval.csv')
    counts_done = eval_sources(counts_csv)
    counts_ok = n_models > 0 and model_names <= counts_done

    norms_needed = needs_norms(lang)
    norms_csv = os.path.join(evaldir, 'norms', f'{lang}_eval.csv')
    norms_done = eval_sources(norms_csv) if norms_needed else set()
    norms_ok = (not norms_needed) or (n_models > 0 and model_names <= norms_done)

    rep_needed = needs_replication(lang)
    rep_csv = os.path.join(evaldir, 'replication', f'{lang}_eval.csv')
    rep_done = eval_sources(rep_csv) if rep_needed else set()
    rep_ok = (not rep_needed) or (n_models > 0 and model_names <= rep_done)

    target = zenodo_target_version.get(lang, zenodo_target_version['_default'])
    zenodo_ok = zenodo_versions.get(lang, 0) >= target

    return {
        'code': lang,
        'name': display_names[lang],
        'n_models': n_models,
        'models_ok': n_models == 60,
        'counts_done': len(counts_done),
        'counts_ok': counts_ok,
        'norms_needed': norms_needed,
        'norms_done': len(norms_done),
        'norms_ok': norms_ok,
        'rep_needed': rep_needed,
        'rep_done': len(rep_done),
        'rep_ok': rep_ok,
        'zenodo_ok': zenodo_ok,
        'zenodo_target': target,
    }


def counts_cell(row):
    # every language eventually needs a counts eval, regardless of whether
    # its models have been trained yet
    return f"[{check(row['counts_ok'])}] {int(row['counts_ok'])}/1"


def norms_cell(row):
    if not row['norms_needed']:
        return 'n/a'
    return f"[{check(row['norms_ok'])}] {int(row['norms_ok'])}/1"


def zenodo_cell(row):
    return f"[{check(row['zenodo_ok'])}] v{row['zenodo_target']}"


def rep_cell(row):
    if not row['rep_needed']:
        return 'n/a'
    return f"[{check(row['rep_ok'])}] {int(row['rep_ok'])}/1"


def build_markdown(rows):
    total = len(rows)
    models_done = sum(r['models_ok'] for r in rows)
    counts_done = sum(r['counts_ok'] for r in rows)
    norms_needed = sum(r['norms_needed'] for r in rows)
    norms_done = sum(r['norms_ok'] for r in rows if r['norms_needed'])
    rep_needed = sum(r['rep_needed'] for r in rows)
    rep_done = sum(r['rep_ok'] for r in rows if r['rep_needed'])
    zenodo_done = sum(r['zenodo_ok'] for r in rows)

    total_items = total + total + norms_needed + rep_needed + total
    done_items = models_done + counts_done + norms_done + rep_done + zenodo_done
    overall_pct = 100 * done_items / total_items

    # Raw object counts -- actual files/rows produced, not just "language
    # fully done or not". Sums (n_models) instead of language-level flags,
    # so partial progress within a language shows up too.
    model_files = sum(r['n_models'] for r in rows)
    model_files_possible = total * 60
    counts_rows = sum(r['counts_done'] for r in rows)
    counts_rows_possible = model_files
    norms_rows = sum(r['norms_done'] for r in rows if r['norms_needed'])
    norms_rows_possible = sum(r['n_models'] for r in rows if r['norms_needed'])
    rep_rows = sum(r['rep_done'] for r in rows if r['rep_needed'])
    rep_rows_possible = sum(r['n_models'] for r in rows if r['rep_needed'])

    lines = []
    lines.append('# word2manylanguages progress tracker')
    lines.append('')
    lines.append('Re-run `python progress_tracker.py` any time to refresh this file.')
    lines.append('')
    lines.append(f'**Overall: {overall_pct:.1f}% ({done_items}/{total_items} items complete)**')
    lines.append('')
    lines.append('Raw object counts (actual files/model-results produced, not just "language fully done"):')
    lines.append(f'- Model files trained: {model_files}/{model_files_possible}')
    lines.append(f'- Counts eval model-results written: {counts_rows}/{counts_rows_possible}')
    lines.append(f'- Norms eval model-results written: {norms_rows}/{norms_rows_possible}')
    lines.append(f'- Replication eval model-results written: {rep_rows}/{rep_rows_possible}')
    lines.append('')
    lines.append('Language-level completion (all applicable model-results present for that language):')
    lines.append(f'- Models complete (60/60): {models_done}/{total}')
    lines.append(f'- Counts eval complete: {counts_done}/{total}')
    lines.append(f'- Norms eval complete: {norms_done}/{norms_needed}')
    lines.append(f'- Replication eval complete: {rep_done}/{rep_needed}')
    lines.append(f'- Zenodo upload at target version: {zenodo_done}/{total}')
    lines.append('')
    lines.append('| | Code | Language | Models | Counts | Norms | Replication | Zenodo |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        row_mark = 'x' if (r['models_ok'] and r['counts_ok'] and r['norms_ok'] and r['rep_ok'] and r['zenodo_ok']) else ' '
        lines.append(
            f"| [{row_mark}] | {r['code']} | {r['name']} | "
            f"[{check(r['models_ok'])}] {r['n_models']}/60 | "
            f"{counts_cell(r)} | "
            f"{norms_cell(r)} | {rep_cell(r)} | {zenodo_cell(r)} |"
        )
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    zenodo_versions = load_zenodo_versions()
    rows = [build_row(lang, zenodo_versions) for lang in languages]
    md = build_markdown(rows)
    outpath = os.path.join(basedir, 'progress_tracker.md')
    with open(outpath, 'w') as f:
        f.write(md)
    print(f'Wrote {outpath}')
