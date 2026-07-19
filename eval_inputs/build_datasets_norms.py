import os, csv, re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "norms")
OUT = os.path.join(HERE, "datasets_norms.csv")

# ---------------------------------------------------------------------------
# Language vocabulary: literal tokens used in this project's word_<language>
# columns (superset of evaluation.py's code2lang, plus tokens observed
# empirically in eval_inputs/norms headers, incl. multi-token names).
# Ordered longest-first so multi-token names match before their prefix does.
# ---------------------------------------------------------------------------
LANGUAGE_SEQUENCES = [
    ('south', 'african', 'english'),
    ('chinese', 'simplified'), ('chinese', 'traditional'),
    ('afrikaans',), ('arabic',), ('armenian',), ('bulgarian',), ('catalan',),
    ('chinese',), ('croatian',), ('czech',), ('danish',), ('dutch',),
    ('english',), ('esperanto',), ('estonian',), ('basque',), ('farsi',),
    ('persian',), ('finnish',), ('french',), ('gaelic',), ('galician',),
    ('galacian',), ('georgian',), ('german',), ('greek',), ('hebrew',),
    ('hindi',), ('hungarian',), ('icelandic',), ('indonesian',), ('irish',),
    ('isixhosa',), ('italian',), ('japanese',), ('kazakh',), ('korean',),
    ('lithuanian',), ('latvian',), ('luxembourgish',), ('macedonian',),
    ('malayalam',), ('malay',), ('maltese',), ('norwegian',), ('polish',),
    ('portuguese',), ('romanian',), ('russian',), ('sinhalese',), ('slovak',),
    ('slovenian',), ('albanian',), ('serbian',), ('swahili',), ('swedish',),
    ('spanish',), ('tamil',), ('telugu',), ('thai',), ('tagalog',), ('turkish',),
    ('ukrainian',), ('urdu',), ('vietnamese',), ('welsh',), ('bengali',),
    ('breton',), ('bosnian',), ('pseudoword',),
]
LANGUAGE_SEQUENCES.sort(key=lambda t: -len(t))

WORD_PREFIXES = [
    'translate_word_', 'translation_word_', 'translated_word_',
    'paired_word_', 'word_',
]

STAT_WORDS = {
    'mean', 'sd', 'n', 'se', 'min', 'minimum', 'max', 'maximum', 'median',
    'mode', 'zscore', 'z', 'rt', 'proportion', 'percent', 'ci', 'unknown',
    'count', 'total', 'blup', 'prop', 'skew', 'kurtosis', 'variance',
    'range', 'h', 'p', 'k',
}

# ---------------------------------------------------------------------------
# Concept normalization table: raw (stat-stripped) concept -> basic/normed
# name, restricted to the constructs we actually want to predict from
# embeddings. Matching: exact match first, then longest-key "startswith"
# fallback. Anything NOT in this table returns None and its row is dropped
# entirely (see normalize_concept) -- this is a curated target list, not a
# full data dictionary of every column in every dataset.
# ---------------------------------------------------------------------------
CONCEPT_MAP = OrderedDict()
def add(basic, *raws):
    for r in raws:
        CONCEPT_MAP[r] = basic

# --- core requested constructs ---
# includes classic Osgood semantic-differential vocabulary: Evaluation /
# Potency / Activity are the textbook equivalents of Valence / Dominance /
# Arousal, and pleasant-unpleasant / good-bad, weak-strong, active-passive
# are their defining bipolar adjective pairs (Heise1965, Jenkins1958,
# Silverstein1968, Bellezza1986, Brown1969, Toglia1978, Schauenburg2015)
add('valence', 'valence', 'valance', 'social_valence', 'iaps_valence_category',
    'pleasant', 'unpleasant', 'evaluation_dimension', 'good_bad')
add('arousal', 'arousal', 'activity_dimension', 'active_passive')
add('dominance', 'dominance', 'pos_dominance', 'potency', 'potency_dimension',
    'weak_strong')
add('concreteness', 'concrete', 'concrete_category', 'abstractness_category')
add('familiarity', 'familiar', 'affective', 'familiar_gpt', 'familiar_gpt_finetuned',
    'familiar_practice', 'familiar_know_meaning', 'familiar_blup')
add('imageability', 'imagine', 'imagery', 'imagne', 'imagine_stresiduals',
    'imagine_known_meaning')
add('aoa', 'aoa', 'aoa_subjective', 'aoa_objective', 'aoa_certainty',
    'aoa_recognition', 'aoa_overall', 'aoa_known_meaning', 'age_of_learning')

# all emotion constructs (discrete emotions + generic emotion-adjacent
# ratings) share the single "emotion" label; the specific emotion (happiness,
# anger, taboo, etc.) is preserved in variable_stat/variable_original instead
add('emotion',
    'emotion',
    'happiness', 'unknown_happiness', 'happy',
    'anger', 'unknown_anger',
    'sad', 'sadness', 'unknown_sadness',
    'fear', 'unknown_fear',
    'disgust', 'unknown_disgust',
    'surprise',
    'emotionality', 'emotionalcharge', 'emotion_intensity', 'emotional',
    'taboo', 'taboo_personal', 'insulting', 'offensiveness')

# sensory / motor norms, consolidated into one bucket per user request
add('sensory',
    'sensory',
    'boi', 'body_object_interaction', 'body_object_interact',
    'auditory', 'sound', 'auditory_strength', 'auditory_modality', 'auditory_exclusivity',
    'visual', 'visual_strength', 'visual_modality', 'visual_exclusivity',
    'haptic', 'touch', 'haptic_strength', 'haptic_modality',
    'gustatory', 'taste', 'gustatory_strength', 'gustatory_modality', 'gustatory_exclusivity',
    'olfactory', 'smell', 'olfactory_strength', 'olfactory_modality', 'olfactory_exclusivity',
    'interoception', 'interoceptive',
    'tactile', 'tactile_exclusivity', 'grasp', 'mime',
    'action_effector', 'action_effector_feet', 'action_effector_hands',
    'action_effector_mouth', 'action_effector_torso', 'motion',
    'manipulability', 'manip', 'manip1', 'manip2')

CONCEPT_KEYS_SORTED = sorted(CONCEPT_MAP.keys(), key=len, reverse=True)

# stat suffixes that count as "the mean" for filtering purposes; anything
# with an explicit non-mean stat (sd, n, min, max, zscore, rt, etc.) is
# dropped -- we only want to predict central-tendency (mean) columns
MEAN_STATS = {'', 'mean'}

# Columns with NO stat suffix at all (e.g. a bare "valence" column) are only
# trustworthy as "the mean" when the column IS the bare root name and
# nothing else -- no modifiers, no aliases. A column like "aoa_subjective"
# or "concrete_category" or "valence_category" is NOT accepted bare: those
# turned out to be regression estimates, p-values, ANOVA output, sex-diff
# scores, and categorical bins, not mean ratings, when spot-checked. When a
# stat suffix (e.g. "_mean") IS present, the full CONCEPT_MAP (with prefix
# fallback for compounds like "aoa_subjective_mean") still applies.
BARE_ROOTS = {
    'valence', 'arousal', 'dominance', 'concrete', 'familiar', 'imagine',
    'aoa',
    'happiness', 'happy', 'anger', 'sad', 'sadness', 'fear', 'disgust', 'surprise',
    'emotionality', 'taboo', 'offensiveness', 'emotion',
    'boi', 'body_object_interaction', 'auditory', 'visual', 'haptic',
    'gustatory', 'olfactory', 'tactile', 'interoceptive', 'interoception',
    'sensory',
    'pleasant', 'unpleasant', 'evaluation_dimension', 'good_bad',
    'activity_dimension', 'active_passive', 'potency', 'potency_dimension',
    'weak_strong',
}


# Compounds that would otherwise prefix-match a sensory root (e.g.
# "visual_complexity_mean" starts with "visual_") but measure something
# else entirely -- picture-naming visual complexity, not how strongly a
# word evokes visual sensory experience. Checked before prefix fallback.
EXCLUDE_CONCEPTS = {'visual_complexity', 'visualcomplexity'}


def normalize_concept(concept, stat):
    if concept in EXCLUDE_CONCEPTS:
        return None
    if stat == '':
        if concept not in BARE_ROOTS:
            return None
        return CONCEPT_MAP.get(concept)
    if concept in CONCEPT_MAP:
        return CONCEPT_MAP[concept]
    for key in CONCEPT_KEYS_SORTED:
        if concept == key or concept.startswith(key + '_'):
            return CONCEPT_MAP[key]
    return None


def match_language(tokens, start):
    """Try to match a language name sequence starting at tokens[start]."""
    for seq in LANGUAGE_SEQUENCES:
        n = len(seq)
        if tuple(tokens[start:start + n]) == seq:
            return '_'.join(seq), start + n
    return None, None


def find_word_column(col_lower):
    for prefix in WORD_PREFIXES:
        if col_lower.startswith(prefix):
            rest = col_lower[len(prefix):]
            tokens = rest.split('_') if rest else []
            # language usually comes first (word_spanish) but sometimes after
            # a modifier (word_idiom_german), so scan all positions
            for i in range(len(tokens)):
                lang, _ = match_language(tokens, i)
                if lang:
                    return lang
            return None  # word_-prefixed but not a recognized language (e.g. word_class)
    return None


def find_language_in_tokens(tokens):
    for i in range(len(tokens)):
        lang, end = match_language(tokens, i)
        if lang:
            return lang, tokens[:i] + tokens[end:]
    return None, tokens


def split_stat(tokens):
    for i in range(1, len(tokens)):
        if tokens[i] in STAT_WORDS:
            return '_'.join(tokens[:i]), tokens[i], tokens[i + 1:]
    return '_'.join(tokens), '', []


rows = []
files = sorted(f for f in os.listdir(D) if f.endswith('.csv'))
for fname in files:
    path = os.path.join(D, fname)
    with open(path, encoding='utf-8', errors='replace') as fh:
        header = fh.readline().strip().lstrip('﻿')
    cols = [c.strip().strip('"') for c in header.split(',') if c.strip()]

    file_languages = []
    measurement_cols = []
    for col in cols:
        low = col.lower()
        lang = find_word_column(low)
        if lang:
            if lang not in file_languages:
                file_languages.append(lang)
        elif any(low.startswith(p) for p in WORD_PREFIXES):
            # word_-prefixed but not a language (e.g. word_class, word_aoa_mean)
            measurement_cols.append(col)
        else:
            measurement_cols.append(col)

    default_lang = file_languages[0] if len(file_languages) == 1 else (
        '|'.join(file_languages) if file_languages else ''
    )

    for col in measurement_cols:
        low = col.lower()
        tokens = low.split('_')
        # "word_aoa_mean" / "translation_concrete_mean" (Prior2007 pairs-style
        # files): word_/translation_ here is a qualifier (primary item vs.
        # its translation), not a language tag or part of the concept name.
        # Strip it if the remainder parses to a real mean-suffixed concept.
        qualifier = None
        if len(tokens) > 2 and tokens[0] in ('word', 'translation'):
            cand_concept, cand_stat, _ = split_stat(tokens[1:])
            if cand_stat and normalize_concept(cand_concept, cand_stat) is not None:
                qualifier = tokens[0]
                tokens = tokens[1:]
        concept, stat, group_tokens = split_stat(tokens)
        # look for an embedded language name in the group (or whole concept
        # if no stat was found) and use it to override the file-level language
        search_tokens = group_tokens if stat else tokens
        lang_found, remaining = find_language_in_tokens(search_tokens)
        if lang_found:
            row_lang = lang_found
            if stat:
                group_tokens = remaining
            else:
                concept = '_'.join(remaining) if remaining else concept
        else:
            row_lang = default_lang

        if stat not in MEAN_STATS:
            continue  # only predicting mean columns -- drop sd/n/se/min/max/etc.

        variable_normed = normalize_concept(concept, stat)
        if variable_normed is None:
            continue  # not one of the constructs we're targeting

        variable_stat = f"{concept}_{stat}" if stat else concept
        if qualifier:
            group_tokens = group_tokens + [qualifier]
        variable_stat_group = '_'.join(group_tokens)

        rows.append({
            'dataset': fname,
            'language': row_lang,
            'variable_normed': variable_normed,
            'variable_stat': variable_stat,
            'variable_stat_group': variable_stat_group,
            'variable_original': col,
        })

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'dataset', 'language', 'variable_normed', 'variable_stat',
        'variable_stat_group', 'variable_original'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")

# quick coverage report
from collections import Counter
c = Counter(r['variable_normed'] for r in rows)
print("\nCategory counts:")
for k, v in c.most_common():
    print(f"  {k:20s} {v}")
print(f"\ndatasets represented: {len(set(r['dataset'] for r in rows))}")
