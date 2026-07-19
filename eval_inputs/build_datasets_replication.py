import os, csv, re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "replication")
OUT = os.path.join(HERE, "datasets_replication.csv")

CODE2LANG = {
    'de': 'german', 'en': 'english', 'es': 'spanish', 'fa': 'farsi',
    'fi': 'finnish', 'fr': 'french', 'id': 'indonesian', 'it': 'italian',
    'ms': 'malay', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese',
    'tr': 'turkish',
}

# Same basic-name vocabulary as datasets_norms.csv (see that script), so
# both catalogs can be filtered/joined on variable_normed. UNLIKE the norms
# catalog, every column is kept here (not just the core constructs) -- the
# evaluation pipeline's evaluate_replication() predicts every column in
# these files (see 03_evaluation/evaluation.py: predict() uses
# targets.columns.values with no filtering), so the catalog needs to cover
# all of them too. Anything that doesn't match a known construct is labeled
# "other" rather than dropped.
REPLICATION_MAP = OrderedDict()
def add(basic, *raws):
    for r in raws:
        REPLICATION_MAP[r] = basic

add('valence', 'valence', 'pleasant', 'unpleasant')
add('arousal', 'arousal')
add('dominance', 'dominance', 'potency')
add('concreteness', 'concreteness')
add('familiarity', 'familiarity')
add('imageability', 'imageability', 'imagery',
    'imageability (young adults)', 'imageability (older adults)')
add('aoa', 'age of acquisition')

# all emotion constructs (discrete emotions + generic emotion-adjacent
# ratings) share the single "emotion" label; the specific emotion (happy,
# anger, taboo, etc.) is preserved in variable_original instead
add('emotion',
    'happy', 'happiness', 'angry', 'anger', 'sad', 'sadness',
    'fearful', 'fear', 'disgusted', 'disgust', 'surprised',
    'emotional charge', 'emotionality (young adults)',
    'emotionality (older adults)', 'tabooness', 'taboo (general)',
    'taboo (personal)', 'insulting', 'offensiveness')

# sensory / motor norms, consolidated into one bucket per user request.
# Includes Binder-style raw attribute names (vision, loud, touch, ...) and
# the Diez-Alamo-style perceptual-dimension names (color vividness, sound
# intensity, ...) since those measure the same modality-strength constructs,
# just under different source vocabularies.
add('sensory',
    'body object interaction',
    'audition', 'sound', 'auditory', 'auditory perceptual strength',
    'loud', 'low', 'high', 'music', 'speech', 'sound intensity',
    'visual', 'visual perceptual strength', 'vision', 'bright', 'dark',
    'color', 'pattern', 'motion', 'biomotion', 'fast', 'slow', 'shape',
    'complexity', 'face', 'near', 'toward', 'away', 'scene', 'large', 'small',
    'color vividness', 'visual motion',
    'touch', 'haptic', 'graspability', 'temperature', 'texture', 'weight',
    'pain', 'risk of pain',
    'taste', 'gustatory', 'pleasant taste',
    'smell', 'olfactory', 'smell intensity',
    'tactile',
    'interoceptive',
    'body', 'head', 'upperlimb', 'lowerlimb', 'practice',
    'foot and leg', 'hand and arm', 'mouth', 'torso', 'motor content',
    'sensory experience', 'modality exclusivity', 'perceptual exclusivity',
    'action exclusivity', 'sensorimotor exclusivity')

CONCEPT_KEYS_SORTED = sorted(REPLICATION_MAP.keys(), key=len, reverse=True)


def normalize(col_lower):
    if col_lower in REPLICATION_MAP:
        return REPLICATION_MAP[col_lower]
    base = re.sub(r'\s*\([^)]*\)\s*$', '', col_lower).strip()
    if base in REPLICATION_MAP:
        return REPLICATION_MAP[base]
    return 'other'


rows = []
files = sorted(f for f in os.listdir(D) if f.endswith('.tsv'))
for fname in files:
    code = fname.split('-', 1)[0]
    language = CODE2LANG.get(code, code)
    path = os.path.join(D, fname)
    with open(path, encoding='utf-8', errors='replace') as fh:
        header = fh.readline().rstrip('\n').lstrip('﻿')
    cols = [c.strip() for c in header.split('\t') if c.strip()]
    for col in cols:
        if col.lower() == 'word':
            continue
        variable_normed = normalize(col.lower())
        rows.append({
            'dataset': fname,
            'language': language,
            'variable_normed': variable_normed,
            'variable_original': col,
        })

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'dataset', 'language', 'variable_normed', 'variable_original'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")

from collections import Counter
c = Counter(r['variable_normed'] for r in rows)
for k, v in c.most_common():
    print(f"  {k:20s} {v}")
print(f"\ndatasets represented: {len(set(r['dataset'] for r in rows))}")
