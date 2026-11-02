"""
Step 3: Create train/val/test splits from {processed-dir}/{lang}.jsonl.

IMPORTANT: xMMMU rows are parallel across languages, identified by a shared
"match_id" (the original MMMU question id, e.g. "validation_Topic_17") -
NOT by row position. The split is decided once per match_id and applied to
every language that has that match_id, so a question never sits in train for
one language and test for another.

Rows from other sources (e.g. MathVista) aren't parallel, so they are split
independently per language.

Default behaviour is identical to the original script (data/processed ->
data/splits, 70/15/15, seed 42), so existing splits are reproduced exactly.

Run (original small matched set):
    python scripts/make_splits.py

Run (full 900-question set from build_full_xmmmu.py, bigger test share,
test questions spread proportionally across the MMMU subjects):
    python scripts/make_splits.py --processed-dir data/full \
        --splits-dir data/splits_full --train-frac 0.4 --val-frac 0.1 \
        --stratify-by-subject

Output:
    {splits-dir}/{lang}_train.jsonl, {lang}_val.jsonl, {lang}_test.jsonl
"""

import os
import json
import glob
import random
import argparse
from collections import defaultdict

SEED = 42


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_parallel(row):
    return row["source_dataset"].startswith("xmmmu") and row.get("match_id")


def subject_of(match_id):
    """'validation_Basic_Medical_Science_17' -> 'Basic_Medical_Science'"""
    parts = match_id.split("_")
    return "_".join(parts[1:-1]) if len(parts) >= 3 else "all"


def assign_parallel_splits(all_data, train_frac, val_frac, stratify):
    match_ids = sorted({r["match_id"] for rows in all_data.values() for r in rows if is_parallel(r)})
    rng = random.Random(SEED)
    mapping = {}

    if not stratify:
        # original behaviour
        rng.shuffle(match_ids)
        n = len(match_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        for i, mid in enumerate(match_ids):
            mapping[mid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        return mapping

    by_subject = defaultdict(list)
    for mid in match_ids:
        by_subject[subject_of(mid)].append(mid)
    for subject in sorted(by_subject):
        ids = by_subject[subject]
        rng.shuffle(ids)
        n_train = round(len(ids) * train_frac)
        n_val = round(len(ids) * val_frac)
        for i, mid in enumerate(ids):
            mapping[mid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return mapping


def main(args):
    os.makedirs(args.splits_dir, exist_ok=True)
    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        raise SystemExit("train-frac + val-frac must be < 1")

    all_data = {}
    for path in glob.glob(os.path.join(args.processed_dir, "*.jsonl")):
        lang = os.path.splitext(os.path.basename(path))[0]
        all_data[lang] = load_jsonl(path)
    if not all_data:
        raise SystemExit(f"No .jsonl files in {args.processed_dir}")

    match_id_split = assign_parallel_splits(all_data, args.train_frac, args.val_frac, args.stratify_by_subject)

    for lang, rows in sorted(all_data.items()):
        splits = {"train": [], "val": [], "test": []}

        for r in rows:
            if is_parallel(r):
                splits[match_id_split[r["match_id"]]].append(r)

        other = [r for r in rows if not is_parallel(r)]
        if len(other) >= 10:
            from sklearn.model_selection import train_test_split  # only needed for non-parallel data
            tr, temp = train_test_split(other, test_size=(args.val_frac + test_frac), random_state=SEED)
            va, te = train_test_split(temp, test_size=test_frac / (args.val_frac + test_frac), random_state=SEED)
            splits["train"] += tr
            splits["val"] += va
            splits["test"] += te
        else:
            splits["train"] += other

        for name, split_rows in splits.items():
            write_jsonl(os.path.join(args.splits_dir, f"{lang}_{name}.jsonl"), split_rows)
        print(f"{lang}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default=os.path.join("data", "processed"))
    p.add_argument("--splits-dir", default=os.path.join("data", "splits"))
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--stratify-by-subject", action="store_true",
                   help="split each MMMU subject separately so test mirrors the subject mix")
    main(p.parse_args())
