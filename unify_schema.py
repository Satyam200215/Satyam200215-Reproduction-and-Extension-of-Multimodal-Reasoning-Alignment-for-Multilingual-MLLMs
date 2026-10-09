"""
Step 2: Convert every source dataset into one common JSONL schema so that
downstream MMB / SFT / GRPO code never needs to know which raw dataset an
example came from.

Common schema (one JSON object per line):
{
    "id": str,            # unique per row, e.g. "xmmmu_hi_validation_Agriculture_8"
    "match_id": str|None, # the ORIGINAL MMMU question id (e.g. "validation_Agriculture_8"),
                           # shared across en/hi/ja/zh/te for the SAME question.
                           # None for non-parallel sources like MathVista.
    "language": str,       # "en", "zh", "ja", "hi", "te"
    "source_dataset": str, # "mathvista" | "xmmmu" | "xmmmu_mt_nllb"
    "image_path": str | None,
    "question": str,
    "choices": list[str] | None,
    "answer": str,
}

IMPORTANT — why this filters to a matched subset:
xMMMU's "en" split has 900 rows (the FULL MMMU validation set), but "hi" has
291 and "ja" has 269 — translations of only a SUBSET of the English
questions, and not necessarily the same subset as each other. Comparing by
row position (what an earlier version of this script did) silently compared
different questions. This version instead:
  1. Finds the question ids common to BOTH hi and ja (the "match_id"s).
  2. Keeps only those questions for en/hi/ja, so all three are guaranteed to
     be the literal same questions.
  3. Writes "match_id" into every row, which translate_questions.py and
     make_splits.py then use to keep zh/te aligned to the same set too.

Run:
    python scripts/unify_schema.py

Output:
    data/processed/{language}.jsonl
"""

import os
import io
import json
import ast
from datasets import load_from_disk
from PIL import Image


def clean_choices(choices):
    """Some sources pad the choices list with empty strings to a fixed
    length (e.g. 4 slots even for a true/false question). Drop those, and
    handle the case where the field is a stringified list rather than a
    real one."""
    if choices is None:
        return None
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            return None
    cleaned = [str(c).strip() for c in choices if str(c).strip()]
    return cleaned or None

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
IMAGE_DIR = os.path.join("data", "images")


def save_image_if_present(example, dataset_name, idx):
    """MathVista stores images as a decoded PIL object directly. Dump to
    disk once and keep a path so the unified JSONL stays lightweight."""
    img = example.get("decoded_image") or example.get("image")
    if img is None:
        return None
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = os.path.join(IMAGE_DIR, f"{dataset_name}_{idx}.png")
    try:
        img.save(path)
        return path
    except Exception:
        return None


def save_xmmmu_image(example, dataset_name, mmmu_id):
    """xMMMU stores images differently: as a dict {'bytes': <raw PNG/JPEG
    bytes>, 'path': ...} under columns image_1..image_7, NOT as an already-
    decoded PIL object. Only image_1 is handled (the primary image) — a
    question whose text references <image 2> or later has extra images
    that are currently dropped. Checked for and reported at the end."""
    img_field = example.get("image_1")
    if not img_field or not isinstance(img_field, dict) or not img_field.get("bytes"):
        return None
    os.makedirs(IMAGE_DIR, exist_ok=True)
    safe_id = mmmu_id.replace("/", "_")
    path = os.path.join(IMAGE_DIR, f"{dataset_name}_{safe_id}.png")
    try:
        img = Image.open(io.BytesIO(img_field["bytes"]))
        img.save(path)
        return path
    except Exception as e:
        print(f"  warning: failed to decode image for {mmmu_id}: {e}")
        return None


def process_mathvista():
    ds_dir = os.path.join(RAW_DIR, "mathvista")
    if not os.path.exists(ds_dir):
        print("Skipping MathVista (not downloaded yet)")
        return []
    ds = load_from_disk(ds_dir)
    rows = []
    for idx, ex in enumerate(ds):
        lang = ex.get("language", "en")
        rows.append({
            "id": f"mathvista_{ex.get('pid', idx)}",
            "match_id": None,  # not a parallel/translated question set
            "language": lang,
            "source_dataset": "mathvista",
            "image_path": save_image_if_present(ex, "mathvista", idx),
            "question": ex.get("question", ex.get("query", "")),
            "choices": clean_choices(ex.get("choices")),
            "answer": str(ex.get("answer", "")),
        })
    return rows


def process_xmmmu():
    base_dir = os.path.join(RAW_DIR, "xmmmu")
    rows = []
    if not os.path.exists(base_dir):
        print("Skipping xMMMU (not downloaded yet)")
        return rows

    splits = {}
    for split in ["en", "hi", "ja"]:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            splits[split] = load_from_disk(split_dir)
        else:
            print(f"  xMMMU '{split}' split not downloaded, skipping it")

    if "hi" not in splits or "ja" not in splits:
        print("  Need both hi and ja splits to compute the matched set — skipping xMMMU.")
        return rows

    # Pull just the "id" column (no image bytes touched) — this is what was
    # slow before: iterating full rows via `for ex in ds` decodes every
    # image field on every row just to read one string field.
    hi_ids = set(splits["hi"]["id"])
    ja_ids = set(splits["ja"]["id"])
    matched_ids = hi_ids & ja_ids
    print(f"  xMMMU: hi has {len(hi_ids)} ids, ja has {len(ja_ids)} ids, "
          f"{len(matched_ids)} ids in common -> this is the matched set used everywhere.")

    multi_image_count = 0
    for split, ds in splits.items():
        # Narrow to just the matched rows BEFORE touching any image data.
        # select() is lazy/cheap (index-based); only these ~N rows (not the
        # full 900 for "en") ever get their images decoded below.
        id_col = ds["id"]
        keep_indices = [i for i, mid in enumerate(id_col) if mid in matched_ids]
        matched_ds = ds.select(keep_indices)
        print(f"  {split}: processing {len(matched_ds)} matched rows "
              f"(out of {len(ds)} total — only these get images decoded)")

        for ex in matched_ds:
            mmmu_id = ex["id"]
            question = ex.get("question", "")
            if "<image 2>" in question or "<image_2>" in question:
                multi_image_count += 1  # only image_1 is extracted; flagged, not silently dropped
            rows.append({
                "id": f"xmmmu_{split}_{mmmu_id}",
                "match_id": mmmu_id,
                "language": split,
                "source_dataset": "xmmmu",
                "image_path": save_xmmmu_image(ex, f"xmmmu_{split}", mmmu_id),
                "question": question,
                "choices": clean_choices(ex.get("options") or ex.get("choices")),
                "answer": str(ex.get("answer", "")),
            })

    if multi_image_count:
        print(f"  NOTE: {multi_image_count} questions reference a 2nd+ image that was "
              f"NOT extracted (only image_1 is handled). Review these before training if "
              f"multi-image questions matter for your evaluation.")
    return rows


def write_by_language(rows):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)
    for lang, lang_rows in by_lang.items():
        out_path = os.path.join(PROCESSED_DIR, f"{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in lang_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{lang}: wrote {len(lang_rows)} examples to {out_path}")


if __name__ == "__main__":
    all_rows = []
    all_rows += process_mathvista()
    all_rows += process_xmmmu()
    write_by_language(all_rows)
    print("\nNote: Telugu has no source here yet — that comes from the")
    print("translation step (see README) and should be appended to")
    print("data/processed/te.jsonl in the same schema before splitting.")
