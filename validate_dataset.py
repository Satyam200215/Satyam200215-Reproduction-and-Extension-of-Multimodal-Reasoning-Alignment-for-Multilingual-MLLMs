"""
Step 2c: Validate the processed dataset before splitting/training.

Checks, per language file in data/processed/:
- every row has a non-empty question and answer
- every image_path actually points to a file that exists and opens correctly
- choices (when present) is a non-empty list with no empty strings
- no duplicate ids within a file
- reports the row count per language, so you can see if anything is
  suspiciously smaller/larger than expected

This does NOT fix problems automatically — it reports them, because
silently dropping or auto-fixing rows can hide real issues (e.g. many
missing images pointing to a bug in an earlier step, not just a few
one-off bad rows).

Run:
    python scripts/validate_dataset.py
"""

import os
import json
import glob
from PIL import Image

PROCESSED_DIR = os.path.join("data", "processed")


def validate_file(path):
    lang = os.path.splitext(os.path.basename(path))[0]
    issues = []
    seen_ids = set()
    row_count = 0

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"line {line_num}: invalid JSON ({e})")
                continue

            rid = row.get("id")
            if not rid:
                issues.append(f"line {line_num}: missing id")
            elif rid in seen_ids:
                issues.append(f"line {line_num}: duplicate id '{rid}'")
            else:
                seen_ids.add(rid)

            if not row.get("question", "").strip():
                issues.append(f"line {line_num} ({rid}): empty question")

            if not str(row.get("answer", "")).strip():
                issues.append(f"line {line_num} ({rid}): empty answer")

            choices = row.get("choices")
            if choices is not None:
                if len(choices) == 0:
                    issues.append(f"line {line_num} ({rid}): choices is an empty list")
                elif any(not str(c).strip() for c in choices):
                    issues.append(f"line {line_num} ({rid}): one or more empty choice strings")

            img_path = row.get("image_path")
            if img_path:
                if not os.path.exists(img_path):
                    issues.append(f"line {line_num} ({rid}): image_path does not exist: {img_path}")
                else:
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception as e:
                        issues.append(f"line {line_num} ({rid}): image file is corrupted ({e})")

    return lang, row_count, issues


def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.jsonl")))
    if not files:
        print(f"No files found in {PROCESSED_DIR} — run unify_schema.py / translate_questions.py first.")
        return

    print(f"{'language':<10} {'rows':>6}  issues")
    print("-" * 40)
    total_issues = 0
    all_issue_details = {}
    for path in files:
        lang, row_count, issues = validate_file(path)
        print(f"{lang:<10} {row_count:>6}  {len(issues)}")
        total_issues += len(issues)
        if issues:
            all_issue_details[lang] = issues

    print()
    if total_issues == 0:
        print("All files passed validation. Safe to run make_splits.py.")
    else:
        print(f"Found {total_issues} issue(s) across {len(all_issue_details)} file(s):\n")
        for lang, issues in all_issue_details.items():
            print(f"--- {lang} ---")
            for issue in issues[:20]:  # cap so one bad file doesn't flood the terminal
                print(f"  {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")
            print()
        print("Fix these before running make_splits.py — a corrupted row or")
        print("missing image is much cheaper to catch now than mid-training.")


if __name__ == "__main__":
    main()
