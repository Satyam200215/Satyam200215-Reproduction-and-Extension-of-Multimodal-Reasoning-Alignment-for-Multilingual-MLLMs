"""
Step 2b: Create MATCHED Chinese and Telugu versions of the same xMMMU
English question set, so every language has a translation of the exact
same underlying question — needed to fairly test "does the model do worse
on this question in language X than in English".

MUST run AFTER unify_schema.py. It reads data/processed/en.jsonl, which
unify_schema.py has already filtered down to exactly the questions that
BOTH Hindi and Japanese have native translations for (the "match_id"
field) — NOT the full 900-question English set. This is what guarantees
zh/te line up with en/hi/ja on the literal same questions, rather than
just the same row position (xMMMU's en/hi/ja splits have different row
counts — 900 vs ~290 — so row position was never a valid way to match them;
"match_id" is the fix for that).

Uses NLLB-200 (facebook/nllb-200-distilled-600M) — one model covers
Chinese, Telugu, and dozens of other languages, so there's no need for a
separate tool per language. Runs on CPU (slow: ~1-2s per sentence) or much
faster on any GPU (recommended if your college lab has one free).

Install (in addition to requirements.txt):
    pip install transformers torch sentencepiece

Run:
    python scripts/translate_questions.py --langs zh te

Output:
    data/processed/zh.jsonl   (match_id-aligned with en/hi/ja)
    data/processed/te.jsonl
"""

import os
import json
import argparse

# The newer "xet" transfer backend can be flaky on slow/unstable connections
# (repeated timeouts and re-downloads of the same large file). Force the
# plain HTTPS download path instead, which resumes more predictably.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

NLLB_LANG_CODES = {
    "zh": "zho_Hans",
    "te": "tel_Telu",
}
SRC_LANG_CODE = "eng_Latn"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

IMAGE_DIR = os.path.join("data", "images")
PROCESSED_DIR = os.path.join("data", "processed")


def load_translator():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Loaded {MODEL_NAME} on {device}")
    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, target_lang_code, batch_size=8, max_length=200):
    import torch
    import time
    if not texts:
        return []
    tokenizer.src_lang = SRC_LANG_CODE
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        t0 = time.time()
        batch_num = i // batch_size + 1
        batch = [t if t else " " for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=1,        # greedy decoding — much faster on CPU, fine for this use case
                do_sample=False,
            )
        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"    batch {batch_num}/{total_batches} done ({time.time() - t0:.1f}s)", flush=True)
    return results


def load_matched_english_rows():
    """Load the English rows unify_schema.py already filtered down to the
    hi/ja matched set (source_dataset == 'xmmmu', each with a match_id)."""
    en_path = os.path.join(PROCESSED_DIR, "en.jsonl")
    if not os.path.exists(en_path):
        print(f"{en_path} not found — run unify_schema.py first.")
        return []
    rows = []
    with open(en_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("source_dataset") == "xmmmu" and row.get("match_id"):
                rows.append(row)
    return rows


def main(target_langs):
    en_rows = load_matched_english_rows()
    if not en_rows:
        print("No matched xMMMU rows found in data/processed/en.jsonl — nothing to translate.")
        return

    questions = [r["question"] for r in en_rows]
    choices_list = [r.get("choices") or [] for r in en_rows]

    total_choice_texts = sum(len(c) for c in choices_list)
    print(f"{len(questions)} matched questions, {total_choice_texts} choice texts total "
          f"(expect roughly {len(questions)}-{len(questions)*5}, not tens of thousands)")
    if choices_list:
        print(f"Example choices for row 0: {choices_list[0]}")

    tokenizer, model, device = load_translator()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for lang in target_langs:
        if lang not in NLLB_LANG_CODES:
            print(f"No NLLB code configured for '{lang}' — add one to NLLB_LANG_CODES. Skipping.")
            continue

        target_code = NLLB_LANG_CODES[lang]
        print(f"Translating {len(questions)} questions to {lang} ({target_code})...")
        translated_questions = translate_batch(questions, tokenizer, model, device, target_code)

        flat_choices, counts = [], []
        for c in choices_list:
            counts.append(len(c))
            flat_choices.extend(c)
        translated_flat = translate_batch(flat_choices, tokenizer, model, device, target_code)
        translated_choices_list, pos = [], 0
        for n in counts:
            translated_choices_list.append(translated_flat[pos:pos + n])
            pos += n

        out_path = os.path.join(PROCESSED_DIR, f"{lang}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, src_row in enumerate(en_rows):
                mmmu_id = src_row["match_id"]
                row = {
                    "id": f"xmmmu_{lang}_{mmmu_id}",
                    "match_id": mmmu_id,  # same key as the source English row -> alignable
                    "language": lang,
                    "source_dataset": "xmmmu_mt_nllb",
                    # same image as the source English row — translation only changes text
                    "image_path": src_row["image_path"],
                    "question": translated_questions[i],
                    "choices": translated_choices_list[i] if translated_choices_list[i] else None,
                    # answer is normally a letter/index (A/B/C/D) — left untranslated on purpose
                    "answer": src_row["answer"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(en_rows)} matched examples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["zh", "te"], help="Target language codes to translate into")
    args = parser.parse_args()
    main(args.langs)
