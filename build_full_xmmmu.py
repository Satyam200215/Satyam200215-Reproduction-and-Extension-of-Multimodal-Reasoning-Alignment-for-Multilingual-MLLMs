"""
Step 2d: Build the FULL xMMMU question set (all 900 English questions) in
all five languages, so the test set is ~3x bigger than the 265-question
subset that has native Hindi + Japanese translations.

Run AFTER download_datasets.py (needs data/raw/xmmmu/en). A GPU is strongly
recommended for the translation (Colab T4 is fine). Reuses the images that
unify_schema.py already saved in data/images/ and only decodes new ones.

    python scripts/build_full_xmmmu.py --langs hi --limit 20   # smoke test
    python scripts/build_full_xmmmu.py                         # hi ja zh te, NLLB for all
    python scripts/build_full_xmmmu.py --prefer-native         # keep native hi/ja where they exist

Design choices:
- By default ALL four non-English languages are NLLB machine translations,
  so languages are compared under the SAME translator. (If native hi/ja are
  mixed in, hi/ja quality differs from zh/te for a reason unrelated to the
  model.) Use --prefer-native if you would rather keep the human translations
  for the 265 questions that have them; rows are tagged either way via
  "source_dataset": "xmmmu" (native) or "xmmmu_mt_nllb" (machine translated).
- Questions are translated sentence by sentence (not as one long string) to
  avoid NLLB truncating or dropping text on long multi-sentence questions.
- Strings with no real words (numbers, formulas, symbols) are copied as-is,
  which also avoids the empty-choice problem seen earlier in zh.
- Every translated row gets "mt_flags" (e.g. "question_numbers_changed") so
  suspicious translations can be inspected or excluded later.
- Translations are cached per language (data/full/_cache_{lang}.json), so a
  Colab disconnect only costs the batch in progress; just rerun.

Output:
    data/full/{en,hi,ja,zh,te}.jsonl     (same schema as data/processed/)
    data/full/mt_report.txt              (rows with quality flags)
"""

import os
import re
import sys
import json
import argparse
import unicodedata

# plain HTTPS downloads resume more reliably than the xet backend
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

FULL_DIR = os.path.join("data", "full")
NATIVE_DIR = os.path.join("data", "processed")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_CODE = "eng_Latn"
NLLB_CODES = {"hi": "hin_Deva", "ja": "jpn_Jpan", "zh": "zho_Hans", "te": "tel_Telu"}
NO_SPACE_LANGS = {"zh", "ja"}  # no spaces between sentences when rejoining

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z<(\[])")
HAS_WORDS = re.compile(r"[A-Za-z]{2,}")
IMG_TAG = re.compile(r"<image[ _]?\d+>")
NUM = re.compile(r"\d+(?:\.\d+)?")


# ---------- text helpers (no model needed) ----------

def segment(text):
    """text -> list of lines; each line -> list of stripped sentences."""
    return [[s.strip() for s in SENT_SPLIT.split(line) if s.strip()]
            for line in text.split("\n")]


def units_from(text):
    """The set of sentences that actually need translating."""
    return {s for line in segment(text) for s in line if HAS_WORDS.search(s)}


def rebuild(text, cache, lang):
    """Re-assemble `text` using translations from `cache`. Sentences without
    real words, or without a (non-empty) translation, are kept as-is."""
    sep = "" if lang in NO_SPACE_LANGS else " "
    lines = []
    for sents in segment(text):
        out = []
        for s in sents:
            t = cache.get(s, "").strip() if HAS_WORDS.search(s) else ""
            out.append(t or s)
        lines.append(sep.join(out))
    return "\n".join(lines)


def ascii_digits(s):
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in s)


def numbers(s):
    return sorted(NUM.findall(ascii_digits(s)))


def restore_image_tags(src, tr):
    """If NLLB dropped an '<image N>' tag, put it back at the start."""
    missing = list(dict.fromkeys(t for t in IMG_TAG.findall(src) if t not in tr))
    if missing:
        return " ".join(missing) + " " + tr, True
    return tr, False


def quality_flags(src_q, tr_q, src_choices, tr_choices):
    flags = []
    if not tr_q.strip():
        flags.append("empty_question")
    if numbers(src_q) != numbers(tr_q):
        flags.append("question_numbers_changed")
    if src_choices and tr_choices and any(
            numbers(a) != numbers(b) for a, b in zip(src_choices, tr_choices)):
        flags.append("choice_numbers_changed")
    return flags


# ---------- data loading ----------

def load_english(limit=None):
    # imported here so the text helpers above can be used without `datasets`
    from datasets import load_from_disk
    from unify_schema import clean_choices, save_xmmmu_image, IMAGE_DIR, RAW_DIR

    ds_dir = os.path.join(RAW_DIR, "xmmmu", "en")
    if not os.path.exists(ds_dir):
        sys.exit(f"{ds_dir} not found - run download_datasets.py first.")
    ds = load_from_disk(ds_dir)
    wanted = [c for c in ["id", "question", "options", "choices", "answer", "image_1"]
              if c in ds.column_names]
    ds = ds.select_columns(wanted)  # skip image_2..image_7 bytes
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows, reused, decoded = [], 0, 0
    for ex in ds:
        mid = ex["id"]
        path = os.path.join(IMAGE_DIR, f"xmmmu_en_{mid.replace('/', '_')}.png")
        if os.path.exists(path):
            reused += 1
        else:
            path = save_xmmmu_image(ex, "xmmmu_en", mid)
            decoded += 1
        rows.append({
            "id": f"xmmmu_en_{mid}",
            "match_id": mid,
            "language": "en",
            "source_dataset": "xmmmu",
            "image_path": path,
            "question": ex.get("question", ""),
            "choices": clean_choices(ex.get("options") or ex.get("choices")),
            "answer": str(ex.get("answer", "")),
            "mt_flags": [],
        })
    print(f"English: {len(rows)} questions ({reused} images reused, {decoded} newly decoded)")
    return rows


def load_native(lang):
    """Native (human) translations kept by unify_schema.py, keyed by match_id."""
    path = os.path.join(NATIVE_DIR, f"{lang}.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source_dataset") == "xmmmu" and r.get("match_id"):
                r["mt_flags"] = []
                out[r["match_id"]] = r
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- translation ----------

def load_translator(fp16):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    kwargs = {"torch_dtype": torch.float16} if (fp16 and device == "cuda") else {}
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **kwargs).to(device).eval()
    print(f"Loaded {MODEL_NAME} on {device}" + (" (fp16)" if kwargs else ""))
    if device == "cpu":
        print("WARNING: no GPU - this will be very slow for ~10k sentences per language.")
    return tok, model, device


def translate_units(units, cache, cache_path, tok, model, device, code, batch_size, beams):
    todo = sorted((u for u in units if u not in cache), key=len, reverse=True)
    print(f"  {len(units)} unique sentences, {len(todo)} still to translate")
    if not todo:
        return
    import torch
    tok.src_lang = SRC_CODE
    forced = tok.convert_tokens_to_ids(code)
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = todo[b * batch_size:(b + 1) * batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=forced, max_new_tokens=256,
                                 num_beams=beams, do_sample=False)
        for src, hyp in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            cache[src] = hyp.strip()
        if b % 10 == 0 or b == n_batches - 1:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"    batch {b + 1}/{n_batches}", flush=True)


def build_language(lang, en_rows, tok, model, device, args):
    native = load_native(lang) if args.prefer_native else {}
    if args.prefer_native:
        print(f"  using {len(native)} native {lang} rows; the rest are machine translated")

    cache_path = os.path.join(FULL_DIR, f"_cache_{lang}.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    units = set()
    for r in en_rows:
        if r["match_id"] in native:
            continue
        units |= units_from(r["question"])
        for c in r["choices"] or []:
            units |= units_from(c)
    translate_units(units, cache, cache_path, tok, model, device,
                    NLLB_CODES[lang], args.batch_size, args.beams)

    rows, flagged = [], []
    for r in en_rows:
        if r["match_id"] in native:
            rows.append(native[r["match_id"]])
            continue
        q = rebuild(r["question"], cache, lang)
        q, restored = restore_image_tags(r["question"], q)
        choices = [rebuild(c, cache, lang) for c in r["choices"]] if r["choices"] else None
        flags = quality_flags(r["question"], q, r["choices"], choices)
        if restored:
            flags.append("image_tag_restored")
        row = {
            "id": f"xmmmu_{lang}_{r['match_id']}",
            "match_id": r["match_id"],
            "language": lang,
            "source_dataset": "xmmmu_mt_nllb",
            "image_path": r["image_path"],
            "question": q,
            "choices": choices,
            "answer": r["answer"],  # letter/short answer, intentionally untranslated
            "mt_flags": flags,
        }
        rows.append(row)
        if flags:
            flagged.append((row["id"], flags))

    write_jsonl(os.path.join(FULL_DIR, f"{lang}.jsonl"), rows)
    print(f"  wrote {len(rows)} rows to {FULL_DIR}/{lang}.jsonl, {len(flagged)} with quality flags")
    return flagged


def main(args):
    os.makedirs(FULL_DIR, exist_ok=True)
    en_rows = load_english(args.limit)
    write_jsonl(os.path.join(FULL_DIR, "en.jsonl"), en_rows)

    tok, model, device = load_translator(args.fp16)
    report = []
    for lang in args.langs:
        if lang not in NLLB_CODES:
            print(f"No NLLB code for '{lang}', skipping.")
            continue
        print(f"\nTranslating to {lang} ({NLLB_CODES[lang]})")
        for rid, flags in build_language(lang, en_rows, tok, model, device, args):
            report.append(f"{lang}\t{rid}\t{','.join(flags)}")

    with open(os.path.join(FULL_DIR, "mt_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nDone. {len(report)} flagged rows listed in {FULL_DIR}/mt_report.txt")
    print("Next: python scripts/make_splits.py --processed-dir data/full "
          "--splits-dir data/splits_full --train-frac 0.4 --val-frac 0.1 --stratify-by-subject")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["hi", "ja", "zh", "te"])
    p.add_argument("--prefer-native", action="store_true",
                   help="use the human hi/ja translations from data/processed where they exist")
    p.add_argument("--limit", type=int, default=None, help="only the first N English questions (smoke test)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beams", type=int, default=2)
    p.add_argument("--fp16", action="store_true", help="half precision on GPU (faster, slightly riskier)")
    main(p.parse_args())
