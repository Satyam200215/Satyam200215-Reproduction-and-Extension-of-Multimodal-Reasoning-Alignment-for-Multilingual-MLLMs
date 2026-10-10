"""
translate_chains_v2.py - a better version of translate_chains.py (which is left untouched).

What changed, and why (from the zh diagnostic):
1. "Step 1:" labels are NOT sent to the translator. NLLB wrote them as words
   ("第一步", "Job 1" -> "一个工作"), which tripped the number check without any real
   error. The labels are now cut out, the text between them is translated, and
   the label is put back in a fixed form per language (e.g. "步骤1:").
2. The number check compares VALUES, not digit strings. "10,308" == "10308",
   "$170,000,000" == "1.70亿", "2.5 million" == "250万". It still flags real
   errors: decimal point -> comma (0.1085 -> 0,1085), dropped values ("X = 50"
   missing), wrong magnitude ($200,000,000 -> 200万). Small integers (< --sig-min,
   default 10) are ignored so "Step 3" / "2 options" style differences do not flag.
   Use --sig-min 0 for the strictest check.
3. --translator lets you try a larger NLLB model (facebook/nllb-200-distilled-1.3B).

Same output schema as translate_chains.py, so:
    python train_sft.py --model qwen --data data/mmb/qwen_sft_data_v2.jsonl --out-dir outputs/qwen_sft_v2

Run (Colab GPU):
    python translate_chains_v2.py --model qwen
    python translate_chains_v2.py --model qwen --translator facebook/nllb-200-distilled-1.3B --out data/mmb/qwen_sft_data_v2_1p3b.jsonl
"""

import os
import re
import sys
import json
import argparse
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

import eval_mmb_sft as eb
import build_full_xmmmu as bf
import translate_chains as tc

STAGE_TEXT = tc.STAGE_TEXT
MMB_DIR = tc.MMB_DIR
DEFAULT_TRANSLATOR = "facebook/nllb-200-distilled-600M"

STEP_RE = re.compile(r"(\bStep\s+\d+\s*:)")
LABELS = {"zh": "步骤{n}:", "ja": "ステップ{n}:", "hi": "चरण {n}:", "te": "దశ {n}:"}

_GROUPED = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])")
_NUMBER = re.compile(
    r"(?<![\d.])(\d+)(\.\d+)?(?:\s*(trillion|billion|million|thousand|百万|千万|万|亿|億|千)(?![A-Za-z]))?", re.I)
_MULT = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12,
         "千": 10 ** 3, "万": 10 ** 4, "百万": 10 ** 6, "千万": 10 ** 7, "亿": 10 ** 8, "億": 10 ** 8}


def number_values(text, sig_min=10):
    """Multiset of significant numeric values in text (scripts/units/commas normalised)."""
    s = bf.ascii_digits(text)
    s = _GROUPED.sub(lambda m: m.group(0).replace(",", ""), s)
    vals = Counter()
    for m in _NUMBER.finditer(s):
        v = Decimal(m.group(1) + (m.group(2) or ""))
        unit = m.group(3)
        if unit:
            v *= _MULT[unit.lower()]
        if m.group(2) or v >= sig_min:
            vals[v.normalize()] += 1
    return vals


def chain_flags(en_parts, tr_parts, sig_min):
    flags = []
    for s in STAGE_TEXT:
        if not tr_parts[s].strip():
            flags.append(f"{s}_empty")
        elif number_values(en_parts[s], sig_min) != number_values(tr_parts[s], sig_min):
            flags.append(f"{s}_numbers_changed")
    return flags


def split_steps(text):
    """-> list of ('text', str) / ('label', n) pieces."""
    out = []
    for i, piece in enumerate(STEP_RE.split(text)):
        if i % 2 == 0:
            if piece:
                out.append(("text", piece))
        else:
            out.append(("label", int(re.search(r"\d+", piece).group())))
    return out


def stage_units(text):
    units = set()
    for kind, v in split_steps(text):
        if kind == "text":
            units |= bf.units_from(v)
    return units


def translate_text(text, cache, lang):
    out = []
    for kind, v in split_steps(text):
        if kind == "label":
            out.append(LABELS[lang].format(n=v) + " ")
        else:
            piece = bf.rebuild(v, cache, lang)
            out.append(piece.lstrip(" ") if out and out[-1].endswith(" ") else piece)
    return re.sub(r" +\n", "\n", "".join(out)).strip()


def cache_file(lang, translator):
    if translator == DEFAULT_TRANSLATOR:
        return tc.cache_path(lang)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", translator.split("/")[-1])
    return os.path.join(MMB_DIR, f"_chain_cache_{lang}_{tag}.json")


def main(args):
    os.makedirs(MMB_DIR, exist_ok=True)
    in_path = args.chains or os.path.join(MMB_DIR, f"{args.model}_en_train_chains.jsonl")
    if not os.path.exists(in_path):
        sys.exit(f"{in_path} not found - run generate_mmb.py first.")
    sel_args = SimpleNamespace(exclude_rationalized=args.exclude_rationalized,
                               no_leak_filter=args.no_leak_filter,
                               chains_per_question=args.chains_per_question)
    selected, dropped = tc.select_chains(in_path, sel_args)
    print(f"{len(selected)} English chains selected from {in_path}")
    if dropped:
        print("  not used: " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    if not selected:
        sys.exit("Nothing to translate.")

    units = set()
    for _, chain, _ in selected:
        for stage in STAGE_TEXT:
            units |= stage_units(chain["parts"][stage])
    print(f"{len(units)} unique sentences per language (step labels excluded)")

    caches, pending = {}, 0
    for lang in args.langs:
        p = cache_file(lang, args.translator)
        caches[lang] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        pending += sum(1 for u in units if u not in caches[lang])
    if pending:
        bf.MODEL_NAME = args.translator
        tok, model, device = bf.load_translator(args.fp16)
        for lang in args.langs:
            print(f"\nTranslating chains to {lang} with {args.translator}")
            bf.translate_units(units, caches[lang], cache_file(lang, args.translator), tok, model, device,
                               bf.NLLB_CODES[lang], args.batch_size, args.beams)
    else:
        print("All sentences already cached, skipping the translation model.")

    questions = {lang: tc.load_train_questions(args.splits_dir, lang) for lang in args.langs}

    rows, missing = [], Counter()
    for rec, chain, ci in selected:
        suffix = f"_c{ci}" if ci else ""
        for lang in ["en"] + list(args.langs):
            if lang == "en":
                question, choices, q_flags = rec["question"], rec["choices"], []
                parts, flags = dict(chain["parts"]), []
            else:
                qrow = questions[lang].get(rec["match_id"])
                if qrow is None:
                    missing[lang] += 1
                    continue
                question, choices, q_flags = qrow["question"], qrow["choices"], qrow.get("mt_flags", [])
                parts = {s: translate_text(chain["parts"][s], caches[lang], lang) for s in STAGE_TEXT}
                parts["answer"] = chain["parts"]["answer"]
                flags = chain_flags(chain["parts"], parts, args.sig_min)
            rows.append({
                "id": f"xmmmu_{lang}_{rec['match_id']}{suffix}",
                "match_id": rec["match_id"], "language": lang,
                "image_path": rec["image_path"],
                "question": question, "choices": choices, "answer": rec["answer"],
                "parts": parts, "target": tc.target_text(parts),
                "source": rec["source"], "flags": flags, "q_flags": q_flags,
            })

    out_path = args.out or os.path.join(MMB_DIR, f"{args.model}_sft_data_v2.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"{'lang':<6}{'rows':>6}{'flagged':>9}{'usable':>8}{'rationalized':>14}")
    for lang in ["en"] + list(args.langs):
        rs = [r for r in rows if r["language"] == lang]
        bad = sum(bool(r["flags"]) for r in rs)
        print(f"{lang:<6}{len(rs):>6}{bad:>9}{len(rs) - bad:>8}{sum(r['source'] == 'rationalized' for r in rs):>14}")
    if missing:
        print("  questions with no translated train row (skipped): " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    fc = Counter(f for r in rows for f in r["flags"])
    if fc:
        print("  chain flags: " + ", ".join(f"{k}={v}" for k, v in fc.most_common()))
    print(f"\nNext: python train_sft.py --model {args.model} --data {out_path} --out-dir outputs/{args.model}_sft_v2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava", "deepseek"], required=True)
    p.add_argument("--langs", nargs="+", default=["hi", "ja", "zh", "te"])
    p.add_argument("--chains", default=None)
    p.add_argument("--splits-dir", default=os.path.join("data", "splits_full"))
    p.add_argument("--chains-per-question", type=int, default=1)
    p.add_argument("--exclude-rationalized", action="store_true")
    p.add_argument("--no-leak-filter", action="store_true")
    p.add_argument("--translator", default=DEFAULT_TRANSLATOR)
    p.add_argument("--out", default=None)
    p.add_argument("--sig-min", type=int, default=10, help="ignore whole numbers below this (0 = strictest)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beams", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    main(p.parse_args())
