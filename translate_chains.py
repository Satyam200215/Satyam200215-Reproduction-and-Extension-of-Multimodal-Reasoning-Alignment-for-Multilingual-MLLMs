"""
Step 4b - Multilingual Modality Bridging (MMB), part 2: translate the verified
English chains into Hindi, Japanese, Chinese and Telugu, and assemble the SFT
dataset.

Input : data/mmb/{model}_en_train_chains.jsonl      (from generate_mmb.py)
        data/splits_full/{lang}_train.jsonl          (translated TRAIN questions)
Output: data/mmb/{model}_sft_data.jsonl              (one row per question x language)

What happens:
1. Pick the verified chain for each training question. Rationalized chains
   (written with the correct option as a hint) are screened for leakage: a chain
   that mentions the hint, or states the answer before its reasoning stage, is
   dropped. Use --exclude-rationalized to drop all of them (ablation), or
   --no-leak-filter to switch the screening off.
2. Translate ONLY the text inside the three stages (visual_abstraction,
   formalisation, reasoning) with NLLB, sentence by sentence. The tags and the
   answer letter stay exactly as they are, so the output format is identical in
   every language.
3. Quality-check each translation: if any number in a stage changed (digits are
   compared across scripts, e.g. Devanagari 3 vs 3), the row gets a flag in
   "flags" and train_sft.py skips it by default. Question-side flags from
   build_full_xmmmu.py are kept separately in "q_flags".
4. Pair every translated chain with the question of the SAME match_id in the
   same language (taken from the TRAIN split, so no test question is involved).

Only the train split is used anywhere here.

Run (Colab GPU):
    python translate_chains.py --model qwen
    python translate_chains.py --model qwen --exclude-rationalized   # ablation file contents; see train_sft.py too

Translations are cached in data/mmb/_chain_cache_{lang}.json, so a disconnect
only costs the batch in progress: rerun the same command.
"""

import os
import re
import sys
import json
import argparse
from collections import Counter

import eval_mmb_sft as eb
import build_full_xmmmu as bf

MMB_DIR = os.path.join("data", "mmb")
STAGE_TEXT = eb.STAGES[:-1]  # the three stages that contain translatable prose

# A rationalized chain was written while the model was told the right option.
LEAK_ANYWHERE = re.compile(
    r"\b(the|this|that|given) (note|hint)\b|as (instructed|requested|told)|"
    r"\bthe instruction\b|do not mention|\bnote:", re.I)
LEAK_EARLY = re.compile(
    r"correct (option|answer)|the answer is|answer *: *[A-J]\b|"
    r"option [A-J] is (the )?correct|\b[A-J] is (the )?correct", re.I)


def leaks(parts):
    """Return a reason string if a chain appears to reveal the hint, else None."""
    if LEAK_ANYWHERE.search(" ".join(parts[s] for s in STAGE_TEXT)):
        return "mentions_hint"
    early = parts["visual_abstraction"] + "\n" + parts["formalisation"]
    if LEAK_EARLY.search(early):
        return "answer_before_reasoning"
    return None


def target_text(parts):
    """Canonical SFT target: same layout generate_mmb.to_text() produces."""
    body = [f"<{t}>\n{parts[t]}\n</{t}>" for t in STAGE_TEXT]
    body.append(f"<answer>{parts['answer']}</answer>")
    return "\n".join(body)


def chain_flags(en_parts, tr_parts):
    flags = []
    for s in STAGE_TEXT:
        if not tr_parts[s].strip():
            flags.append(f"{s}_empty")
        elif bf.numbers(en_parts[s]) != bf.numbers(tr_parts[s]):
            flags.append(f"{s}_numbers_changed")
    return flags


def select_chains(path, args):
    """-> (list of (record, chain, chain_index), Counter of why records were dropped)."""
    selected, dropped = [], Counter()
    for rec in eb.load_jsonl(path):
        if not rec["chains"]:
            dropped["unsolved_question"] += 1
            continue
        if rec["source"] == "rationalized":
            if args.exclude_rationalized:
                dropped["rationalized_excluded"] += 1
                continue
            if not args.no_leak_filter:
                why = leaks(rec["chains"][0]["parts"])
                if why:
                    dropped[f"leak:{why}"] += 1
                    continue
        for ci, chain in enumerate(rec["chains"][:args.chains_per_question]):
            selected.append((rec, chain, ci))
    return selected, dropped


def cache_path(lang):
    return os.path.join(MMB_DIR, f"_chain_cache_{lang}.json")


def load_train_questions(splits_dir, lang):
    path = os.path.join(splits_dir, f"{lang}_train.jsonl")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - run make_splits.py for the full set first.")
    return {r["match_id"]: r for r in eb.load_jsonl(path) if r.get("match_id")}


def main(args):
    os.makedirs(MMB_DIR, exist_ok=True)
    in_path = args.chains or os.path.join(MMB_DIR, f"{args.model}_en_train_chains.jsonl")
    if not os.path.exists(in_path):
        sys.exit(f"{in_path} not found - run generate_mmb.py first.")

    selected, dropped = select_chains(in_path, args)
    print(f"{len(selected)} English chains selected from {in_path}")
    if dropped:
        print("  not used: " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    if not selected:
        sys.exit("Nothing to translate.")

    # sentences that need translating (shared across languages)
    units = set()
    for _, chain, _ in selected:
        for stage in STAGE_TEXT:
            units |= bf.units_from(chain["parts"][stage])
    print(f"{len(units)} unique sentences per language")

    caches, pending = {}, 0
    for lang in args.langs:
        p = cache_path(lang)
        caches[lang] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        pending += sum(1 for u in units if u not in caches[lang])
    if pending:
        tok, model, device = bf.load_translator(args.fp16)
        for lang in args.langs:
            print(f"\nTranslating chains to {lang} ({bf.NLLB_CODES[lang]})")
            bf.translate_units(units, caches[lang], cache_path(lang), tok, model, device,
                               bf.NLLB_CODES[lang], args.batch_size, args.beams)
    else:
        print("All sentences already cached, skipping the translation model.")

    questions = {lang: load_train_questions(args.splits_dir, lang) for lang in args.langs}

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
                parts = {s: bf.rebuild(chain["parts"][s], caches[lang], lang) for s in STAGE_TEXT}
                parts["answer"] = chain["parts"]["answer"]
                flags = chain_flags(chain["parts"], parts)
            rows.append({
                "id": f"xmmmu_{lang}_{rec['match_id']}{suffix}",
                "match_id": rec["match_id"], "language": lang,
                "image_path": rec["image_path"],
                "question": question, "choices": choices, "answer": rec["answer"],
                "parts": parts, "target": target_text(parts),
                "source": rec["source"], "flags": flags, "q_flags": q_flags,
            })

    out_path = os.path.join(MMB_DIR, f"{args.model}_sft_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"{'lang':<6}{'rows':>6}{'chain flagged':>15}{'question flagged':>18}{'rationalized':>14}")
    for lang in ["en"] + list(args.langs):
        rs = [r for r in rows if r["language"] == lang]
        print(f"{lang:<6}{len(rs):>6}{sum(bool(r['flags']) for r in rs):>15}"
              f"{sum(bool(r['q_flags']) for r in rs):>18}{sum(r['source'] == 'rationalized' for r in rs):>14}")
    if missing:
        print("  questions with no translated train row (skipped): " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    flag_counts = Counter(f for r in rows for f in r["flags"])
    if flag_counts:
        print("  chain flags: " + ", ".join(f"{k}={v}" for k, v in flag_counts.most_common()))
    usable = sum(1 for r in rows if not r["flags"])
    print(f"  rows train_sft.py will use by default (no chain flags): {usable} of {len(rows)}")

    sample = next((r for r in rows if r["language"] == args.langs[0]), None) if args.langs else None
    if sample:
        print(f"\nExample ({sample['language']}), first 600 characters of the target:")
        print(sample["target"][:600])
    print("\nNext: python train_sft.py --model", args.model, "--limit 16 --epochs 1   (smoke test)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava", "deepseek"], required=True)
    p.add_argument("--langs", nargs="+", default=["hi", "ja", "zh", "te"])
    p.add_argument("--chains", default=None, help="override path of the English chains file")
    p.add_argument("--splits-dir", default=os.path.join("data", "splits_full"))
    p.add_argument("--chains-per-question", type=int, default=1)
    p.add_argument("--exclude-rationalized", action="store_true",
                   help="use only chains the model produced WITHOUT being told the answer")
    p.add_argument("--no-leak-filter", action="store_true", help="keep rationalized chains that look like they mention the hint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beams", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    main(p.parse_args())
