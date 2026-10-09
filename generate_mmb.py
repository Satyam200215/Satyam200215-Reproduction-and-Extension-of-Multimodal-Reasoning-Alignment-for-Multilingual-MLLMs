"""
Step 4a - Multilingual Modality Bridging (MMB), part 1: build structured
reasoning chains in English from the TRAIN split only.

The model is prompted to break each question into the four stages from the
R2-MultiOmnia figure:
    <visual_abstraction> -> <formalisation> -> <reasoning> -> <answer>
For every training question it samples several chains (self-training), then
VERIFIES them: a chain is kept only if (1) all four stages are present, in
order and non-empty, and (2) its final answer equals the gold answer. This is
the "verification / refinement" box in the figure. Wrong or malformed chains
are discarded, not repaired.

Optional --rationalize: for questions where no sampled chain was correct, ask
again while telling the model the correct option (STaR-style). Those chains
are tagged "source": "rationalized" so you can run an ablation without them.
The hint is NOT part of the stored prompt/training example.

Only the train split is used, so no test or validation question ever
contributes a gold answer to the training data.

Run (Colab GPU, same environment as eval_baseline.py):
    python scripts/generate_mmb.py --model qwen --limit 10          # smoke test + timing
    python scripts/generate_mmb.py --model qwen                     # full train split
    python scripts/generate_mmb.py --model qwen --rationalize       # also rescue unsolved questions

Output (resumable; reruns skip finished ids):
    data/mmb/{model}_en_{split}_chains.jsonl
Next: translate_chains.py (chains -> hi/ja/zh/te), then train_sft.py.
"""

import os
import json
import time
import argparse
from collections import Counter
from PIL import Image

import eval_mmb_sft as eb

MMB_DIR = os.path.join("data", "mmb")


def verify(text, row):
    """Return (chain, 'ok') for a well-formed, correct sample, else (None, reason)."""
    chain = eb.parse_chain(text)
    if chain is None:
        return None, "malformed"
    choices = eb.normalize_choices(row.get("choices"))
    if choices:
        ans = eb.extract_answer(chain["answer"], True)
        if not ans or ans not in eb.LETTERS[:len(choices)]:
            return None, "bad_answer"
        chain["answer"] = ans  # keep only the letter, e.g. "B. Not very likely" -> "B"
    if eb.norm(chain["answer"]) != eb.norm(row["answer"]):
        return None, "wrong"
    if len(chain["reasoning"]) < 20 or sum(len(v) for v in chain.values()) > 3500:
        return None, "degenerate"
    return chain, "ok"


def to_text(chain):
    """Canonical target text used for SFT (drops any junk outside the tags)."""
    parts = [f"<{t}>\n{chain[t]}\n</{t}>" for t in eb.STAGES[:-1]]
    parts.append(f"<answer>{chain['answer']}</answer>")
    return "\n".join(parts)


def collect(samples, row, keep):
    """Verify samples; return (up to `keep` distinct valid chains, reason counts)."""
    valid, seen, reasons = [], set(), Counter()
    for text in samples:
        chain, why = verify(text, row)
        reasons[why] += 1
        if chain and chain["reasoning"] not in seen and len(valid) < keep:
            seen.add(chain["reasoning"])
            valid.append(chain)
    return valid, reasons


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def summarize(path):
    recs = eb.load_jsonl(path)
    if not recs:
        return
    solved = [r for r in recs if r["chains"]]
    by_source = Counter(r["source"] for r in solved)
    reasons = Counter()
    for r in recs:
        reasons.update(r["reasons"])
    n_samples = sum(reasons.values())
    print(f"\n{len(recs)} questions processed, {len(solved)} with at least one verified chain "
          f"({len(solved) / len(recs):.1%})")
    print(f"  sources: {dict(by_source)}")
    print(f"  sample outcomes ({n_samples} samples): "
          + ", ".join(f"{k}={v} ({v / n_samples:.0%})" for k, v in reasons.most_common()))
    print(f"  verified chains available for SFT: {sum(len(r['chains']) for r in recs)} (English)")


def main(args):
    os.makedirs(MMB_DIR, exist_ok=True)
    out_path = os.path.join(MMB_DIR, f"{args.model}_en_{args.split}_chains.jsonl")

    src = os.path.join(args.splits_dir, f"en_{args.split}.jsonl")
    rows = [r for r in eb.load_jsonl(src) if r["source_dataset"].startswith("xmmmu")]
    if args.limit:
        rows = rows[:args.limit]
    done = load_done(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} English {args.split} questions, {len(done)} already done, {len(todo)} to go.")

    if todo:
        kwargs = {"model_id": args.model_id} if args.model_id else {}
        if args.load_4bit and args.model == "llava":
            kwargs["load_4bit"] = True
        model = eb.MODELS[args.model](**kwargs)
        gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo, 1):
                img_path = row.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    print(f"  skip {row['id']}: image missing")
                    continue
                image = Image.open(img_path).convert("RGB")

                samples, shrunk = eb.generate_with_retry(
                    model, image, eb.build_structured_prompt(row), n=args.samples, **gen)
                if samples is None:
                    print(f"  skip {row['id']}: out of memory even at the smallest image size", flush=True)
                    continue
                samples = [samples] if isinstance(samples, str) else samples
                chains, reasons = collect(samples, row, args.keep)
                source = "sampled"

                if not chains and args.rationalize and eb.normalize_choices(row.get("choices")):
                    hint = (f"Note: the correct option is {row['answer']}. Write the reasoning that "
                            f"leads to it, and do not mention this note.")
                    h_samples, _ = eb.generate_with_retry(
                        model, image, eb.build_structured_prompt(row, hint=hint), n=args.samples, **gen)
                    if h_samples is not None:
                        h_samples = [h_samples] if isinstance(h_samples, str) else h_samples
                        chains, _ = collect(h_samples, row, args.keep)
                        source = "rationalized"

                rec = {
                    "id": row["id"], "match_id": row.get("match_id"), "image_path": img_path,
                    "question": row["question"], "choices": row.get("choices"), "answer": row["answer"],
                    "source": source if chains else None,
                    "chains": [{"parts": c, "text": to_text(c)} for c in chains],
                    "reasons": dict(reasons), "shrunk_to": shrunk,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el / i:.1f}s/question, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                          flush=True)

    summarize(out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava", "deepseek"], required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--load-4bit", action="store_true", help="llava only")
    p.add_argument("--splits-dir", default=os.path.join("data", "splits_full"))
    p.add_argument("--split", default="train")
    p.add_argument("--samples", type=int, default=4, help="chains sampled per question")
    p.add_argument("--keep", type=int, default=1, help="max verified chains stored per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=640)
    p.add_argument("--rationalize", action="store_true",
                   help="for unsolved questions, retry with the correct option as a hint")
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args())
