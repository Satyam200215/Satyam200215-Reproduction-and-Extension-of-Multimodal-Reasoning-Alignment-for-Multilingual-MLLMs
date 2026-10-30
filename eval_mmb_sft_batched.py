"""
Batched evaluation (eval_mmb_sft_batched.py).

Same prompts, same scoring and the SAME result files as eval_mmb_sft.py, but
Qwen answers several questions per forward pass. This matters for the
structured (--structured) runs: each answer is hundreds of tokens, and one
question at a time on 2,250 test rows would take most of a day.

It is a separate file on purpose: eval_baseline.py and eval_mmb_sft.py are not
touched. Results land in the same place with the same naming, so a run started
with one script can be resumed with the other.

Usage (same options as eval_mmb_sft.py, plus --batch-size):
    python eval_mmb_sft_batched.py --model qwen --splits-dir data/splits_full --structured --run-name structured --limit 20   # smoke test
    python eval_mmb_sft_batched.py --model qwen --splits-dir data/splits_full --structured --run-name structured
    python eval_mmb_sft_batched.py --model qwen --splits-dir data/splits_full --structured --adapter outputs/qwen_sft --run-name sft

Notes:
- Batching is implemented for Qwen. For llava it runs one question at a time
  (same speed as eval_mmb_sft.py); deepseek is not supported with --adapter.
- Greedy decoding is deterministic, but padding in a batch can change bf16
  rounding slightly, so a few answers may differ from a one-at-a-time run.
  For a fair comparison, evaluate every condition you compare with the same
  script (e.g. structured baseline AND SFT both with this one).
- If a batch runs out of GPU memory it is split in half automatically;
  a single question that still does not fit is retried with a smaller image.
"""

import os
import gc
import json
import argparse
from PIL import Image

import eval_mmb_sft as eb


class QwenBatched(eb.QwenVL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor.tokenizer.padding_side = "left"  # required for batched generation

    def generate_batch(self, images, prompts, max_new_tokens=16):
        texts = []
        for p in prompts:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        if not bool((inputs["attention_mask"][:, -1] == 1).all()):
            raise RuntimeError("Batch is padded on the right, which would corrupt generation. "
                               "Run with --batch-size 1 or update transformers.")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **eb.gen_kwargs(max_new_tokens, 1, 0.0))
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def run_batch(model, images, prompts, max_new):
    """-> list of (raw_text or None, shrunk_to). Splits the batch on out-of-memory."""
    if len(prompts) == 1 or not hasattr(model, "generate_batch"):
        out = []
        for img, p in zip(images, prompts):
            raw, shrunk = eb.generate_with_retry(model, img, p, max_new_tokens=max_new)
            out.append((raw, shrunk))
        return out
    try:
        return [(t, None) for t in model.generate_batch(images, prompts, max_new)]
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        free_gpu()
        mid = len(prompts) // 2
        print(f"    out of memory with batch of {len(prompts)}, splitting", flush=True)
        return run_batch(model, images[:mid], prompts[:mid], max_new) + \
               run_batch(model, images[mid:], prompts[mid:], max_new)


def evaluate(args):
    os.makedirs(eb.RESULTS_DIR, exist_ok=True)
    tag = "" if os.path.normpath(args.splits_dir) == os.path.normpath(eb.SPLITS_DIR) \
        else "_" + os.path.basename(os.path.normpath(args.splits_dir))
    run = f"_{args.run_name}" if args.run_name else ""
    pred_path = os.path.join(eb.RESULTS_DIR, f"{args.model}{run}{tag}_{args.split}_predictions.jsonl")

    done = set()
    if os.path.exists(pred_path):
        done = {r["id"] for r in eb.load_jsonl(pred_path)}
        print(f"Resuming: {len(done)} predictions already saved.")

    todo = []
    for lang in args.langs:
        path = os.path.join(args.splits_dir, f"{lang}_{args.split}.jsonl")
        if not os.path.exists(path):
            print(f"Missing {path}, skipping {lang}")
            continue
        rows = [r for r in eb.load_jsonl(path) if r["source_dataset"].startswith(args.source)]
        if args.limit:
            rows = rows[:args.limit]
        todo += [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} rows to evaluate (batch size {args.batch_size}).")

    if todo:
        kwargs = {"model_id": args.model_id} if args.model_id else {}
        if args.load_4bit and args.model == "llava":
            kwargs["load_4bit"] = True
        if args.adapter:
            if args.model == "deepseek":
                raise SystemExit("--adapter is not supported for deepseek yet")
            kwargs["adapter"] = args.adapter
        cls = QwenBatched if args.model == "qwen" else eb.MODELS[args.model]
        model = cls(**kwargs)
        max_new = args.max_new_tokens or (640 if args.structured else 16)

        import time
        t0, finished = time.time(), 0
        with open(pred_path, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), args.batch_size):
                chunk, images, prompts = [], [], []
                for row in todo[start:start + args.batch_size]:
                    img_path = row.get("image_path")
                    if not img_path or not os.path.exists(img_path):
                        print(f"  skip {row['id']}: image missing")
                        continue
                    chunk.append(row)
                    images.append(Image.open(img_path).convert("RGB"))
                    prompts.append(eb.build_structured_prompt(row) if args.structured else eb.build_prompt(row))
                if not chunk:
                    continue
                results = run_batch(model, images, prompts, max_new)
                for row, (raw, shrunk) in zip(chunk, results):
                    if raw is None:
                        print(f"  skip {row['id']}: out of memory even at the smallest image size", flush=True)
                        continue
                    has_choices = bool(eb.normalize_choices(row.get("choices")))
                    if args.structured:
                        pred, tag_found = eb.extract_structured_answer(raw, has_choices)
                    else:
                        pred, tag_found = eb.extract_answer(raw, has_choices), None
                    gold = str(row["answer"]).strip()
                    f.write(json.dumps({
                        "id": row["id"], "match_id": row.get("match_id"), "language": row["language"],
                        "gold": gold, "pred": pred, "raw": raw, "correct": eb.norm(pred) == eb.norm(gold),
                        "shrunk_to": shrunk, "answer_tag": tag_found,
                    }, ensure_ascii=False) + "\n")
                f.flush()
                finished += len(chunk)
                if (start // args.batch_size) % max(1, 100 // args.batch_size) == 0:
                    el = time.time() - t0
                    print(f"  {finished}/{len(todo)}  ({el / finished:.1f}s/row, ~{el / finished * (len(todo) - finished) / 60:.0f} min left)",
                          flush=True)

    eb.summarize(pred_path, args.langs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(eb.MODELS), required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--langs", nargs="+", default=["en", "zh", "ja", "hi", "te"])
    p.add_argument("--source", default="xmmmu")
    p.add_argument("--splits-dir", default=eb.SPLITS_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--load-4bit", action="store_true", help="llava only")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=None, help="default 16, or 640 with --structured")
    p.add_argument("--adapter", default=None, help="LoRA adapter folder from train_sft.py")
    p.add_argument("--run-name", default="")
    p.add_argument("--batch-size", type=int, default=8, help="questions per forward pass (qwen)")
    evaluate(p.parse_args())
