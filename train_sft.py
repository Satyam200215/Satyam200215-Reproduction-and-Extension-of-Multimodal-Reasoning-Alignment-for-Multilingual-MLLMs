"""
Step 4c - Supervised fine-tuning (SFT) on the multilingual structured chains.

Teaches the model to answer in the four-stage format
    <visual_abstraction> -> <formalisation> -> <reasoning> -> <answer>
in every language. It uses LoRA, so only a small set of added weights is trained
and the result is a few-hundred-MB "adapter" instead of a full model copy.

Input : data/mmb/{model}_sft_data.jsonl      (from translate_chains.py)
Output: outputs/{model}_sft/                 (the adapter; also epoch1/, epoch2/ ...)
        outputs/{model}_sft/train_log.jsonl, sft_config.json

What is trained on:
- Prompt = the same structured prompt used by generate_mmb.py and by
  eval_mmb_sft.py --structured (English instruction + the question in language L).
- Target = the chain in language L, ending with <answer>LETTER</answer>.
- The loss is computed ONLY on the target (the prompt and image tokens are
  masked), so the model learns to write chains, not to repeat questions.
- Only TRAIN-split questions are used. A small share of them (--holdout-frac)
  is kept aside, by question, just to report a held-out loss each epoch; it is
  not the test set. Use --holdout-frac 0 to train on everything.
- Rows whose translation changed a number are skipped (--keep-flagged to keep).

Needs an A100-class GPU for Qwen2.5-VL-3B in bf16.
    pip install -q accelerate peft bitsandbytes

Usage:
    python train_sft.py --model qwen --limit 16 --epochs 1       # smoke test; prints speed
    python train_sft.py --model qwen                             # full run
    python train_sft.py --model qwen --exclude-rationalized --out-dir outputs/qwen_sft_sampled_only   # ablation
Then evaluate on the unseen test set:
    python eval_mmb_sft.py --model qwen --splits-dir data/splits_full --structured \
        --adapter outputs/qwen_sft --run-name sft

LLaVA (--model llava --load-4bit) is included as a best-effort path (QLoRA) and
has not been run yet; Qwen is the tested route. DeepSeek is not supported yet.
"""

import os
import json
import math
import time
import random
import argparse
from collections import Counter
from PIL import Image

import eval_mmb_sft as eb

MMB_DIR = os.path.join("data", "mmb")
END_OF_TURN = "<|im_end|>\n"
# LoRA on the language model's attention + MLP layers only. The negative
# lookahead skips the vision encoder / projector, which use the same layer names.
LORA_TARGETS = (r"^(?!.*(visual|vision_tower|multi_modal_projector))"
                r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


# ---------- data (no torch needed) ----------

def load_records(args):
    path = args.data or os.path.join(MMB_DIR, f"{args.model}_sft_data.jsonl")
    keep, dropped = [], Counter()
    for r in eb.load_jsonl(path):
        if r["language"] not in args.langs:
            continue
        if args.exclude_rationalized and r.get("source") == "rationalized":
            dropped["rationalized_excluded"] += 1
        elif r.get("flags") and not args.keep_flagged:
            dropped["chain_flagged"] += 1
        elif r.get("q_flags") and args.drop_q_flagged:
            dropped["question_flagged"] += 1
        elif not r.get("image_path") or not os.path.exists(r["image_path"]):
            dropped["image_missing"] += 1
        else:
            keep.append(r)
    return keep, dropped


def split_holdout(recs, frac, seed):
    """Hold out whole questions (every language of a question together)."""
    ids = sorted({r["match_id"] for r in recs})
    random.Random(seed).shuffle(ids)
    held = set(ids[:int(round(len(ids) * frac))]) if frac > 0 else set()
    return ([r for r in recs if r["match_id"] not in held],
            [r for r in recs if r["match_id"] in held])


def render(processor, prompt, target):
    """-> (prompt_text with generation prompt, full text = prompt + target + end of turn)."""
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        user + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
        tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt_text):  # unusual template: build it by hand
        full = prompt_text + target + END_OF_TURN
    return prompt_text, full


class Ctx:
    """Everything make_example needs: processor, image size cap, length cap."""
    def __init__(self, processor, max_len, max_side=None):
        self.processor, self.max_len, self.max_side = processor, max_len, max_side


def make_example(row, ctx):
    """Tokenise one training row. Returns a dict of tensors with `labels`
    (-100 everywhere except the target), or None if it is too long / empty."""
    image = Image.open(row["image_path"]).convert("RGB")
    if ctx.max_side and max(image.size) > ctx.max_side:
        s = ctx.max_side / max(image.size)
        image = image.resize((max(1, int(image.width * s)), max(1, int(image.height * s))))
    prompt = eb.build_structured_prompt(row)
    prompt_text, full_text = render(ctx.processor, prompt, row["target"])
    full = ctx.processor(text=[full_text], images=[image], return_tensors="pt")
    n_prompt = ctx.processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"].shape[1]
    ids = full["input_ids"]
    if ids.shape[1] > ctx.max_len or ids.shape[1] <= n_prompt:
        return None
    labels = ids.clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return full


def lr_at(step, total, warmup, base):
    """Linear warm-up, then cosine decay down to 10% of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def summarize_data(train, val, dropped, args):
    print(f"Training rows: {len(train)}   held-out rows (loss only): {len(val)}")
    if dropped:
        print("  skipped: " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    print("  per language:", dict(sorted(Counter(r["language"] for r in train).items())))
    print("  chain source:", dict(Counter(r.get("source") for r in train)))
    steps = math.ceil(len(train) * args.epochs / args.grad_accum)
    print(f"  {args.epochs} epochs, batch {args.grad_accum} (1 x {args.grad_accum} accumulation) "
          f"-> about {steps} optimizer steps")


# ---------- model / training (needs torch, peft, a GPU) ----------

def to_device(batch, device, dtype):
    return {k: (v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device))
            for k, v in batch.items()}


def load_model(args):
    kw = {"model_id": args.model_id} if args.model_id else {}
    if args.model == "qwen":
        obj = eb.QwenVL(**kw)
        max_side = None  # the Qwen processor already caps pixels (max_pixels)
    elif args.model == "llava":
        obj = eb.LlavaOV(load_4bit=args.load_4bit, **kw)
        max_side = obj.MAX_SIDE  # same downscaling that evaluation applies
    else:
        raise SystemExit("--model deepseek is not supported by train_sft.py yet.")
    return obj, max_side


def run_training(args, train, val):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    obj, max_side = load_model(args)
    model, processor = obj.model, obj.processor

    if args.model == "llava" and args.load_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    try:
        model.config.use_cache = False
    except Exception:
        pass

    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                     target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    device, dtype = obj.model.device, next(obj.model.parameters()).dtype
    if dtype not in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = Ctx(processor, args.max_len, max_side)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(train) * args.epochs / args.grad_accum))
    warmup = max(1, int(0.05 * total_steps))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sft_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "train_rows": len(train), "heldout_rows": len(val),
                   "total_optimizer_steps": total_steps}, f, indent=2)
    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "a", encoding="utf-8")

    def forward_loss(row):
        ex = make_example(row, ctx)
        if ex is None:
            return None
        ex = to_device(ex, device, dtype)
        with torch.autocast(device_type="cuda", dtype=amp):
            return model(**ex).loss

    def heldout_loss():
        if not val:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for row in val:
                loss = forward_loss(row)
                if loss is not None:
                    losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses) if losses else None

    model.train()
    skipped, window, seen, step, micro = Counter(), [], 0, 0, 0
    n_total = len(train) * args.epochs
    t0 = time.time()

    def optimizer_step():
        nonlocal step, micro
        lr = lr_at(step, total_steps, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        micro = 0
        return lr

    for epoch in range(1, args.epochs + 1):
        order = train[:]
        random.shuffle(order)
        for row in order:
            try:
                loss = forward_loss(row)
                if loss is None:
                    skipped["too_long_or_empty"] += 1
                    continue
                (loss / args.grad_accum).backward()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                skipped["out_of_memory"] += 1
                opt.zero_grad(set_to_none=True)
                micro = 0
                torch.cuda.empty_cache()
                continue
            micro += 1
            seen += 1
            window.append(loss.item())
            if micro == args.grad_accum:
                lr = optimizer_step()
                if step % args.log_every == 0 or step == 1:
                    mean = sum(window) / len(window)
                    el = time.time() - t0
                    eta = el / seen * max(0, n_total - seen) / 60
                    print(f"  epoch {epoch}  step {step}/{total_steps}  loss {mean:.4f}  lr {lr:.2e}  "
                          f"{el / seen:.1f}s/example  ~{eta:.0f} min left", flush=True)
                    log.write(json.dumps({"epoch": epoch, "step": step, "loss": mean, "lr": lr}) + "\n")
                    log.flush()
                    window = []
        if micro:  # leftover examples at the end of the epoch
            optimizer_step()
        hl = heldout_loss()
        msg = f"epoch {epoch} done."
        if hl is not None:
            msg += f" held-out loss {hl:.4f}"
        print(msg + (f"   skipped so far: {dict(skipped)}" if skipped else ""), flush=True)
        log.write(json.dumps({"epoch": epoch, "heldout_loss": hl, "skipped": dict(skipped)}) + "\n")
        log.flush()
        model.save_pretrained(os.path.join(args.out_dir, f"epoch{epoch}"))

    model.save_pretrained(args.out_dir)
    log.close()
    print(f"\nSaved the adapter to {args.out_dir}")
    print("Evaluate it on the unseen test set:")
    print(f"  python eval_mmb_sft.py --model {args.model} --splits-dir data/splits_full --structured "
          f"--adapter {args.out_dir} --run-name sft")


def main(args):
    if args.model == "deepseek":
        raise SystemExit("--model deepseek is not supported by train_sft.py yet.")
    recs, dropped = load_records(args)
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        raise SystemExit("No training rows found - run translate_chains.py first.")
    train, val = split_holdout(recs, args.holdout_frac, args.seed)
    summarize_data(train, val, dropped, args)
    run_training(args, train, val)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava", "deepseek"], required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--load-4bit", action="store_true", help="llava only (QLoRA)")
    p.add_argument("--data", default=None, help="override path of the SFT data file")
    p.add_argument("--out-dir", default=None, help="default outputs/{model}_sft")
    p.add_argument("--langs", nargs="+", default=["en", "hi", "ja", "zh", "te"])
    p.add_argument("--exclude-rationalized", action="store_true")
    p.add_argument("--keep-flagged", action="store_true", help="also train on rows whose translation changed a number")
    p.add_argument("--drop-q-flagged", action="store_true", help="skip rows whose translated QUESTION was flagged")
    p.add_argument("--holdout-frac", type=float, default=0.08, help="share of questions kept aside for a held-out loss; 0 = use all")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-len", type=int, default=4096, help="skip examples longer than this many tokens")
    p.add_argument("--limit", type=int, default=None, help="use only the first N rows (smoke test)")
    p.add_argument("--log-every", type=int, default=5, help="print every N optimizer steps")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    a.out_dir = a.out_dir or os.path.join("outputs", f"{a.model}_sft")
    main(a)
