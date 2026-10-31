"""
deepseek_r2.py - R2-MultiOmnia stages (SFT, GRPO and their evaluation) for DeepSeek-VL2-Tiny.

DeepSeek-VL2 only runs in its own environment (deepseek_vl2 package + transformers==4.38.2), so
it gets this separate script instead of train_sft.py / train_grpo.py / eval_mmb_sft.py. It reuses
their logic so everything matches the Qwen/LLaVA runs:
  - same structured 4-stage prompt (eval_mmb_sft.build_structured_prompt)
  - same SFT data handling (train_sft: flagged rows skipped, held-out questions, LR schedule)
  - same GRPO rewards, group-normalised advantage, KL annealing and length curriculum (train_grpo)
  - same results file format and summary table (eval_mmb_sft), so the result tables line up.

Environment (Kaggle, same as stage2b_deepseek_baseline.ipynb, plus peft):
    pip install -e DeepSeek-VL2 --no-deps
    pip install "transformers==4.38.2" "timm>=0.9.16" attrdict3 einops sentencepiece accelerate "peft==0.10.0"
    + the xformers stand-in on PYTHONPATH

Usage (run from the folder that contains data/):
    # 1. SFT on the translated MMB chains (paper setup: 1 epoch)
    python deepseek_r2.py sft  --data data/mmb/qwen_sft_data_v2.jsonl --out-dir outputs/deepseek_sft
    # 2. evaluation (no --adapter = untrained baseline)
    python deepseek_r2.py eval --run-name structured_sub --limit 100 --max-new-tokens 1024
    python deepseek_r2.py eval --adapter outputs/deepseek_sft --run-name sft_sub --limit 100 --max-new-tokens 1024
    # 3. GRPO on top of the SFT adapter, then evaluate it
    python deepseek_r2.py grpo --adapter outputs/deepseek_sft --out-dir outputs/deepseek_grpo
    python deepseek_r2.py eval --adapter outputs/deepseek_grpo --run-name grpo_sub --limit 100 --max-new-tokens 1024

Results: results/deepseek_{run-name}_splits_full_test_predictions.jsonl (resumable, like eval_mmb_sft.py)
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
import train_sft as ts
import train_grpo as tg

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"
# LoRA on the language model only: attention (MHA in Tiny; MLA names included for Small) and the
# dense / shared-expert MLPs. The routed experts and the vision encoder stay frozen.
LORA_TARGETS = (r"^language\.model\.layers\.\d+\.("
                r"self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
                r"|mlp\.(gate_proj|up_proj|down_proj)"
                r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$")


# ---------- model ----------

def load_model(model_id, adapter=None, trainable=False, ref_adapter=False):
    """Returns (peft_or_plain_model, vl_model, processor, dtype).
    vl_model is the underlying DeepseekVLV2ForCausalLM (LoRA layers are injected into it)."""
    import torch
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor

    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    dtype = torch.bfloat16  # same as the baseline run (eval_baseline.py)
    base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).cuda()
    if trainable:
        base.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.language.config.use_cache = False
    if adapter is None and not trainable:
        return base.eval(), base, processor, dtype

    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
        print(f"Loaded LoRA adapter from {adapter}", flush=True)
        if ref_adapter:
            model.load_adapter(adapter, adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(base, cfg)
    if trainable:
        for prm in model.parameters():  # keep the trainable LoRA weights in fp32 for stable updates
            if prm.requires_grad:
                prm.data = prm.data.float()
        model.print_trainable_parameters()
    else:
        model.eval()
    return model, model.base_model.model, processor, dtype


def conversation(row, answer=""):
    prompt = eb.build_structured_prompt(row)
    return [{"role": "<|User|>", "content": f"<image>\n{prompt}", "images": ["in-memory"]},
            {"role": "<|Assistant|>", "content": answer}]


def encode_prompt(processor, row, dtype):
    image = Image.open(row["image_path"]).convert("RGB")
    return processor(conversations=conversation(row), images=[image], force_batchify=True,
                     inference_mode=True, system_prompt="").to("cuda", dtype=dtype)


def prompt_embeds(vl, inputs):
    import torch
    with torch.no_grad():  # the image encoder and token embeddings are frozen
        return vl.prepare_inputs_embeds(input_ids=inputs.input_ids, images=inputs.images,
                                        images_seq_mask=inputs.images_seq_mask,
                                        images_spatial_crop=inputs.images_spatial_crop)


def generate(vl, processor, inputs, max_new_tokens, n=1, temperature=0.0):
    """Returns a list of n generated token-id lists (new tokens only)."""
    import torch
    tok = processor.tokenizer
    emb = prompt_embeds(vl, inputs)
    mask = inputs.attention_mask
    if n > 1:
        emb, mask = emb.repeat(n, 1, 1), mask.repeat(n, 1)
    sample = n > 1 or temperature > 0
    with torch.no_grad():
        out = vl.language.generate(
            inputs_embeds=emb, attention_mask=mask,
            pad_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
            max_new_tokens=max_new_tokens, use_cache=True,
            **({"do_sample": True, "temperature": temperature or 0.9, "top_p": 1.0} if sample
               else {"do_sample": False}))
    return [seq.tolist() for seq in out]  # generate() on inputs_embeds returns only the new tokens


def completion_logprobs(vl, inputs, emb, completion):
    """Log-prob of each completion token, given the prompt embeddings `emb` (1, T, D)."""
    import torch
    comp = torch.tensor([completion], device=emb.device)
    with torch.no_grad():
        comp_emb = vl.language.get_input_embeddings()(comp)
    full = torch.cat([emb, comp_emb.to(emb.dtype)], dim=1)
    mask = torch.ones(full.shape[:2], dtype=torch.long, device=emb.device)
    logits = vl.language(inputs_embeds=full, attention_mask=mask, use_cache=False).logits
    n_prompt = emb.shape[1]
    logits = logits[:, n_prompt - 1:-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, comp.view(1, -1, 1)).squeeze(-1).squeeze(0)


# ---------- SFT ----------

def sft_example(processor, row, dtype, max_len):
    """Tokenise prompt + target chain with the processor's own SFT format; loss only on the target."""
    import torch
    image = Image.open(row["image_path"]).convert("RGB")
    full = processor(conversations=conversation(row, row["target"]), images=[image], force_batchify=True,
                     inference_mode=False, system_prompt="")
    prompt = processor(conversations=conversation(row), images=[image], force_batchify=True,
                       inference_mode=True, system_prompt="")
    n_prompt, n_full = prompt.input_ids.shape[1], full.input_ids.shape[1]
    if n_full > max_len or n_full <= n_prompt or not torch.equal(full.input_ids[0, :n_prompt], prompt.input_ids[0]):
        return None
    labels = full.input_ids.clone()
    labels[:, :n_prompt] = -100
    full = full.to("cuda", dtype=dtype)
    return full, labels.cuda()


def run_sft(args):
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.model, args.exclude_rationalized, args.drop_q_flagged = "deepseek", args.exclude_rationalized, False
    recs, dropped = ts.load_records(args)
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        raise SystemExit("No training rows found.")
    train, val = ts.split_holdout(recs, args.holdout_frac, args.seed)
    ts.summarize_data(train, val, dropped, args)

    model, vl, processor, dtype = load_model(args.model_id, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = max(1, math.ceil(len(train) * args.epochs / args.grad_accum))
    warmup = max(1, int(0.05 * total))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "sft_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "train_rows": len(train), "heldout_rows": len(val),
                   "total_optimizer_steps": total}, f, indent=2)
    log = open(os.path.join(args.out_dir, "train_log.jsonl"), "a", encoding="utf-8")

    def loss_of(row):
        ex = sft_example(processor, row, dtype, args.max_len)
        if ex is None:
            return None
        inputs, labels = ex
        emb = prompt_embeds(vl, inputs)
        return vl.language(inputs_embeds=emb, attention_mask=inputs.attention_mask,
                           labels=labels, use_cache=False).loss

    skipped, window, step, micro, seen, t0 = Counter(), [], 0, 0, 0, time.time()
    model.train()

    def opt_step():
        nonlocal step, micro
        lr = ts.lr_at(step, total, warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step, micro = step + 1, 0
        return lr

    for epoch in range(1, args.epochs + 1):
        order = train[:]
        random.shuffle(order)
        for row in order:
            try:
                loss = loss_of(row)
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
            micro, seen = micro + 1, seen + 1
            window.append(loss.item())
            if micro == args.grad_accum:
                lr = opt_step()
                if step % args.log_every == 0 or step == 1:
                    el = time.time() - t0
                    eta = el / seen * max(0, len(train) * args.epochs - seen) / 60
                    mean = sum(window) / len(window)
                    print(f"  epoch {epoch}  step {step}/{total}  loss {mean:.4f}  lr {lr:.2e}  "
                          f"{el / seen:.1f}s/example  ~{eta:.0f} min left", flush=True)
                    log.write(json.dumps({"epoch": epoch, "step": step, "loss": mean, "lr": lr}) + "\n")
                    log.flush()
                    window = []
        if micro:
            opt_step()
        if val:
            model.eval()
            with torch.no_grad():
                ls = [l.item() for l in (loss_of(r) for r in val) if l is not None]
            model.train()
            hl = sum(ls) / len(ls) if ls else None
        else:
            hl = None
        print(f"epoch {epoch} done. held-out loss {hl}   skipped so far: {dict(skipped)}", flush=True)
        log.write(json.dumps({"epoch": epoch, "heldout_loss": hl, "skipped": dict(skipped)}) + "\n")
        log.flush()
        tg.save_adapter(model, os.path.join(args.out_dir, f"epoch{epoch}"))
    tg.save_adapter(model, args.out_dir)
    log.close()
    print(f"\nSaved the adapter to {args.out_dir}")


# ---------- evaluation ----------

def run_eval(args):
    os.makedirs(eb.RESULTS_DIR, exist_ok=True)
    tag = "" if os.path.normpath(args.splits_dir) == os.path.normpath(eb.SPLITS_DIR) \
        else "_" + os.path.basename(os.path.normpath(args.splits_dir))
    run = f"_{args.run_name}" if args.run_name else ""
    pred_path = os.path.join(eb.RESULTS_DIR, f"deepseek{run}{tag}_{args.split}_predictions.jsonl")

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
    print(f"{len(todo)} rows to evaluate.", flush=True)

    if todo:
        _, vl, processor, dtype = load_model(args.model_id, adapter=args.adapter)
        tok = processor.tokenizer
        t0 = time.time()
        with open(pred_path, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo, 1):
                if not row.get("image_path") or not os.path.exists(row["image_path"]):
                    print(f"  skip {row['id']}: image missing")
                    continue
                try:
                    ids = generate(vl, processor, encode_prompt(processor, row, dtype), args.max_new_tokens)[0]
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    print(f"  skip {row['id']}: out of memory", flush=True)
                    continue
                raw = tok.decode(ids, skip_special_tokens=True)
                choices = eb.normalize_choices(row.get("choices"))
                pred, tag_found = eb.extract_structured_answer(raw, bool(choices))
                gold = str(row["answer"]).strip()
                f.write(json.dumps({
                    "id": row["id"], "match_id": row.get("match_id"), "language": row["language"],
                    "gold": gold, "pred": pred, "raw": raw, "correct": eb.norm(pred) == eb.norm(gold),
                    "shrunk_to": None, "answer_tag": tag_found}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0:
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el / i:.1f}s/row, ~{el / i * (len(todo) - i) / 60:.0f} min left)",
                          flush=True)
    eb.summarize(pred_path, args.langs)


# ---------- GRPO ----------

def run_grpo(args):
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    questions = tg.load_questions(args)
    stages = list(zip(args.stage_lengths, args.stage_groups))
    total_steps = len(stages) * args.steps_per_stage
    print(f"{len(questions)} training questions; this run uses "
          f"{total_steps * args.questions_per_step}")

    model, vl, processor, dtype = load_model(args.model_id, adapter=args.adapter, trainable=True,
                                             ref_adapter=args.beta > 0)
    tok = processor.tokenizer
    stops = {tok.eos_token_id}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "grpo_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(args.out_dir, "grpo_log.jsonl"), "a", encoding="utf-8")

    step, q_pos, t0 = 0, 0, time.time()
    for s_idx, (max_len, group) in enumerate(stages, 1):
        print(f"\n=== Stage {s_idx}/{len(stages)}: max {max_len} new tokens, group size {group} ===", flush=True)
        for _ in range(args.steps_per_stage):
            beta = args.beta / 2 * (1 + math.cos(math.pi * step / max(1, total_steps)))  # KL annealing (Eq. 6)
            stats = {"reward": [], "correct": [], "format": [], "finished": [], "length": []}
            n_used, no_signal = 0, 0
            opt.zero_grad(set_to_none=True)
            for _q in range(args.questions_per_step):
                row = questions[q_pos % len(questions)]
                q_pos += 1
                try:
                    inputs = encode_prompt(processor, row, dtype)
                    model.eval()
                    outs = generate(vl, processor, inputs, max_len, n=group, temperature=args.temperature)
                    samples = []
                    for ids in outs:
                        cut = next((j for j, t in enumerate(ids) if t in stops), None)
                        samples.append((ids[:cut + 1], True) if cut is not None else (ids, False))
                    rewards = []
                    for comp, fin in samples:
                        r, parts = tg.compute_reward(tok.decode(comp, skip_special_tokens=True), row, fin)
                        rewards.append(r)
                        stats["reward"].append(r)
                        stats["correct"].append(parts["strict"] > 0)
                        stats["format"].append(parts["format"] > 0)
                        stats["finished"].append(fin)
                        stats["length"].append(len(comp))
                    rw = torch.tensor(rewards)
                    if rw.std() < 1e-6:
                        no_signal += 1
                        continue
                    adv = (rw - rw.mean()) / (rw.std() + 1e-4)  # group-normalised advantage (Eq. 4)
                    emb = prompt_embeds(vl, inputs)
                    model.train()
                    for (comp, _), a in zip(samples, adv.tolist()):
                        if not comp:
                            continue
                        logp = completion_logprobs(vl, inputs, emb, comp)
                        loss = -(a * logp).mean()
                        if beta > 0:
                            model.set_adapter("ref")
                            with torch.no_grad():
                                ref = completion_logprobs(vl, inputs, emb, comp)
                            model.set_adapter("default")
                            d = ref - logp
                            loss = loss + beta * (torch.exp(d) - d - 1).mean()
                        (loss / (group * args.questions_per_step)).backward()
                    n_used += 1
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    print("    out of memory on one question - skipped", flush=True)
                    opt.zero_grad(set_to_none=True)
                    n_used = 0
                    torch.cuda.empty_cache()
                    if args.beta > 0:
                        model.set_adapter("default")
            if n_used:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            step += 1
            mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            rec = {"stage": s_idx, "step": step, "beta": round(beta, 5), "reward": round(mean(stats["reward"]), 4),
                   "sample_acc": round(mean(stats["correct"]), 4), "format_ok": round(mean(stats["format"]), 4),
                   "finished": round(mean(stats["finished"]), 4), "mean_len": round(mean(stats["length"]), 1),
                   "questions_used": n_used, "no_signal": no_signal}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            el = time.time() - t0
            print(f"  step {step}/{total_steps}  reward {rec['reward']:.3f}  acc {rec['sample_acc']:.2f}  "
                  f"format {rec['format_ok']:.2f}  finished {rec['finished']:.2f}  len {rec['mean_len']:.0f}  "
                  f"used {n_used}/{args.questions_per_step}  ~{el / step * (total_steps - step) / 60:.0f} min left",
                  flush=True)
        tg.save_adapter(model, os.path.join(args.out_dir, f"stage{s_idx}"))
        print(f"saved stage {s_idx} adapter", flush=True)
    tg.save_adapter(model, args.out_dir)
    log.close()
    print(f"\nSaved the GRPO adapter to {args.out_dir}")


# ---------- command line ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-id", default=MODEL_ID)
    common.add_argument("--splits-dir", default=os.path.join("data", "splits_full"))
    common.add_argument("--langs", nargs="+", default=["en", "zh", "ja", "hi", "te"])
    common.add_argument("--source", default="xmmmu")
    common.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("sft", parents=[common])
    s.add_argument("--data", default=os.path.join("data", "mmb", "qwen_sft_data_v2.jsonl"))
    s.add_argument("--out-dir", default=os.path.join("outputs", "deepseek_sft"))
    s.add_argument("--exclude-rationalized", action="store_true")
    s.add_argument("--keep-flagged", action="store_true")
    s.add_argument("--holdout-frac", type=float, default=0.08)
    s.add_argument("--epochs", type=int, default=1, help="paper: 1 epoch warm-up")
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--max-len", type=int, default=4096)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--log-every", type=int, default=5)

    e = sub.add_parser("eval", parents=[common])
    e.add_argument("--adapter", default=None, help="omit for the untrained (baseline) model")
    e.add_argument("--run-name", default="")
    e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=None, help="questions per language")
    e.add_argument("--max-new-tokens", type=int, default=1024)

    g = sub.add_parser("grpo", parents=[common])
    g.add_argument("--adapter", required=True, help="the SFT adapter to start from")
    g.add_argument("--out-dir", default=os.path.join("outputs", "deepseek_grpo"))
    g.add_argument("--split", default="train")
    g.add_argument("--stage-lengths", nargs="+", type=int, default=[512, 768, 1024])
    g.add_argument("--stage-groups", nargs="+", type=int, default=[4, 4, 4])
    g.add_argument("--steps-per-stage", type=int, default=4)
    g.add_argument("--questions-per-step", type=int, default=3)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--beta", type=float, default=0.04)
    g.add_argument("--temperature", type=float, default=0.9)

    a = p.parse_args()
    if a.mode == "sft":
        run_sft(a)
    elif a.mode == "eval":
        run_eval(a)
    else:
        if len(a.stage_lengths) != len(a.stage_groups):
            raise SystemExit("--stage-lengths and --stage-groups need the same number of values")
        run_grpo(a)
