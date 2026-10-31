"""
Step 5 - Language-Agnostic Reasoning Alignment (LAA): GRPO on top of the SFT model,
following R2-MultiOmnia (ACL 2025) Sec. 2.2-2.3 and Appendix K, scaled down for one GPU.

What it does, per training question:
  1. Sample a GROUP of answers from the current model (temperature 0.9).
  2. Score each answer with the paper's composite reward (Appendix K):
       strict reward        2.0  if the <answer> matches the gold answer
       answer-format reward 0.5  if an <answer> tag with a valid option letter is present
       structure reward     0.5  if all four tags appear, in order
                                 (<visual_abstraction> <formalisation> <reasoning> <answer>)
       structural integrity 0.125 per tag present, minus 0.001 per character outside the tags
     An answer that is cut off by the length limit never reaches <answer>, so it loses the
     format and strict rewards - this is how the length limit is enforced.
  3. Group-normalised advantage: A_i = (R_i - mean(R)) / std(R)   (Eq. 4)
  4. Policy-gradient update of the LoRA weights, with a KL penalty to the SFT model
     (the reference policy) whose weight beta is cosine-annealed (Eq. 5-6).
Curriculum Learning Stabilisation (Sec. 2.3): training runs in stages with a growing answer
length limit and a shrinking group size. Paper: 4K/8K/16K tokens, groups 16/8/4 on 4x48GB GPUs.
Default here: 512/768/1024 tokens, groups 8/6/4 (fits a 40GB A100 in a few hours).

Starts from the SFT LoRA adapter (train_sft.py output) and keeps training that adapter.
Supports Qwen2.5-VL (--model qwen, default) and LLaVA-OneVision (--model llava, use --load-4bit
and the 7B --model-id, same as train_sft.py / eval_mmb_sft.py).

Run (Colab A100):
    pip install -q "transformers>=4.49" accelerate peft pillow
    python train_grpo.py --adapter outputs/qwen_sft --steps-per-stage 2 --questions-per-step 1   # smoke test
    python train_grpo.py --adapter outputs/qwen_sft --out-dir outputs/qwen_grpo                  # full run
    python train_grpo.py --model llava --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf --load-4bit \
        --adapter outputs/llava_sft --out-dir outputs/llava_grpo                                  # LLaVA
Then evaluate exactly like SFT:
    python eval_mmb_sft.py --model qwen --splits-dir data/splits_full --structured \
        --adapter outputs/qwen_grpo --run-name grpo --max-new-tokens 1024

Output: outputs/qwen_grpo/ (final adapter), stage1/ stage2/ stage3/, grpo_log.jsonl, grpo_config.json
"""

import os
import re
import json
import math
import time
import random
import argparse
from PIL import Image

import eval_mmb_sft as eb

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TAG_RE = re.compile(r"</?(visual_abstraction|formalisation|reasoning|answer)>", re.I)
BLOCK_RE = re.compile(r"<(visual_abstraction|formalisation|reasoning|answer)>.*?</\1>", re.I | re.S)


# ---------- rewards (Appendix K) ----------

def compute_reward(text, row, finished):
    """Return (total reward, parts dict) for one generated answer."""
    choices = eb.normalize_choices(row.get("choices"))
    pred, tag_found = eb.extract_structured_answer(text, bool(choices)) if finished else ("", False)
    gold = str(row["answer"]).strip()

    correct = bool(tag_found and pred and eb.norm(pred) == eb.norm(gold))
    valid_letter = bool(tag_found and pred and (not choices or pred in eb.LETTERS[:len(choices)]))
    structured = finished and eb.parse_chain(text) is not None

    present = sum(1 for t in eb.STAGES if re.search(rf"<{t}>.*?</{t}>", text, re.I | re.S))
    outside = len(BLOCK_RE.sub("", text).strip())
    integrity = 0.125 * present - min(0.5, 0.001 * outside)

    parts = {
        "strict": 2.0 if correct else 0.0,
        "answer_format": 0.5 if valid_letter else 0.0,
        "format": 0.5 if structured else 0.0,
        "integrity": integrity,
    }
    return sum(parts.values()), parts


# ---------- data ----------

def load_questions(args):
    """Training-split questions in every language, interleaved so each step mixes languages."""
    per_lang = {}
    for lang in args.langs:
        path = os.path.join(args.splits_dir, f"{lang}_{args.split}.jsonl")
        if not os.path.exists(path):
            print(f"missing {path}, skipping {lang}")
            continue
        rows = [r for r in eb.load_jsonl(path)
                if r.get("source_dataset", "").startswith(args.source)
                and r.get("image_path") and os.path.exists(r["image_path"])]
        random.Random(args.seed).shuffle(rows)
        per_lang[lang] = rows
        print(f"  {lang}: {len(rows)} training questions")
    mixed, i = [], 0
    while any(i < len(v) for v in per_lang.values()):
        for lang in args.langs:
            if lang in per_lang and i < len(per_lang[lang]):
                mixed.append(per_lang[lang][i])
        i += 1
    return mixed


# ---------- model ----------

LLAVA_MAX_SIDE = 768  # same image downscaling as eval_mmb_sft.py / train_sft.py use for LLaVA


def load_policy(args):
    import torch
    from transformers import AutoProcessor
    from peft import PeftModel, prepare_model_for_kbit_training

    if args.model == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=768 * 28 * 28)
    else:
        from transformers import LlavaOnevisionForConditionalGeneration
        kw = {"torch_dtype": torch.float16, "device_map": "auto"}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        base = LlavaOnevisionForConditionalGeneration.from_pretrained(args.model_id, **kw)
        processor = AutoProcessor.from_pretrained(args.model_id)

    if args.model == "llava" and args.load_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    # "default" = the policy being trained (starts as the SFT adapter);
    # "ref" = a frozen copy of the SFT adapter, used as the KL reference policy.
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    if args.beta > 0:
        model.load_adapter(args.adapter, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
    model.print_trainable_parameters()
    return model, processor


def encode_prompt(processor, row, device, model_name="qwen"):
    import torch
    image = Image.open(row["image_path"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": eb.build_structured_prompt(row)}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if model_name == "llava":
        if max(image.size) > LLAVA_MAX_SIDE:
            s = LLAVA_MAX_SIDE / max(image.size)
            image = image.resize((max(1, int(image.width * s)), max(1, int(image.height * s))))
        return processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    return processor(text=[text], images=[image], return_tensors="pt").to(device)


def stop_ids(model, processor):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids])
    tok = processor.tokenizer
    for t in ["<|im_end|>", "<|endoftext|>"]:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    if tok.pad_token_id is not None:
        ids.add(tok.pad_token_id)
    return ids


def trim(completion, stops):
    """Cut a sampled completion at its first stop token (kept). Returns (ids, finished)."""
    seq = completion.tolist()
    for j, t in enumerate(seq):
        if t in stops:
            return completion[:j + 1], True
    return completion, False


def token_logprobs(model, inputs, completion):
    """Log-probability of each completion token given the prompt (and image)."""
    import torch
    ids = torch.cat([inputs["input_ids"], completion.unsqueeze(0)], dim=1)
    mask = torch.ones_like(ids)
    n_prompt, n_new = inputs["input_ids"].shape[1], completion.shape[0]
    extra = {}
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        # Per-token inputs (e.g. mm_token_type_ids / token_type_ids in newer transformers)
        # cover only the prompt; extend them over the completion (0 = ordinary text token).
        if torch.is_tensor(v) and v.dim() == 2 and v.shape == inputs["input_ids"].shape:
            v = torch.cat([v, torch.zeros((1, n_new), dtype=v.dtype, device=v.device)], dim=1)
        extra[k] = v
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=amp):
        logits = model(input_ids=ids, attention_mask=mask, **extra).logits
    logits = logits[:, n_prompt - 1:-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, completion.view(1, -1, 1)).squeeze(-1).squeeze(0)


def save_adapter(model, path):
    """Save only the trained ("default") LoRA adapter.
    Saved to local disk first and then copied, because writing straight into a Google Drive
    folder from Colab can fail ("No such file or directory"). If safetensors refuses the
    tensors, falls back to the plain PyTorch .bin format; both load with --adapter."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="adapter_")
    try:
        try:
            model.save_pretrained(tmp, selected_adapters=["default"])
        except Exception as e:
            print(f"  safetensors save failed ({type(e).__name__}: {str(e)[:120]}); saving as .bin instead",
                  flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            model.save_pretrained(tmp, selected_adapters=["default"], safe_serialization=False)
        os.makedirs(path, exist_ok=True)
        for root, _, files in os.walk(tmp):
            dst = os.path.join(path, os.path.relpath(root, tmp))
            os.makedirs(dst, exist_ok=True)
            for fn in files:
                shutil.copy2(os.path.join(root, fn), os.path.join(dst, fn))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- training ----------

def main(args):
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    questions = load_questions(args)
    stages = list(zip(args.stage_lengths, args.stage_groups))
    needed = len(stages) * args.steps_per_stage * args.questions_per_step
    print(f"{len(questions)} training questions; this run uses {needed} "
          f"({len(stages)} stages x {args.steps_per_stage} steps x {args.questions_per_step} questions)")
    if not questions:
        raise SystemExit("No training questions found.")

    model, processor = load_policy(args)
    device = next(model.parameters()).device
    stops = stop_ids(model, processor)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "grpo_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(args.out_dir, "grpo_log.jsonl"), "a", encoding="utf-8")

    total_steps = len(stages) * args.steps_per_stage
    step, q_pos, t0 = 0, 0, time.time()
    for s_idx, (max_len, group) in enumerate(stages, 1):
        print(f"\n=== Stage {s_idx}/{len(stages)}: max {max_len} new tokens, group size {group} ===", flush=True)
        for _ in range(args.steps_per_stage):
            beta = args.beta / 2 * (1 + math.cos(math.pi * step / max(1, total_steps)))  # Eq. 6
            stats = {"reward": [], "correct": [], "format": [], "finished": [], "length": []}
            n_used, skipped = 0, 0
            opt.zero_grad(set_to_none=True)

            for _q in range(args.questions_per_step):
                row = questions[q_pos % len(questions)]
                q_pos += 1
                try:
                    inputs = encode_prompt(processor, row, device, args.model)
                    model.eval()
                    with torch.no_grad():
                        out = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                             top_p=1.0, max_new_tokens=max_len,
                                             num_return_sequences=group, pad_token_id=pad_id, use_cache=True)
                    n_prompt = inputs["input_ids"].shape[1]
                    samples = [trim(c, stops) for c in out[:, n_prompt:]]
                    texts = processor.batch_decode([c for c, _ in samples], skip_special_tokens=True)
                    rewards = []
                    for text, (comp, fin) in zip(texts, samples):
                        r, parts = compute_reward(text, row, fin)
                        rewards.append(r)
                        stats["reward"].append(r)
                        stats["correct"].append(parts["strict"] > 0)
                        stats["format"].append(parts["format"] > 0)
                        stats["finished"].append(fin)
                        stats["length"].append(len(comp))
                    rw = torch.tensor(rewards)
                    if rw.std() < 1e-6:  # identical rewards -> no learning signal for this question
                        skipped += 1
                        continue
                    adv = (rw - rw.mean()) / (rw.std() + 1e-4)  # Eq. 4

                    model.train()
                    for (comp, _), a in zip(samples, adv.tolist()):
                        comp = comp.to(device)
                        logp = token_logprobs(model, inputs, comp)
                        # ratio pi/pi_old = 1 at this point (one update per batch), so the GRPO
                        # surrogate reduces to the advantage-weighted log-likelihood (Eq. 5).
                        loss = -(a * logp).mean()
                        if beta > 0:
                            model.set_adapter("ref")
                            with torch.no_grad():
                                ref = token_logprobs(model, inputs, comp)
                            model.set_adapter("default")
                            d = ref - logp
                            loss = loss + beta * (torch.exp(d) - d - 1).mean()  # k3 KL estimator
                        (loss / (group * args.questions_per_step)).backward()
                    n_used += 1
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    print("    out of memory on one question - skipped", flush=True)
                    opt.zero_grad(set_to_none=True)
                    n_used = 0
                    torch.cuda.empty_cache()
                    model.set_adapter("default")

            if n_used:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            step += 1

            mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
            rec = {"stage": s_idx, "step": step, "beta": round(beta, 5),
                   "reward": round(mean(stats["reward"]), 4),
                   "sample_acc": round(mean(stats["correct"]), 4),
                   "format_ok": round(mean(stats["format"]), 4),
                   "finished": round(mean(stats["finished"]), 4),
                   "mean_len": round(mean(stats["length"]), 1),
                   "questions_used": n_used, "no_signal": skipped}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            el = time.time() - t0
            print(f"  step {step}/{total_steps}  reward {rec['reward']:.3f}  acc {rec['sample_acc']:.2f}  "
                  f"format {rec['format_ok']:.2f}  finished {rec['finished']:.2f}  len {rec['mean_len']:.0f}  "
                  f"used {n_used}/{args.questions_per_step}  ~{el / step * (total_steps - step) / 60:.0f} min left",
                  flush=True)

        stage_dir = os.path.join(args.out_dir, f"stage{s_idx}")
        save_adapter(model, stage_dir)
        print(f"saved stage {s_idx} adapter to {stage_dir}", flush=True)

    save_adapter(model, args.out_dir)
    log.close()
    print(f"\nSaved the GRPO adapter to {args.out_dir}")
    print("Evaluate it on the unseen test set:")
    mid = f" --model-id {args.model_id}" + (" --load-4bit" if args.load_4bit else "") if args.model == "llava" else ""
    print(f"  python eval_mmb_sft.py --model {args.model}{mid} --splits-dir {args.splits_dir} --structured "
          f"--adapter {args.out_dir} --run-name grpo --max-new-tokens {args.stage_lengths[-1]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "llava"], default="qwen")
    p.add_argument("--model-id", default=None,
                   help="default Qwen/Qwen2.5-VL-3B-Instruct; for llava pass llava-hf/llava-onevision-qwen2-7b-ov-hf")
    p.add_argument("--load-4bit", action="store_true", help="llava only (QLoRA), as in train_sft.py")
    p.add_argument("--adapter", required=True, help="SFT adapter to start from, e.g. outputs/qwen_sft")
    p.add_argument("--out-dir", default=os.path.join("outputs", "qwen_grpo"))
    p.add_argument("--splits-dir", default=os.path.join("data", "splits_full"))
    p.add_argument("--split", default="train", help="never the test split")
    p.add_argument("--langs", nargs="+", default=["en", "zh", "ja", "hi", "te"])
    p.add_argument("--source", default="xmmmu")
    p.add_argument("--stage-lengths", nargs="+", type=int, default=[512, 768, 1024],
                   help="max new tokens per curriculum stage (paper: 4096 8192 16384)")
    p.add_argument("--stage-groups", nargs="+", type=int, default=[8, 6, 4],
                   help="answers sampled per question in each stage (paper: 16 8 4)")
    p.add_argument("--steps-per-stage", type=int, default=25)
    p.add_argument("--questions-per-step", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5, help="for LoRA; the paper's 5e-7 is for full fine-tuning")
    p.add_argument("--beta", type=float, default=0.04, help="KL weight (paper 0.04); 0 = no KL, faster")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.model_id is None:
        if a.model == "llava":
            raise SystemExit("For --model llava pass --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf "
                             "(the 0.5B default used elsewhere is not the model in this project)")
        a.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    if len(a.stage_lengths) != len(a.stage_groups):
        raise SystemExit("--stage-lengths and --stage-groups need the same number of values")
    main(a)
