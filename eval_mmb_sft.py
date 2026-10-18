"""
Evaluation for the MMB / SFT stages (eval_mmb_sft.py).

This is a SEPARATE file from eval_baseline.py on purpose: eval_baseline.py is
left untouched as the record of how accuracy was measured with the unmodified
models (direct answer, no tuning). This file contains everything in that
script PLUS what the structured-reasoning stages need:
    --structured      ask for <visual_abstraction> / <formalisation> /
                      <reasoning> / <answer> and score the <answer> tag
    --adapter PATH    load LoRA weights produced by train_sft.py
    --run-name NAME   label added to the results file name
    --max-new-tokens  default 16 (direct answer) or 640 (--structured)
It also provides the shared helpers (structured prompt, chain parser, model
classes) that generate_mmb.py imports. Without the new flags it behaves like
eval_baseline.py.

Needs a GPU. Run on Colab / the college lab with transformers>=4.49:
    pip install "transformers>=4.49" accelerate pillow torch peft bitsandbytes

Usage:
    # untrained model, structured prompt only (control row)
    python eval_mmb_sft.py --model qwen --splits-dir data/splits_full --structured --run-name structured
    # after SFT
    python eval_mmb_sft.py --model qwen --splits-dir data/splits_full --structured --adapter outputs/qwen_sft --run-name sft
    # LLaVA (16 GB GPU)
    python eval_mmb_sft.py --model llava --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf --load-4bit --splits-dir data/splits_full --structured --run-name structured
    # quick smoke test
    python eval_mmb_sft.py --model qwen --splits-dir data/splits_full --structured --limit 5

Other options (same as eval_baseline.py): --split, --langs, --source, --limit,
--splits-dir, --model-id, --load-4bit.

Output:
    results/{model}[_{run-name}][_{splits-dir}]_{split}_predictions.jsonl
    (resumable: reruns skip done ids) and a printed summary table.

DeepSeek-VL2-Small needs the separate deepseek_vl2 package (see the DeepSeekVL2
class docstring) and must be run in its own environment.
"""

import os
import re
import ast
import json
import argparse
from collections import defaultdict
from PIL import Image

# Reduces GPU memory fragmentation (must be set before torch is imported;
# torch is only imported lazily inside the model classes below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SPLITS_DIR = os.path.join("data", "splits")
RESULTS_DIR = "results"
LETTERS = "ABCDEFGHIJ"


# ---------- data helpers ----------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize_choices(choices):
    """Choices should be a list; some sources store a stringified list."""
    if choices is None:
        return []
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            return []
    return [str(c) for c in choices]


def build_prompt(row):
    q = row["question"].strip()
    choices = normalize_choices(row.get("choices"))
    if choices:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices[:len(LETTERS)]))
        return f"{q}\n{opts}\nAnswer with the option's letter only."
    return f"{q}\nAnswer with a short final answer only."


def extract_answer(text, has_choices):
    text = text.strip()
    if has_choices:
        m = re.match(r"^\(?([A-J])(?:[\).:\s]|$)", text)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-J])\b", text)
        return m.group(1) if m else ""
    return text


def norm(s):
    return re.sub(r"[^\w]+", "", str(s).lower())


# ---------- structured (MMB) prompt and parsing ----------
# The four stages of Multilingual Modality Bridging. The SAME prompt is used
# when generating training chains, during SFT, and at evaluation time.

STAGES = ["visual_abstraction", "formalisation", "reasoning", "answer"]


def build_structured_prompt(row, hint=None):
    q = row["question"].strip()
    choices = normalize_choices(row.get("choices"))
    answer_hint = "the letter of the correct option only" if choices else "the final answer only"
    instruction = (
        "Solve the problem in four stages, using exactly these tags in this order:\n"
        "<visual_abstraction> describe only the information in the image that matters for the question "
        "(objects, numbers, labels, relations) </visual_abstraction>\n"
        "<formalisation> restate the problem precisely: what is given, what is asked, and the formulas "
        "or relations you will use </formalisation>\n"
        "<reasoning> Step 1: ... Step 2: ... show every step, including calculations </reasoning>\n"
        f"<answer> {answer_hint} </answer>"
    )
    body = f"{instruction}\n\nQuestion:\n{q}"
    if choices:
        body += "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices[:len(LETTERS)]))
    if hint:
        body += f"\n\n{hint}"
    return body


def parse_chain(text):
    """Return {stage: content} if all four stages are present, non-empty and
    in the right order, else None."""
    out, last_pos = {}, -1
    for tag in STAGES:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
        if not m or m.start() < last_pos or not m.group(1).strip():
            return None
        out[tag] = m.group(1).strip()
        last_pos = m.start()
    return out


def extract_structured_answer(text, has_choices):
    """Pull the final answer out of the <answer> tag (also accepts an
    unclosed tag from a truncated generation). Returns (answer, tag_found)."""
    ms = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if ms:
        body = ms[-1].strip()
    else:
        m = re.search(r"<answer>(.*)$", text, flags=re.S | re.I)
        body = m.group(1).strip() if m else ""
    if not body:
        return "", False
    return (extract_answer(body, True) if has_choices else body), True


def gen_kwargs(max_new_tokens, n, temperature):
    if n > 1 or temperature > 0:
        return dict(max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95,
                    temperature=temperature or 0.7, num_return_sequences=n)
    return dict(max_new_tokens=max_new_tokens, do_sample=False)


# ---------- model adapters ----------

class QwenVL:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", adapter=None):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:  # LoRA weights from train_sft.py
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
            print(f"Loaded LoRA adapter from {adapter}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=768 * 28 * 28)

    def generate(self, image, prompt, max_new_tokens=16, n=1, temperature=0.0):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs(max_new_tokens, n, temperature))
        out = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(out, skip_special_tokens=True)
        return texts if n > 1 else texts[0]


class LlavaOV:
    # LLaVA-OneVision splits big images into many tiles (up to ~10k visual
    # tokens), which blows up attention memory on a 16 GB GPU. Capping the
    # longest image side keeps it to a few thousand tokens.
    MAX_SIDE = 768

    def __init__(self, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", load_4bit=False, adapter=None):
        import torch
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        self.torch = torch
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, **kwargs)
        print(f"LLaVA loaded, 4-bit={load_4bit}, model memory: {self.model.get_memory_footprint() / 1e9:.1f} GB "
              f"(expect ~6 GB for 7B in 4-bit, ~15 GB if fp16)", flush=True)
        if adapter:  # LoRA weights from train_sft.py
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
            print(f"Loaded LoRA adapter from {adapter}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, image, prompt, max_new_tokens=16, n=1, temperature=0.0):
        if max(image.size) > self.MAX_SIDE:
            scale = self.MAX_SIDE / max(image.size)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.model.device, self.torch.float16)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs(max_new_tokens, n, temperature))
        out = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(out, skip_special_tokens=True)
        return texts if n > 1 else texts[0]


class DeepSeekVL2:
    """DeepSeek-VL2-Small (~16B total params, ~2.8B active; MoE).
    Needs the deepseek_vl2 package, which pins an OLD transformers version,
    so run this in its OWN environment (separate from Qwen/LLaVA):
        git clone https://github.com/deepseek-ai/DeepSeek-VL2
        cd DeepSeek-VL2 && pip install -e .
    bf16 weights are ~32 GB, so this needs a 40 GB-class GPU (A100)."""

    def __init__(self, model_id="deepseek-ai/deepseek-vl2-small"):
        import torch
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor  # noqa: F401 (also registers the model class)
        self.torch = torch
        self.processor = DeepseekVLV2Processor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model = model.cuda().eval()

    def generate(self, image, prompt, max_new_tokens=16, n=1, temperature=0.0):
        texts = [self._one(image, prompt, max_new_tokens, temperature > 0 or n > 1, temperature or 0.7)
                 for _ in range(n)]
        return texts if n > 1 else texts[0]

    def _one(self, image, prompt, max_new_tokens, sample, temperature):
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n{prompt}", "images": ["in-memory"]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = self.processor(
            conversations=conversation, images=[image],
            force_batchify=True, system_prompt="",
        ).to(self.model.device)
        with self.torch.no_grad():
            embeds = self.model.prepare_inputs_embeds(**inputs)
            out = self.model.language.generate(
                inputs_embeds=embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens, use_cache=True,
                **({"do_sample": True, "temperature": temperature, "top_p": 0.95} if sample
                   else {"do_sample": False}))
        # generate() on inputs_embeds returns only the new tokens
        return self.tokenizer.decode(out[0].cpu().tolist(), skip_special_tokens=True)


MODELS = {"qwen": QwenVL, "llava": LlavaOV, "deepseek": DeepSeekVL2}


# ---------- evaluation ----------

def generate_with_retry(model, image, prompt, sizes=(None, 512, 384), **gen):
    """Run model.generate; if the GPU runs out of memory (rare, very long
    prompts + large image), free memory and retry with a smaller image.
    Returns (raw_text, max_side_used) or (None, None) if every size failed."""
    import gc
    for max_side in sizes:
        img = image
        if max_side and max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side))
        try:
            return model.generate(img, prompt, **gen), max_side
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            print(f"    out of memory (image max side {max_side or 'original'}), retrying smaller...", flush=True)
    return None, None


def evaluate(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Default splits dir keeps the original result file names; any other dir
    # (e.g. data/splits_full) gets its own suffix so results never mix.
    tag = "" if os.path.normpath(args.splits_dir) == os.path.normpath(SPLITS_DIR) \
        else "_" + os.path.basename(os.path.normpath(args.splits_dir))
    run = f"_{args.run_name}" if args.run_name else ""
    pred_path = os.path.join(RESULTS_DIR, f"{args.model}{run}{tag}_{args.split}_predictions.jsonl")

    done = set()
    if os.path.exists(pred_path):
        for r in load_jsonl(pred_path):
            done.add(r["id"])
        print(f"Resuming: {len(done)} predictions already saved.")

    todo = []
    for lang in args.langs:
        path = os.path.join(args.splits_dir, f"{lang}_{args.split}.jsonl")
        if not os.path.exists(path):
            print(f"Missing {path}, skipping {lang}")
            continue
        rows = [r for r in load_jsonl(path) if r["source_dataset"].startswith(args.source)]
        if args.limit:
            rows = rows[:args.limit]
        todo += [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} rows to evaluate.")

    if todo:
        kwargs = {"model_id": args.model_id} if args.model_id else {}
        if args.load_4bit and args.model == "llava":
            kwargs["load_4bit"] = True
        if args.adapter:
            if args.model == "deepseek":
                raise SystemExit("--adapter is not supported for deepseek yet")
            kwargs["adapter"] = args.adapter
        max_new = args.max_new_tokens or (640 if args.structured else 16)
        model = MODELS[args.model](**kwargs)
        with open(pred_path, "a", encoding="utf-8") as f:
            for i, row in enumerate(todo, 1):
                img_path = row.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    print(f"  skip {row['id']}: image missing")
                    continue
                image = Image.open(img_path).convert("RGB")
                choices = normalize_choices(row.get("choices"))
                prompt = build_structured_prompt(row) if args.structured else build_prompt(row)
                raw, shrunk_to = generate_with_retry(model, image, prompt, max_new_tokens=max_new)
                if raw is None:
                    print(f"  skip {row['id']}: out of memory even at the smallest image size", flush=True)
                    continue
                if args.structured:
                    pred, tag_found = extract_structured_answer(raw, bool(choices))
                else:
                    pred, tag_found = extract_answer(raw, bool(choices)), None
                gold = str(row["answer"]).strip()
                rec = {
                    "id": row["id"], "match_id": row.get("match_id"), "language": row["language"],
                    "gold": gold, "pred": pred, "raw": raw,
                    "correct": norm(pred) == norm(gold),
                    "shrunk_to": shrunk_to,  # None = normal size; 512/384 = retried after OOM
                    "answer_tag": tag_found,  # structured mode only: did the output contain <answer>?
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    summarize(pred_path, args.langs)


def summarize(pred_path, langs):
    rows = load_jsonl(pred_path)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)

    print("\nAccuracy by language")
    print(f"{'lang':<6}{'n':>6}{'acc':>9}")
    for lang in langs:
        rs = by_lang.get(lang, [])
        if rs:
            acc = sum(r["correct"] for r in rs) / len(rs)
            print(f"{lang:<6}{len(rs):>6}{acc:>9.3f}")

    if any(r.get("answer_tag") is not None for r in rows):
        print("\nOutput followed the format (found an <answer> tag)")
        for lang in langs:
            rs = by_lang.get(lang, [])
            if rs:
                print(f"{lang:<6}{sum(bool(r.get('answer_tag')) for r in rs) / len(rs):>9.3f}")

    # same-question consistency vs English, matched by the shared match_id
    # (NOT by parsing the id string — en/hi/ja/zh/te don't share row counts,
    # so match_id is the only reliable join key; see unify_schema.py)
    en = {r["match_id"]: r["correct"] for r in by_lang.get("en", []) if r.get("match_id")}
    if en:
        print("\nSame-question comparison against English")
        print(f"{'lang':<6}{'pairs':>7}{'en right, lang wrong':>22}{'en wrong, lang right':>22}")
        for lang in langs:
            if lang == "en":
                continue
            pairs = [(en[r["match_id"]], r["correct"]) for r in by_lang.get(lang, [])
                     if r.get("match_id") and r["match_id"] in en]
            if pairs:
                a = sum(1 for e, l in pairs if e and not l)
                b = sum(1 for e, l in pairs if not e and l)
                print(f"{lang:<6}{len(pairs):>7}{a:>22}{b:>22}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--langs", nargs="+", default=["en", "zh", "ja", "hi", "te"])
    p.add_argument("--source", default="xmmmu")
    p.add_argument("--splits-dir", default=SPLITS_DIR,
                   help="e.g. data/splits_full for the 900-question set (results get a _splits_full suffix)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--load-4bit", action="store_true", help="llava only: load in 4-bit (needs bitsandbytes) for 16GB GPUs")
    p.add_argument("--structured", action="store_true",
                   help="ask for the 4-stage structured reasoning and score the <answer> tag")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="default 16 (direct answer) or 640 (--structured)")
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter saved by train_sft.py")
    p.add_argument("--run-name", default="", help="suffix for the results file, e.g. structured or sft")
    evaluate(p.parse_args())
