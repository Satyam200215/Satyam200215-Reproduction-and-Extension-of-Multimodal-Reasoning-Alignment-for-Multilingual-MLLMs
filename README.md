# Dataset collection pipeline

Dataset preparation and baseline evaluation are separate stages. Prepare the data first,
then run the baseline evaluator against the resulting splits.

```
scripts/download_datasets.py    ->  data/raw/...
scripts/unify_schema.py         ->  data/processed/{lang}.jsonl, data/images/...
scripts/translate_questions.py  ->  data/processed/zh.jsonl, te.jsonl
scripts/make_splits.py          ->  data/splits/{lang}_{train,val,test}.jsonl
scripts/eval_baseline.py        ->  baseline predictions and summary results
```

**Run `translate_questions.py` AFTER `unify_schema.py`, not before** — it reuses
the English images that `unify_schema.py` extracts to `data/images/`, since
translating the question text doesn't change the image.

### Why translate_questions.py exists

Your report requires checking whether the model performs worse on the *same*
question in a non-English language than in English. That only works if the
non-English version is an actual translation of the same question — not a
different question that happens to be in that language.

- **en / hi / ja** are already matched this way — xMMMU ships the same 300
  questions translated into Hindi and Japanese, in the same row order as
  English. No extra step needed; `unify_schema.py` already handles these.
- **zh / te** are not matched by anything downloaded so far. MathVista's
  Chinese subset, for example, is a *different* set of questions, not a
  translation of the English MathVista set — useful as extra Chinese data,
  but not valid for a same-question comparison. `translate_questions.py`
  fixes this by machine-translating the same English xMMMU set into Chinese
  and Telugu, row-aligned so `xmmmu_zh_17` and `xmmmu_en_17` are the same
  question in two languages.

Run it after `unify_schema.py`:
```bash
pip install transformers torch sentencepiece
python scripts/translate_questions.py --langs zh te
```
Then re-run `python scripts/make_splits.py` so the new zh/te files get split too.

As with the Telugu step described below, spot-check a sample of the machine
translations before treating them as ground truth — the checklist in section 3
below applies here too.

## 1. Where to run this

These scripts only need a CPU and internet access (no GPU needed for dataset
collection). You have two easy options:

**Option A — Your own laptop / college lab machine**
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — Google Colab (recommended if your laptop has a slow/limited connection)**
1. Open a new Colab notebook, upload this `project/` folder (or clone it if you push it to a GitHub repo you and K. Srinu share).
2. In a cell:
   ```python
   !pip install -r requirements.txt
   ```
3. Run the three scripts as shown below using `!python scripts/...` instead of `python scripts/...`.
4. Save `data/` to your Google Drive at the end of the session (Colab storage is wiped when the runtime disconnects):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp -r data /content/drive/MyDrive/mllm-project-data
   ```

## 2. Running the pipeline

From inside the `project/` folder:

```bash
python scripts/download_datasets.py
python scripts/unify_schema.py
python scripts/translate_questions.py --langs zh te
python scripts/make_splits.py
python scripts/eval_baseline.py --model qwen --limit 20
```

Run them in that exact order — each one depends on the previous one's output.

First run will take a few minutes since `datasets` has to download and cache
MathVista and xMMMU from Hugging Face (a few hundred MB total).

## 3. Telugu (and Chinese matched set)

`scripts/translate_questions.py` (see above) now handles this automatically
using NLLB-200. It's a solid first pass, but for your final report you should
still spot-check quality rather than trust it blindly:

1. After running `translate_questions.py --langs zh te`, sample ~10% of
   `data/processed/te.jsonl` (and `zh.jsonl`) and review against the checklist
   from the report: numbers/units unchanged, answer choices still map 1:1 in
   meaning, no simplification of the question.
2. If NLLB's Telugu quality looks weak on review (general-purpose translation
   models are noticeably behind India-specific ones for Telugu), swap in
   **IndicTrans2** (AI4Bharat) for just the Telugu pass — it's the standard
   choice in academic work for English→Telugu and usually a clear step up:
   ```bash
   pip install IndicTransToolkit
   ```
   (See https://github.com/AI4Bharat/IndicTrans2 for the inference snippet —
   swap it in for the `load_translator`/`translate_batch` functions in
   `translate_questions.py`, everything else in the script stays the same.)
3. Re-run `python scripts/make_splits.py` after any changes — it picks up any
   `.jsonl` file in `data/processed/` automatically.


## 5. Baseline model evaluation (`eval_baseline.py`)

After preparing and splitting the multilingual dataset, run `scripts/eval_baseline.py`
to measure model performance **before SFT or GRPO training**. These results are the
baseline for the later comparison: **Baseline → SFT → SFT + GRPO**.

### What it does

- Reads examples from `data/splits/` (the default split is `test`).
- Supports the languages `en`, `zh`, `ja`, `hi`, and `te`.
- Records model predictions and summary metrics, including accuracy and the
  English-versus-other-language gap.
- Saves prediction JSONL files and summary results; check the script's output for
  the exact paths used by your current version.
- Supports a small `--limit` for smoke-testing before a full run.

### Before running

1. Confirm that `data/splits/` contains the language/split files you intend to evaluate.
2. In Colab, select a GPU runtime and verify it with:
   ```python
   !nvidia-smi
   ```
3. Install the project's requirements. The Qwen2.5-VL path notes that
   `transformers>=4.49` is required.
4. Begin with a small run such as `--limit 20`; model downloads and memory needs can
   be substantial.

### Example commands

Run from the project root. In Colab, prefix commands with `!`.

```bash
# Small Qwen smoke test
python scripts/eval_baseline.py --model qwen --limit 20

# Full Qwen baseline
python scripts/eval_baseline.py --model qwen

# LLaVA-OneVision default checkpoint
python scripts/eval_baseline.py --model llava --limit 20

# Optional larger LLaVA-OneVision checkpoint in 4-bit
python scripts/eval_baseline.py --model llava \
  --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf --load-4bit

# DeepSeek-VL2 (only if its separate environment and GPU memory requirements are met)
python scripts/eval_baseline.py --model deepseek --limit 20
```

The script also supports choosing the split, languages, source dataset, and model ID.
Run `python scripts/eval_baseline.py --help` to inspect the options in your copy.

### Recommended order

1. Finish downloading, schema unification, and translation.
2. Run `python scripts/make_splits.py` after all intended language JSONL files have
   been created or updated.
3. Run a smoke test, for example:
   `python scripts/eval_baseline.py --model qwen --limit 20`.
4. Inspect prediction files, output summaries, example counts, and whether images load
   correctly.
5. Save the baseline results before starting SFT.

### Important caveats

- Use the held-out test split for final reporting. If validation results are used to
  tune prompts or reward settings, keep the test split untouched.
- The default LLaVA checkpoint is a small 0.5B LLaVA-OneVision variant. Record the
  exact `--model-id` used so comparisons are reproducible.
- DeepSeek-VL2-Small has substantially higher environment and GPU-memory requirements
  than the Qwen smoke-test path; do not start with it on a limited Colab GPU.
- The question remains in its target language, but the instruction to answer with
  the option letter is in English. Record this prompt detail and keep prompts
  consistent across stages.
- This script evaluates the baseline; it does not train SFT or GRPO models.

## 6. Sanity checks worth running after step 3

```python
import json, glob
for path in sorted(glob.glob("data/splits/*.jsonl")):
    n = sum(1 for _ in open(path, encoding="utf-8"))
    print(path, n)
```
This just prints example counts per language/split — check nothing is empty
and the train/val/test proportions look right before starting MMB.
