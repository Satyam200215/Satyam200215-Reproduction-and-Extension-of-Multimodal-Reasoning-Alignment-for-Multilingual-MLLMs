# Reproduction and Extension of Multimodal Reasoning Alignment for Multilingual MLLMs

Course project reproducing **R2-MultiOmnia** (Ranaldi, Ranaldi and Pucci, ACL 2025) on small multilingual vision-language models, and extending it with an extra model and two Indian languages.

**Team:** K. Srinu (M26DS007), Satyam Das (M26DS012)
**Emails:** m26ds007@iitbhilai.ac.in, m26ds012@iitbhilai.ac.in
**Criterion followed:** Criterion 2, reproduce paper results because the original code is not available.

---

## 1. Overview

Multimodal LLMs often reason worse when the same visual question is asked in a language other than English. R2-MultiOmnia addresses this with two stages:

1. **Multilingual Modality Bridging (MMB):** build structured reasoning demonstrations (visual abstraction, formalisation, reasoning, answer), translate them into every language, filter them, and use them for supervised fine-tuning (SFT).
2. **Language-Agnostic Reasoning Alignment (LAA):** refine the SFT model with reinforcement learning using **GRPO**, with a composite reward, KL annealing and curriculum learning on reasoning length.

### What we added to the original work

| Extension | Detail |
|---|---|
| Additional MLLM | LLaVA-OneVision, alongside Qwen2.5-VL-3B-Instruct and DeepSeek-VL2 |
| Additional languages | Hindi and Telugu, alongside English, Chinese and Japanese |
| Stage-wise comparison | Baseline, SFT, SFT+RL, SFT+GRPO for every model and language |

---

## 2. Pipeline

```
Dataset Preparation -> Baseline Evaluation (separate reference run)
        |
        v
   MMB (structured reasoning chains, translated, filtered)
        |
        v
   SFT warm-up
        |
        v
   LAA with GRPO (composite reward, KL annealing, curriculum)
        |
        v
   Multilingual evaluation and comparison
```

Structured output format used for MMB and rewards:

```
<visual_abstraction> ... </visual_abstraction>
<formalisation> ... </formalisation>
<reasoning> ... </reasoning>
<answer> ... </answer>
```

---

## 3. Models and Languages

**Models**

- Qwen2.5-VL-3B-Instruct
- LLaVA-OneVision (our added model)
- DeepSeek-VL2-Tiny (used instead of DeepSeek-VL2-Small, which did not fit the GPUs available to us)

**Languages:** English (En), Chinese (Zh), Japanese (Ja), Hindi (Hi), Telugu (Te)

---

## 4. Data

- Questions are based on **xMMMU** (from `neulab/PangeaBench-xmmmu`); MathVista and MathVision were also collected during dataset preparation.
- English questions (900 in the full set) were translated into Chinese, Japanese, Hindi and Telugu with **NLLB-200**, keeping numbers, expressions and answer options unchanged. Images are shared across languages.
- Reasoning chains from MMB were translated into each language (the paper's method, not an English-pivot variant) and filtered by rule-based checks.
- Splits are stored under `data/splits_full`. The **test set has 450 questions per language** and is never used for training or alignment.

---

## 5. Repository Structure

Only files and folders that exist in the project are listed here.

```
scripts/
  download_datasets.py     # download xMMMU, MathVista, MathVision
  unify_schema.py          # unify all datasets to one schema
  translate_questions.py   # NLLB-200 translation of questions
  validate_dataset.py      # sanity checks on the translated data
  make_splits.py           # train / validation / test splits
  eval_baseline.py         # baseline evaluation (Qwen and LLaVA adapters)
data/
  splits_full/             # final splits (test = 450 questions per language)
paper.tex                  # final report (IEEEtran)
fig_language.pdf           # per-language baseline vs SFT+GRPO figure
fig_gain_heatmap.pdf       # gain heatmap figure
```

`eval_baseline.py` is kept unchanged on purpose. It shows how baseline accuracy was measured without any tuning. Any modified evaluation code lives in separate files.

---

## 6. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch transformers datasets trl
# plus the extra libraries each MLLM needs, e.g. the DeepSeek-VL2 package
```

Main libraries: PyTorch, Hugging Face Transformers, Hugging Face Datasets, TRL.

---

## 7. How to Reproduce

1. **Prepare the data**
   ```bash
   python scripts/download_datasets.py
   python scripts/unify_schema.py
   python scripts/translate_questions.py
   python scripts/validate_dataset.py
   python scripts/make_splits.py
   ```
2. **Baseline evaluation** (no tuning)
   ```bash
   python scripts/eval_baseline.py
   ```
   Run it once per model adapter and compare per-language accuracy.
3. **MMB:** generate structured reasoning chains for the training questions, translate them into every language and filter them.
4. **SFT:** fine-tune each model for one epoch on the filtered chains, then evaluate on the test split.
5. **GRPO (LAA):** run GRPO from the SFT checkpoint using the composite reward, cosine KL annealing and the length curriculum, then evaluate on the test split.

The GRPO settings follow the original paper: learning rate 5e-7, KL coefficient 0.04, sampling temperature 0.9, curriculum with increasing length limits and decreasing group sizes. Our own runs used smaller scales because of GPU limits (see Section 9).

---

## 8. Results

Accuracy (%) on the held-out test set, per language and averaged over the five languages.

| Model | Stage | En | Zh | Ja | Hi | Te | Avg. |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL | Baseline | 38.9 | 32.0 | 32.0 | 33.8 | 28.0 | 32.9 |
| | SFT | 46.7 | 37.8 | 32.2 | 34.0 | 28.2 | 35.8 |
| | SFT+RL | 46.1 | 32.2 | 33.6 | 34.0 | 28.2 | 34.8 |
| | SFT+GRPO | 48.1 | 41.7 | 38.9 | 38.3 | 35.4 | **40.5** |
| LLaVA-OneVision | Baseline | 34.6 | 37.0 | 36.1 | 30.7 | 25.3 | 32.7 |
| | SFT | 37.3 | 38.0 | 36.1 | 32.5 | 25.3 | 33.8 |
| | SFT+RL | 44.4 | 40.3 | 40.3 | 34.6 | 26.1 | 37.1 |
| | SFT+GRPO | 49.6 | 46.1 | 44.0 | 37.3 | 31.2 | **41.6** |
| DeepSeek-VL2 | Baseline | 22.0 | 18.0 | 21.3 | 21.0 | 22.1 | 20.9 |
| | SFT | 30.0 | 26.8 | 26.8 | 23.1 | 25.8 | 26.5 |
| | SFT+RL | 32.0 | 33.1 | 33.9 | 24.1 | 27.3 | 30.1 |
| | SFT+GRPO | 38.4 | 39.7 | 40.3 | 25.2 | 30.7 | **34.9** |

**Key findings**

- SFT+GRPO is the best configuration for every model in every language. Average gains over baseline are +7.6 (Qwen2.5-VL), +8.9 (LLaVA-OneVision) and +14.0 (DeepSeek-VL2).
- Hindi gains the least for Qwen2.5-VL and DeepSeek-VL2, and Telugu the least for LLaVA-OneVision.
- The spread between the best and worst language grew after training for all three models, so the paper's claim of aligned performance across languages is only partly reproduced in our setting.
- For Qwen2.5-VL, standard SFT+RL scored slightly below SFT (34.8 against 35.8).

---

## 9. Differences from the Original Paper (Resource Constraints)

- **GPUs:** no A100-class GPU was available. Early work used free-tier Colab and Kaggle T4 GPUs (about 15 GB), where sessions were interrupted. DeepSeek-VL2-Small did not fit, so DeepSeek-VL2-Tiny was used, and LLaVA-OneVision ran in reduced precision with a cap on image resolution. Full SFT and GRPO were moved to a shared college laboratory GPU machine.
- **Training scale:** smaller dataset, shorter generation lengths, smaller GRPO groups and fewer optimisation steps than the original 4K/8K/16K curriculum.
- **Token and API budget:** no budget for paid API tokens, so MMB chains were generated with open models and GPT-4o or Gemini are not used as reference systems.
- **Translation:** machine translation with NLLB-200 instead of human translation, which adds noise, especially for Hindi and Telugu.

Absolute numbers are therefore not comparable with the original paper. Compare the relative gain from Baseline to SFT+GRPO within each model.

---

## 10. Limitations

- Translated questions and reasoning chains may contain errors that change meaning or difficulty.
- The test set is a few hundred questions per language, and each configuration was run once, so differences of a point or two may be within noise.
- KL annealing, curriculum learning and individual reward components were not ablated separately.
- Only final-answer accuracy was measured, not the quality of the generated reasoning.

---

## 11. Future Work

- Separate ablations of KL annealing, curriculum learning and each reward component.
- Higher-quality or human-checked translation for Hindi and Telugu.
- Larger test sets, more benchmarks and repeated runs with confidence intervals.
- Evaluation of reasoning quality, not only accuracy.
- Methods aimed at low-resource languages, such as language-specific demonstrations or larger models.

---

## 12. References

1. L. Ranaldi, F. Ranaldi, G. Pucci. *R2-MultiOmnia: Leading Multilingual Multimodal Reasoning via Self-Training.* ACL 2025, pp. 8220-8234.
2. Z. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024 (GRPO).
3. S. Bai et al. *Qwen2.5-VL Technical Report.* arXiv:2502.13923, 2025.
4. Z. Wu et al. *DeepSeek-VL2.* arXiv:2412.10302, 2024.
5. B. Li et al. *LLaVA-OneVision: Easy Visual Task Transfer.* arXiv:2408.03326, 2024.
6. X. Yue et al. *Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages.* ICLR 2025 (xMMMU).
7. NLLB Team. *No Language Left Behind: Scaling Human-Centered Machine Translation.* arXiv:2207.04672, 2022.

The full reference list is in `paper.tex`.

---

## 13. Acknowledgements

Prompts and tools used while working on the project are listed in the Appendix of `paper.tex`. The large language model used was Claude (Anthropic).
