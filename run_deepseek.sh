#!/usr/bin/env bash
# Runs the DeepSeek-VL2-Tiny stages on the lab machine (after setup_deepseek_env.sh).
# Settings match the LLaVA runs: translated MMB chains, 100 test questions per language,
# 1,024-token answers; max-len 2048 for SFT because the RTX 3060 has 12 GB.
#
#     bash run_deepseek.sh smoke       # ~10 min check that everything works (do this first)
#     bash run_deepseek.sh sft         # SFT training            -> outputs/deepseek_sft
#     bash run_deepseek.sh eval_base   # untrained baseline       -> results/deepseek_structured_sub_...
#     bash run_deepseek.sh eval_sft    # SFT evaluation           -> results/deepseek_sft_sub_...
#     bash run_deepseek.sh grpo        # GRPO training            -> outputs/deepseek_grpo
#     bash run_deepseek.sh eval_grpo   # GRPO evaluation          -> results/deepseek_grpo_sub_...
#     bash run_deepseek.sh all         # everything above after smoke, one after the other
#     bash run_deepseek.sh summary     # print all DeepSeek results
# Every stage also writes a log to logs/deepseek_<stage>.log
set -e
cd "$(dirname "$0")"
source ds-venv/bin/activate
export PYTHONPATH=~/xshim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
mkdir -p logs outputs results

DATA=data/mmb/qwen_sft_data_v2.jsonl
EVAL="--limit 100 --max-new-tokens 1024"

run() {  # run <log name> <deepseek_r2.py arguments...>
    local name=$1; shift
    echo "== $name: python deepseek_r2.py $*"
    python deepseek_r2.py "$@" 2>&1 | tee "logs/deepseek_$name.log"
}

stage() {
    case "$1" in
        smoke)     run smoke sft --data $DATA --limit 16 --max-len 2048 --out-dir outputs/deepseek_sft_smoke ;;
        sft)       run sft sft --data $DATA --max-len 2048 --out-dir outputs/deepseek_sft ;;
        eval_base) run eval_base eval --run-name structured_sub $EVAL ;;
        eval_sft)  run eval_sft eval --adapter outputs/deepseek_sft --run-name sft_sub $EVAL ;;
        grpo)      run grpo grpo --adapter outputs/deepseek_sft --out-dir outputs/deepseek_grpo ;;
        eval_grpo) run eval_grpo eval --adapter outputs/deepseek_grpo --run-name grpo_sub $EVAL ;;
        summary)
            python - <<'EOF'
import json, os
for run in ["structured_sub", "sft_sub", "grpo_sub"]:
    p = f"results/deepseek_{run}_splits_full_test_predictions.jsonl"
    if not os.path.exists(p):
        print(f"{run:15} (not run yet)")
        continue
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    out = []
    for lang in ["en", "zh", "ja", "hi", "te"]:
        r = [x for x in rs if x["language"] == lang]
        if r:
            acc = sum(x["correct"] for x in r) / len(r)
            fin = sum(bool(x.get("answer_tag")) for x in r) / len(r)
            out.append(f"{lang} {acc:.3f} ({fin:.2f})")
    print(f"{run:15} n={len(rs):4}  " + "  ".join(out))
EOF
            ;;
        *) echo "unknown stage '$1' (use: smoke sft eval_base eval_sft grpo eval_grpo all summary)"; exit 1 ;;
    esac
}

if [ "$1" = "all" ]; then
    for s in sft eval_base eval_sft grpo eval_grpo summary; do stage $s; done
else
    stage "${1:?give a stage, e.g. bash run_deepseek.sh smoke}"
fi
