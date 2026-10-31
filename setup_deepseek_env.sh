#!/usr/bin/env bash
# One-time setup of a separate Python 3.11 environment for DeepSeek-VL2-Tiny (lab machine).
# DeepSeek-VL2 needs transformers==4.38.2, which does not install on Python 3.14, so it gets
# its own environment (ds-venv); the existing venv for Qwen/LLaVA is not touched.
#
# Run from the project folder (the one that contains data/, train_sft.py, eval_mmb_sft.py):
#     bash setup_deepseek_env.sh
set -e
cd "$(dirname "$0")"

echo "== 1/5 uv (installs Python 3.11 without admin rights)"
if [ -x venv/bin/pip ]; then venv/bin/pip install -q uv; UV=venv/bin/uv; else python3 -m pip install -q --user uv; UV=~/.local/bin/uv; fi
$UV python install 3.11
[ -d ds-venv ] || $UV venv ds-venv --python 3.11
source ds-venv/bin/activate

echo "== 2/5 PyTorch (CUDA 12.4 build; runs on the lab's newer driver)"
$UV pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "== 3/5 DeepSeek-era libraries"
$UV pip install "transformers==4.38.2" "timm>=0.9.16" attrdict3 einops sentencepiece accelerate "peft==0.10.0" pillow

echo "== 4/5 DeepSeek-VL2 code"
[ -d ~/DeepSeek-VL2 ] || git clone -q --depth 1 https://github.com/deepseek-ai/DeepSeek-VL2 ~/DeepSeek-VL2
$UV pip install --no-deps -e ~/DeepSeek-VL2

echo "== 5/5 xformers stand-in (same maths via PyTorch attention)"
mkdir -p ~/xshim/xformers
touch ~/xshim/xformers/__init__.py
cat > ~/xshim/xformers/ops.py <<'EOF'
import torch.nn.functional as F


def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None, **kwargs):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale)
    return out.transpose(1, 2)
EOF

PYTHONPATH=~/xshim python -c "import torch, transformers, peft, deepseek_vl2.models, xformers.ops; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| peft', peft.__version__, '| DeepSeek environment OK')"
