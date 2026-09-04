#!/bin/bash
# Bootstrap a bare Ubuntu GPU pod (no pip, no sudo needed): uv venv + torch cu128 + transformers + hf CLI, then download Qwen3.5-27B.
set -e
cd ~/work
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv venv -q .venv --python 3.12
uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv/bin/python "transformers>=4.46" accelerate einops tqdm "huggingface_hub[hf_xet]" hf_transfer
.venv/bin/python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0))'
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download Qwen/Qwen3.5-27B --quiet
echo SETUP_AND_DOWNLOAD_DONE
