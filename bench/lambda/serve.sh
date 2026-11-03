#!/usr/bin/env bash
# Serve one open-weight model with vLLM on this instance's 127.0.0.1:8000. It is reached only through the SSH tunnel
# that bench/lambda/pilot.sh opens from the laptop, never on a public port, so it needs no API key.
# Usage (on the instance): bash serve.sh <hugging-face-model-id or local path> [name to serve it under].
# Ctrl-C, or `pkill -f "vllm serve"`, stops it.
set -euo pipefail
MODEL="$1"
NAME="${2:-$1}"
MAX_LEN=131072  # bench/prices.json caps output at 32768 tokens, and PURSE's longest input is about 80k tokens
VENV="$HOME/vllm"
if [ ! -x "$VENV/bin/vllm" ]; then
  python3 -m pip install --user --quiet uv
  "$HOME/.local/bin/uv" venv --quiet --python 3.12 "$VENV"
  "$HOME/.local/bin/uv" pip install --quiet --python "$VENV/bin/python" vllm
fi
"$VENV/bin/python" -c 'import vllm; print("vllm", vllm.__version__)'
export PATH="$VENV/bin:$PATH"  # kernels compiled at startup need the venv's ninja
# HASHIRU calls at temperature 0 (greedy), so FlashInfer's top-k/top-p sampler, compiled at startup, is never used
export VLLM_USE_FLASHINFER_SAMPLER=0
# Flags from each model card (read 2026-11-02). Tool calling stays off: calibration and the splitter need none, and
# vLLM 0.33's tool grammar fails every request on Ministral ("get_lark_from_jinja() missing ... force_tool_names").
case "$MODEL" in
  Qwen/*|*splitter*)  # thinks by default; the thinking goes to reasoning_content and the answer stays in content
    EXTRA=(--reasoning-parser qwen3 --language-model-only) ;;
  mistralai/*)
    EXTRA=(--tokenizer_mode mistral --config_format mistral --load_format mistral) ;;
  google/gemma-4-31B-it)  # answers directly (it thinks only when the system prompt asks). Its 62 GB of bf16 weights
    # leave room for only ~12k tokens of context on one 80 GB H100, and PURSE's longest input is ~84k: weights in FP8
    EXTRA=(--quantization fp8) ;;
  *)
    EXTRA=() ;;
esac
exec "$VENV/bin/vllm" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port 8000 --max-model-len "$MAX_LEN" \
  --tensor-parallel-size "$(nvidia-smi -L | wc -l)" --gpu-memory-utilization 0.92 "${EXTRA[@]}"
