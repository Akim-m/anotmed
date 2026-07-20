#!/usr/bin/env bash
# Launch the vLLM OpenAI server for MedGemma (anotmed's VLM path).
#
# anotmed's app is a thin HTTP client (backends/vllm_medgemma.py); this script
# runs the model. The in-process MedSAM-2 segmenter shares the same GPU, so the
# memory cap here is deliberately conservative — see PLAN.md §2.1:
#
#   vLLM claims --gpu-memory-utilization 0.60, leaving the rest for SAM2. On an
#   8 GB card that is the whole budget; on 12-16 GB you can raise it and consider
#   bf16 (QUANTIZATION="") for quality, still behind the Phase 3 Dice gate.
#
# Env knobs (all optional):
#   MODEL          HF model id            (default google/medgemma-4b-it)
#   PORT           server port            (default 8000)
#   QUANTIZATION   fp8 | ""(=bf16)        (default fp8; on-the-fly, no offline step)
#   GPU_MEM_UTIL   fraction for vLLM      (default 0.60)
#   MAX_MODEL_LEN  context length         (default 2048)
#   HF_TOKEN       for gated model pulls  (optional)
set -euo pipefail

MODEL="${MODEL:-google/medgemma-4b-it}"
PORT="${PORT:-8000}"
QUANTIZATION="${QUANTIZATION-fp8}"   # note '-' not ':-': explicit "" means bf16
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

# bf16 path omits the flag entirely (vLLM uses the model's native dtype).
QUANT_ARG=()
if [ -n "$QUANTIZATION" ]; then
  QUANT_ARG=(--quantization "$QUANTIZATION")
fi

echo "Serving $MODEL on :$PORT  (quant=${QUANTIZATION:-bf16}, gpu-util=$GPU_MEM_UTIL)"

# Docker form (matches the vllm/vllm-openai image). For a native WSL2 install,
# swap the docker preamble for:  vllm serve "$MODEL" --port "$PORT" <same flags>
exec docker run --rm --gpus all -p "${PORT}:8000" --ipc=host \
  -e "VLLM_USE_FLASHINFER_SAMPLER=0" \
  ${HF_TOKEN:+-e "HF_TOKEN=$HF_TOKEN"} \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  "${QUANT_ARG[@]}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs 1 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1}'
