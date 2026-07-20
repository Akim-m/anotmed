#!/usr/bin/env bash
# Launch the vLLM OpenAI server for MedGemma (anotmed's VLM path).
#
# anotmed's app is a thin HTTP client (backends/vllm_medgemma.py); this script
# runs the model. Verified working natively on an RTX 4060 (8 GiB) in WSL2.
#
# Memory (PLAN.md §2.1, and it's tight — 8 GiB):
#   * bf16 MedGemma-4b (~8.5 GiB) does NOT fit; FP8 (~4.5 GiB) does -> fp8 default.
#   * GPU_MEM_UTIL is auto-derived from *free* VRAM (nvidia-smi) minus headroom,
#     so a straggler process or a co-resident MedSAM-2 doesn't OOM startup.
#     Override GPU_MEM_UTIL to pin it (use ~0.60 when MedSAM-2 must coexist).
#
# Gated model + cached weights: google/medgemma-4b-it is gated. If the weights
# are already cached (they are on Akim's box), we run HF_HUB_OFFLINE=1 so vLLM
# never hits the 401, and pass the cached chat_template.jinja explicitly (the
# repo ships .jinja, but transformers looks for chat_template.json first).
#
# Env knobs: MODEL, PORT, QUANTIZATION(fp8|""=bf16), GPU_MEM_UTIL, MAX_MODEL_LEN,
#            MODE(native|docker), HF_TOKEN (only needed if weights aren't cached).
set -euo pipefail

MODEL="${MODEL:-google/medgemma-4b-it}"
PORT="${PORT:-8000}"
QUANTIZATION="${QUANTIZATION-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MODE="${MODE:-native}"

# --- auto gpu-memory-utilization from free VRAM (leave ~0.6 GiB headroom) -----
if [ -z "${GPU_MEM_UTIL:-}" ]; then
  read -r free_mib total_mib < <(nvidia-smi --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ') || true
  if [ -n "${total_mib:-}" ] && [ "${total_mib:-0}" -gt 0 ]; then
    GPU_MEM_UTIL=$(awk -v f="$free_mib" -v t="$total_mib" \
      'BEGIN{u=(f-600)/t; if(u>0.92)u=0.92; if(u<0.30)u=0.30; printf "%.2f", u}')
  else
    GPU_MEM_UTIL=0.85
  fi
fi

# --- gated-but-cached: run offline + supply the cached chat template ----------
# HF cache dir uses '--' between org and model: models--google--medgemma-4b-it
SNAP="$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--"$(echo "$MODEL" | sed 's#/#--#g')"/snapshots/*/ 2>/dev/null | head -1 || true)"
CHAT_ARG=()
OFFLINE=()
if [ -n "$SNAP" ]; then
  OFFLINE=(env HF_HUB_OFFLINE=1)
  if [ ! -e "${SNAP}chat_template.json" ] && [ -e "${SNAP}chat_template.jinja" ]; then
    CHAT_ARG=(--chat-template "${SNAP}chat_template.jinja")
  fi
fi

QUANT_ARG=()
[ -n "$QUANTIZATION" ] && QUANT_ARG=(--quantization "$QUANTIZATION")

# SLEEP_MODE=1 lets a warm server offload weights to RAM between phases (POST
# /sleep, /wake_up) instead of restarting — no 2-3 min reload, no co-residency.
# The sleep endpoints are admin endpoints: only mounted under VLLM_SERVER_DEV_MODE=1,
# and we bind localhost so they're never exposed.
# NOTE: sleep mode needs CUDA Virtual Memory Management, which is BROKEN on WSL2
# (OOMs during weight reload with a garbage 2^64-byte accounting value). Leave
# SLEEP_MODE=0 on WSL2; the reliable default there is ANOTMED_SEG_DEVICE=cpu.
# This works on native Linux with proper VMM support.
SLEEP_MODE="${SLEEP_MODE:-0}"
SLEEP_ARGS=()
DEV_ENV=()
if [ "$SLEEP_MODE" = "1" ]; then
  SLEEP_ARGS=(--enable-sleep-mode)
  DEV_ENV=(VLLM_SERVER_DEV_MODE=1)
fi

echo "Serving $MODEL on :$PORT  (mode=$MODE, quant=${QUANTIZATION:-bf16}, gpu-util=$GPU_MEM_UTIL, sleep=$SLEEP_MODE)"
[ -n "$SNAP" ] && echo "  cached weights: $SNAP (running HF_HUB_OFFLINE=1)"

COMMON_ARGS=(
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs 1
  --enforce-eager
  --limit-mm-per-prompt '{"image":1}'
  "${QUANT_ARG[@]}"
  "${CHAT_ARG[@]}"
  "${SLEEP_ARGS[@]}"
)

if [ "$MODE" = "docker" ]; then
  # bind host publish to localhost when the admin endpoints are enabled
  PUBLISH="${PORT}:8000"; [ "$SLEEP_MODE" = "1" ] && PUBLISH="127.0.0.1:${PORT}:8000"
  exec docker run --rm --gpus all -p "$PUBLISH" --ipc=host \
    -e "VLLM_USE_FLASHINFER_SAMPLER=0" ${SLEEP_MODE:+-e "VLLM_SERVER_DEV_MODE=$([ "$SLEEP_MODE" = 1 ] && echo 1 || echo 0)"} \
    ${HF_TOKEN:+-e "HF_TOKEN=$HF_TOKEN"} \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    vllm/vllm-openai:latest --model "$MODEL" --port 8000 "${COMMON_ARGS[@]}"
else
  exec "${OFFLINE[@]}" "${DEV_ENV[@]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve "$MODEL" --port "$PORT" --host 127.0.0.1 "${COMMON_ARGS[@]}"
fi
