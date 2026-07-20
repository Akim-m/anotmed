#!/usr/bin/env bash
# Preflight for the two-model GPU setup (run first on the WSL2/GPU box).
# Checks, in order: NVIDIA GPU visible, VRAM headroom, vLLM reachable, MedSAM-2
# checkpoint present. Non-fatal warnings for the model-serving bits so you can
# run it before either model is up. Exits non-zero only if the GPU is invisible.
set -uo pipefail

VLLM_URL="${ANOTMED_VLLM_URL:-http://127.0.0.1:8000/v1}"
SAM_CKPT="${ANOTMED_SAM_CHECKPOINT:-}"
fail=0

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  FAIL: nvidia-smi not found. In WSL2 you need the NVIDIA driver on Windows"
  echo "        + the CUDA/WSL passthrough. https://docs.nvidia.com/cuda/wsl-user-guide/"
  fail=1
else
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
    | sed 's/^/  /' || { echo "  FAIL: nvidia-smi ran but returned no GPU"; fail=1; }
fi

echo "== vLLM (MedGemma) =="
health="${VLLM_URL%/v1}/health"
if curl -fsS --max-time 3 "$health" >/dev/null 2>&1; then
  echo "  OK: reachable at $health"
else
  echo "  warn: not reachable at $health — start it with scripts/serve_vllm.sh"
fi

echo "== MedSAM-2 checkpoint =="
if [ -z "$SAM_CKPT" ]; then
  echo "  warn: ANOTMED_SAM_CHECKPOINT is unset"
elif [ -f "$SAM_CKPT" ]; then
  echo "  OK: $SAM_CKPT ($(du -h "$SAM_CKPT" | cut -f1))"
else
  echo "  warn: ANOTMED_SAM_CHECKPOINT set but file not found: $SAM_CKPT"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "GPU preflight FAILED — fix the GPU visibility above before running anotmed on GPU."
  exit 1
fi
echo "GPU visible. Warnings above (if any) are for services you can start next."
