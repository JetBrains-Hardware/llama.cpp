#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 LLAMA_SERVER TARGET_GGUF DRAFTER_GGUF [server options...]" >&2
    exit 2
fi

server=$1
target=$2
drafter=$3
shift 3

export GGML_CUDA_GLU_MMQ_FUSION=1
export GGML_CUDA_SSM_CONV_SPLIT=1
export GGML_CUDA_FA_PREFILL_GQA=2
export LLAMA_GDN_REPLAY=1
export GGML_CUDA_ASYNC_INPUTS=1
export LLAMA_DFLASH_ASYNC_INJECT=1
export LLAMA_DFLASH_TREE=1

exec "$server" \
    -m "$target" -md "$drafter" --alias Qwen3.8-27B \
    -ngl 99 -ngld 99 -np 1 -c 40960 -b 2048 -ub 2048 \
    --fit off --kv-unified -ctk f16 -ctv f16 -fa on \
    --spec-type draft-dflash2 --spec-draft-n-max 15 \
    --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
    --backend-sampling --temp 1 --top-p .95 --top-k 20 --min-p 0 \
    --presence-penalty 0 --frequency-penalty 0 --repeat-penalty 1 \
    --reasoning on --reasoning-effort low --seed 42 \
    --cache-prompt --cache-ram 0 --no-cache-idle-slots \
    --host 127.0.0.1 --port 8080 "$@"
