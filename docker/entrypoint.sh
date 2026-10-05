#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

data_root="${CLIPPING_TOOL_DATA_ROOT:-/data/clipping-tool}"
export WORK_DIR="${WORK_DIR:-${data_root}/work}"
export OUTPUT_DIR="${OUTPUT_DIR:-${data_root}/output}"
export MODELS_DIR="${MODELS_DIR:-${data_root}/models}"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR" "$MODELS_DIR"

exec python -m app.main
