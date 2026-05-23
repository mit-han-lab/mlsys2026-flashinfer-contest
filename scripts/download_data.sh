#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${FIB_DATASET_PATH:-$ROOT_DIR/data/flashinfer-trace}"

mkdir -p "$DATA_DIR"
hf download flashinfer-ai/mlsys26-contest --repo-type=dataset --local-dir "$DATA_DIR"

echo "Dataset written to: $DATA_DIR"
