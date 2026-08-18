#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting Open Model Serving Container"
echo "=========================================="

exec python3 -m vllm.entrypoints.openai.api_server "$@"
