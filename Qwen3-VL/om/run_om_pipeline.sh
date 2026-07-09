#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL OM pipeline — serial entry (inline msame)
#
# Usage:
#   RUN_MSAME=1 bash run_om_pipeline.sh
#   RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
#   GEN_STEPS=50 RUN_MSAME=1 bash run_om_pipeline.sh batch 50
#
# Multi-image OM stage pipeline: use run_om_pipeline_pipe.sh
# =============================================================================

set -euo pipefail

OM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${OM_DIR}/pipeline"
export OM_DIR PIPELINE_DIR

# shellcheck source=pipeline/serial.sh
source "${PIPELINE_DIR}/serial.sh"
serial_dispatch "$@"
