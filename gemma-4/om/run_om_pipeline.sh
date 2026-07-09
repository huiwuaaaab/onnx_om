#!/usr/bin/env bash
# =============================================================================
# Gemma-4 OM pipeline — serial entry (inline msame + assistant speculative decode)
#
# Usage:
#   RUN_MSAME=1 bash run_om_pipeline.sh
#   RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
#
# Multi-image main-chain pipeline: use run_om_pipeline_pipe.sh (MODE=main_decode)
# =============================================================================

set -euo pipefail

OM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${OM_DIR}/pipeline"
export OM_DIR PIPELINE_DIR

# shellcheck source=pipeline/serial.sh
source "${PIPELINE_DIR}/serial.sh"
serial_dispatch "$@"
