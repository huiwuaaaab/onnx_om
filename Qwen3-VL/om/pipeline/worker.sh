#!/usr/bin/env bash
# OM worker — FIFO job queue for pipe scheduler (pipeline/worker.sh)

set -euo pipefail

WORKER_NAME="${1:?worker name}"
OM_PATH="${2:?om path}"
QUEUE_DIR="${3:?queue dir}"

MSAME_BIN="${MSAME_BIN:-./msame}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"
OM_SCRIPT_DIR="${OM_SCRIPT_DIR:-${OM_DIR:-$(cd "$(dirname "${MSAME_BIN}")" 2>/dev/null && pwd || pwd)}}"
JOBS_DIR="${QUEUE_DIR}/jobs"
RUN_SH="${OM_SCRIPT_DIR}/run_pipe_${WORKER_NAME}.sh"

log() { echo "[$(date '+%H:%M:%S')][pipe:${WORKER_NAME}] $*"; }

msame_input_arg() {
  local dir="$1" n="$2"
  if [[ "${n}" -le 1 ]]; then echo "${dir}/0.bin"; return; fi
  local i p=""
  for ((i = 0; i < n; i++)); do
    [[ -n "${p}" ]] && p+=","
    p+="${dir}/${i}.bin"
  done
  echo "${p}"
}

check_om_output_bins() {
  local output_dir="$1" tag="$2"
  find "${output_dir}" -name '*.bin' -print -quit | grep -q . || {
    log "ERROR: OM ${tag} produced no .bin under ${output_dir}"
    return 1
  }
}

[[ -f "${OM_PATH}" ]] || { log "ERROR: OM not found: ${OM_PATH}"; exit 1; }

mkdir -p "${JOBS_DIR}"
touch "${QUEUE_DIR}/ready"
log "ready  om=${OM_PATH}  jobs=${JOBS_DIR}"

while [[ ! -f "${QUEUE_DIR}/exit" ]]; do
  shopt -s nullglob
  pending_jobs=("${JOBS_DIR}"/*.pending)
  shopt -u nullglob

  if [[ "${#pending_jobs[@]}" -eq 0 ]]; then
    sleep 0.005
    continue
  fi

  job_pending="$(printf '%s\n' "${pending_jobs[@]}" | LC_ALL=C sort | head -n1)"
  job_id="$(basename "${job_pending}" .pending)"
  job_env="${JOBS_DIR}/${job_id}.env"

  rm -f "${job_pending}"

  # shellcheck disable=SC1090
  source "${job_env}"
  : "${TAG:?}" "${INPUT_DIR:?}" "${OUTPUT_DIR:?}" "${NUM_INPUTS:=1}" "${RUN_MSAME:=0}"

  mkdir -p "${OUTPUT_DIR}"
  local_input_arg="$(msame_input_arg "${INPUT_DIR}" "${NUM_INPUTS}")"

  log "job ${job_id}  tag=${TAG}"
  log "  input : ${local_input_arg}"
  log "  output: ${OUTPUT_DIR}"

  msame_line="pmupload ${MSAME_BIN} --model ${OM_PATH} --input \"${local_input_arg}\" --output ${OUTPUT_DIR} --outfmt BIN --loop 1"
  echo "${msame_line}" > "${RUN_SH}"
  chmod 777 "${RUN_SH}"

  if [[ "${RUN_MSAME}" == "1" ]]; then
    mkdir -p "${OUTPUT_DIR}/msprof"
    if ! (cd "${OM_SCRIPT_DIR}" && "${MSPROF_BIN}" --application="./run_pipe_${WORKER_NAME}.sh" --output="${OUTPUT_DIR}/msprof"); then
      log "ERROR: msame failed ${job_id}"
      log "  cwd   : ${OM_SCRIPT_DIR}"
      log "  run.sh: ${RUN_SH}"
      touch "${JOBS_DIR}/${job_id}.failed"
      continue
    fi
    if ! check_om_output_bins "${OUTPUT_DIR}" "${TAG}"; then
      log "ERROR: no output bins ${job_id}"
      touch "${JOBS_DIR}/${job_id}.failed"
      continue
    fi
  else
    log "  [dry-run] ${msame_line}"
    python3 - "${WORKER_NAME}" "${OUTPUT_DIR}" "${EXPORT_PROFILE:-448_512}" <<'PY'
import os, sys, pathlib
worker, out_s, profile = sys.argv[1], sys.argv[2], sys.argv[3]
out = pathlib.Path(out_s)
if profile == "256_256":
    MAX_SEQ_LEN, NUM_IMAGE_TOKENS = 256, 64
else:
    MAX_SEQ_LEN, NUM_IMAGE_TOKENS = 512, 196
H = 2048
IMAGE_EMBEDS = NUM_IMAGE_TOKENS * H * 2
BLOCK_HIDDEN = MAX_SEQ_LEN * H * 2
DEEPSTACK = NUM_IMAGE_TOKENS * H * 2
MASK = MAX_SEQ_LEN * MAX_SEQ_LEN * 2
LOGITS = 151936 * 2

def w(name, n):
    (out / name).write_bytes(b"\x00" * n)

out.mkdir(parents=True, exist_ok=True)
if worker == "vision":
    w("merged_hidden_states.bin", IMAGE_EMBEDS)
    w("deepstack_feat_5.bin", DEEPSTACK)
    w("deepstack_feat_11.bin", DEEPSTACK)
    w("deepstack_feat_17.bin", DEEPSTACK)
elif worker == "preblock":
    w("inputs_embeds_out.bin", BLOCK_HIDDEN)
    w("attention_mask_out.bin", MASK)
elif worker in ("block1", "block2", "block3"):
    w("hidden_states_out.bin", BLOCK_HIDDEN)
elif worker == "lm_head":
    w("logits.bin", LOGITS)
else:
    w("0.bin", 16)
PY
  fi

  touch "${JOBS_DIR}/${job_id}.done"
done

log "exit"
