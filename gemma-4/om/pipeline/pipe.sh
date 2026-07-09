# Pipe OM pipeline — sourced by ../run_om_pipeline_pipe.sh
# Gemma-4 multi-image main-chain pipeline: vision/mm_proj/preblock/block* workers in parallel.

: "${OM_DIR:?OM_DIR required}"
: "${PIPELINE_DIR:?PIPELINE_DIR required}"
SCRIPT_DIR="${OM_DIR}"
# shellcheck source=paths.sh
source "${PIPELINE_DIR}/paths.sh"
REPO_ROOT="${REPO_ROOT:-${OM_DIR}/..}"
GEMMA4_ROOT="${GEMMA4_ROOT:-${SCRIPT_DIR}}"

OM_EXPORT_DIR="${OM_EXPORT_DIR:-${SCRIPT_DIR}/om_export}"
PY_HELPER="${PY_HELPER:-${SCRIPT_DIR}/om_bin_utils.py}"
PARSE_SCRIPT="${PARSE_SCRIPT:-${SCRIPT_DIR}/parse_state.py}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/gemma-4-E2B-it}"

MODE="${MODE:-main_decode}"
WITH_ASSISTANT="${WITH_ASSISTANT:-0}"
GEN_STEPS="${GEN_STEPS:-50}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
RUN_MSAME="${RUN_MSAME:-0}"
DO_PARSE="${DO_PARSE:-0}"
SKIP_EXIST="${SKIP_EXIST:-0}"
STOP_ON_EOS="${STOP_ON_EOS:-1}"
MAX_SEQ_LEN=512
VISION_OM_PREFIX="vision_"

PLE_TABLE_DIR="${PLE_TABLE_DIR:-${SCRIPT_DIR}/ple_table}"
PLE_TABLE_BIN="${PLE_TABLE_BIN:-${PLE_TABLE_DIR}/embed_tokens_per_layer.bin}"
PAD_TOKEN_ID="${PAD_TOKEN_ID:-0}"

OM_WORKER_SH="${OM_WORKER_SH:-${PIPELINE_DIR}/worker.sh}"
MSAME_BIN="${MSAME_BIN:-${SCRIPT_DIR}/msame}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"

declare -a STAGE_NAMES=(
  vision mm_proj preblock
  block1 block2 block3 block4 block5 block6 block7
  lm_head
)
declare -a STAGE_WORKERS=(
  vision mm_proj preblock
  block1 block2 block3 block4 block5 block6 block7
  lm_head
)
declare -a STAGE_NUM_INS=(2 1 5 8 8 8 12 12 12 12 1)
declare -a OM_WORKER_NAMES=(
  vision mm_proj preblock
  block1 block2 block3 block4 block5 block6 block7
  lm_head
)

PIPE_ROOT=""
OM_WORKERS_STARTED=0
declare -A OM_WORKER_PATH=()
declare -A WORKER_BUSY=()
declare -A WORKER_JOB=()

declare -a ITEM_DIRS=()
declare -a ITEM_STEMS=()
declare -a ITEM_VISION=()
declare -a ITEM_PROMPT=()
declare -a ITEM_OUT=()
declare -a ITEM_WORK=()
declare -a ITEM_STATE=()
declare -a IMG_ACTIVE=()
declare -a IMG_DECODE_STEP=()
declare -a IMG_CUR_LEN=()
declare -a IMG_NEXT_STAGE=()
declare -a IMG_INFLIGHT=()

PIPE_TOTAL_ACTIVE=0
PIPE_DONE_COUNT=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  cat <<EOF
Usage: bash run_om_pipeline_pipe.sh <batch_root> [gen_steps]

  --batch-root PATH     batch directory (required unless positional path given)
  path                  batch root (item/vision_bin/ per image)

Env: MODE=main_decode (required)  WITH_ASSISTANT=0  RUN_MSAME=1
     SHARED_PROMPT_BIN=om/prompt_bin  PLE_TABLE_DIR=om/ple_table

Pipeline: up to 11 images in-flight (one per OM stage worker).
Use serial mode for speculative decode (WITH_ASSISTANT=1).
EOF
}

resolve_om() {
  local env_val="$1" glob_prefix="$2" label="$3"
  if [[ -n "${env_val}" && -f "${env_val}" ]]; then echo "${env_val}"; return 0; fi
  local -a matches=()
  local f
  shopt -s nullglob
  for f in "${OM_EXPORT_DIR}/${glob_prefix}"*.om; do matches+=("${f}"); done
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 1 ]]; then echo "${matches[0]}"; return 0; fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "ERROR: ambiguous ${label}: ${matches[*]}" >&2; return 1
  fi
  echo "ERROR: ${label} not found under ${OM_EXPORT_DIR}" >&2; return 1
}

resolve_ple_table_dir() {
  local d
  if [[ -n "${PLE_TABLE_DIR:-}" && -f "${PLE_TABLE_DIR}/embed_tokens_per_layer.bin" ]]; then
    echo "${PLE_TABLE_DIR}"
    return 0
  fi
  for d in "${SCRIPT_DIR}/ple_table"; do
    [[ -f "${d}/embed_tokens_per_layer.bin" ]] && { echo "${d}"; return 0; }
  done
  return 1
}

check_ple_table() {
  [[ -f "${PLE_TABLE_BIN}" ]] || {
    echo "ERROR: PLE table not found: ${PLE_TABLE_BIN}" >&2
    exit 1
  }
}

run_py_for_item() {
  local idx="$1"; shift
  GEMMA4_ROOT="${GEMMA4_ROOT}" \
    DUMP_ROOT="${ITEM_DIRS[$idx]}" \
    DUMP_PREBLOCK="${ITEM_PROMPT[$idx]}" \
    python3 "${PY_HELPER}" "$@"
}

check_om_output_bins() {
  local output_dir="$1" tag="$2"
  find "${output_dir}" -name '*.bin' -print -quit | grep -q . || {
    echo "ERROR: OM ${tag} produced no .bin under ${output_dir}" >&2
    return 1
  }
}

_rm_if_not_keep() {
  [[ "${KEEP_INTERMEDIATE}" == "1" ]] && return 0
  rm -rf "$@"
}

img_step_tag() {
  printf "step_%04d" "${IMG_DECODE_STEP[$1]}"
}

img_step_base() {
  echo "${ITEM_WORK[$1]}/$(img_step_tag "$1")"
}

stage_om_path() {
  local stage="$1"
  case "${stage}" in
    0) echo "${OM_VISION}" ;;
    1) echo "${OM_MM_PROJ}" ;;
    2) echo "${OM_PREBLOCK}" ;;
    3) echo "${OM_B1}" ;;
    4) echo "${OM_B2}" ;;
    5) echo "${OM_B3}" ;;
    6) echo "${OM_B4}" ;;
    7) echo "${OM_B5}" ;;
    8) echo "${OM_B6}" ;;
    9) echo "${OM_B7}" ;;
    10) echo "${OM_LM_HEAD}" ;;
  esac
}

stage_tag_suffix() {
  local stage="$1"
  case "${stage}" in
    0) echo "${VISION_OM_PREFIX}" ;;
    1) echo "mm_proj_" ;;
    2) echo "llm_preblock_" ;;
    3) echo "llm_block_0_5_" ;;
    4) echo "llm_block_5_10_" ;;
    5) echo "llm_block_10_15_" ;;
    6) echo "llm_block_15_20_" ;;
    7) echo "llm_block_20_25_" ;;
    8) echo "llm_block_25_30_" ;;
    9) echo "llm_block_30_35_" ;;
    10) echo "lm_head_" ;;
  esac
}

wait_for_worker_ready() {
  local name="$1" qdir="${PIPE_ROOT}/${1}"
  local waited=0
  while [[ ! -f "${qdir}/ready" ]]; do
    sleep 0.01
    waited=$((waited + 1))
    (( waited <= 60000 )) || { echo "ERROR: worker ${name} not ready" >&2; return 1; }
  done
}

verify_worker_resident() {
  local name="$1" qdir="${PIPE_ROOT}/${1}" logf="${qdir}/worker.log"
  [[ -f "${logf}" ]] || { echo "ERROR: missing ${logf}" >&2; return 1; }
  if grep -qE "resident=ctypes(-perjob)?" "${logf}"; then
    return 0
  fi
  echo "ERROR: worker ${name} is NOT ctypes resident (see ${logf})" >&2
  echo "  hint: tail -30 ${logf}" >&2
  grep -E "WARN:|ERROR:|resident=" "${logf}" | tail -5 >&2 || true
  return 1
}

start_om_workers() {
  [[ "${OM_WORKERS_STARTED}" == "1" ]] && return 0
  [[ -f "${OM_WORKER_SH}" ]] || { echo "ERROR: missing ${OM_WORKER_SH}" >&2; exit 1; }

  mkdir -p "${PIPE_ROOT}"
  local name qdir

  _start_one_worker() {
    name="$1"
    qdir="${PIPE_ROOT}/${name}"
    mkdir -p "${qdir}/jobs"
    rm -f "${qdir}/exit" "${qdir}/ready"
    find "${qdir}/jobs" -maxdepth 1 -type f \( -name '*.pending' -o -name '*.done' -o -name '*.failed' -o -name '*.env' \) -delete 2>/dev/null || true

    WORKER_BUSY["${name}"]=0
    WORKER_JOB["${name}"]=""

    MSAME_BIN="${MSAME_BIN}" MSPROF_BIN="${MSPROF_BIN}" OM_SCRIPT_DIR="${SCRIPT_DIR}" \
      OM_PER_JOB_LOAD="${OM_PER_JOB_LOAD:-1}" \
      bash "${OM_WORKER_SH}" "${name}" "${OM_WORKER_PATH[$name]}" "${qdir}" \
      >> "${qdir}/worker.log" 2>&1 &
    echo $! > "${qdir}/worker.pid"
  }

  if [[ "${OM_RESIDENT:-0}" == "1" ]]; then
    for name in "${OM_WORKER_NAMES[@]}"; do
      log "starting worker: ${name} (per-job OM load/unload) ..."
      _start_one_worker "${name}"
      wait_for_worker_ready "${name}" || exit 1
      verify_worker_resident "${name}" || exit 1
      log "worker ready: ${name}"
    done
  else
    for name in "${OM_WORKER_NAMES[@]}"; do
      _start_one_worker "${name}"
    done
    for name in "${OM_WORKER_NAMES[@]}"; do
      wait_for_worker_ready "${name}" || exit 1
    done
  fi

  OM_WORKERS_STARTED=1
  if [[ "${OM_RESIDENT:-0}" == "1" ]]; then
    if [[ "${OM_WORKER_SH##*/}" == "worker_cpp_resident.sh" ]]; then
      log "pipe workers up (11 cpp, per-job OM)  root=${PIPE_ROOT}"
    elif [[ "${OM_PER_JOB_LOAD:-1}" == "1" ]]; then
      log "pipe workers up (11 processes, per-job OM)  root=${PIPE_ROOT}"
    else
      log "pipe workers up (11 resident)  root=${PIPE_ROOT}"
    fi
  else
    log "pipe workers up (11 processes)  root=${PIPE_ROOT}"
  fi
}

stop_om_workers() {
  [[ "${OM_WORKERS_STARTED}" == "1" ]] || return 0
  local name qdir pid

  for name in "${OM_WORKER_NAMES[@]}"; do
    qdir="${PIPE_ROOT}/${name}"
    [[ -d "${qdir}" ]] || continue
    touch "${qdir}/exit"
    if [[ -f "${qdir}/worker.pid" ]]; then
      pid="$(cat "${qdir}/worker.pid")"
      wait "${pid}" 2>/dev/null || true
    fi
  done
  OM_WORKERS_STARTED=0
  log "pipe workers stopped"
}

submit_om_job_async() {
  local worker="$1" job_id="$2" tag="$3" input_dir="$4" output_dir="$5" num_inputs="${6:-1}"
  local jdir="${PIPE_ROOT}/${worker}/jobs"

  cat > "${jdir}/${job_id}.env" <<EOF
TAG=${tag}
INPUT_DIR=${input_dir}
OUTPUT_DIR=${output_dir}
NUM_INPUTS=${num_inputs}
RUN_MSAME=${RUN_MSAME}
EOF
  touch "${jdir}/${job_id}.pending"
  WORKER_BUSY["${worker}"]=1
  WORKER_JOB["${worker}"]="${job_id}"
}

pipe_poll_workers() {
  local worker job_id jdir donef failf
  for worker in "${OM_WORKER_NAMES[@]}"; do
    [[ "${WORKER_BUSY[$worker]:-0}" == "1" ]] || continue
    job_id="${WORKER_JOB[$worker]}"
    jdir="${PIPE_ROOT}/${worker}/jobs"
    donef="${jdir}/${job_id}.done"
    failf="${jdir}/${job_id}.failed"

    if [[ -f "${failf}" ]]; then
      echo "ERROR: worker ${worker} failed job ${job_id} (see ${PIPE_ROOT}/${worker}/worker.log)" >&2
      if [[ -f "${PIPE_ROOT}/${worker}/worker.log" ]]; then
        echo "--- tail ${PIPE_ROOT}/${worker}/worker.log ---" >&2
        tail -n 20 "${PIPE_ROOT}/${worker}/worker.log" >&2 || true
      fi
      exit 1
    fi
    [[ -f "${donef}" ]] || continue

    if [[ "${RUN_MSAME}" == "1" ]]; then
      # shellcheck disable=SC1090
      source "${jdir}/${job_id}.env"
      check_om_output_bins "${OUTPUT_DIR}" "${TAG}"
    fi

    pipe_on_job_complete "${worker}" "${job_id}"
    WORKER_BUSY["${worker}"]=0
    WORKER_JOB["${worker}"]=""
    rm -f "${donef}" "${jdir}/${job_id}.env"
  done
}

setup_step_mmproj_path() {
  local idx="$1"
  local base
  base="$(img_step_base "${idx}")"
  mkdir -p "${base}"
  if [[ "${IMG_DECODE_STEP[$idx]}" -gt 0 ]]; then
    echo "${ITEM_WORK[$idx]}/step_0000/mmproj_out" > "${base}/mmproj_out.path"
  fi
}

pipe_prep_stage() {
  local idx="$1" stage="$2"
  local base tag mmproj_out pre_out prev_out b3_out bi
  base="$(img_step_base "${idx}")"
  tag="$(img_step_tag "${idx}")"
  mkdir -p "${base}"

  case "${stage}" in
    0)
      run_py_for_item "${idx}" prepare-vision-input \
        --dump-vision-dir "${ITEM_VISION[$idx]}" \
        --out-dir "${base}/vision_in"
      ;;
    1)
      run_py_for_item "${idx}" prepare-mmproj-input \
        --vision-out-dir "${base}/vision_out" \
        --out-dir "${base}/mmproj_in"
      ;;
    2)
      mmproj_out="$(cat "${base}/mmproj_out.path")"
      run_py_for_item "${idx}" prepare-preblock-input \
        --state-dir "${ITEM_STATE[$idx]}" \
        --static-preblock-dir "${ITEM_PROMPT[$idx]}" \
        --mm-proj-out-dir "${mmproj_out}" \
        --out-dir "${base}/pre_in"
      ;;
    3|4|5|6|7|8|9)
      bi=$((stage - 2))
      prev_out="${base}/pre_out"
      [[ "${bi}" -gt 1 ]] && prev_out="${base}/b$((bi - 1))_out"
      b3_out="${base}/b3_out"
      local -a py_args=(
        prepare-block-input
        --pre-out-dir "${base}/pre_out"
        --out-dir "${base}/b${bi}_in"
        --block-idx "${bi}"
      )
      [[ "${bi}" -gt 1 ]] && py_args+=(--prev-block-out-dir "${prev_out}")
      [[ "${bi}" -ge 4 ]] && py_args+=(--b3-out-dir "${b3_out}")
      run_py_for_item "${idx}" "${py_args[@]}"
      ;;
    10)
      run_py_for_item "${idx}" prepare-lm-head-input \
        --b7-out-dir "${base}/b7_out" \
        --out-dir "${base}/lm_head_in" \
        --cur-len "${IMG_CUR_LEN[$idx]}"
      ;;
  esac
}

pipe_submit_stage() {
  local idx="$1" stage="$2"
  local worker="${STAGE_WORKERS[$stage]}"
  local base tag stem job_id om_path tag_suffix input_dir output_dir num_ins bi
  base="$(img_step_base "${idx}")"
  tag="$(img_step_tag "${idx}")"
  stem="${ITEM_STEMS[$idx]}"
  job_id="i${idx}_${tag}_s${stage}"
  om_path="$(stage_om_path "${stage}")"
  tag_suffix="$(stage_tag_suffix "${stage}")"
  num_ins="${STAGE_NUM_INS[$stage]}"

  case "${stage}" in
    0) input_dir="${base}/vision_in"; output_dir="${base}/vision_out" ;;
    1) input_dir="${base}/mmproj_in"; output_dir="${base}/mmproj_out" ;;
    2) input_dir="${base}/pre_in"; output_dir="${base}/pre_out" ;;
    3|4|5|6|7|8|9)
      bi=$((stage - 2))
      input_dir="${base}/b${bi}_in"
      output_dir="${base}/b${bi}_out"
      ;;
    10) input_dir="${base}/lm_head_in"; output_dir="${base}/lm_head_out" ;;
  esac

  local full_tag="${tag}/${tag_suffix}"
  log "submit [${stem} ${tag} ${STAGE_NAMES[$stage]}] -> worker:${worker}  job=${job_id}"
  submit_om_job_async "${worker}" "${job_id}" "${full_tag}" "${input_dir}" "${output_dir}" "${num_ins}"
  IMG_INFLIGHT["${idx}"]=1
}

pipe_cleanup_step_scratch() {
  local idx="$1"
  local base keep_mmproj
  base="$(img_step_base "${idx}")"
  keep_mmproj=0
  [[ "${IMG_DECODE_STEP[$idx]}" -eq 0 ]] && keep_mmproj=1
  _rm_if_not_keep \
    "${base}/vision_in" "${base}/pre_in" "${base}/pre_out" \
    "${base}/b1_in" "${base}/b2_in" "${base}/b3_in" \
    "${base}/b4_in" "${base}/b5_in" "${base}/b6_in" "${base}/b7_in" \
    "${base}/b1_out" "${base}/b2_out" \
    "${base}/b4_out" "${base}/b5_out" "${base}/b6_out" \
    "${base}/lm_head_in" "${base}/lm_head_out" \
    "${base}/lm_head_out.path" "${base}/b7_out.path" "${base}/b3_out.path"
  if [[ "${keep_mmproj}" != "1" ]]; then
    _rm_if_not_keep "${base}/vision_out" "${base}/mmproj_in" "${base}/mmproj_out" "${base}/mmproj_out.path"
  fi
}

write_item_final_output() {
  local idx="$1"
  local out="${ITEM_OUT[$idx]}" state="${ITEM_STATE[$idx]}" work="${ITEM_WORK[$idx]}"
  mkdir -p "${out}" "${state}"

  local lm_out="${work}/$(img_step_tag "${idx}")/lm_head_out"
  if [[ -d "${lm_out}" ]]; then
    python3 - <<PY || true
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "${SCRIPT_DIR}")
from om_bin_utils import LM_HEAD_LOGITS_BYTES, _find_bin_by_size
src = _find_bin_by_size(Path("${lm_out}"), LM_HEAD_LOGITS_BYTES, "logits")
dest = Path("${out}/final_logits.bin")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dest)
print(json.dumps({"src": str(src), "bytes": dest.stat().st_size}))
PY
  fi

  [[ -f "${state}/input_ids.bin" ]] && cp -f "${state}/input_ids.bin" "${out}/final_input_ids.bin"
  [[ -f "${state}/attention_mask.bin" ]] && cp -f "${state}/attention_mask.bin" "${out}/final_attention_mask.bin"
  echo "${IMG_CUR_LEN[$idx]}" > "${out}/final_cur_len.txt"

  python3 - <<PY
import json
from pathlib import Path
meta = {
    "mode": "${MODE}",
    "cur_len": int("${IMG_CUR_LEN[$idx]}"),
    "gen_steps": int("${GEN_STEPS}"),
    "stem": "${ITEM_STEMS[$idx]}",
    "output_root": str(Path("${out}").resolve()),
}
Path("${out}/final.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY

  if [[ "${KEEP_INTERMEDIATE}" != "1" ]]; then
    rm -rf "${work}"
  fi
}

pipe_finish_decode_step() {
  local idx="$1"
  local base tag lm_out stem
  base="$(img_step_base "${idx}")"
  tag="$(img_step_tag "${idx}")"
  stem="${ITEM_STEMS[$idx]}"

  echo "${base}/b3_out" > "${base}/b3_out.path"
  echo "${base}/b7_out" > "${base}/b7_out.path"
  echo "${base}/lm_head_out" > "${base}/lm_head_out.path"
  lm_out="${base}/lm_head_out"

  local decode_line hit_eos=0
  decode_line="$(run_py_for_item "${idx}" update-decode-state \
    --lm-head-out-dir "${lm_out}" \
    --state-dir "${ITEM_STATE[$idx]}" \
    --cur-len "${IMG_CUR_LEN[$idx]}" \
    --step "${IMG_DECODE_STEP[$idx]}" \
    --ple-table "${PLE_TABLE_BIN}" \
    --pad-token-id "${PAD_TOKEN_ID}")"
  log "[${stem}] ${decode_line}"
  [[ "${decode_line}" == *" eos=1"* ]] && hit_eos=1

  IMG_CUR_LEN["${idx}"]=$((IMG_CUR_LEN[$idx] + 1))

  if [[ "${hit_eos}" == "1" && "${STOP_ON_EOS}" == "1" ]]; then
    write_item_final_output "${idx}"
    pipe_cleanup_step_scratch "${idx}"
    _rm_if_not_keep "${ITEM_WORK[$idx]}/step_0000"
    IMG_ACTIVE["${idx}"]=0
    IMG_NEXT_STAGE["${idx}"]=-1
    IMG_INFLIGHT["${idx}"]=0
    PIPE_DONE_COUNT=$((PIPE_DONE_COUNT + 1))
    log "done [${stem}] EOS cur_len=${IMG_CUR_LEN[$idx]}"
    return 0
  fi

  if [[ "${IMG_CUR_LEN[$idx]}" -ge "${MAX_SEQ_LEN}" ]] || \
     [[ $((IMG_DECODE_STEP[$idx] + 1)) -ge "${GEN_STEPS}" ]]; then
    write_item_final_output "${idx}"
    pipe_cleanup_step_scratch "${idx}"
    _rm_if_not_keep "${ITEM_WORK[$idx]}/step_0000"
    IMG_ACTIVE["${idx}"]=0
    IMG_NEXT_STAGE["${idx}"]=-1
    IMG_INFLIGHT["${idx}"]=0
    PIPE_DONE_COUNT=$((PIPE_DONE_COUNT + 1))
    log "done [${stem}] decode cur_len=${IMG_CUR_LEN[$idx]}"
    return 0
  fi

  pipe_cleanup_step_scratch "${idx}"
  IMG_DECODE_STEP["${idx}"]=$((IMG_DECODE_STEP[$idx] + 1))
  setup_step_mmproj_path "${idx}"
  IMG_NEXT_STAGE["${idx}"]=2
  IMG_INFLIGHT["${idx}"]=0
  log "advance [${stem}] -> step $(img_step_tag "${idx}") cur_len=${IMG_CUR_LEN[$idx]}"
}

pipe_on_job_complete() {
  local worker="$1" job_id="$2"
  local idx stage base
  if [[ "${job_id}" =~ ^i([0-9]+)_ ]]; then
    idx="${BASH_REMATCH[1]}"
  else
    echo "ERROR: bad job_id ${job_id}" >&2
    exit 1
  fi
  base="$(img_step_base "${idx}")"

  stage="${IMG_NEXT_STAGE[$idx]}"
  IMG_INFLIGHT["${idx}"]=0

  case "${stage}" in
    0)
      _rm_if_not_keep "${base}/vision_in"
      IMG_NEXT_STAGE["${idx}"]=1
      ;;
    1)
      _rm_if_not_keep "${base}/mmproj_in"
      echo "${base}/mmproj_out" > "${base}/mmproj_out.path"
      IMG_NEXT_STAGE["${idx}"]=2
      ;;
    2) _rm_if_not_keep "${base}/pre_in"; IMG_NEXT_STAGE["${idx}"]=3 ;;
    3) _rm_if_not_keep "${base}/b1_in"; IMG_NEXT_STAGE["${idx}"]=4 ;;
    4) _rm_if_not_keep "${base}/b2_in"; IMG_NEXT_STAGE["${idx}"]=5 ;;
    5) _rm_if_not_keep "${base}/b3_in"; IMG_NEXT_STAGE["${idx}"]=6 ;;
    6) _rm_if_not_keep "${base}/b4_in"; IMG_NEXT_STAGE["${idx}"]=7 ;;
    7) _rm_if_not_keep "${base}/b5_in"; IMG_NEXT_STAGE["${idx}"]=8 ;;
    8) _rm_if_not_keep "${base}/b6_in"; IMG_NEXT_STAGE["${idx}"]=9 ;;
    9) _rm_if_not_keep "${base}/b7_in"; IMG_NEXT_STAGE["${idx}"]=10 ;;
    10)
      _rm_if_not_keep "${base}/lm_head_in"
      pipe_finish_decode_step "${idx}"
      ;;
  esac

  log "complete [${ITEM_STEMS[$idx]}] worker:${worker} stage:${STAGE_NAMES[$stage]:-done}"
}

pipe_find_image_for_stage() {
  local stage="$1" idx
  for idx in "${!ITEM_DIRS[@]}"; do
    [[ "${IMG_ACTIVE[$idx]:-0}" == "1" ]] || continue
    [[ "${IMG_INFLIGHT[$idx]:-0}" == "1" ]] && continue
    [[ "${IMG_NEXT_STAGE[$idx]}" == "${stage}" ]] || continue
    if [[ "${stage}" -le 1 && "${IMG_DECODE_STEP[$idx]}" -gt 0 ]]; then
      continue
    fi
    echo "${idx}"
    return 0
  done
  return 1
}

pipe_try_fill() {
  local stage worker idx
  for stage in 0 1 2 3 4 5 6 7 8 9 10; do
    worker="${STAGE_WORKERS[$stage]}"
    [[ "${WORKER_BUSY[$worker]:-0}" == "1" ]] && continue
    if ! idx="$(pipe_find_image_for_stage "${stage}")"; then
      continue
    fi
    pipe_prep_stage "${idx}" "${stage}"
    pipe_submit_stage "${idx}" "${stage}"
  done
}

pipe_init_image() {
  local idx="$1"
  local item_dir="${ITEM_DIRS[$idx]}"
  local stem="${ITEM_STEMS[$idx]}"
  local out_dir="${ITEM_OUT[$idx]}"
  local work_dir="${ITEM_WORK[$idx]}"
  local state_dir="${ITEM_STATE[$idx]}"
  local cur_len_file="${work_dir}/cur_len.txt"

  mkdir -p "${out_dir}" "${work_dir}" "${state_dir}"

  run_py_for_item "${idx}" sync-preblock-state \
    --dump-preblock-dir "${ITEM_PROMPT[$idx]}" \
    --state-dir "${state_dir}"
  run_py_for_item "${idx}" init-cur-len \
    --state-dir "${state_dir}" \
    --out-file "${cur_len_file}"

  IMG_DECODE_STEP["${idx}"]=0
  IMG_CUR_LEN["${idx}"]="$(cat "${cur_len_file}")"
  setup_step_mmproj_path "${idx}"
  IMG_NEXT_STAGE["${idx}"]=0
  IMG_INFLIGHT["${idx}"]=0
  IMG_ACTIVE["${idx}"]=1
  PIPE_TOTAL_ACTIVE=$((PIPE_TOTAL_ACTIVE + 1))
  log "init [${stem}] cur_len=${IMG_CUR_LEN[$idx]}"
}

collect_batch_items() {
  local batch_root="$1"
  local -a all_dirs=() d stem
  while IFS= read -r d; do
    [[ -n "${d}" ]] && all_dirs+=("${d}")
  done <<EOF
$(find "${batch_root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort)
EOF

  ITEM_DIRS=()
  ITEM_STEMS=()
  ITEM_VISION=()
  ITEM_PROMPT=()
  ITEM_OUT=()
  ITEM_WORK=()
  ITEM_STATE=()

  for d in "${all_dirs[@]}"; do
    stem="$(basename "${d}")"
    [[ "${stem}" == .* ]] && continue
    local vision_bin prompt_bin
    if ! vision_bin="$(resolve_vision_bin "${d}")"; then
      log "SKIP ${stem}: missing vision_bin"
      continue
    fi
    if ! prompt_bin="$(resolve_prompt_bin "${d}")"; then
      log "SKIP ${stem}: missing prompt_bin"
      continue
    fi
    if [[ "${SKIP_EXIST}" == "1" && -d "${d}/om_output/work/step_0000" ]]; then
      log "SKIP_EXIST ${stem}"
      continue
    fi
    ITEM_DIRS+=("${d}")
    ITEM_STEMS+=("${stem}")
    ITEM_VISION+=("${vision_bin}")
    ITEM_PROMPT+=("${prompt_bin}")
    ITEM_OUT+=("${d}/om_output")
    ITEM_WORK+=("${d}/om_output/work")
    ITEM_STATE+=("${d}/om_output/state")
  done

  ((${#ITEM_DIRS[@]} > 0)) || {
    echo "ERROR: no runnable items under ${batch_root}" >&2
    exit 1
  }
}

run_pipeline_batch() {
  local batch_root="$1"
  local summary="${batch_root}/summary_run_pipe.tsv"
  local idx stem

  batch_root="$(cd "${batch_root}" && pwd)"
  PIPE_ROOT="${batch_root}/.om_pipe"
  collect_batch_items "${batch_root}"

  PLE_TABLE_DIR="$(resolve_ple_table_dir)" || {
    echo "ERROR: PLE table not found under ${SCRIPT_DIR}/ple_table" >&2
    exit 1
  }
  PLE_TABLE_BIN="${PLE_TABLE_DIR}/embed_tokens_per_layer.bin"
  check_ple_table

  log "PIPE batch=${batch_root}  images=${#ITEM_DIRS[@]}  MODE=${MODE}  GEN_STEPS=${GEN_STEPS}"

  if [[ "${RUN_MSAME}" == "1" && ! -x "${MSAME_BIN}" ]]; then
    echo "ERROR: MSAME_BIN not found or not executable: ${MSAME_BIN}" >&2
    echo "  set MSAME_BIN=/path/to/msame" >&2
    exit 1
  fi

  OM_VISION="$(resolve_om "${OM_VISION:-}" "vision_" "vision")" || exit 1
  OM_MM_PROJ="$(resolve_om "${OM_MM_PROJ:-}" "mm_proj_" "mm_proj")" || exit 1
  OM_PREBLOCK="$(resolve_om "${OM_PREBLOCK:-}" "llm_preblock_" "llm_preblock")" || exit 1
  OM_B1="$(resolve_om "${OM_B1:-}" "llm_block_0_5_" "llm_block_1")" || exit 1
  OM_B2="$(resolve_om "${OM_B2:-}" "llm_block_5_10_" "llm_block_2")" || exit 1
  OM_B3="$(resolve_om "${OM_B3:-}" "llm_block_10_15_" "llm_block_3")" || exit 1
  OM_B4="$(resolve_om "${OM_B4:-}" "llm_block_15_20_" "llm_block_4")" || exit 1
  OM_B5="$(resolve_om "${OM_B5:-}" "llm_block_20_25_" "llm_block_5")" || exit 1
  OM_B6="$(resolve_om "${OM_B6:-}" "llm_block_25_30_" "llm_block_6")" || exit 1
  OM_B7="$(resolve_om "${OM_B7:-}" "llm_block_30_35_" "llm_block_7")" || exit 1
  OM_LM_HEAD="$(resolve_om "${OM_LM_HEAD:-}" "lm_head_" "lm_head")" || exit 1

  OM_WORKER_PATH=(
    [vision]="${OM_VISION}"
    [mm_proj]="${OM_MM_PROJ}"
    [preblock]="${OM_PREBLOCK}"
    [block1]="${OM_B1}"
    [block2]="${OM_B2}"
    [block3]="${OM_B3}"
    [block4]="${OM_B4}"
    [block5]="${OM_B5}"
    [block6]="${OM_B6}"
    [block7]="${OM_B7}"
    [lm_head]="${OM_LM_HEAD}"
  )

  PIPE_TOTAL_ACTIVE=0
  PIPE_DONE_COUNT=0

  trap 'stop_om_workers' EXIT
  start_om_workers

  for idx in "${!ITEM_DIRS[@]}"; do
    pipe_init_image "${idx}"
  done

  echo "stem	status	item_dir	output_dir" > "${summary}"

  while (( PIPE_DONE_COUNT < PIPE_TOTAL_ACTIVE )); do
    pipe_poll_workers
    pipe_try_fill
    sleep 0.005
  done

  for idx in "${!ITEM_DIRS[@]}"; do
    stem="${ITEM_STEMS[$idx]}"
    if [[ "${DO_PARSE}" == "1" ]]; then
      python3 "${PARSE_SCRIPT}" \
        --output-dir "${ITEM_OUT[$idx]}" \
        --dump-dir "${ITEM_PROMPT[$idx]}" \
        --model-dir "${MODEL_DIR}" \
        --response-out "${ITEM_DIRS[$idx]}/response.txt" \
        > "${ITEM_DIRS[$idx]}/parse.log" 2>&1 || true
    fi
    echo "${stem}	ok	${ITEM_DIRS[$idx]}	${ITEM_OUT[$idx]}" >> "${summary}"
  done

  log "Pipeline batch done. summary=${summary}"
}

pipe_dispatch() {
  if [[ "${MODE}" != "main_decode" ]]; then
    echo "ERROR: pipe mode only supports MODE=main_decode (got MODE=${MODE})" >&2
    echo "  Use serial mode for prefill_only/full/decode/speculative decode." >&2
    exit 1
  fi
  if [[ "${WITH_ASSISTANT}" == "1" ]]; then
    echo "ERROR: pipe mode does not support WITH_ASSISTANT=1" >&2
    echo "  Use serial mode for speculative decode." >&2
    exit 1
  fi

  local input="" cli_batch=""
  local -a positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --batch-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_batch="$(cd "$2" && pwd)"; shift 2 ;;
      --) shift; while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done; break ;;
      -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) positional+=("$1"); shift ;;
    esac
  done

  [[ "${#positional[@]}" -ge 2 ]] && GEN_STEPS="${positional[1]}"
  input="${positional[0]:-${cli_batch}}"

  [[ -n "${input}" ]] || { echo "ERROR: batch root required" >&2; usage >&2; exit 1; }
  [[ -d "${input}" ]] || { echo "ERROR: not a directory: ${input}" >&2; exit 1; }
  input="$(cd "${input}" && pwd)"

  if ! is_batch_root "${input}"; then
    echo "ERROR: ${input} is not a batch root (need item/vision_bin/pixel_values.bin)" >&2
    exit 1
  fi

  run_pipeline_batch "${input}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: use ../run_om_pipeline_pipe.sh" >&2
  exit 1
fi
