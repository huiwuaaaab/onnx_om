# Pipe OM pipeline — sourced by ../run_om_pipeline_pipe.sh
# Multi-image OM stage pipeline: vision/preblock/block* workers run in parallel.

: "${OM_DIR:?OM_DIR required}"
: "${PIPELINE_DIR:?PIPELINE_DIR required}"
SCRIPT_DIR="${OM_DIR}"
# shellcheck source=paths.sh
source "${PIPELINE_DIR}/paths.sh"
REPO_ROOT="${REPO_ROOT:-${OM_DIR}/..}"

OM_EXPORT_DIR="${OM_EXPORT_DIR:-${SCRIPT_DIR}/om_export}"
PY_HELPER="${PY_HELPER:-${SCRIPT_DIR}/om_bin_utils.py}"
PARSE_SCRIPT="${PARSE_SCRIPT:-${SCRIPT_DIR}/parse_state.py}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/Qwen3-VL-2B-Instruct}"

MODE="${MODE:-full}"
GEN_STEPS="${GEN_STEPS:-50}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
RUN_MSAME="${RUN_MSAME:-0}"
DO_PARSE="${DO_PARSE:-0}"
SKIP_EXIST="${SKIP_EXIST:-0}"
STOP_ON_EOS="${STOP_ON_EOS:-1}"
EXPORT_PROFILE="${QWEN3_EXPORT_PROFILE:-448_512}"

OM_WORKER_SH="${OM_WORKER_SH:-${PIPELINE_DIR}/worker.sh}"
if [[ "${OM_RESIDENT:-0}" == "1" ]]; then
  OM_WORKER_SH="${PIPELINE_DIR}/worker_resident.sh"
fi
MSAME_BIN="${MSAME_BIN:-${SCRIPT_DIR}/msame}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"

declare -a STAGE_NAMES=(vision preblock block1 block2 block3 lm_head)
declare -a STAGE_WORKERS=(vision preblock block1 block2 block3 lm_head)
declare -a STAGE_NUM_INS=(1 4 5 6 4 1)
declare -a OM_WORKER_NAMES=(vision preblock block1 block2 block3 lm_head)

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
declare -a IMG_ACTIVE=()       # 1=in pipeline
declare -a IMG_DECODE_STEP=()
declare -a IMG_CUR_LEN=()
declare -a IMG_NEXT_STAGE=()   # 0..5 OM stage index; -1=done
declare -a IMG_INFLIGHT=()     # 1=job submitted for current stage

PIPE_TOTAL_ACTIVE=0
PIPE_DONE_COUNT=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

configure_export_profile() {
  case "${EXPORT_PROFILE}" in
    256_256) MAX_SEQ_LEN=256; VISION_OM_PREFIX="vision_256" ;;
    448_512) MAX_SEQ_LEN=512; VISION_OM_PREFIX="vision_448" ;;
    *) echo "ERROR: unknown EXPORT_PROFILE=${EXPORT_PROFILE}" >&2; exit 1 ;;
  esac
}

configure_export_profile

usage() {
  cat <<EOF
Usage: bash run_om_pipeline_pipe.sh <batch_root> [gen_steps]

  --batch-root PATH     batch directory (required unless positional path given)
  --profile NAME        256_256 | 448_512
  path                  batch root (item/vision_bin/ per image)

Env: MODE=prefill_only|full|decode  RUN_MSAME=1  OM_RESIDENT=1  SHARED_PROMPT_BIN=om/prompt_bin

Pipeline: up to 6 images in-flight (one per OM stage worker).
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

run_py_for_item() {
  local idx="$1"; shift
  QWEN3_EXPORT_PROFILE="${EXPORT_PROFILE}" \
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
    1) echo "${OM_PREBLOCK}" ;;
    2) echo "${OM_B1}" ;;
    3) echo "${OM_B2}" ;;
    4) echo "${OM_B3}" ;;
    5) echo "${OM_LM_HEAD}" ;;
  esac
}

stage_tag_suffix() {
  local stage="$1"
  case "${stage}" in
    0) echo "${VISION_OM_PREFIX}" ;;
    1) echo "llm_preblock" ;;
    2) echo "llm_block1" ;;
    3) echo "llm_block2" ;;
    4) echo "llm_block3" ;;
    5) echo "lm_head" ;;
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
      bash "${OM_WORKER_SH}" "${name}" "${OM_WORKER_PATH[$name]}" "${qdir}" \
      >> "${qdir}/worker.log" 2>&1 &
    echo $! > "${qdir}/worker.pid"
  }

  if [[ "${OM_RESIDENT:-0}" == "1" ]]; then
    # Load OM one worker at a time — avoids concurrent acl.init() hangs on MDC.
    for name in "${OM_WORKER_NAMES[@]}"; do
      log "loading resident OM worker: ${name} ..."
      _start_one_worker "${name}"
      wait_for_worker_ready "${name}" || exit 1
      log "resident OM ready: ${name}"
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
      log "pipe workers up (6 cpp resident)  root=${PIPE_ROOT}"
    else
      log "pipe workers up (6 resident)  root=${PIPE_ROOT}"
    fi
  else
    log "pipe workers up (6 processes)  root=${PIPE_ROOT}"
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

setup_step_vision_path() {
  local idx="$1"
  local base tag
  base="$(img_step_base "${idx}")"
  tag="$(img_step_tag "${idx}")"
  mkdir -p "${base}"
  if [[ "${IMG_DECODE_STEP[$idx]}" -eq 0 ]]; then
  :
  else
    echo "${ITEM_WORK[$idx]}/step_0000/vision_out" > "${base}/vision_out.path"
  fi
}

pipe_prep_stage() {
  local idx="$1" stage="$2"
  local base tag vision_out_path vision_out pre_in pre_out bi b_in prev_out
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
      vision_out="$(cat "${base}/vision_out.path")"
      run_py_for_item "${idx}" prepare-preblock-input \
        --state-dir "${ITEM_STATE[$idx]}" \
        --static-preblock-dir "${ITEM_PROMPT[$idx]}" \
        --vision-out-dir "${vision_out}" \
        --out-dir "${base}/pre_in"
      ;;
    2|3|4)
      vision_out="$(cat "${base}/vision_out.path")"
      bi=$((stage - 1))
      prev_out="${base}/pre_out"
      [[ "${bi}" -gt 1 ]] && prev_out="${base}/b$((bi - 1))_out"
      local -a py_args=(
        prepare-block-input
        --pre-out-dir "${base}/pre_out"
        --static-preblock-dir "${ITEM_PROMPT[$idx]}"
        --vision-out-dir "${vision_out}"
        --out-dir "${base}/b${bi}_in"
        --block-idx "${bi}"
      )
      [[ "${bi}" -gt 1 ]] && py_args+=(--prev-block-out-dir "${prev_out}")
      run_py_for_item "${idx}" "${py_args[@]}"
      ;;
    5)
      run_py_for_item "${idx}" prepare-lm-head-input \
        --b3-out-dir "${base}/b3_out" \
        --out-dir "${base}/lm_head_in" \
        --cur-len "${IMG_CUR_LEN[$idx]}"
      ;;
  esac
}

pipe_submit_stage() {
  local idx="$1" stage="$2"
  local worker="${STAGE_WORKERS[$stage]}"
  local base tag stem job_id om_path tag_suffix input_dir output_dir num_ins
  base="$(img_step_base "${idx}")"
  tag="$(img_step_tag "${idx}")"
  stem="${ITEM_STEMS[$idx]}"
  job_id="i${idx}_${tag}_s${stage}"
  om_path="$(stage_om_path "${stage}")"
  tag_suffix="$(stage_tag_suffix "${stage}")"
  num_ins="${STAGE_NUM_INS[$stage]}"

  case "${stage}" in
    0) input_dir="${base}/vision_in"; output_dir="${base}/vision_out" ;;
    1) input_dir="${base}/pre_in"; output_dir="${base}/pre_out" ;;
    2|3|4)
      local bi=$((stage - 1))
      input_dir="${base}/b${bi}_in"
      output_dir="${base}/b${bi}_out"
      ;;
    5) input_dir="${base}/lm_head_in"; output_dir="${base}/lm_head_out" ;;
  esac

  local full_tag="${tag}/${tag_suffix}"
  log "submit [${stem} ${tag} ${STAGE_NAMES[$stage]}] -> worker:${worker}  job=${job_id}"
  submit_om_job_async "${worker}" "${job_id}" "${full_tag}" "${input_dir}" "${output_dir}" "${num_ins}"
  IMG_INFLIGHT["${idx}"]=1
}

pipe_cleanup_step_scratch() {
  local idx="$1"
  local base keep_vision
  base="$(img_step_base "${idx}")"
  keep_vision=0
  [[ "${IMG_DECODE_STEP[$idx]}" -eq 0 ]] && keep_vision=1
  _rm_if_not_keep \
    "${base}/vision_in" "${base}/pre_in" "${base}/pre_out" \
    "${base}/b1_in" "${base}/b2_in" "${base}/b3_in" \
    "${base}/b1_out" "${base}/b2_out" \
    "${base}/lm_head_in" "${base}/lm_head_out" \
    "${base}/lm_head_out.path" "${base}/b3_out.path"
  if [[ "${keep_vision}" != "1" ]]; then
    _rm_if_not_keep "${base}/vision_out" "${base}/vision_out.path"
  fi
}

write_item_final_output() {
  local idx="$1"
  local out="${ITEM_OUT[$idx]}" state="${ITEM_STATE[$idx]}" work="${ITEM_WORK[$idx]}"
  mkdir -p "${out}" "${state}"

  local lm_out="${work}/$(img_step_tag "${idx}")/lm_head_out"
  if [[ -d "${lm_out}" ]]; then
    python3 - <<PY || true
import json, os, shutil, sys
from pathlib import Path
os.environ["QWEN3_EXPORT_PROFILE"] = "${EXPORT_PROFILE}"
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
  echo "${base}/lm_head_out" > "${base}/lm_head_out.path"
  lm_out="${base}/lm_head_out"

  if [[ "${MODE}" == "prefill_only" ]]; then
    write_item_final_output "${idx}"
    IMG_ACTIVE["${idx}"]=0
    IMG_NEXT_STAGE["${idx}"]=-1
    IMG_INFLIGHT["${idx}"]=0
    PIPE_DONE_COUNT=$((PIPE_DONE_COUNT + 1))
    log "done [${stem}] prefill"
    return 0
  fi

  local decode_line hit_eos=0
  decode_line="$(run_py_for_item "${idx}" update-decode-state \
    --lm-head-out-dir "${lm_out}" \
    --state-dir "${ITEM_STATE[$idx]}" \
    --cur-len "${IMG_CUR_LEN[$idx]}" \
    --step "${IMG_DECODE_STEP[$idx]}")"
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
  setup_step_vision_path "${idx}"
  IMG_NEXT_STAGE["${idx}"]=1
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
      echo "${base}/vision_out" > "${base}/vision_out.path"
      IMG_NEXT_STAGE["${idx}"]=1
      ;;
    1) _rm_if_not_keep "${base}/pre_in"; IMG_NEXT_STAGE["${idx}"]=2 ;;
    2) _rm_if_not_keep "${base}/b1_in"; IMG_NEXT_STAGE["${idx}"]=3 ;;
    3) _rm_if_not_keep "${base}/b2_in"; IMG_NEXT_STAGE["${idx}"]=4 ;;
    4) _rm_if_not_keep "${base}/b3_in"; IMG_NEXT_STAGE["${idx}"]=5 ;;
    5)
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
    if [[ "${stage}" -eq 0 && "${IMG_DECODE_STEP[$idx]}" -gt 0 ]]; then
      continue
    fi
    echo "${idx}"
    return 0
  done
  return 1
}

pipe_try_fill() {
  local stage worker idx
  for stage in 0 1 2 3 4 5; do
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

  run_py_for_item "${idx}" sync-state \
    --dump-preblock-dir "${ITEM_PROMPT[$idx]}" \
    --state-dir "${state_dir}"
  run_py_for_item "${idx}" init-cur-len \
    --state-dir "${state_dir}" \
    --out-file "${cur_len_file}"

  IMG_DECODE_STEP["${idx}"]=0
  IMG_CUR_LEN["${idx}"]="$(cat "${cur_len_file}")"
  setup_step_vision_path "${idx}"
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
  local idx

  batch_root="$(cd "${batch_root}" && pwd)"
  PIPE_ROOT="${batch_root}/.om_pipe"
  collect_batch_items "${batch_root}"

  log "PIPE batch=${batch_root}  images=${#ITEM_DIRS[@]}  MODE=${MODE}  GEN_STEPS=${GEN_STEPS}"

  if [[ "${RUN_MSAME}" == "1" && ! -x "${MSAME_BIN}" ]]; then
    echo "ERROR: MSAME_BIN not found or not executable: ${MSAME_BIN}" >&2
    echo "  set MSAME_BIN=/path/to/msame" >&2
    exit 1
  fi

  OM_VISION="$(resolve_om "${OM_VISION:-}" "${VISION_OM_PREFIX}" "${VISION_OM_PREFIX}")" || exit 1
  OM_PREBLOCK="$(resolve_om "${OM_PREBLOCK:-}" "llm_preblock" "llm_preblock")" || exit 1
  OM_B1="$(resolve_om "${OM_B1:-}" "llm_block1" "llm_block1")" || exit 1
  OM_B2="$(resolve_om "${OM_B2:-}" "llm_block2" "llm_block2")" || exit 1
  OM_B3="$(resolve_om "${OM_B3:-}" "llm_block3" "llm_block3")" || exit 1
  OM_LM_HEAD="$(resolve_om "${OM_LM_HEAD:-}" "lm_head" "lm_head")" || exit 1

  OM_WORKER_PATH=(
    [vision]="${OM_VISION}"
    [preblock]="${OM_PREBLOCK}"
    [block1]="${OM_B1}"
    [block2]="${OM_B2}"
    [block3]="${OM_B3}"
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

dispatch() {
  local input="" cli_batch="" cli_profile=""
  local -a positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --profile)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires name" >&2; exit 1; }
        cli_profile="$2"; shift 2 ;;
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

  if [[ -n "${cli_profile}" ]]; then
    EXPORT_PROFILE="${cli_profile}"
    configure_export_profile
  fi

  [[ -n "${input}" ]] || { echo "ERROR: batch root required" >&2; usage >&2; exit 1; }
  [[ -d "${input}" ]] || { echo "ERROR: not a directory: ${input}" >&2; exit 1; }
  input="$(cd "${input}" && pwd)"

  if ! is_batch_root "${input}"; then
    echo "ERROR: ${input} is not a batch root (need item/vision_bin/pixel_values.bin)" >&2
    exit 1
  fi

  run_pipeline_batch "${input}"
}

pipe_dispatch() {
  dispatch "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: use ../run_om_pipeline_pipe.sh" >&2
  exit 1
fi
