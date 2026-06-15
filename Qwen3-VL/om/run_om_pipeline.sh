#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL OM pipeline
#
# [MDC] vision_448 -> llm_preblock -> b1..b3 -> lm_head  (default profile 448_512)
# [本地] dump_om_inputs.py → scp dump/ → MDC → scp om_output/ → parse_state.py
#
# Static bins: dump/vision/, dump/llm_preblock/
# Output root: om_output/ (work/ + state/ + final_*)
#
# Usage:
#   RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir dump
#   RUN_MSAME=1 bash run_om_pipeline.sh --batch-root batch
#   MODE=full RUN_MSAME=1 bash run_om_pipeline.sh ./batch 50
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}/..}"

DUMP_ROOT="${DUMP_ROOT:-${SCRIPT_DIR}/dump}"
DUMP_VISION="${DUMP_VISION:-${DUMP_ROOT}/vision}"
DUMP_PREBLOCK="${DUMP_PREBLOCK:-${DUMP_ROOT}/llm_preblock}"

OM_DIR="${OM_DIR:-${SCRIPT_DIR}/om_export}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/om_output}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/work}"
STATE_DIR="${STATE_DIR:-${OUTPUT_ROOT}/state}"

PY_HELPER="${PY_HELPER:-${SCRIPT_DIR}/om_bin_utils.py}"
PARSE_SCRIPT="${PARSE_SCRIPT:-${SCRIPT_DIR}/parse_state.py}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/Qwen3-VL-2B-Instruct}"

MODE="${MODE:-full}"
GEN_STEPS="${GEN_STEPS:-50}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
RUN_MSAME="${RUN_MSAME:-0}"
DO_PARSE="${DO_PARSE:-0}"
SKIP_EXIST="${SKIP_EXIST:-0}"
EXPORT_PROFILE="${QWEN3_EXPORT_PROFILE:-448_512}"

configure_export_profile() {
  case "${EXPORT_PROFILE}" in
    256_256)
      MAX_SEQ_LEN=256
      VISION_OM_PREFIX="vision_256"
      ;;
    448_512)
      MAX_SEQ_LEN=512
      VISION_OM_PREFIX="vision_448"
      ;;
    *)
      echo "ERROR: unknown EXPORT_PROFILE=${EXPORT_PROFILE} (256_256|448_512)" >&2
      exit 1
      ;;
  esac
}

configure_export_profile

MSAME_BIN="${MSAME_BIN:-${SCRIPT_DIR}/msame}"
RUN_SH="${RUN_SH:-${SCRIPT_DIR}/run.sh}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"
LAST_LM_HEAD_OUT=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

refresh_dump_paths() {
  DUMP_VISION="${DUMP_ROOT}/vision"
  DUMP_PREBLOCK="${DUMP_ROOT}/llm_preblock"
}

is_batch_root() {
  local root="$1" d
  [[ -d "${root}" ]] || return 1
  for d in "${root}"/*; do
    [[ -d "${d}" && -f "${d}/dump/vision/pixel_values.bin" ]] && return 0
  done
  return 1
}

configure_item_paths() {
  DUMP_ROOT="${1}/dump"
  OUTPUT_ROOT="${1}/om_output"
  WORK_ROOT="${OUTPUT_ROOT}/work"
  STATE_DIR="${OUTPUT_ROOT}/state"
  refresh_dump_paths
}

configure_dump_root_paths() {
  DUMP_ROOT="$1"
  refresh_dump_paths
}

configure_output_layout() {
  OUTPUT_ROOT="$1"
  [[ "${2:-0}" != "1" ]] && WORK_ROOT="${OUTPUT_ROOT}/work"
  [[ "${3:-0}" != "1" ]] && STATE_DIR="${OUTPUT_ROOT}/state"
}

usage() {
  cat <<EOF
Usage: bash run_om_pipeline.sh [options] [path] [gen_steps]

  --dump-dir PATH       static input root (vision/, llm_preblock/)
  --output-dir PATH     unified output root (default: om_output/)
  --batch-root PATH     batch mode
  --profile NAME        256_256 | 448_512 (default: 448_512)
  path                  auto-detect batch / item / dump root

Env: QWEN3_EXPORT_PROFILE  MODE=prefill_only|full|decode  RUN_MSAME=1
EOF
}

resolve_om() {
  local env_val="$1" glob_prefix="$2" label="$3"
  if [[ -n "${env_val}" && -f "${env_val}" ]]; then
    echo "${env_val}"; return 0
  fi
  local -a matches=()
  local f
  shopt -s nullglob
  for f in "${OM_DIR}/${glob_prefix}"*.om; do matches+=("${f}"); done
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 1 ]]; then echo "${matches[0]}"; return 0; fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "ERROR: ambiguous ${label}: ${matches[*]}" >&2; return 1
  fi
  echo "ERROR: ${label} not found under ${OM_DIR}" >&2
  return 1
}

check_dump() {
  for f in "${DUMP_VISION}/pixel_values.bin" \
           "${DUMP_PREBLOCK}/input_ids.bin" \
           "${DUMP_PREBLOCK}/attention_mask.bin" \
           "${DUMP_PREBLOCK}/position_ids.bin"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2
      echo "  run: python dump_om_inputs.py --image path/img.jpg" >&2
      exit 1
    }
  done
}

check_om_output_bins() {
  local output_dir="$1" tag="$2"
  find "${output_dir}" -name '*.bin' -print -quit | grep -q . || {
    echo "ERROR: OM ${tag} produced no .bin under ${output_dir}" >&2
    return 1
  }
}

run_py() {
  QWEN3_EXPORT_PROFILE="${EXPORT_PROFILE}" \
    DUMP_ROOT="${DUMP_ROOT}" DUMP_PREBLOCK="${DUMP_PREBLOCK}" \
    python3 "${PY_HELPER}" "$@"
}

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

run_om_model() {
  local tag="$1" om_path="$2" input_dir="$3" output_dir="$4"
  local num_inputs="${5:-1}"
  mkdir -p "${output_dir}"
  local input_arg
  input_arg="$(msame_input_arg "${input_dir}" "${num_inputs}")"
  [[ -f "${om_path}" ]] || { echo "ERROR: OM not found: ${om_path}" >&2; exit 1; }

  log "=== OM ${tag} ==="
  log "  model : ${om_path}"
  log "  input : ${input_arg}"
  log "  output: ${output_dir}"

  local msame_line="pmupload ${MSAME_BIN} --model ${om_path} --input \"${input_arg}\" --output ${output_dir} --outfmt BIN --loop 1"
  echo "${msame_line}" > "${RUN_SH}"
  chmod 777 "${RUN_SH}"

  if [[ "${RUN_MSAME}" == "1" ]]; then
    mkdir -p "${output_dir}/msprof"
    if ! (cd "${SCRIPT_DIR}" && "${MSPROF_BIN}" --application=./run.sh --output="${output_dir}/msprof"); then
      echo "ERROR: msame failed for ${tag}" >&2
      exit 1
    fi
    check_om_output_bins "${output_dir}" "${tag}"
  else
    log "  [dry-run] ${msame_line}"
  fi
}

_rm_if_not_keep() {
  [[ "${KEEP_INTERMEDIATE}" == "1" ]] && return 0
  rm -rf "$@"
}

cleanup_step_scratch() {
  local tag="$1" keep_vision="${2:-0}"
  local base="${WORK_ROOT}/${tag}"
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

copy_lm_logits_bin() {
  local lm_out_dir="$1" dest="$2"
  [[ -d "${lm_out_dir}" ]] || return 0
  python3 - <<PY
import json, os, shutil, sys
from pathlib import Path
os.environ["QWEN3_EXPORT_PROFILE"] = "${EXPORT_PROFILE}"
sys.path.insert(0, "${SCRIPT_DIR}")
from om_bin_utils import LM_HEAD_LOGITS_BYTES, _find_bin_by_size
src = _find_bin_by_size(Path("${lm_out_dir}"), LM_HEAD_LOGITS_BYTES, "logits")
dest = Path("${dest}")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dest)
print(json.dumps({"src": str(src), "bytes": dest.stat().st_size}))
PY
}

update_final_logits() {
  [[ -n "${LAST_LM_HEAD_OUT}" ]] || return 0
  mkdir -p "${OUTPUT_ROOT}"
  if copy_lm_logits_bin "${LAST_LM_HEAD_OUT}" "${OUTPUT_ROOT}/final_logits.bin" 2>/dev/null; then
    log "updated final_logits.bin"
  fi
}

write_final_output() {
  local cur_len="$1"
  mkdir -p "${OUTPUT_ROOT}" "${STATE_DIR}"

  update_final_logits || true
  [[ -f "${STATE_DIR}/input_ids.bin" ]] && \
    cp -f "${STATE_DIR}/input_ids.bin" "${OUTPUT_ROOT}/final_input_ids.bin"
  [[ -f "${STATE_DIR}/attention_mask.bin" ]] && \
    cp -f "${STATE_DIR}/attention_mask.bin" "${OUTPUT_ROOT}/final_attention_mask.bin"
  echo "${cur_len}" > "${OUTPUT_ROOT}/final_cur_len.txt"

  python3 - <<PY
import json
from pathlib import Path
meta = {
    "mode": "${MODE}",
    "cur_len": int("${cur_len}"),
    "gen_steps": int("${GEN_STEPS}"),
    "final_logits": str(Path("${OUTPUT_ROOT}/final_logits.bin").resolve()),
    "output_root": str(Path("${OUTPUT_ROOT}").resolve()),
    "state_dir": str(Path("${STATE_DIR}").resolve()),
    "work_root": str(Path("${WORK_ROOT}").resolve()),
}
Path("${OUTPUT_ROOT}/final.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

  if [[ "${KEEP_INTERMEDIATE}" != "1" ]]; then
    rm -rf "${WORK_ROOT}"
    log "removed scratch ${WORK_ROOT}"
  fi
  log "final output: ${OUTPUT_ROOT}/final_logits.bin"
}

run_vision() {
  local prefix="$1"
  local vision_in="${WORK_ROOT}/${prefix}/vision_in"
  local vision_out="${WORK_ROOT}/${prefix}/vision_out"

  run_py prepare-vision-input --dump-vision-dir "${DUMP_VISION}" --out-dir "${vision_in}"
  run_om_model "${prefix}/${VISION_OM_PREFIX}" "${OM_VISION}" "${vision_in}" "${vision_out}" 1
  _rm_if_not_keep "${vision_in}"
  echo "${vision_out}" > "${WORK_ROOT}/${prefix}/vision_out.path"
}

run_llm_chain() {
  local prefix="$1" state_dir="$2" cur_len="$3"
  local vision_out
  vision_out="$(cat "${WORK_ROOT}/${prefix}/vision_out.path")"

  local pre_in="${WORK_ROOT}/${prefix}/pre_in"
  local pre_out="${WORK_ROOT}/${prefix}/pre_out"

  run_py prepare-preblock-input \
    --state-dir "${state_dir}" \
    --static-preblock-dir "${DUMP_PREBLOCK}" \
    --vision-out-dir "${vision_out}" \
    --out-dir "${pre_in}"
  run_om_model "${prefix}/llm_preblock" "${OM_PREBLOCK}" "${pre_in}" "${pre_out}" 4

  local prev_out="${pre_out}" bi b_in b_out
  local -a blocks=(1 2 3)
  local -a om_paths=("" "${OM_B1}" "${OM_B2}" "${OM_B3}")
  local -a num_ins=(0 5 6 4)

  for bi in "${blocks[@]}"; do
    b_in="${WORK_ROOT}/${prefix}/b${bi}_in"
    b_out="${WORK_ROOT}/${prefix}/b${bi}_out"
    local -a py_args=(
      prepare-block-input
      --pre-out-dir "${pre_out}"
      --static-preblock-dir "${DUMP_PREBLOCK}"
      --vision-out-dir "${vision_out}"
      --out-dir "${b_in}"
      --block-idx "${bi}"
    )
    [[ "${bi}" -gt 1 ]] && py_args+=(--prev-block-out-dir "${prev_out}")
    run_py "${py_args[@]}"
    run_om_model "${prefix}/llm_block${bi}" "${om_paths[$bi]}" "${b_in}" "${b_out}" "${num_ins[$bi]}"
    _rm_if_not_keep "${b_in}"
    prev_out="${b_out}"
  done

  local lm_in="${WORK_ROOT}/${prefix}/lm_head_in"
  local lm_out="${WORK_ROOT}/${prefix}/lm_head_out"
  run_py prepare-lm-head-input --b3-out-dir "${prev_out}" --out-dir "${lm_in}" --cur-len "${cur_len}"
  run_om_model "${prefix}/lm_head" "${OM_LM_HEAD}" "${lm_in}" "${lm_out}" 1
  _rm_if_not_keep "${lm_in}"

  echo "${prev_out}" > "${WORK_ROOT}/${prefix}/b3_out.path"
  echo "${lm_out}" > "${WORK_ROOT}/${prefix}/lm_head_out.path"
  LAST_LM_HEAD_OUT="${lm_out}"
  update_final_logits
}

sync_state_from_dump() {
  run_py sync-state --dump-preblock-dir "${DUMP_PREBLOCK}" --state-dir "$1"
}

run_prefill() {
  local tag="prefill" state_dir="${STATE_DIR}"
  sync_state_from_dump "${state_dir}"

  local cur_len_file="${WORK_ROOT}/cur_len.txt"
  run_py init-cur-len --state-dir "${state_dir}" --out-file "${cur_len_file}"
  local cur_len
  cur_len="$(cat "${cur_len_file}")"
  log "prefill cur_len=${cur_len}"

  run_vision "${tag}"
  run_llm_chain "${tag}" "${state_dir}" "${cur_len}"
  write_final_output "${cur_len}"
  log "prefill done: ${OUTPUT_ROOT}/"
}

run_decode_loop() {
  local state_dir="${STATE_DIR}"
  sync_state_from_dump "${state_dir}"

  local cur_len_file="${WORK_ROOT}/cur_len.txt"
  run_py init-cur-len --state-dir "${state_dir}" --out-file "${cur_len_file}"
  local cur_len step tag
  cur_len="$(cat "${cur_len_file}")"
  log "decode initial cur_len=${cur_len}"

  for ((step = 0; step < GEN_STEPS; step++)); do
    if [[ "${cur_len}" -ge "${MAX_SEQ_LEN}" ]]; then
      log "reach MAX_SEQ_LEN=${MAX_SEQ_LEN}, stop"
      break
    fi

    tag=$(printf "step_%04d" "${step}")

    if [[ "${step}" -eq 0 ]]; then
      run_vision "${tag}"
    else
      mkdir -p "${WORK_ROOT}/${tag}"
      echo "${WORK_ROOT}/step_0000/vision_out" > "${WORK_ROOT}/${tag}/vision_out.path"
    fi

    run_llm_chain "${tag}" "${state_dir}" "${cur_len}"

    local lm_out
    lm_out="$(cat "${WORK_ROOT}/${tag}/lm_head_out.path")"
    run_py update-decode-state \
      --lm-head-out-dir "${lm_out}" \
      --state-dir "${state_dir}" \
      --cur-len "${cur_len}" \
      --step "${step}"

    cur_len=$((cur_len + 1))
    echo "${cur_len}" > "${cur_len_file}"

    cleanup_step_scratch "${tag}" "$([[ "${step}" -eq 0 ]] && echo 1 || echo 0)"
  done

  _rm_if_not_keep "${WORK_ROOT}/step_0000"
  write_final_output "${cur_len}"
  log "decode done cur_len=${cur_len}"
}

main() {
  refresh_dump_paths
  check_dump
  mkdir -p "${WORK_ROOT}" "${OUTPUT_ROOT}" "${OM_DIR}" "${STATE_DIR}"

  OM_VISION="$(resolve_om "${OM_VISION:-}" "${VISION_OM_PREFIX}" "${VISION_OM_PREFIX}")" || exit 1
  OM_PREBLOCK="$(resolve_om "${OM_PREBLOCK:-}" "llm_preblock" "llm_preblock")" || exit 1
  OM_B1="$(resolve_om "${OM_B1:-}" "llm_block1" "llm_block1")" || exit 1
  OM_B2="$(resolve_om "${OM_B2:-}" "llm_block2" "llm_block2")" || exit 1
  OM_B3="$(resolve_om "${OM_B3:-}" "llm_block3" "llm_block3")" || exit 1
  OM_LM_HEAD="$(resolve_om "${OM_LM_HEAD:-}" "lm_head" "lm_head")" || exit 1

  log "PROFILE=${EXPORT_PROFILE}  MAX_SEQ_LEN=${MAX_SEQ_LEN}  vision=${VISION_OM_PREFIX}"
  log "DUMP_ROOT=${DUMP_ROOT}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}  MODE=${MODE}  GEN_STEPS=${GEN_STEPS}  RUN_MSAME=${RUN_MSAME}"

  case "${MODE}" in
    prefill_only) run_prefill ;;
    full|decode)  run_decode_loop ;;
    *)
      echo "ERROR: unknown MODE=${MODE} (prefill_only|full|decode)" >&2
      exit 1
      ;;
  esac

  log "Done. final: ${OUTPUT_ROOT}/final_logits.bin"
}

run_batch() {
  local batch_root="$1"
  local summary="${batch_root}/summary_run.tsv"
  local -a item_dirs=()

  while IFS= read -r _d; do
    [[ -n "${_d}" ]] && item_dirs+=("${_d}")
  done <<EOF
$(find "${batch_root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort)
EOF

  if [[ "${#item_dirs[@]}" -eq 0 ]]; then
    echo "ERROR: no item dirs under ${batch_root}" >&2
    exit 1
  fi

  log "BATCH_ROOT=${batch_root}  items=${#item_dirs[@]}  MODE=${MODE}  GEN_STEPS=${GEN_STEPS}"
  echo "stem	status	item_dir	output_dir" > "${summary}"

  local idx=0 total="${#item_dirs[@]}"
  for item_dir in "${item_dirs[@]}"; do
    idx=$((idx + 1))
    local stem dump_dir
    stem="$(basename "${item_dir}")"
    dump_dir="${item_dir}/dump"

    if [[ ! -f "${dump_dir}/vision/pixel_values.bin" ]]; then
      log "[${idx}/${total}] SKIP ${stem}: missing dump"
      echo "${stem}	skip_no_dump	${item_dir}	" >> "${summary}"
      continue
    fi

    if [[ "${SKIP_EXIST}" == "1" && -d "${item_dir}/om_output/work/step_0000" ]]; then
      log "[${idx}/${total}] SKIP_EXIST ${stem}"
      echo "${stem}	skip_exist	${item_dir}	${item_dir}/om_output" >> "${summary}"
      continue
    fi

    log "========== [${idx}/${total}] pipeline ${stem} =========="
    configure_item_paths "${item_dir}"
    main

    if [[ "${DO_PARSE}" == "1" ]]; then
      python3 "${PARSE_SCRIPT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --dump-dir "${DUMP_ROOT}" \
        --model-dir "${MODEL_DIR}" \
        --response-out "${item_dir}/response.txt" \
        > "${item_dir}/parse.log" 2>&1 || true
    fi

    echo "${stem}	ok	${item_dir}	${OUTPUT_ROOT}" >> "${summary}"
  done

  log "Batch done. summary: ${summary}"
}

run_single_configured() {
  refresh_dump_paths
  log "DUMP_ROOT=${DUMP_ROOT}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}  work=${WORK_ROOT}  state=${STATE_DIR}"
  main
}

dispatch() {
  local input="" cli_dump="" cli_batch="" cli_output="" cli_work="" cli_state="" cli_profile=""
  local -a positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --profile)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires name" >&2; exit 1; }
        cli_profile="$2"; shift 2 ;;
      --dump-dir|--dump-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_dump="$(cd "$2" && pwd)"; shift 2 ;;
      --batch-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_batch="$(cd "$2" && pwd)"; shift 2 ;;
      --output-dir|--output-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_output="$(cd "$2" && pwd)"; shift 2 ;;
      --work-dir|--work-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_work="$(cd "$2" && pwd)"; shift 2 ;;
      --state-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_state="$(cd "$2" && pwd)"; shift 2 ;;
      --) shift; while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done; break ;;
      -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) positional+=("$1"); shift ;;
    esac
  done

  [[ "${#positional[@]}" -ge 2 ]] && GEN_STEPS="${positional[1]}"
  input="${positional[0]:-}"

  if [[ -n "${cli_profile}" ]]; then
    EXPORT_PROFILE="${cli_profile}"
    configure_export_profile
  fi

  local work_override=0 state_override=0
  [[ -n "${cli_work}" ]] && work_override=1
  [[ -n "${cli_state}" ]] && state_override=1
  [[ -n "${cli_output}" ]] && OUTPUT_ROOT="${cli_output}"
  [[ -n "${cli_work}" ]] && WORK_ROOT="${cli_work}"
  [[ -n "${cli_state}" ]] && STATE_DIR="${cli_state}"
  configure_output_layout "${OUTPUT_ROOT}" "${work_override}" "${state_override}"

  if [[ -n "${cli_batch}" ]]; then
    run_batch "${cli_batch}"
    return
  fi

  if [[ -n "${cli_dump}" ]]; then
    configure_dump_root_paths "${cli_dump}"
    run_single_configured
    return
  fi

  if [[ -z "${input}" ]]; then
    run_single_configured
    return
  fi

  [[ -d "${input}" ]] || { echo "ERROR: not a directory: ${input}" >&2; exit 1; }
  input="$(cd "${input}" && pwd)"

  if is_batch_root "${input}"; then
    run_batch "${input}"
    return
  fi

  if [[ -f "${input}/dump/vision/pixel_values.bin" ]]; then
    configure_item_paths "${input}"
  elif [[ -f "${input}/vision/pixel_values.bin" ]]; then
    configure_dump_root_paths "${input}"
  else
    echo "ERROR: no dump bins under ${input}" >&2
    exit 1
  fi

  run_single_configured
}

dispatch "$@"
