# Serial OM pipeline — sourced by ../run_om_pipeline.sh
# One sample at a time; each OM step runs inline (no worker processes).

: "${OM_DIR:?OM_DIR required}"
: "${PIPELINE_DIR:?PIPELINE_DIR required}"
SCRIPT_DIR="${OM_DIR}"
# shellcheck source=paths.sh
source "${PIPELINE_DIR}/paths.sh"
REPO_ROOT="${REPO_ROOT:-${OM_DIR}/..}"

VISION_BIN="${VISION_BIN:-${SCRIPT_DIR}/vision_bin}"
PROMPT_BIN="${PROMPT_BIN:-${SCRIPT_DIR}/prompt_bin}"
INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}}"
DUMP_VISION="${DUMP_VISION:-${VISION_BIN}}"
DUMP_PREBLOCK="${DUMP_PREBLOCK:-${PROMPT_BIN}}"
DUMP_ROOT="${DUMP_ROOT:-${INPUT_ROOT}}"

OM_EXPORT_DIR="${OM_EXPORT_DIR:-${SCRIPT_DIR}/om_export}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/om_output}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/work}"
STATE_DIR="${STATE_DIR:-${OUTPUT_ROOT}/state}"

PY_HELPER="${PY_HELPER:-${SCRIPT_DIR}/om_bin_utils.py}"
PARSE_SCRIPT="${PARSE_SCRIPT:-${SCRIPT_DIR}/parse_state.py}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/InternVL3_5-1B-HF}"

MODE="${MODE:-full}"
GEN_STEPS="${GEN_STEPS:-50}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
RUN_MSAME="${RUN_MSAME:-0}"
DO_PARSE="${DO_PARSE:-0}"
SKIP_EXIST="${SKIP_EXIST:-0}"
STOP_ON_EOS="${STOP_ON_EOS:-1}"
MAX_SEQ_LEN=512
VISION_OM_PREFIX="vision_448"

MSAME_BIN="${MSAME_BIN:-${SCRIPT_DIR}/msame}"
RUN_SH="${RUN_SH:-${SCRIPT_DIR}/run.sh}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"
LAST_LM_HEAD_OUT=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
  cat <<EOF
Usage: bash run_om_pipeline.sh [options] [path] [gen_steps]

  --vision-bin PATH     vision input dir (default: om/vision_bin)
  --prompt-bin PATH     preblock static bins (default: om/prompt_bin)
  --dump-dir PATH       legacy alias: auto-detect vision_bin/prompt_bin or dump/
  --output-dir PATH     unified output root (default: om_output/)
  --batch-root PATH     batch mode (item/vision_bin per image)
  path                  auto-detect batch / item / om root

Env: MODE=prefill_only|full|decode  RUN_MSAME=1  STOP_ON_EOS=1
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
  for f in "${OM_EXPORT_DIR}/${glob_prefix}"*.om; do matches+=("${f}"); done
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 1 ]]; then echo "${matches[0]}"; return 0; fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "ERROR: ambiguous ${label}: ${matches[*]}" >&2; return 1
  fi
  echo "ERROR: ${label} not found under ${OM_EXPORT_DIR}" >&2
  return 1
}

configure_output_layout() {
  OUTPUT_ROOT="$1"
  [[ "${2:-0}" != "1" ]] && WORK_ROOT="${OUTPUT_ROOT}/work"
  [[ "${3:-0}" != "1" ]] && STATE_DIR="${OUTPUT_ROOT}/state"
}

check_dump() {
  check_om_inputs
}

check_om_output_bins() {
  local output_dir="$1" tag="$2"
  find "${output_dir}" -name '*.bin' -print -quit | grep -q . || {
    echo "ERROR: OM ${tag} produced no .bin under ${output_dir}" >&2
    return 1
  }
}

run_py() {
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
  local tag="$1" keep_mmproj="${2:-0}"
  local base="${WORK_ROOT}/${tag}"
  _rm_if_not_keep \
    "${base}/vision_in" "${base}/pre_in" "${base}/pre_out" \
    "${base}/b1_in" "${base}/b2_in" "${base}/b3_in" \
    "${base}/b1_out" "${base}/b2_out" \
    "${base}/lm_head_in" "${base}/lm_head_out" \
    "${base}/lm_head_out.path" "${base}/b3_out.path"
  if [[ "${keep_mmproj}" != "1" ]]; then
    _rm_if_not_keep "${base}/vision_out" "${base}/mmproj_in" "${base}/mmproj_out" "${base}/mmproj_out.path"
  fi
}

copy_lm_logits_bin() {
  local lm_out_dir="$1" dest="$2"
  [[ -d "${lm_out_dir}" ]] || return 0
  python3 - <<PY
import json, shutil, sys
from pathlib import Path
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

run_vision_mmproj() {
  local prefix="$1"
  local vision_in="${WORK_ROOT}/${prefix}/vision_in"
  local vision_out="${WORK_ROOT}/${prefix}/vision_out"
  local mmproj_in="${WORK_ROOT}/${prefix}/mmproj_in"
  local mmproj_out="${WORK_ROOT}/${prefix}/mmproj_out"

  run_py prepare-vision-input --dump-vision-dir "${DUMP_VISION}" --out-dir "${vision_in}"
  run_om_model "${prefix}/${VISION_OM_PREFIX}" "${OM_VISION}" "${vision_in}" "${vision_out}" 1
  _rm_if_not_keep "${vision_in}"

  run_py prepare-mmproj-input --vision-out-dir "${vision_out}" --out-dir "${mmproj_in}"
  _rm_if_not_keep "${vision_out}"

  run_om_model "${prefix}/mm_proj" "${OM_MM_PROJ}" "${mmproj_in}" "${mmproj_out}" 1
  _rm_if_not_keep "${mmproj_in}"
  echo "${mmproj_out}" > "${WORK_ROOT}/${prefix}/mmproj_out.path"
}

run_llm_chain() {
  local prefix="$1" state_dir="$2" cur_len="$3"
  local mmproj_out
  mmproj_out="$(cat "${WORK_ROOT}/${prefix}/mmproj_out.path")"

  local pre_in="${WORK_ROOT}/${prefix}/pre_in"
  local pre_out="${WORK_ROOT}/${prefix}/pre_out"

  run_py prepare-preblock-input \
    --state-dir "${state_dir}" \
    --static-preblock-dir "${DUMP_PREBLOCK}" \
    --mm-proj-out-dir "${mmproj_out}" \
    --out-dir "${pre_in}"
  run_om_model "${prefix}/llm_preblock" "${OM_PREBLOCK}" "${pre_in}" "${pre_out}" 4

  local prev_out="${pre_out}" bi b_in b_out
  local -a blocks=(1 2 3)
  local -a om_paths=("" "${OM_B1}" "${OM_B2}" "${OM_B3}")

  for bi in "${blocks[@]}"; do
    b_in="${WORK_ROOT}/${prefix}/b${bi}_in"
    b_out="${WORK_ROOT}/${prefix}/b${bi}_out"
    local -a py_args=(prepare-block-input --pre-out-dir "${pre_out}" --out-dir "${b_in}" --block-idx "${bi}")
    [[ "${bi}" -gt 1 ]] && py_args+=(--prev-block-out-dir "${prev_out}")
    run_py "${py_args[@]}"
    run_om_model "${prefix}/llm_block${bi}" "${om_paths[$bi]}" "${b_in}" "${b_out}" 4
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

  run_vision_mmproj "${tag}"
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
      run_vision_mmproj "${tag}"
    else
      mkdir -p "${WORK_ROOT}/${tag}"
      echo "${WORK_ROOT}/step_0000/mmproj_out" > "${WORK_ROOT}/${tag}/mmproj_out.path"
    fi

    run_llm_chain "${tag}" "${state_dir}" "${cur_len}"

    local lm_out decode_line hit_eos=0
    lm_out="$(cat "${WORK_ROOT}/${tag}/lm_head_out.path")"
    decode_line="$(run_py update-decode-state \
      --lm-head-out-dir "${lm_out}" \
      --state-dir "${state_dir}" \
      --cur-len "${cur_len}" \
      --step "${step}")"
    log "${decode_line}"
    [[ "${decode_line}" == *" eos=1"* ]] && hit_eos=1

    cur_len=$((cur_len + 1))
    echo "${cur_len}" > "${cur_len_file}"

    cleanup_step_scratch "${tag}" "$([[ "${step}" -eq 0 ]] && echo 1 || echo 0)"

    if [[ "${hit_eos}" == "1" && "${STOP_ON_EOS}" == "1" ]]; then
      log "EOS at step ${step}, stop decode"
      break
    fi
  done

  _rm_if_not_keep "${WORK_ROOT}/step_0000"
  write_final_output "${cur_len}"
  log "decode done cur_len=${cur_len}"
}

main() {
  sync_om_input_env
  check_dump
  mkdir -p "${WORK_ROOT}" "${OUTPUT_ROOT}" "${OM_EXPORT_DIR}" "${STATE_DIR}"

  OM_VISION="$(resolve_om "${OM_VISION:-}" "${VISION_OM_PREFIX}" "${VISION_OM_PREFIX}")" || exit 1
  OM_MM_PROJ="$(resolve_om "${OM_MM_PROJ:-}" "mm_proj" "mm_proj")" || exit 1
  OM_PREBLOCK="$(resolve_om "${OM_PREBLOCK:-}" "llm_preblock" "llm_preblock")" || exit 1
  OM_B1="$(resolve_om "${OM_B1:-}" "llm_block1" "llm_block1")" || exit 1
  OM_B2="$(resolve_om "${OM_B2:-}" "llm_block2" "llm_block2")" || exit 1
  OM_B3="$(resolve_om "${OM_B3:-}" "llm_block3" "llm_block3")" || exit 1
  OM_LM_HEAD="$(resolve_om "${OM_LM_HEAD:-}" "lm_head" "lm_head")" || exit 1

  log "MAX_SEQ_LEN=${MAX_SEQ_LEN}  vision=${VISION_OM_PREFIX}"
  log "VISION_BIN=${VISION_BIN}  PROMPT_BIN=${PROMPT_BIN}"
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
    local stem
    stem="$(basename "${item_dir}")"
    [[ "${stem}" == .* ]] && continue

    if ! configure_item_inputs "${item_dir}"; then
      local skip_reason
      skip_reason="$(diagnose_item_inputs "${item_dir}")"
      log "[${idx}/${total}] SKIP ${stem}: ${skip_reason}"
      echo "${stem}	skip_no_input	${item_dir}	" >> "${summary}"
      continue
    fi

    if [[ "${SKIP_EXIST}" == "1" && -d "${item_dir}/om_output/work/step_0000" ]]; then
      log "[${idx}/${total}] SKIP_EXIST ${stem}"
      echo "${stem}	skip_exist	${item_dir}	${item_dir}/om_output" >> "${summary}"
      continue
    fi

    log "========== [${idx}/${total}] pipeline ${stem} =========="
    main

    if [[ "${DO_PARSE}" == "1" ]]; then
      python3 "${PARSE_SCRIPT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --dump-dir "${PROMPT_BIN}" \
        --model-dir "${MODEL_DIR}" \
        --response-out "${item_dir}/response.txt" \
        > "${item_dir}/parse.log" 2>&1 || true
    fi

    echo "${stem}	ok	${item_dir}	${OUTPUT_ROOT}" >> "${summary}"
  done

  log "Batch done. summary: ${summary}"
}

run_single_configured() {
  if [[ -z "${VISION_BIN:-}" || -z "${PROMPT_BIN:-}" ]]; then
    configure_default_single_inputs || exit 1
  else
    sync_om_input_env
  fi
  log "VISION_BIN=${VISION_BIN}  PROMPT_BIN=${PROMPT_BIN}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}  work=${WORK_ROOT}  state=${STATE_DIR}"
  main
}

serial_dispatch() {
  local input="" cli_dump="" cli_batch="" cli_output="" cli_work="" cli_state=""
  local cli_vision="" cli_prompt=""
  local -a positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --vision-bin)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_vision="$(cd "$2" && pwd)"; shift 2 ;;
      --prompt-bin)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires path" >&2; exit 1; }
        cli_prompt="$(cd "$2" && pwd)"; shift 2 ;;
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

  [[ -n "${cli_vision}" ]] && VISION_BIN="${cli_vision}"
  [[ -n "${cli_prompt}" ]] && PROMPT_BIN="${cli_prompt}"
  if [[ -n "${cli_vision}" || -n "${cli_prompt}" ]]; then
    sync_om_input_env
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
    configure_input_root "${cli_dump}" || exit 1
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

  if resolve_vision_bin "${input}" &>/dev/null; then
    configure_item_inputs "${input}" || exit 1
  elif configure_input_root "${input}" 2>/dev/null; then
    :
  else
    echo "ERROR: no input bins under ${input}" >&2
    echo "  expected vision_bin/pixel_values.bin (+ shared prompt_bin/)" >&2
    exit 1
  fi

  run_single_configured
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: use ../run_om_pipeline.sh" >&2
  exit 1
fi
