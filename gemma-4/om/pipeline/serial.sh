# Serial OM pipeline — sourced by ../run_om_pipeline.sh
# Gemma-4: main chain + assistant speculative decode (inline msame).

: "${OM_DIR:?OM_DIR required}"
: "${PIPELINE_DIR:?PIPELINE_DIR required}"
SCRIPT_DIR="${OM_DIR}"
# shellcheck source=paths.sh
source "${PIPELINE_DIR}/paths.sh"
REPO_ROOT="${REPO_ROOT:-${OM_DIR}/..}"
GEMMA4_ROOT="${GEMMA4_ROOT:-${SCRIPT_DIR}}"

VISION_BIN="${VISION_BIN:-${SCRIPT_DIR}/vision_bin}"
PROMPT_BIN="${PROMPT_BIN:-${SCRIPT_DIR}/prompt_bin}"
INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}}"
DUMP_VISION="${DUMP_VISION:-${VISION_BIN}}"
DUMP_PREBLOCK="${DUMP_PREBLOCK:-${PROMPT_BIN}}"
DUMP_ROOT="${DUMP_ROOT:-${INPUT_ROOT}}"

OM_EXPORT_DIR="${OM_EXPORT_DIR:-${SCRIPT_DIR}/om_export}"
OM_ASSISTANT="${OM_ASSISTANT:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/om_output}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/work}"
STATE_DIR="${STATE_DIR:-${OUTPUT_ROOT}/state}"

PY_HELPER="${PY_HELPER:-${SCRIPT_DIR}/om_bin_utils.py}"
PY_ASSISTANT="${PY_ASSISTANT:-${SCRIPT_DIR}/om_bin_utils_it_assistant.py}"

PLE_TABLE_DIR="${PLE_TABLE_DIR:-${SCRIPT_DIR}/ple_table}"
PLE_TABLE_BIN="${PLE_TABLE_BIN:-${PLE_TABLE_DIR}/embed_tokens_per_layer.bin}"
PAD_TOKEN_ID="${PAD_TOKEN_ID:-0}"

MODE="${MODE:-full}"              # prefill_only | main_only | main_decode | full
GEN_STEPS="${GEN_STEPS:-50}"
NUM_ASSISTANT_TOKENS="${NUM_ASSISTANT_TOKENS:-6}"
WITH_ASSISTANT="${WITH_ASSISTANT:-1}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"
export MODE GEN_STEPS NUM_ASSISTANT_TOKENS WITH_ASSISTANT
MAX_SEQ_LEN=512

MSAME_BIN="${MSAME_BIN:-${SCRIPT_DIR}/msame}"
RUN_SH="${RUN_SH:-${SCRIPT_DIR}/run.sh}"
MSPROF_BIN="${MSPROF_BIN:-/var/msprof}"
RUN_MSAME="${RUN_MSAME:-0}"

STOP_ON_EOS="${STOP_ON_EOS:-1}"
DO_PARSE="${DO_PARSE:-0}"
SKIP_EXIST="${SKIP_EXIST:-0}"
PARSE_SCRIPT="${PARSE_SCRIPT:-${SCRIPT_DIR}/parse_state.py}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/gemma-4-E2B-it}"

# Updated each lm_head run; used for final_logits.bin
LAST_LM_HEAD_OUT=""

# ----------------------------- helpers ---------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

refresh_dump_paths() {
  sync_om_input_env
  PLE_TABLE_DIR="${PLE_TABLE_DIR:-${SCRIPT_DIR}/ple_table}"
  PLE_TABLE_BIN="${PLE_TABLE_BIN:-${PLE_TABLE_DIR}/embed_tokens_per_layer.bin}"
}

is_batch_root() {
  local root="$1"
  [[ -d "${root}" ]] || return 1
  local d
  for d in "${root}"/*; do
    [[ -d "${d}" ]] || continue
    resolve_vision_bin "${d}" &>/dev/null && return 0
  done
  return 1
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

configure_item_paths() {
  configure_item_inputs "$1" || return 1
  configure_output_layout "${1}/om_output"
}

configure_output_layout() {
  local root="$1"
  local work_override="${2:-0}"
  local state_override="${3:-0}"
  OUTPUT_ROOT="${root}"
  if [[ "${work_override}" != "1" ]]; then
    WORK_ROOT="${OUTPUT_ROOT}/work"
  fi
  if [[ "${state_override}" != "1" ]]; then
    STATE_DIR="${OUTPUT_ROOT}/state"
  fi
}

configure_dump_root_paths() {
  configure_input_root "$1" || return 1
}

usage() {
  cat <<EOF
Usage: bash run_om_pipeline.sh [options] [path] [gen_steps] [num_assistant_tokens]

  --vision-bin PATH     vision input dir (default: om/vision_bin)
  --prompt-bin PATH     preblock static bins (default: om/prompt_bin)
  --dump-dir PATH       legacy alias: auto-detect vision_bin/prompt_bin or dump/
  --output-dir PATH                   unified output root (default: om/om_output)
                                      ├── work/    scratch
                                      ├── state/   decode state
                                      └── final_*  final bins
  --work-dir PATH                     override scratch (default: <output-dir>/work)
  --state-dir PATH                    override state (default: <output-dir>/state)
  --ple-table-dir PATH                PLE table dir (default: om/ple_table)

Batch run:
  --batch-root PATH                   batch root (<stem>/dump/ per image)
  path                                auto-detect: batch root | item dir | dump root

Env equivalents: DUMP_ROOT, WORK_ROOT, OUTPUT_ROOT, STATE_DIR, PLE_TABLE_DIR

Examples:
  RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir /data/my_dump
  RUN_MSAME=1 bash run_om_pipeline.sh --dump-dir ./dump --output-dir ./om_output
  DUMP_ROOT=/data/dump RUN_MSAME=1 bash run_om_pipeline.sh
  RUN_MSAME=1 bash run_om_pipeline.sh ./batch 100 8
EOF
}

resolve_om() {
  local env_val="$1" glob_prefix="$2" label="$3"
  if [[ -n "${env_val}" && -f "${env_val}" ]]; then
    echo "${env_val}"
    return 0
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

resolve_om_assistant() {
  if [[ -n "${OM_ASSISTANT}" ]]; then
    [[ -f "${OM_ASSISTANT}" ]] && echo "${OM_ASSISTANT}" && return 0
    echo "ERROR: OM_ASSISTANT not found: ${OM_ASSISTANT}" >&2
    return 1
  fi
  local -a found=()
  local f
  shopt -s nullglob
  for f in "${OM_EXPORT_DIR}"/assistant*.om; do [[ -f "${f}" ]] && found+=("${f}"); done
  shopt -u nullglob
  if [[ ${#found[@]} -eq 1 ]]; then echo "${found[0]}"; return 0; fi
  echo "ERROR: assistant OM not found under ${OM_EXPORT_DIR}" >&2
  return 1
}

check_dump() {
  check_om_inputs
}

check_ple_table() {
  [[ -f "${PLE_TABLE_BIN}" ]] || {
    echo "ERROR: PLE table not found: ${PLE_TABLE_BIN}" >&2
    exit 1
  }
}

check_om_output_bins() {
  local output_dir="$1" tag="$2"
  find "${output_dir}" -name '*.bin' -print -quit | grep -q . || {
    echo "ERROR: OM ${tag} produced no .bin under ${output_dir}" >&2
    return 1
  }
}

run_py() {
  GEMMA4_ROOT="${GEMMA4_ROOT}" DUMP_PREBLOCK="${DUMP_PREBLOCK}" DUMP_ROOT="${DUMP_ROOT}" \
    python3 "${PY_HELPER}" "$@"
}

run_py_assistant() {
  GEMMA4_ROOT="${GEMMA4_ROOT}" DUMP_ROOT="${DUMP_ROOT}" \
    python3 "${PY_ASSISTANT}" "$@"
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

# ----------------------------- cleanup ---------------------------------------
_rm_if_not_keep() {
  [[ "${KEEP_INTERMEDIATE}" == "1" ]] && return 0
  rm -rf "$@"
}

# After llm chain: drop inputs and block outs except b3/b7/lm_head
cleanup_llm_chain_scratch() {
  local prefix="$1"
  local base="${WORK_ROOT}/${prefix}"
  _rm_if_not_keep \
    "${base}/pre_in" "${base}/pre_out" \
    "${base}/b1_in" "${base}/b2_in" "${base}/b3_in" \
    "${base}/b4_in" "${base}/b5_in" "${base}/b6_in" "${base}/b7_in" \
    "${base}/b1_out" "${base}/b2_out" \
    "${base}/b4_out" "${base}/b5_out" "${base}/b6_out" \
    "${base}/lm_head_in"
}

cleanup_vision_scratch() {
  local prefix="$1"
  local base="${WORK_ROOT}/${prefix}"
  _rm_if_not_keep "${base}/vision_in" "${base}/vision_out" "${base}/mmproj_in"
}

# Drop entire step workspace (after speculative step completes)
cleanup_step_workspace() {
  local tag="$1"
  local keep_mmproj="${2:-0}"
  local base="${WORK_ROOT}/${tag}"
  cleanup_vision_scratch "${tag}"
  cleanup_llm_chain_scratch "${tag}"
  _rm_if_not_keep \
    "${base}/assistant_in" "${base}/assistant_out" \
    "${base}/assistant_chain_in" "${base}/assistant_chain_out" \
    "${base}/lm_head_out" \
    "${base}/lm_head_out.path" \
    "${base}/b7_out.path" "${base}/b3_out.path" \
    "${base}/assistant_candidates.path" "${base}/main_preds.txt" \
    "${base}/accept_count.txt"
  local k
  for ((k = 0; k < NUM_ASSISTANT_TOKENS; k++)); do
    _rm_if_not_keep "${WORK_ROOT}/${tag}_draft_${k}"
  done
  _rm_if_not_keep "${WORK_ROOT}/${tag}_verify" "${WORK_ROOT}/${tag}_verify_"*
  if [[ "${keep_mmproj}" != "1" ]]; then
    _rm_if_not_keep "${base}/mmproj_out" "${base}/mmproj_out.path"
  fi
}

# ----------------------------- final output ----------------------------------
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

  if [[ ! -f "${OUTPUT_ROOT}/final_logits.bin" ]]; then
    update_final_logits || true
  fi
  [[ -f "${OUTPUT_ROOT}/final_logits.bin" ]] || {
    echo "WARN: final_logits.bin missing (dry-run or no lm_head output yet)" >&2
  }

  if [[ -f "${STATE_DIR}/input_ids.bin" ]]; then
    cp -f "${STATE_DIR}/input_ids.bin" "${OUTPUT_ROOT}/final_input_ids.bin"
  fi
  if [[ -f "${STATE_DIR}/attention_mask.bin" ]]; then
    cp -f "${STATE_DIR}/attention_mask.bin" "${OUTPUT_ROOT}/final_attention_mask.bin"
  fi
  echo "${cur_len}" > "${OUTPUT_ROOT}/final_cur_len.txt"

  python3 - <<PY
import json
from pathlib import Path
meta = {
    "mode": "${MODE}",
    "cur_len": int("${cur_len}"),
    "gen_steps": int("${GEN_STEPS}"),
    "num_assistant_tokens": int("${NUM_ASSISTANT_TOKENS}"),
    "final_logits": str(Path("${OUTPUT_ROOT}/final_logits.bin").resolve()),
    "output_root": str(Path("${OUTPUT_ROOT}").resolve()),
    "state_dir": str(Path("${STATE_DIR}").resolve()),
    "work_root": str(Path("${WORK_ROOT}").resolve()),
}
Path("${OUTPUT_ROOT}/final.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

  # Trim scratch under WORK_ROOT (final bins live in OUTPUT_ROOT)
  if [[ "${KEEP_INTERMEDIATE}" != "1" ]]; then
    rm -rf "${WORK_ROOT}"
    log "removed scratch ${WORK_ROOT}"
  fi
  log "final output: ${OUTPUT_ROOT}/final_logits.bin"
}

# ----------------------------- vision / mm_proj --------------------------------
run_vision_mmproj() {
  local prefix="$1"
  local vision_in="${WORK_ROOT}/${prefix}/vision_in"
  local vision_out="${WORK_ROOT}/${prefix}/vision_out"
  local mmproj_in="${WORK_ROOT}/${prefix}/mmproj_in"
  local mmproj_out="${WORK_ROOT}/${prefix}/mmproj_out"

  run_py prepare-vision-input --dump-vision-dir "${DUMP_VISION}" --out-dir "${vision_in}"
  run_om_model "${prefix}/vision" "${OM_VISION}" "${vision_in}" "${vision_out}" 2
  _rm_if_not_keep "${vision_in}"

  run_py prepare-mmproj-input --vision-out-dir "${vision_out}" --out-dir "${mmproj_in}"
  _rm_if_not_keep "${vision_out}"

  run_om_model "${prefix}/mm_proj" "${OM_MM_PROJ}" "${mmproj_in}" "${mmproj_out}" 1
  _rm_if_not_keep "${mmproj_in}"
  echo "${mmproj_out}" > "${WORK_ROOT}/${prefix}/mmproj_out.path"
}

# ----------------------------- LLM chain ---------------------------------------
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
  run_om_model "${prefix}/llm_preblock" "${OM_PREBLOCK}" "${pre_in}" "${pre_out}" 5

  local prev_out="${pre_out}" b3_out="" bi
  local -a blocks=(1 2 3 4 5 6 7)
  local -a om_paths=("" "${OM_B1}" "${OM_B2}" "${OM_B3}" "${OM_B4}" "${OM_B5}" "${OM_B6}" "${OM_B7}")

  for bi in "${blocks[@]}"; do
    local b_in="${WORK_ROOT}/${prefix}/b${bi}_in"
    local b_out="${WORK_ROOT}/${prefix}/b${bi}_out"
    local -a py_args=(prepare-block-input --pre-out-dir "${pre_out}" --out-dir "${b_in}" --block-idx "${bi}")
    [[ "${bi}" -gt 1 ]] && py_args+=(--prev-block-out-dir "${prev_out}")
    [[ "${bi}" -ge 4 ]] && py_args+=(--b3-out-dir "${b3_out}")
    run_py "${py_args[@]}"
    if [[ "${bi}" -ge 4 ]]; then
      run_om_model "${prefix}/llm_block_${bi}" "${om_paths[$bi]}" "${b_in}" "${b_out}" 12
    else
      run_om_model "${prefix}/llm_block_${bi}" "${om_paths[$bi]}" "${b_in}" "${b_out}" 8
    fi
    _rm_if_not_keep "${b_in}"
    prev_out="${b_out}"
    [[ "${bi}" -eq 3 ]] && b3_out="${b_out}"
  done

  local lm_in="${WORK_ROOT}/${prefix}/lm_head_in"
  local lm_out="${WORK_ROOT}/${prefix}/lm_head_out"
  run_py prepare-lm-head-input --b7-out-dir "${prev_out}" --out-dir "${lm_in}" --cur-len "${cur_len}"
  run_om_model "${prefix}/lm_head" "${OM_LM_HEAD}" "${lm_in}" "${lm_out}" 1
  _rm_if_not_keep "${lm_in}"

  echo "${prev_out}" > "${WORK_ROOT}/${prefix}/b7_out.path"
  echo "${b3_out}" > "${WORK_ROOT}/${prefix}/b3_out.path"
  echo "${lm_out}" > "${WORK_ROOT}/${prefix}/lm_head_out.path"
  LAST_LM_HEAD_OUT="${lm_out}"
  update_final_logits
}

# ----------------------------- assistant -------------------------------------
run_assistant_om() {
  run_om_model "$1/assistant" "${OM_ASSISTANT}" "$2" "$3" 8
}

run_assistant_from_main_chain() {
  local tag="$1" state_dir="$2" cur_len="$3"
  local b7_out b3_out
  b7_out="$(cat "${WORK_ROOT}/${tag}/b7_out.path")"
  b3_out="$(cat "${WORK_ROOT}/${tag}/b3_out.path")"
  local ass_in="${WORK_ROOT}/${tag}/assistant_chain_in"
  local ass_out="${WORK_ROOT}/${tag}/assistant_chain_out"
  run_py_assistant prepare-assistant-input-chain \
    --state-dir "${state_dir}" \
    --b7-out-dir "${b7_out}" \
    --b3-out-dir "${b3_out}" \
    --cur-len "${cur_len}" \
    --out-dir "${ass_in}"
  run_assistant_om "${tag}/chain" "${ass_in}" "${ass_out}"
  _rm_if_not_keep "${ass_in}"
  log "assistant chain output: ${ass_out}/"
}

run_assistant_draft_loop() {
  local tag="$1" state_dir="$2" cur_len="$3"
  local b3_out cand_file k draft_tag ass_in ass_out
  b3_out="$(cat "${WORK_ROOT}/${tag}/b3_out.path")"
  cand_file="${WORK_ROOT}/${tag}/assistant_candidates.txt"
  : > "${cand_file}"

  for ((k = 0; k < NUM_ASSISTANT_TOKENS; k++)); do
    draft_tag="${tag}_draft_${k}"
    ass_in="${WORK_ROOT}/${draft_tag}/assistant_in"
    ass_out="${WORK_ROOT}/${draft_tag}/assistant_out"

    if [[ "${k}" -eq 0 ]]; then
      run_py_assistant prepare-assistant-input-chain \
        --state-dir "${state_dir}" \
        --b7-out-dir "$(cat "${WORK_ROOT}/${tag}/b7_out.path")" \
        --b3-out-dir "${b3_out}" \
        --cur-len "${cur_len}" \
        --out-dir "${ass_in}"
    else
      run_py_assistant prepare-assistant-draft-step \
        --prev-assistant-out-dir "${WORK_ROOT}/${tag}_draft_$((k - 1))/assistant_out" \
        --state-dir "${state_dir}" \
        --b3-out-dir "${b3_out}" \
        --pos "$((cur_len + k - 1))" \
        --out-dir "${ass_in}"
    fi

    run_assistant_om "${draft_tag}" "${ass_in}" "${ass_out}"
    _rm_if_not_keep "${ass_in}"
    [[ "${k}" -gt 0 ]] && _rm_if_not_keep "${WORK_ROOT}/${tag}_draft_$((k - 1))/assistant_out"

    local tok_file="${WORK_ROOT}/${draft_tag}/token.txt"
    run_py_assistant parse-assistant-argmax --assistant-out-dir "${ass_out}" --out-file "${tok_file}"
    tr -d ' \n\r' < "${tok_file}" >> "${cand_file}"
    [[ "${k}" -lt $((NUM_ASSISTANT_TOKENS - 1)) ]] && printf ' ' >> "${cand_file}"
  done

  echo "${cand_file}" > "${WORK_ROOT}/${tag}/assistant_candidates.path"
  log "assistant candidates: $(cat "${cand_file}")"
}

run_main_verify_lm_range() {
  local tag="$1" b7_out_dir="$2" cur_len="$3"
  local cand_file main_preds_file vi lm_tag lm_in lm_out pred_file
  cand_file="$(cat "${WORK_ROOT}/${tag}/assistant_candidates.path")"
  read -r -a _cand_arr <<< "$(tr '\n' ' ' < "${cand_file}" | xargs)"
  local cand_len="${#_cand_arr[@]}"
  main_preds_file="${WORK_ROOT}/${tag}/main_preds.txt"
  : > "${main_preds_file}"

  for ((vi = 0; vi < cand_len; vi++)); do
    local pos=$((cur_len - 1 + vi))
    lm_tag="${tag}_verify_${vi}"
    lm_in="${WORK_ROOT}/${lm_tag}/lm_head_in"
    lm_out="${WORK_ROOT}/${lm_tag}/lm_head_out"
    pred_file="${WORK_ROOT}/${lm_tag}/token.txt"

    run_py prepare-lm-head-input \
      --b7-out-dir "${b7_out_dir}" \
      --out-dir "${lm_in}" \
      --cur-len "$((pos + 1))"
    run_om_model "${lm_tag}/lm_head" "${OM_LM_HEAD}" "${lm_in}" "${lm_out}" 1
    LAST_LM_HEAD_OUT="${lm_out}"
    update_final_logits
    _rm_if_not_keep "${lm_in}"

    run_py_assistant parse-lm-argmax --lm-head-out-dir "${lm_out}" --out-file "${pred_file}"
    tr -d ' \n\r' < "${pred_file}" >> "${main_preds_file}"
    [[ "${vi}" -lt $((cand_len - 1)) ]] && printf ' ' >> "${main_preds_file}"
    _rm_if_not_keep "${lm_out}"
  done
  log "main verify preds: $(cat "${main_preds_file}")"
}

# ----------------------------- modes -----------------------------------------
sync_state_from_dump() {
  run_py sync-preblock-state --dump-preblock-dir "${DUMP_PREBLOCK}" --state-dir "$1"
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
  cleanup_llm_chain_scratch "${tag}"

  if [[ "${WITH_ASSISTANT}" == "1" ]]; then
    run_assistant_from_main_chain "${tag}" "${state_dir}" "${cur_len}"
    _rm_if_not_keep "${WORK_ROOT}/${tag}/assistant_chain_out"
  fi

  cleanup_step_workspace "${tag}" 0
  write_final_output "${cur_len}"
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
    [[ "${cur_len}" -ge "${MAX_SEQ_LEN}" ]] && { log "MAX_SEQ_LEN reached"; break; }

    tag=$(printf "step_%04d" "${step}")
    if [[ "${step}" -eq 0 ]]; then
      run_vision_mmproj "${tag}"
    else
      mkdir -p "${WORK_ROOT}/${tag}"
      echo "${WORK_ROOT}/step_0000/mmproj_out" > "${WORK_ROOT}/${tag}/mmproj_out.path"
    fi

    run_llm_chain "${tag}" "${state_dir}" "${cur_len}"
    cleanup_llm_chain_scratch "${tag}"

    local lm_out decode_line hit_eos=0
    lm_out="$(cat "${WORK_ROOT}/${tag}/lm_head_out.path")"
    decode_line="$(run_py update-decode-state \
      --lm-head-out-dir "${lm_out}" \
      --state-dir "${state_dir}" \
      --cur-len "${cur_len}" \
      --step "${step}" \
      --ple-table "${PLE_TABLE_BIN}" \
      --pad-token-id "${PAD_TOKEN_ID}")"
    log "${decode_line}"
    [[ "${decode_line}" == *" eos=1"* ]] && hit_eos=1

    cur_len=$((cur_len + 1))
    echo "${cur_len}" > "${cur_len_file}"

    cleanup_step_workspace "${tag}" "$([[ "${step}" -eq 0 ]] && echo 1 || echo 0)"

    if [[ "${hit_eos}" == "1" && "${STOP_ON_EOS}" == "1" ]]; then
      log "EOS at step ${step}, stop decode"
      break
    fi
  done

  _rm_if_not_keep "${WORK_ROOT}/step_0000"
  write_final_output "${cur_len}"
  log "decode done cur_len=${cur_len}"
}

run_speculative_decode_loop() {
  local state_dir="${STATE_DIR}"
  sync_state_from_dump "${state_dir}"

  local cur_len_file="${WORK_ROOT}/cur_len.txt"
  run_py init-cur-len --state-dir "${state_dir}" --out-file "${cur_len_file}"
  local cur_len step tag accept_count
  cur_len="$(cat "${cur_len_file}")"
  log "speculative decode initial cur_len=${cur_len}"

  for ((step = 0; step < GEN_STEPS; step++)); do
    [[ "${cur_len}" -ge "${MAX_SEQ_LEN}" ]] && { log "MAX_SEQ_LEN reached"; break; }

    tag=$(printf "step_%04d" "${step}")
    if [[ "${step}" -eq 0 ]]; then
      run_vision_mmproj "${tag}"
    else
      mkdir -p "${WORK_ROOT}/${tag}"
      echo "${WORK_ROOT}/step_0000/mmproj_out" > "${WORK_ROOT}/${tag}/mmproj_out.path"
    fi

    run_llm_chain "${tag}" "${state_dir}" "${cur_len}"
    cleanup_llm_chain_scratch "${tag}"

    run_assistant_draft_loop "${tag}" "${state_dir}" "${cur_len}"

    local verify_state="${WORK_ROOT}/${tag}/verify_state"
    run_py_assistant prepare-speculative-verify-state \
      --state-dir "${state_dir}" \
      --verify-state-dir "${verify_state}" \
      --cur-len "${cur_len}" \
      --candidates-file "$(cat "${WORK_ROOT}/${tag}/assistant_candidates.path")" \
      --ple-table "${PLE_TABLE_BIN}" \
      --pad-token-id "${PAD_TOKEN_ID}"

    mkdir -p "${WORK_ROOT}/${tag}_verify"
    cat "${WORK_ROOT}/${tag}/mmproj_out.path" > "${WORK_ROOT}/${tag}_verify/mmproj_out.path"
    run_llm_chain "${tag}_verify" "${verify_state}" "${cur_len}"
    cleanup_llm_chain_scratch "${tag}_verify"

    run_main_verify_lm_range "${tag}" "$(cat "${WORK_ROOT}/${tag}_verify/b7_out.path")" "${cur_len}"

    local accept_file="${WORK_ROOT}/${tag}/accept_count.txt"
    local accept_line hit_eos=0
    accept_line="$(run_py_assistant process-speculative-accept \
      --state-dir "${state_dir}" \
      --cur-len "${cur_len}" \
      --candidates-file "$(cat "${WORK_ROOT}/${tag}/assistant_candidates.path")" \
      --main-preds-file "${WORK_ROOT}/${tag}/main_preds.txt" \
      --accept-count-file "${accept_file}" \
      --step "${step}" \
      --ple-table "${PLE_TABLE_BIN}" \
      --pad-token-id "${PAD_TOKEN_ID}")"
    log "${accept_line}"
    [[ "${accept_line}" == *" eos=1"* ]] && hit_eos=1

    accept_count="$(cat "${accept_file}")"
    cur_len=$((cur_len + accept_count))
    echo "${cur_len}" > "${cur_len_file}"
    log "step ${step} accepted ${accept_count}, cur_len=${cur_len}"

    cleanup_step_workspace "${tag}" "$([[ "${step}" -eq 0 ]] && echo 1 || echo 0)"

    if [[ "${hit_eos}" == "1" && "${STOP_ON_EOS}" == "1" ]]; then
      log "EOS at step ${step}, stop speculative decode"
      break
    fi
  done

  _rm_if_not_keep "${WORK_ROOT}/step_0000"
  write_final_output "${cur_len}"
  log "speculative decode done cur_len=${cur_len}"
}

main() {
  refresh_dump_paths
  sync_om_input_env
  check_dump
  mkdir -p "${WORK_ROOT}" "${OUTPUT_ROOT}" "${OM_EXPORT_DIR}" "${STATE_DIR}"

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

  log "defaults: MODE=${MODE} GEN_STEPS=${GEN_STEPS} NUM_ASSISTANT_TOKENS=${NUM_ASSISTANT_TOKENS} WITH_ASSISTANT=${WITH_ASSISTANT}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}  (work/ state/ final_*)  KEEP_INTERMEDIATE=${KEEP_INTERMEDIATE}"

  case "${MODE}" in
    prefill_only|full|decode)
      if [[ "${MODE}" == "prefill_only" && "${WITH_ASSISTANT}" == "1" ]] || [[ "${MODE}" == "full" || "${MODE}" == "decode" ]]; then
        OM_ASSISTANT="$(resolve_om_assistant)" || exit 1
        log "OM_ASSISTANT=${OM_ASSISTANT}"
      fi
      ;;
  esac

  case "${MODE}" in
    prefill_only)
      run_prefill
      ;;
    main_only)
      WITH_ASSISTANT=0 run_prefill
      ;;
    main_decode)
      check_ple_table
      WITH_ASSISTANT=0 run_decode_loop
      ;;
    full|decode)
      check_ple_table
      run_speculative_decode_loop
      ;;
    *)
      echo "ERROR: unknown MODE=${MODE}" >&2
      echo "  prefill_only | main_only | main_decode | full" >&2
      exit 1
      ;;
  esac

  log "Done. final: ${OUTPUT_ROOT}/final_logits.bin"
}

run_batch() {
  local batch_root="$1"
  local summary="${batch_root}/summary_run.tsv"
  local item_dirs=() item_dir stem dump_dir idx total

  if ! PLE_TABLE_DIR="$(resolve_ple_table_dir)"; then
    echo "ERROR: PLE table not found: ${SCRIPT_DIR}/ple_table/embed_tokens_per_layer.bin" >&2
    echo "本地生成（仅一次，已存在则跳过）: python dump_om_inputs.py --ple-only" >&2
    echo "然后与 dump/ 或 batch/ 一并 scp 到 MDC" >&2
    echo "Or set PLE_TABLE_DIR=/path/to/ple_table" >&2
    exit 1
  fi
  export PLE_TABLE_DIR
  export PLE_TABLE_BIN="${PLE_TABLE_DIR}/embed_tokens_per_layer.bin"
  log "PLE_TABLE_DIR=${PLE_TABLE_DIR}"

  while IFS= read -r _d; do
    [[ -n "${_d}" ]] && item_dirs+=("${_d}")
  done <<EOF
$(find "${batch_root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort)
EOF

  if [[ "${#item_dirs[@]}" -eq 0 ]]; then
    echo "ERROR: no item dirs under ${batch_root}" >&2
    exit 1
  fi

  log "BATCH_ROOT=${batch_root}  items=${#item_dirs[@]}  MODE=${MODE}  GEN_STEPS=${GEN_STEPS}  NUM_ASSISTANT_TOKENS=${NUM_ASSISTANT_TOKENS}  RUN_MSAME=${RUN_MSAME}"
  echo "stem	status	item_dir	output_dir" > "${summary}"

  idx=0
  total="${#item_dirs[@]}"
  for item_dir in "${item_dirs[@]}"; do
    idx=$((idx + 1))
    stem="$(basename "${item_dir}")"
    [[ "${stem}" == .* ]] && continue

    if ! configure_item_inputs "${item_dir}"; then
      local skip_reason
      skip_reason="$(diagnose_item_inputs "${item_dir}")"
      log "[${idx}/${total}] SKIP ${stem}: ${skip_reason}"
      echo "${stem}	skip_no_input	${item_dir}	" >> "${summary}"
      continue
    fi
    configure_output_layout "${item_dir}/om_output"

    if [[ "${SKIP_EXIST}" == "1" && -d "${item_dir}/om_output/work/step_0000" ]]; then
      log "[${idx}/${total}] SKIP_EXIST ${stem}"
      echo "${stem}	skip_exist	${item_dir}	${item_dir}/om_output" >> "${summary}"
      continue
    fi

    log "========== [${idx}/${total}] pipeline ${stem} =========="
    log "  VISION_BIN=${VISION_BIN}  PROMPT_BIN=${PROMPT_BIN}"
    log "  OUTPUT_ROOT=${OUTPUT_ROOT}  (work/ state/ final_*)"

    GEMMA4_ROOT="${SCRIPT_DIR}" main

    if [[ "${DO_PARSE}" == "1" ]]; then
      python3 "${PARSE_SCRIPT}" \
        --output-dir "${OUTPUT_ROOT}" \
        --dump-dir "${PROMPT_BIN}" \
        --state-dir "${STATE_DIR}" \
        --model-dir "${MODEL_DIR}" \
        --response-out "${item_dir}/response.txt" \
        > "${item_dir}/parse.log" 2>&1 || true
      log "  response: ${item_dir}/response.txt (see parse.log)"
    fi

    echo "${stem}	ok	${item_dir}	${OUTPUT_ROOT}" >> "${summary}"
  done

  log "Batch done. summary: ${summary}"
}

run_single_configured() {
  if ! PLE_TABLE_DIR="$(resolve_ple_table_dir)"; then
    PLE_TABLE_DIR="${SCRIPT_DIR}/ple_table"
    PLE_TABLE_BIN="${PLE_TABLE_DIR}/embed_tokens_per_layer.bin"
  else
    PLE_TABLE_BIN="${PLE_TABLE_DIR}/embed_tokens_per_layer.bin"
  fi
  if [[ -z "${VISION_BIN:-}" || -z "${PROMPT_BIN:-}" ]]; then
    configure_default_single_inputs || exit 1
  else
    sync_om_input_env
  fi
  refresh_dump_paths
  log "VISION_BIN=${VISION_BIN}  PROMPT_BIN=${PROMPT_BIN}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}  work=${WORK_ROOT}  state=${STATE_DIR}"
  log "PLE_TABLE_DIR=${PLE_TABLE_DIR}"
  main
}

serial_dispatch() {
  local input=""
  local cli_dump="" cli_batch="" cli_work="" cli_output="" cli_state="" cli_ple=""
  local cli_vision="" cli_prompt=""
  local -a positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --vision-bin)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_vision="$(cd "$2" && pwd)"
        shift 2
        ;;
      --prompt-bin)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_prompt="$(cd "$2" && pwd)"
        shift 2
        ;;
      --dump-dir|--dump-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_dump="$(cd "$2" && pwd)"
        shift 2
        ;;
      --batch-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_batch="$(cd "$2" && pwd)"
        shift 2
        ;;
      --work-dir|--work-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_work="$(cd "$2" && pwd)"
        shift 2
        ;;
      --output-dir|--output-root)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_output="$(cd "$2" && pwd)"
        shift 2
        ;;
      --state-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_state="$(cd "$2" && pwd)"
        shift 2
        ;;
      --ple-table-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path" >&2; exit 1; }
        cli_ple="$(cd "$2" && pwd)"
        shift 2
        ;;
      --)
        shift
        while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done
        break
        ;;
      -*)
        echo "ERROR: unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${#positional[@]}" -ge 2 ]]; then
    GEN_STEPS="${positional[1]}"
    export GEN_STEPS
  fi
  if [[ "${#positional[@]}" -ge 3 ]]; then
    NUM_ASSISTANT_TOKENS="${positional[2]}"
    export NUM_ASSISTANT_TOKENS
  fi
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
  [[ -n "${cli_ple}" ]] && PLE_TABLE_DIR="${cli_ple}"
  configure_output_layout "${OUTPUT_ROOT}" "${work_override}" "${state_override}"

  if [[ -n "${cli_batch}" ]]; then
    [[ -z "${input}" ]] || echo "WARN: positional path ignored when --batch-root is set" >&2
    run_batch "${cli_batch}"
    return
  fi

  if [[ -n "${cli_dump}" ]]; then
    [[ -z "${input}" ]] || echo "WARN: positional path ignored when --dump-dir is set" >&2
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
    configure_output_layout "${input}/om_output"
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
