#!/usr/bin/env bash
# Shared static input path resolution for Gemma-4 OM pipeline scripts.
#
# New layout:
#   single:  om/vision_bin/  +  om/prompt_bin/
#   batch:   batch/<stem>/vision_bin/  +  shared om/prompt_bin/
#
# Legacy: dump/vision/ + dump/llm_preblock/

om_paths_script_dir() {
  if [[ -n "${OM_DIR:-}" ]]; then
    echo "${OM_DIR}"
  elif [[ -n "${SCRIPT_DIR:-}" ]]; then
    echo "${SCRIPT_DIR}"
  else
    echo "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
}

om_paths_shared_prompt_bin() {
  echo "${SHARED_PROMPT_BIN:-$(om_paths_script_dir)/prompt_bin}"
}

resolve_vision_bin() {
  local base="$1"
  [[ -d "${base}" ]] || return 1
  if [[ -f "${base}/pixel_values.bin" && -f "${base}/image_position_ids.bin" ]]; then
    echo "${base}"; return 0
  fi
  if [[ -f "${base}/vision_bin/pixel_values.bin" && -f "${base}/vision_bin/image_position_ids.bin" ]]; then
    echo "${base}/vision_bin"; return 0
  fi
  if [[ -f "${base}/vision/pixel_values.bin" && -f "${base}/vision/image_position_ids.bin" ]]; then
    echo "${base}/vision"; return 0
  fi
  if [[ -f "${base}/dump/vision/pixel_values.bin" && -f "${base}/dump/vision/image_position_ids.bin" ]]; then
    echo "${base}/dump/vision"; return 0
  fi
  return 1
}

resolve_prompt_bin() {
  local base="${1:-}"
  if [[ -n "${base}" && -d "${base}" ]]; then
    if [[ -f "${base}/input_ids.bin" ]]; then
      echo "${base}"; return 0
    fi
    if [[ -f "${base}/prompt_bin/input_ids.bin" ]]; then
      echo "${base}/prompt_bin"; return 0
    fi
    if [[ -f "${base}/llm_preblock/input_ids.bin" ]]; then
      echo "${base}/llm_preblock"; return 0
    fi
    if [[ -f "${base}/dump/llm_preblock/input_ids.bin" ]]; then
      echo "${base}/dump/llm_preblock"; return 0
    fi
  fi
  local shared
  shared="$(om_paths_shared_prompt_bin)"
  if [[ -f "${shared}/input_ids.bin" ]]; then
    echo "${shared}"; return 0
  fi
  return 1
}

sync_om_input_env() {
  DUMP_VISION="${VISION_BIN}"
  DUMP_PREBLOCK="${PROMPT_BIN}"
  DUMP_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}}"
}

configure_default_single_inputs() {
  local root
  root="$(om_paths_script_dir)"
  VISION_BIN="$(resolve_vision_bin "${root}")" || {
    echo "ERROR: missing ${root}/vision_bin/pixel_values.bin (+ image_position_ids.bin)" >&2
    echo "  run: python dump_vision_om_inputs.py --image path/img.jpg" >&2
    return 1
  }
  PROMPT_BIN="$(resolve_prompt_bin "${root}")" || {
    echo "ERROR: missing prompt bins under ${root}/prompt_bin" >&2
    echo "  run: python dump_llm_preblock_inputs.py --prompt '...'" >&2
    return 1
  }
  INPUT_ROOT="${root}"
  sync_om_input_env
}

diagnose_item_inputs() {
  local item_dir="$1"
  local shared v_paths p_paths
  shared="$(om_paths_shared_prompt_bin)"

  if resolve_vision_bin "${item_dir}" &>/dev/null; then
    :
  else
    v_paths="${item_dir}/vision_bin/pixel_values.bin (+ image_position_ids.bin)"
    v_paths+=", ${item_dir}/dump/vision/pixel_values.bin"
    echo "missing vision: need one of ${v_paths}"
    return 0
  fi

  if resolve_prompt_bin "${item_dir}" &>/dev/null; then
    echo "ok"
    return 0
  fi

  p_paths="${item_dir}/prompt_bin/input_ids.bin, ${shared}/input_ids.bin"
  p_paths+=", ${item_dir}/dump/llm_preblock/input_ids.bin"
  echo "missing prompt: need one of ${p_paths} (run dump_llm_preblock_inputs.py)"
}

configure_item_inputs() {
  local item_dir="$1"
  VISION_BIN="$(resolve_vision_bin "${item_dir}")" || {
    echo "ERROR: missing vision_bin for ${item_dir}" >&2
    diagnose_item_inputs "${item_dir}" >&2
    return 1
  }
  PROMPT_BIN="$(resolve_prompt_bin "${item_dir}")" || {
    echo "ERROR: missing prompt_bin for ${item_dir}" >&2
    diagnose_item_inputs "${item_dir}" >&2
    return 1
  }
  INPUT_ROOT="${item_dir}"
  OUTPUT_ROOT="${item_dir}/om_output"
  WORK_ROOT="${OUTPUT_ROOT}/work"
  STATE_DIR="${OUTPUT_ROOT}/state"
  sync_om_input_env
}

configure_input_root() {
  local root="$1"
  VISION_BIN="$(resolve_vision_bin "${root}")" || {
    echo "ERROR: no vision bins under ${root}" >&2
    return 1
  }
  PROMPT_BIN="$(resolve_prompt_bin "${root}")" || {
    echo "ERROR: no prompt bins for ${root}" >&2
    return 1
  }
  INPUT_ROOT="${root}"
  sync_om_input_env
}

is_batch_root() {
  local root="$1" d
  [[ -d "${root}" ]] || return 1
  for d in "${root}"/*; do
    [[ -d "${d}" ]] || continue
    resolve_vision_bin "${d}" &>/dev/null && return 0
  done
  return 1
}

check_om_inputs() {
  local missing=0 f
  for f in "${VISION_BIN}/pixel_values.bin" \
           "${VISION_BIN}/image_position_ids.bin" \
           "${PROMPT_BIN}/input_ids.bin" \
           "${PROMPT_BIN}/attention_mask.bin" \
           "${PROMPT_BIN}/per_layer_inputs.bin" \
           "${PROMPT_BIN}/position_ids.bin"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; missing=1; }
  done
  if (( missing )); then
    echo "  vision: python dump_vision_om_inputs.py --image path/img.jpg" >&2
    echo "  prompt: python dump_llm_preblock_inputs.py --prompt '...'" >&2
    return 1
  fi
  return 0
}
