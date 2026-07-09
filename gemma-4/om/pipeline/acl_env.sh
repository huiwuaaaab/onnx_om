#!/usr/bin/env bash
# CANN / pyACL runtime env — shared by msame launcher and resident OM workers.
#
# Usage:
#   source pipeline/acl_env.sh
#   python3 -c "import acl"
#
# Override:
#   ASCEND_ENV_SH=/path/to/set_env.sh source pipeline/acl_env.sh

: "${ACL_ENV_LOADED:=0}"
[[ "${ACL_ENV_LOADED}" == "1" ]] && return 0

export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-1}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"

_acl_source_env() {
  local env_sh="$1"
  [[ -f "${env_sh}" ]] || return 1
  # shellcheck source=/dev/null
  source "${env_sh}"
  return 0
}

ACL_ENV_SOURCE=""
if [[ -n "${ASCEND_ENV_SH:-}" ]]; then
  _acl_source_env "${ASCEND_ENV_SH}" && ACL_ENV_SOURCE="${ASCEND_ENV_SH}" || {
    echo "ERROR: ASCEND_ENV_SH not found: ${ASCEND_ENV_SH}" >&2
    return 1
  }
else
  for env_sh in \
    /var/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/nnrt/set_env.sh \
    /opt/ascend/nnrt/set_env.sh \
    /opt/ascend/nnrt/latest/set_env.sh \
    "${HOME}/Ascend/ascend-toolkit/set_env.sh" \
    "${HOME}/Ascend/nnrt/set_env.sh"
  do
    if _acl_source_env "${env_sh}"; then
      ACL_ENV_SOURCE="${env_sh}"
      break
    fi
  done
fi

_acl_prepend_path() {
  local p="$1"
  [[ -d "${p}" ]] || return 0
  case ":${PYTHONPATH:-}:" in
    *":${p}:"*) ;;
    *) export PYTHONPATH="${p}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

# AOS / MDC / Atlas: ensure acl.so is on PYTHONPATH even if set_env.sh missed it.
for p in \
  /usr/local/Ascend/ascend-toolkit/latest/python/site-packages \
  /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages \
  /usr/local/Ascend/ascend-toolkit/latest/acllib/python/site-packages \
  /usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe \
  /home/data/miniD/driver/lib64 \
  /var/ascend/python/site-packages \
  /opt/ascend/python/site-packages
do
  if [[ -d "${p}" ]] && { [[ -f "${p}/acl.so" ]] || compgen -G "${p}/acl*.so" >/dev/null; }; then
    _acl_prepend_path "${p}"
  fi
done

if [[ -z "${LD_LIBRARY_PATH:-}" && -d /usr/local/Ascend/acllib/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/Ascend/acllib/lib64"
fi

if [[ -d /usr/local/Ascend/acllib/lib64 ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/local/Ascend/acllib/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="/usr/local/Ascend/acllib/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

if [[ -z "${LD_LIBRARY_PATH:-}" && -d /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub ]]; then
  export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub"
fi

export ACL_ENV_LOADED=1
export ACL_ENV_SOURCE

pick_acl_python() {
  local py c candidates=()

  if [[ -n "${ACL_PYTHON:-}" ]]; then
    echo "${ACL_PYTHON}"
    return 0
  fi

  while IFS= read -r c; do
    [[ -n "${c}" ]] && candidates+=("${c}")
  done <<EOF
$(command -v python3 2>/dev/null || true)
$(command -v python3.11 2>/dev/null || true)
$(command -v python3.10 2>/dev/null || true)
$(command -v python3.9 2>/dev/null || true)
$(command -v python3.8 2>/dev/null || true)
$(command -v python3.7 2>/dev/null || true)
/usr/local/Ascend/ascend-toolkit/latest/bin/python3
/usr/local/Ascend/ascend-toolkit/latest/python/bin/python3
/usr/local/Ascend/ascend-toolkit/latest/tools/msame/python/bin/python3
EOF

  if [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    while IFS= read -r c; do
      candidates+=("${c}")
    done < <(find /usr/local/Ascend/ascend-toolkit/latest -maxdepth 5 -type f -name 'python3' 2>/dev/null | head -20)
  fi

  for py in "${candidates[@]}"; do
    [[ -n "${py}" && -x "${py}" ]] || continue
    if "${py}" -c "import acl" >/dev/null 2>&1; then
      echo "${py}"
      return 0
    fi
  done
  echo "python3"
}

acl_import_ok() {
  local py
  py="$(pick_acl_python)"
  "${py}" -c "import acl" >/dev/null 2>&1
}
