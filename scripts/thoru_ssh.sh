#!/bin/bash
# ThorU SSH helper (double jump via jump2 + SSH_ASKPASS).
#
# Usage:
#   export THORU_JUMP1=user@jump1.example.com
#   export THORU_JUMP1_PASS=...
#   export THORU_JUMP2=user@jump2.example.com
#   export THORU_JUMP2_PASS=...
#   export THORU_TARGET=user@device.example.com
#   export THORU_TARGET_PASS=...
#   ./scripts/thoru_ssh.sh                    # interactive shell on ThorU
#   ./scripts/thoru_ssh.sh 'nvidia-smi'       # run remote command
#   ./scripts/thoru_ssh.sh jump2 'hostname'   # run on jump2 only

set -euo pipefail

JUMP1="${THORU_JUMP1:?Set THORU_JUMP1, e.g. user@jump1.example.com}"
JUMP1_PASS="${THORU_JUMP1_PASS:?Set THORU_JUMP1_PASS}"
JUMP2="${THORU_JUMP2:?Set THORU_JUMP2, e.g. user@jump2.example.com}"
JUMP2_PASS="${THORU_JUMP2_PASS:?Set THORU_JUMP2_PASS}"
TARGET="${THORU_TARGET:?Set THORU_TARGET, e.g. user@device.example.com}"
TARGET_PASS="${THORU_TARGET_PASS:?Set THORU_TARGET_PASS}"

SSH_BASE=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=15
)

ssh_jump1() {
  sshpass -p "${JUMP1_PASS}" ssh "${SSH_BASE[@]}" \
    -o ProxyCommand="sshpass -p '${JUMP1_PASS}' ssh ${SSH_BASE[*]} -W %h:%p ${JUMP1}" \
    "$@"
}

ssh_thoru() {
  local remote_cmd=$1
  ssh_jump1 "${JUMP2}" bash -s <<EOF
cat > /tmp/thoru_pass.sh <<'PASS'
#!/bin/sh
echo ${TARGET_PASS}
PASS
chmod +x /tmp/thoru_pass.sh
DISPLAY=:0 SSH_ASKPASS=/tmp/thoru_pass.sh SSH_ASKPASS_REQUIRE=force \\
  ssh ${SSH_BASE[*]} \\
    -o PreferredAuthentications=password \\
    -o PubkeyAuthentication=no \\
    ${TARGET} ${remote_cmd}
EOF
}

case "${1:-shell}" in
  jump2)
    shift
    ssh_jump1 "${JUMP2}" "$@"
    ;;
  shell)
    ssh_thoru ""
    ;;
  *)
    # quote for remote shell
    cmd=$(printf '%q ' "$@")
    ssh_thoru "${cmd}"
    ;;
esac
