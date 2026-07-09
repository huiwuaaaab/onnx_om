#!/bin/bash
# ThorU SSH helper (double jump via jump2 + SSH_ASKPASS).
#
# Usage:
#   ./scripts/thoru_ssh.sh                    # interactive shell on ThorU
#   ./scripts/thoru_ssh.sh 'nvidia-smi'       # run remote command
#   ./scripts/thoru_ssh.sh jump2 'hostname'   # run on jump2 only
#
# Env (optional):
#   THORU_JUMP1_PASS  THORU_JUMP2_PASS  THORU_TARGET_PASS

set -euo pipefail

JUMP1="test@10.10.84.172"
JUMP1_PASS="${THORU_JUMP1_PASS:-123456}"
JUMP2="root@192.168.17.104"
JUMP2_PASS="${THORU_JUMP2_PASS:-123456}"
TARGET="nvidia@192.168.11.103"
TARGET_PASS="${THORU_TARGET_PASS:-nvidia}"

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
