#!/bin/bash
# Stream a local directory to ThorU via jump2 (files only, no pip).
# Usage:
#   ./scripts/thoru_rsync.sh /local/src /cus_app_data/guanxj/qwen3-vl
set -euo pipefail

SRC=${1:?src dir}
DEST=${2:?remote dest dir}

JUMP1="test@10.10.84.172"
JUMP1_PASS="${THORU_JUMP1_PASS:-123456}"
JUMP2="root@192.168.17.104"
TARGET="nvidia@192.168.11.103"
TARGET_PASS="${THORU_TARGET_PASS:-nvidia}"

SSH_BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30)

echo "[$(date +%H:%M:%S)] tar -> ThorU:${DEST}"
tar cf - -C "$SRC" . | sshpass -p "${JUMP1_PASS}" ssh "${SSH_BASE[@]}" \
  -o ProxyCommand="sshpass -p '${JUMP1_PASS}' ssh ${SSH_BASE[*]} -W %h:%p ${JUMP1}" \
  "${JUMP2}" \
  "DISPLAY=:0 SSH_ASKPASS=/tmp/thoru_pass.sh SSH_ASKPASS_REQUIRE=force ssh ${SSH_BASE[*]} -o PreferredAuthentications=password -o PubkeyAuthentication=no ${TARGET} 'mkdir -p \"${DEST}\" && tar xf - -C \"${DEST}\"'"
echo "[$(date +%H:%M:%S)] done -> ${DEST}"
