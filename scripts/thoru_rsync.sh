#!/bin/bash
# Stream a local directory to ThorU via jump2 (files only, no pip).
#
# Usage:
#   export THORU_JUMP1=... THORU_JUMP1_PASS=...
#   export THORU_JUMP2=... THORU_JUMP2_PASS=...
#   export THORU_TARGET=... THORU_TARGET_PASS=...
#   ./scripts/thoru_rsync.sh ./InternVL3_5-1B /opt/vlm/internvl3_5
set -euo pipefail

SRC=${1:?src dir}
DEST=${2:?remote dest dir}

JUMP1="${THORU_JUMP1:?Set THORU_JUMP1, e.g. user@jump1.example.com}"
JUMP1_PASS="${THORU_JUMP1_PASS:?Set THORU_JUMP1_PASS}"
JUMP2="${THORU_JUMP2:?Set THORU_JUMP2, e.g. user@jump2.example.com}"
JUMP2_PASS="${THORU_JUMP2_PASS:?Set THORU_JUMP2_PASS}"
TARGET="${THORU_TARGET:?Set THORU_TARGET, e.g. user@device.example.com}"
TARGET_PASS="${THORU_TARGET_PASS:?Set THORU_TARGET_PASS}"

SSH_BASE=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30)

echo "[$(date +%H:%M:%S)] tar -> ThorU:${DEST}"
tar cf - -C "$SRC" . | sshpass -p "${JUMP1_PASS}" ssh "${SSH_BASE[@]}" \
  -o ProxyCommand="sshpass -p '${JUMP1_PASS}' ssh ${SSH_BASE[*]} -W %h:%p ${JUMP1}" \
  "${JUMP2}" \
  "DISPLAY=:0 SSH_ASKPASS=/tmp/thoru_pass.sh SSH_ASKPASS_REQUIRE=force ssh ${SSH_BASE[*]} -o PreferredAuthentications=password -o PubkeyAuthentication=no ${TARGET} 'mkdir -p \"${DEST}\" && tar xf - -C \"${DEST}\"'"
echo "[$(date +%H:%M:%S)] done -> ${DEST}"
