#!/bin/bash
# Copy ONLY the read-only Drive remote [gdrive-ro] of this Mac's rclone config to one pod
# (/root/.config/rclone/rclone.conf, mode 600). The token is never printed. Use only with the user's approval.
# Usage: put_rclone.sh <tag>
set -euo pipefail
source "$(dirname "$0")/common.sh"
TAG=$1
CONF=$(rclone config file | tail -1)
SECTION=$(python3 - "$CONF" <<'PY'
import sys, configparser
c = configparser.RawConfigParser(); c.read(sys.argv[1])
if "gdrive-ro" not in c or c["gdrive-ro"].get("scope") != "drive.readonly":
    sys.exit("no [gdrive-ro] remote with scope drive.readonly in the local rclone config")
print("[gdrive-ro]")
for k, v in c["gdrive-ro"].items():
    print(f"{k} = {v}")
PY
)
printf '%s\n' "$SECTION" | bash "$INFRA/rsh.sh" "$TAG" 'mkdir -p /root/.config/rclone && umask 077 && cat > /root/.config/rclone/rclone.conf'
echo "gdrive-ro remote installed on $TAG"
