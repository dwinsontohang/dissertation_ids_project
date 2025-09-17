#!/usr/bin/env bash
set -euo pipefail

IFACE="${IFACE:-ens4}"
WORKERS="${WORKERS:-$(nproc)}"
ROLL_SEC="${ROLL_SEC:-3}"

PCAP_DIR="$HOME/IDSProject/data/pcaps"
TMP_DIR="$HOME/IDSProject/data/flows/tmp"
MASTER="$HOME/IDSProject/flows.csv"
WORKER="$HOME/IDSProject/process_pcap.sh"

# --- helpers ---
die() { echo "[-] $*" >&2; exit 1; }

for cmd in dumpcap inotifywait parallel; do
  command -v "$cmd" >/dev/null 2>&1 || die "Required command '$cmd' not found."
done
[ -x "$WORKER" ] || die "Worker script not executable: $WORKER"

# --- ensure dirs exist ---
mkdir -p "$PCAP_DIR" "$TMP_DIR" "$(dirname "$MASTER")"

# --- reset outputs as requested ---
echo "[*] Deleting old pcaps in $PCAP_DIR …"
find "$PCAP_DIR" -maxdepth 1 -type f -name '*.pcap' -delete || true

echo "[*] Recreating $MASTER …"
rm -f "$MASTER"
: > "$MASTER"   # create empty file

# --- start capture ---
echo "[*] Starting dumpcap on $IFACE (rotate every $ROLL_SEC s)…"
dumpcap -i "$IFACE" -B 32768 \
  -b duration:"$ROLL_SEC" -b files:200 \
  -w "$PCAP_DIR/cap_%F_%H-%M-%S.pcap" &
CAP_PID=$!
echo "[*] dumpcap PID: $CAP_PID"

cleanup() {
  echo "[*] Stopping dumpcap ($CAP_PID)…"
  kill "$CAP_PID" 2>/dev/null || true
  wait "$CAP_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- fanout workers ---
echo "[*] Spawning up to $WORKERS workers to convert pcaps to CSV…"
inotifywait -m -e close_write --format '%w%f' "$PCAP_DIR" \
| parallel -j "$WORKERS" --lb "$WORKER" {} "$MASTER"

