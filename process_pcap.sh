#!/usr/bin/env bash
set -euo pipefail
PCAP="$1"
MASTER="${2:-$HOME/IDSProject/flows.csv}"
PY="$HOME/cicenv/bin/python"

TMP_DIR="$HOME/IDSProject/data/flows/tmp"
mkdir -p "$TMP_DIR" "$(dirname "$MASTER")"
OUT_TMP="$TMP_DIR/$(basename "$PCAP" .pcap).csv"

# Use cicflowmeter offline mode (your sniffer.py supports -f for pcap)
# NOTE: -c is a flag (no value); the next positional arg is the output path.
"$PY" -m cicflowmeter.sniffer -f "$PCAP" -c "$OUT_TMP"

# Append atomically; write header once
LOCK="${MASTER}.lock"
{
  flock 200
  if [ ! -s "$MASTER" ]; then
    head -n 1 "$OUT_TMP" >> "$MASTER"
  fi
  tail -n +2 "$OUT_TMP" >> "$MASTER"
} 200>>"$LOCK"

rm -f "$OUT_TMP"

