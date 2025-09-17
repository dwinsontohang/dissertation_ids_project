#!/usr/bin/env python3
import os
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# === Configuration ===
CSV_PATH = "flows.csv"
MERGED_OUTPUT = "realtime_merged_dataset.csv"

# Flow pair cache & offsets
FLOW_CACHE = {}           # key -> (row_series, first_seen_utc)
PROCESSED_LINES = 0       # how many data lines (rows) have been consumed
LAST_FILE_SIZE = 0

# Timeouts & housekeeping
CACHE_TIMEOUT = 15        # seconds before we flush unmatched (one-direction) flow as partial
FLUSH_INTERVAL = 2        # seconds between background flush checks
MIN_TOTAL_PKTS = 1        # optional: skip rows with very few packets (0/1) if desired

# === Features to keep + metadata ===
KEEP_FEATURES = [
    'src_ip','dst_ip','src_port','dst_port','protocol','flow_duration',
    'tot_fwd_pkts','tot_bwd_pkts','totlen_fwd_pkts','totlen_bwd_pkts',
    'fwd_pkt_len_max','fwd_pkt_len_min','fwd_pkt_len_mean','fwd_pkt_len_std',
    'bwd_pkt_len_max','bwd_pkt_len_min','bwd_pkt_len_mean','bwd_pkt_len_std',
    'flow_byts_s','flow_pkts_s','flow_iat_mean','flow_iat_std','flow_iat_max','flow_iat_min',
    'fwd_iat_tot','fwd_iat_mean','fwd_iat_std','fwd_iat_max','fwd_iat_min',
    'bwd_iat_tot','bwd_iat_mean','bwd_iat_std','bwd_iat_max','bwd_iat_min',
    'fwd_psh_flags','bwd_psh_flags','fwd_urg_flags','bwd_urg_flags',
    'fwd_header_len','bwd_header_len','fwd_pkts_s','bwd_pkts_s',
    'pkt_len_min','pkt_len_max','pkt_len_mean','pkt_len_std','pkt_len_var',
    'fin_flag_cnt','syn_flag_cnt','rst_flag_cnt','psh_flag_cnt','ack_flag_cnt','urg_flag_cnt',
    'cwe_flag_count','ece_flag_cnt','down_up_ratio','pkt_size_avg',
    'fwd_seg_size_avg','bwd_seg_size_avg','fwd_byts_b_avg','fwd_pkts_b_avg','fwd_blk_rate_avg',
    'bwd_byts_b_avg','bwd_pkts_b_avg','bwd_blk_rate_avg',
    'subflow_fwd_pkts','subflow_fwd_byts','subflow_bwd_pkts','subflow_bwd_byts',
    'init_fwd_win_byts','init_bwd_win_byts','fwd_act_data_pkts','fwd_seg_size_min',
    'active_mean','active_std','active_max','active_min',
    'idle_mean','idle_std','idle_max','idle_min'
]

# We’ll output columns in this exact order: metadata + all KEEP_FEATURES (without duplicates) + 'is_partial'
OUTPUT_COLUMNS = (
    ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
    + [c for c in KEEP_FEATURES if c not in {'src_ip','dst_ip','src_port','dst_port','protocol'}]
    + ['is_partial']
)

# === Helpers ===

def _now_utc():
    # timezone-aware UTC (no deprecation warnings)
    return datetime.now(timezone.utc)

def _is_numeric(x):
    return isinstance(x, (int, float, np.number))

def _safe_get(series, key, default=0):
    # pandas Series.get returns default if label missing
    try:
        return series.get(key, default)
    except Exception:
        return default

def recreate_output():
    """Always (re)create the output CSV with just the header."""
    try:
        if os.path.exists(MERGED_OUTPUT):
            os.remove(MERGED_OUTPUT)
            print(f"[INIT] Removed existing {MERGED_OUTPUT}")
    except Exception as e:
        print(f"[INIT][WARN] Could not remove {MERGED_OUTPUT}: {e}")

    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(MERGED_OUTPUT, index=False)
    print(f"[INIT] Created fresh output file with header: {MERGED_OUTPUT}")

def _write_merged_flow(flow_dict):
    # Enforce stable column order and ensure missing keys become 0/empty
    row = {col: flow_dict.get(col, 0) for col in OUTPUT_COLUMNS}
    pd.DataFrame([row]).to_csv(MERGED_OUTPUT, mode='a', index=False, header=False)
    print(f"[WRITE] Flow written to {MERGED_OUTPUT} (is_partial={row['is_partial']})")

def _merge_flows(fwd: pd.Series, bwd: 'pd.Series | None', is_partial: int):
    """
    Merge forward + backward into a single bidirectional record.
    For *_fwd* / *_bwd* families we take the appropriate direction.
    For other numeric features we SUM forward+backward.
    For non-numeric, prefer forward, else backward, else 0.
    """
    merged = {}

    # Metadata always from forward direction
    for meta in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']:
        merged[meta] = _safe_get(fwd, meta, 0)

    for col in KEEP_FEATURES:
        if col in merged:
            continue  # already set metadata

        fwd_val = _safe_get(fwd, col, 0)
        bwd_val = _safe_get(bwd, col, 0) if bwd is not None else 0

        # Direction-specific families
        if col.startswith("tot_fwd") or col.startswith("fwd_"):
            merged[col] = fwd_val
        elif col.startswith("tot_bwd") or col.startswith("bwd_"):
            merged[col] = bwd_val
        else:
            # For non-directional numeric features, sum
            if _is_numeric(fwd_val) and _is_numeric(bwd_val):
                merged[col] = (fwd_val or 0) + (bwd_val or 0)
            else:
                merged[col] = fwd_val if fwd_val not in (None, "") else (bwd_val or 0)

    # Place is_partial at the end
    merged['is_partial'] = is_partial
    return merged

def _flush_expired_flows():
    """Flush any cached flows that have exceeded CACHE_TIMEOUT without seeing a reverse."""
    if not FLOW_CACHE:
        return

    now = _now_utc()
    expired = []
    for key, (fwd_series, first_seen) in FLOW_CACHE.items():
        age = (now - first_seen).total_seconds()
        if age > CACHE_TIMEOUT:
            # Write as partial (no backward partner)
            merged = _merge_flows(fwd_series, None, is_partial=1)
            _write_merged_flow(merged)
            expired.append(key)
            print(f"[TIMEOUT] Flushed partial after {age:.1f}s: {key}")

    for key in expired:
        FLOW_CACHE.pop(key, None)

# === Background flusher thread ===
class Flusher(threading.Thread):
    def __init__(self, interval=FLUSH_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        print(f"[FLUSHER] Started background flusher every {self.interval}s")
        while not self._stop.is_set():
            try:
                _flush_expired_flows()
            except Exception as e:
                print(f"[FLUSHER][ERROR] {e}")
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

# === Watchdog Handler ===
class FlowHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global PROCESSED_LINES, LAST_FILE_SIZE

        if not event.src_path.endswith(os.path.basename(CSV_PATH)):
            return

        # Detect rotations/truncation by file size shrink
        try:
            current_size = os.path.getsize(CSV_PATH)
        except FileNotFoundError:
            print("[WARN] flows.csv not found (maybe rotating).")
            return

        if current_size < LAST_FILE_SIZE:
            print("[INFO] Detected flows.csv shrink/rotate. Resetting offsets & cache.")
            PROCESSED_LINES = 0
            FLOW_CACHE.clear()
        LAST_FILE_SIZE = current_size

        # Read only new rows (skip header + already processed data rows)
        try:
            df = pd.read_csv(
                CSV_PATH,
                skiprows=range(1, PROCESSED_LINES + 1),
                low_memory=False
            )
        except pd.errors.EmptyDataError:
            print("[WARN] No columns to parse (file being written). Skipping this tick.")
            return
        except Exception as e:
            print(f"[WARN] Could not read flows.csv: {e}")
            return

        if df.empty:
            # nothing new yet
            return

        # Validate expected core columns exist
        core_cols = {'src_ip','dst_ip','src_port','dst_port','protocol'}
        if not core_cols.issubset(df.columns):
            print(f"[WARN] Missing core columns in flows.csv: {core_cols - set(df.columns)}")
            return

        # Optionally skip rows with almost no packets (helps avoid degenerate rows)
        if 'tot_fwd_pkts' in df.columns and 'tot_bwd_pkts' in df.columns:
            before = len(df)
            df = df[(df['tot_fwd_pkts'].fillna(0) + df['tot_bwd_pkts'].fillna(0)) >= MIN_TOTAL_PKTS]
            dropped = before - len(df)
            if dropped:
                print(f"[INFO] Dropped {dropped} rows with < {MIN_TOTAL_PKTS} total packets")

        new_rows = len(df)
        PROCESSED_LINES += new_rows
        print(f"[INFO] New flows received: {new_rows}")

        now = _now_utc()

        # Process each flow: merge if reverse exists; else cache it
        for _, flow in df.iterrows():
            try:
                meta_key = (flow['src_ip'], flow['dst_ip'], flow['src_port'], flow['dst_port'], flow['protocol'])
                reverse_key = (flow['dst_ip'], flow['src_ip'], flow['dst_port'], flow['src_port'], flow['protocol'])

                if reverse_key in FLOW_CACHE:
                    rev_series, first_seen = FLOW_CACHE.pop(reverse_key)
                    merged = _merge_flows(rev_series, flow, is_partial=0)  # rev_series is fwd, flow is bwd
                    _write_merged_flow(merged)
                    print(f"[DEBUG] Merged and wrote: {reverse_key} <-> {meta_key}")
                else:
                    FLOW_CACHE[meta_key] = (flow, now)
                    print(f"[DEBUG] Cached forward-only, waiting for reverse: {meta_key}")
            except Exception as e:
                print(f"[ERROR] Failed to process row: {e}")

# === Main ===
if __name__ == "__main__":
    # Always start fresh on every run
    recreate_output()

    # Initialize file size for rotation detection
    if os.path.exists(CSV_PATH):
        LAST_FILE_SIZE = os.path.getsize(CSV_PATH)

    # Start background flusher
    flusher = Flusher(interval=FLUSH_INTERVAL)
    flusher.start()

    # Start watchdog observer
    observer = Observer()
    handler = FlowHandler()
    observer.schedule(handler, path=".", recursive=False)
    observer.start()

    print("Real-time Bi-Directional Flow Merger with Timeout is running. Watching flows.csv ... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping flusher and observer...")
        flusher.stop()
        observer.stop()
    observer.join()

