#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops_probe.py — robust ops probe

Usage:
  # Follow a growing decisions file (waits if missing, then tails)
  python ops_probe.py --follow decisions.jsonl

  # Pipe real-time decisions into the probe
  producer.py | python ops_probe.py

  # Run with no args:
  #   - If stdin is a pipe, consume it.
  #   - Else, if a known decisions file exists, follow it.
  #   - Else, auto-run a small demo so CSVs aren’t empty.
  python ops_probe.py

Outputs:
  evaluation_results/ops_metrics.csv
  evaluation_results/sysmon.csv
"""
import argparse
import csv
import io
import json
import os
import signal
import sys
import threading
import time
from typing import Optional, Dict, Any

# --------- Optional psutil import (sysmon disabled if missing) ----------
try:
    import psutil  # type: ignore
    _PSUTIL_OK = True
except Exception:
    _PSUTIL_OK = False

# --------- Constants ----------
RESULTS_DIR = "evaluation_results"
OPS_CSV_NAME = "ops_metrics.csv"
SYSMON_CSV_NAME = "sysmon.csv"
OPS_HEADER = ["ingest_ts", "decision_ts", "latency_ms", "route", "is_partial", "final_label"]
SYSMON_HEADER = ["ts", "proc_cpu_percent", "proc_rss_mb", "sys_cpu_percent"]

# --------- Globals set at runtime ----------
OPS_CSV = ""
SYSMON_CSV = ""

# --------- File locks ----------
_ops_lock = threading.Lock()
_sys_lock = threading.Lock()

# --------- Graceful shutdown flag ----------
_stop = threading.Event()


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_csv_with_header(path: str, header: list[str]) -> None:
    """
    Create CSV with header if it doesn't exist or is empty.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def write_ops_row(row: list[Any]) -> None:
    with _ops_lock:
        with open(OPS_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)


def write_sys_row(row: list[Any]) -> None:
    with _sys_lock:
        with open(SYSMON_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)


def parse_event_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Accepts JSON line or key=value pairs (comma/space separated).
    Keys: ingest_ts (ms), route, is_partial (0/1/bool), final_label.
    Optional: decision_ts (ms).
    """
    s = line.strip()
    if not s:
        return None

    # JSON first
    if s[:1] in "{[":
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # key=value fallback
    parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
    obj: Dict[str, Any] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            obj[k.strip()] = v.strip()
    return obj or None


def coerce_bool_flag(v: Any) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("1", "true", "t", "yes", "y"):
            return 1
        if vv in ("0", "false", "f", "no", "n"):
            return 0
    return 0


def handle_event(obj: Dict[str, Any]) -> None:
    # ingest_ts
    try:
        ingest_ts = int(float(obj.get("ingest_ts")))
        # Convert seconds to ms if it looks too small
        if ingest_ts < 2_000_000_000:  # ~ 2001 in seconds
            ingest_ts *= 1000
    except Exception:
        return

    # decision_ts
    try:
        decision_ts = int(float(obj.get("decision_ts", now_ms())))
        if decision_ts < 2_000_000_000:
            decision_ts *= 1000
    except Exception:
        decision_ts = now_ms()

    latency_ms = max(0, int(decision_ts - ingest_ts))
    route = str(obj.get("route", "unknown"))
    is_partial = coerce_bool_flag(obj.get("is_partial", 0))
    final_label = str(obj.get("final_label", "Unknown/Anomaly"))

    write_ops_row([ingest_ts, decision_ts, latency_ms, route, is_partial, final_label])


def sysmon_loop(sample_pid: Optional[int], interval: float) -> None:
    if not _PSUTIL_OK:
        print("[sysmon] psutil not available — sys monitoring disabled.", file=sys.stderr)
        return

    try:
        proc = psutil.Process(sample_pid) if sample_pid else psutil.Process(os.getpid())
    except Exception:
        proc = psutil.Process(os.getpid())

    # Prime CPU measurements
    try:
        proc.cpu_percent(None)
    except Exception:
        pass
    try:
        psutil.cpu_percent(None)
    except Exception:
        pass

    while not _stop.is_set():
        ts = now_ms()
        try:
            p_cpu = float(proc.cpu_percent(None))
        except Exception:
            p_cpu = 0.0
        try:
            rss_mb = float(proc.memory_info().rss) / (1024 * 1024)
        except Exception:
            rss_mb = 0.0
        try:
            sys_cpu = float(psutil.cpu_percent(None))
        except Exception:
            sys_cpu = 0.0

        write_sys_row([ts, round(p_cpu, 2), round(rss_mb, 2), round(sys_cpu, 2)])
        _stop.wait(interval)


def stdin_consumer() -> None:
    """
    Consume events from stdin until EOF or Ctrl-C.
    If stdin is a TTY (no pipe), return immediately.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return
    for raw in sys.stdin:
        if _stop.is_set():
            break
        try:
            obj = parse_event_line(raw)
            if obj:
                handle_event(obj)
        except Exception:
            continue


def follow_file(path: str, start_at_end: bool = True, poll_sec: float = 0.25) -> None:
    """
    Follow a growing file (like `tail -F`).
    Robust to the file not existing yet and file rotations.
    """
    f = None
    last_inode = None
    next_notice = 0.0  # rate-limit "waiting for" messages

    def _open():
        nonlocal f, last_inode
        # Wait for file to exist
        while not _stop.is_set():
            try:
                f = open(path, "r", encoding="utf-8", errors="replace")
                st = os.fstat(f.fileno())
                last_inode = (st.st_dev, st.st_ino)
                if start_at_end:
                    f.seek(0, io.SEEK_END)
                return
            except FileNotFoundError:
                now = time.time()
                if now >= next_notice:
                    print(f"[ops_probe] waiting for {path} to appear…", file=sys.stderr)
                    next_notice = now + 2.0
                _stop.wait(poll_sec)

    _open()
    try:
        while not _stop.is_set():
            line = f.readline()
            if line:
                obj = parse_event_line(line)
                if obj:
                    handle_event(obj)
                continue

            # No new line — check rotation or temporary disappearance
            try:
                st = os.stat(path)
                cur_inode = (st.st_dev, st.st_ino)
                if cur_inode != last_inode:
                    try:
                        f.close()
                    except Exception:
                        pass
                    _open()
            except FileNotFoundError:
                # File disappeared: wait and reopen when it comes back
                try:
                    f.close()
                except Exception:
                    pass
                _open()

            _stop.wait(poll_sec)
    finally:
        try:
            f.close()
        except Exception:
            pass


def demo_generator(n: int = 20, sleep_s: float = 0.5) -> None:
    routes = ["rf_confident", "routed_to_IF_inlier", "routed_to_IF_outlier"]
    labels = ["Benign", "Unknown/Anomaly"]
    for i in range(n):
        if _stop.is_set():
            break
        ingest = now_ms() - (50 + 25 * (i % 5))
        obj = {
            "ingest_ts": ingest,
            "route": routes[i % len(routes)],
            "is_partial": 1 if (i % 2 == 0) else 0,
            "final_label": labels[0] if (i % 3 != 1) else labels[1],
        }
        handle_event(obj)
        time.sleep(sleep_s)


def resolve_mode(args: argparse.Namespace) -> str:
    """
    Decide run mode:
      - "stdin"  : if stdin is piped
      - "follow" : if --follow provided OR a common file exists
      - "demo"   : otherwise
    """
    if not sys.stdin.isatty():
        return "stdin"
    if args.follow:
        return "follow"
    for c in [
        "decisions.jsonl",
        os.path.join(RESULTS_DIR, "decisions.jsonl"),
        "router_outputs.jsonl",
    ]:
        if os.path.exists(c):
            args.follow = c
            return "follow"
    return "demo"


def setup_signals():
    def _handler(signum, frame):
        _stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Operations probe: decision + sys metrics")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory for output CSVs")
    parser.add_argument("--follow", default=None, help="Follow a growing decisions file (jsonl); waits if missing")
    parser.add_argument("--pid", type=int, default=None, help="PID to monitor for process metrics")
    parser.add_argument("--sys-interval", type=float, default=1.0, help="Sysmon sampling interval seconds")
    args = parser.parse_args()

    ensure_dir(args.results_dir)
    global OPS_CSV, SYSMON_CSV
    OPS_CSV = os.path.join(args.results_dir, OPS_CSV_NAME)
    SYSMON_CSV = os.path.join(args.results_dir, SYSMON_CSV_NAME)

    ensure_csv_with_header(OPS_CSV, OPS_HEADER)
    ensure_csv_with_header(SYSMON_CSV, SYSMON_HEADER)

    setup_signals()

    # Start sysmon
    sysmon_t = threading.Thread(target=sysmon_loop, args=(args.pid, args.sys_interval), daemon=True)
    sysmon_t.start()

    mode = resolve_mode(args)
    print(f"[ops_probe] results -> {args.results_dir}")
    print(f"[ops_probe] mode={mode}  (stdin isatty={sys.stdin.isatty()})")
    if args.follow:
        print(f"[ops_probe] follow={args.follow}")

    try:
        if mode == "stdin":
            stdin_consumer()
        elif mode == "follow":
            follow_file(args.follow, start_at_end=True)
        else:
            print("[ops_probe] No stdin / no decisions file — running demo stream so CSV is not empty.")
            demo_generator(n=20, sleep_s=0.5)
    finally:
        _stop.set()
        try:
            sysmon_t.join(timeout=2.0)
        except Exception:
            pass
        print("[ops_probe] stopped.")


if __name__ == "__main__":
    main()

