#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Realtime Hybrid IDS (uses tuned thresholds.json; original hybrid logic intact)
# - Watches realtime_merged_dataset.csv for appended flows
# - Uses RF (supervised) + IF (unsupervised) with confidence-gated hybrid
# - Low-confidence RF → consult IF:
#       IF score >= threshold → Benign
#       IF score <  threshold → Unknown  (escalate)
# - Loads thresholds from models/thresholds.json containing ONLY:
#     { "conf_gate": <float>, "if_complete_threshold": <float>, "if_partial_threshold": <float> }
# - Prints per-flow decisions with confidence and partial flag
# - Minimal realtime_metrics.csv logging

import os
import time
import json
import csv
import numpy as np
import pandas as pd
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CSV_PATH = "realtime_merged_dataset.csv"   # output from generate_dataset_all.py
MODEL_DIR = "models"

RF_PATH           = os.path.join(MODEL_DIR, "rf_model.pkl")
ISO_COMPLETE_PATH = os.path.join(MODEL_DIR, "iso_complete.pkl")
ISO_PARTIAL_PATH  = os.path.join(MODEL_DIR, "iso_partial.pkl")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH     = os.path.join(MODEL_DIR, "features.txt")
LABEL_MAP_PATH    = os.path.join(MODEL_DIR, "label_mapping.json")
THRESHOLDS_PATH   = os.path.join(MODEL_DIR, "thresholds.json")

# Defaults if thresholds.json is missing
DEFAULT_CONF_GATE = 0.85
DEFAULT_IF_THR    = 0.0

# Globals for loaded artefacts
RF = None
ISO_COMPLETE = None
ISO_PARTIAL = None
SCALER = None
FEATURES = None
STR_TO_INT = {}
INT_TO_STR = {}

# Tunable thresholds (populated from thresholds.json when present)
CONF_GATE = DEFAULT_CONF_GATE
IF_THR_COMPLETE = DEFAULT_IF_THR
IF_THR_PARTIAL  = DEFAULT_IF_THR

# Metrics logging controls
ATTACKER_IP = os.getenv("ATTACKER_IP", "10.128.0.8")
METRICS_CSV_PATH = os.path.join("evaluation_results", "realtime_metrics.csv")

# File-watching state
PROCESSED_ROWS = 0
LAST_MTIME = 0.0


def _load_list(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_model_artifacts():
    """Load RF/IF models, scaler, feature list, and label map."""
    global RF, ISO_COMPLETE, ISO_PARTIAL, SCALER, FEATURES, STR_TO_INT, INT_TO_STR

    required = [RF_PATH, ISO_COMPLETE_PATH, ISO_PARTIAL_PATH, SCALER_PATH, FEATURES_PATH, LABEL_MAP_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required artefacts: {missing}")

    RF = joblib.load(RF_PATH)
    ISO_COMPLETE = joblib.load(ISO_COMPLETE_PATH)
    ISO_PARTIAL = joblib.load(ISO_PARTIAL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    FEATURES = _load_list(FEATURES_PATH)

    # Prefer the scaler’s recorded feature names/order if present
    if hasattr(SCALER, "feature_names_in_"):
        scaler_feats = list(SCALER.feature_names_in_)
        if scaler_feats != FEATURES:
            print(f"[WARN] features.txt ({len(FEATURES)}) != scaler.feature_names_in_ "
                  f"({len(scaler_feats)}). Using scaler order for safety.")
        FEATURES[:] = scaler_feats  # enforce scaler's training-time order

    with open(LABEL_MAP_PATH, "r") as f:
        STR_TO_INT = json.load(f)
    # Reverse map for printing human labels if RF.classes_ are ints
    INT_TO_STR = {int(v): str(k) for k, v in STR_TO_INT.items()}

    print("Loaded artefacts:")
    print(f"   • RF classes: {getattr(RF, 'classes_', None)}")
    print(f"   • Features ({len(FEATURES)}): first 5 → {FEATURES[:5]}")
    print(f"   • Labels: {STR_TO_INT}")


def _load_thresholds():
    """
    Load tuned thresholds from models/thresholds.json.
    Expected minimal schema:
      {
        "conf_gate": <float>,
        "if_complete_threshold": <float>,
        "if_partial_threshold": <float>
      }
    If file is missing, fall back to defaults.
    """
    global CONF_GATE, IF_THR_COMPLETE, IF_THR_PARTIAL

    # Reset to defaults first
    CONF_GATE = DEFAULT_CONF_GATE
    IF_THR_COMPLETE = DEFAULT_IF_THR
    IF_THR_PARTIAL = DEFAULT_IF_THR

    if not os.path.exists(THRESHOLDS_PATH):
        print(f"{THRESHOLDS_PATH} not found → using defaults: "
              f"conf_gate={CONF_GATE:.2f}, if_thr={DEFAULT_IF_THR:.2f}")
        return

    try:
        with open(THRESHOLDS_PATH, "r") as f:
            data = json.load(f)

        # Strictly use only the three fields written by tune_thresholds.py
        CONF_GATE = float(data["conf_gate"])
        IF_THR_COMPLETE = float(data["if_complete_threshold"])
        IF_THR_PARTIAL  = float(data["if_partial_threshold"])

        print(f"Loaded tuned thresholds from {THRESHOLDS_PATH}: "
              f"conf_gate={CONF_GATE:.3f}, if_complete={IF_THR_COMPLETE:.6f}, if_partial={IF_THR_PARTIAL:.6f}")
    except Exception as e:
        # If anything goes wrong, keep defaults and warn
        print(f"[WARN] Failed to parse {THRESHOLDS_PATH} ({e}) → using defaults: "
              f"conf_gate={CONF_GATE:.2f}, if_thr={DEFAULT_IF_THR:.2f}")


def _prepare_vector(row: pd.Series):
    """
    Build a one-row DataFrame with correct column names in canonical order,
    align to scaler.feature_names_in_ (if present), then scale.
    """
    # Build dict in canonical order
    data = {}
    for col in FEATURES:
        v = row.get(col, 0.0)
        try:
            data[col] = float(v) if pd.notna(v) else 0.0
        except Exception:
            data[col] = 0.0

    X_df = pd.DataFrame([data])

    # Align to the scaler's recorded training-time names if available
    if hasattr(SCALER, "feature_names_in_"):
        X_df = X_df.reindex(columns=SCALER.feature_names_in_, fill_value=0.0)

    return SCALER.transform(X_df)  # returns numpy array


def _rf_predict(Xs: np.ndarray):
    """Return (label_str, confidence) from RF probabilities."""
    if not hasattr(RF, "predict_proba"):
        raise AttributeError("RF model has no predict_proba; ensure RandomForestClassifier was trained.")

    proba = RF.predict_proba(Xs)
    idx = int(np.argmax(proba, axis=1)[0])
    cls_value = RF.classes_[idx]
    label_str = INT_TO_STR.get(int(cls_value), "Unknown/Anomaly") if isinstance(cls_value, (int, np.integer)) else str(cls_value)
    conf = float(np.max(proba, axis=1)[0])
    return label_str, conf

def _if_score(Xs: np.ndarray, is_partial: int):
    """IsolationForest decision_function score (higher is more normal/inlier)."""
    model = ISO_PARTIAL if is_partial == 1 else ISO_COMPLETE
    return float(model.decision_function(Xs)[0])


def _hybrid_decision(rf_label: str, rf_conf: float, if_score: float, is_partial: int):
    """
    Desired policy:
      - If RF confidence < gate → consult IF:
            if_score >= IF_threshold → Benign
            else                   → Unknown/Anomaly  (escalate)
      - Else accept RF prediction directly.
    """
    if rf_conf < CONF_GATE:
        thr = IF_THR_PARTIAL if is_partial == 1 else IF_THR_COMPLETE
        if if_score >= thr:
            return "Benign", "routed_to_IF_inlier"
        else:
            return "Unknown/Anomaly", "routed_to_IF_outlier"   # escalate, don't keep rf_label
    else:
        return rf_label, "rf_confident"


def _format_flow(row: pd.Series):
    src = f"{row.get('src_ip','?')}:{row.get('src_port','?')}"
    dst = f"{row.get('dst_ip','?')}:{row.get('dst_port','?')}"
    proto = row.get('protocol','?')
    return f"{src} → {dst} ({proto})"


def _maybe_log_metrics(row: pd.Series, final_label: str, rf_label: str, rf_conf: float,
                       if_score: float, route: str, is_partial: int):
    """Append a CSV row with minimal runtime metrics (kept robust, never breaks detection path)."""
    try:
        ts = time.time()
        thr = IF_THR_PARTIAL if is_partial == 1 else IF_THR_COMPLETE
        unsup_verdict = "inlier" if if_score >= thr else "outlier"

        os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
        is_new_file = not os.path.exists(METRICS_CSV_PATH)

        with open(METRICS_CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if is_new_file:
                w.writerow([
                    "ts","final_label","rf_label","rf_conf","if_score",
                    "route","unsup_verdict","thr_in","thr_out","conf_gate",
                    "is_partial","src_ip","src_port","dst_ip","dst_port","latency_ms"
                ])
            w.writerow([
                f"{ts:.6f}",
                final_label,
                rf_label,
                f"{rf_conf:.6f}",
                f"{if_score:.6f}",
                route,
                unsup_verdict,
                f"{thr:.6f}",  # thr_in (single-value compat)
                f"{thr:.6f}",  # thr_out (kept for compat)
                f"{CONF_GATE:.3f}",
                int(is_partial),
                row.get("src_ip",""),
                row.get("src_port",""),
                row.get("dst_ip",""),
                row.get("dst_port",""),
                ""  # latency_ms (not measured here)
            ])
    except Exception:
        pass  # Never let logging affect detection path


def _predict_row(row: pd.Series):
    Xs = _prepare_vector(row)
    rf_label, rf_conf = _rf_predict(Xs)
    is_partial = int(row.get("is_partial", 0)) if pd.notna(row.get("is_partial", 0)) else 0
    score = _if_score(Xs, is_partial)
    final_label, route = _hybrid_decision(rf_label, rf_conf, score, is_partial)
    meta = _format_flow(row)
    print(f"FINAL RESULT: {final_label} (conf={rf_conf:.3f}, rf={rf_label}, IF={score:.3f}, route={route}) | {meta} (partial={is_partial})")
    _maybe_log_metrics(row, final_label, rf_label, rf_conf, score, route, is_partial)


def _process_new_rows():
    """Read any newly appended rows and process them in order."""
    global PROCESSED_ROWS, LAST_MTIME
    try:
        if not os.path.exists(CSV_PATH):
            return
        mtime = os.path.getmtime(CSV_PATH)
        if mtime == LAST_MTIME:
            return
        # small delay to avoid partial writes
        time.sleep(0.05)
        df = pd.read_csv(CSV_PATH)
        LAST_MTIME = mtime
    except Exception:
        return

    if df.empty:
        return

    total = len(df)
    if PROCESSED_ROWS > total:
        # file was truncated/rotated; start over
        PROCESSED_ROWS = 0
    if PROCESSED_ROWS == total:
        return

    new = df.iloc[PROCESSED_ROWS:]
    PROCESSED_ROWS = total

    for _, row in new.iterrows():
        try:
            _predict_row(row)
        except Exception as e:
            print(f"[WARN] prediction failed for a row: {e}")


class FlowHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and os.path.abspath(event.src_path) == os.path.abspath(CSV_PATH):
            _process_new_rows()


def main():
    _load_model_artifacts()
    _load_thresholds()

    # Snapshot of active operating point
    print(f"Active operating point → conf_gate={CONF_GATE:.3f}, "
          f"if_thr_complete={IF_THR_COMPLETE:.6f}, if_thr_partial={IF_THR_PARTIAL:.6f}")

    # Prime with existing rows (if any), so we only process new appends
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            global PROCESSED_ROWS, LAST_MTIME
            PROCESSED_ROWS = len(df)
            LAST_MTIME = os.path.getmtime(CSV_PATH)
            if PROCESSED_ROWS:
                print(f"Primed with existing rows: {PROCESSED_ROWS}")
        except Exception:
            pass

    observer = Observer()
    handler = FlowHandler()
    observer.schedule(handler, path=".", recursive=False)
    observer.start()

    print(f"Hybrid Realtime IDS is running. Watching {CSV_PATH} ... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
