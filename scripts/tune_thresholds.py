#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json
from typing import List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# ----------------- utils -----------------

def load_features_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_paths(paths: List[str]):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Missing required artefacts:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(2)

def invert_label_map(label_map_path: str):
    """Return int->str class mapping from the JSON label map."""
    m = json.load(open(label_map_path, "r"))
    inv = {}
    for k, v in m.items():
        if isinstance(v, int):
            inv[int(v)] = str(k)
        else:
            try:
                inv[int(k)] = str(v)
            except Exception:
                pass
    return inv

def parse_percentiles(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        try:
            p = float(tok)
            if 0 <= p <= 100:
                out.append(p/100.0)
        except Exception:
            pass
    return sorted(set(out))

def bu_metrics(y_true_str: np.ndarray, y_pred_str: np.ndarray):
    """Compute FPR/FNR on Benign-vs-Unknown subset only."""
    mask = (y_true_str == "Benign") | (y_true_str == "Unknown")
    if not np.any(mask):
        return 0.0, 0.0
    yt = (y_true_str[mask] == "Unknown").astype(int)
    yp = (y_pred_str[mask] == "Unknown").astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return fpr, fnr

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    # Data & artefacts
    ap.add_argument("--val-csv", default="data/processed/val_dataset.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--features", default="models/features.txt")
    ap.add_argument("--label-map", default="models/label_mapping.json")
    ap.add_argument("--scaler", default="models/scaler.pkl")
    ap.add_argument("--rf", default="models/rf_model.pkl")
    ap.add_argument("--if-complete", default="models/iso_complete.pkl")
    ap.add_argument("--if-partial", default="models/iso_partial.pkl")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--partial-col", default="is_partial")
    ap.add_argument("--out", default="models/thresholds.json")

    # Gate search (practical range; 0.01 steps)
    ap.add_argument("--gate-min", type=float, default=0.75)
    ap.add_argument("--gate-max", type=float, default=0.90)
    ap.add_argument("--gate-step", type=float, default=0.01)

    # IF threshold candidates from moderate low benign percentiles (+ median for stability)
    ap.add_argument("--q-complete", default="5,10,15,20,50")
    ap.add_argument("--q-partial",  default="5,10,15,20,50")

    # Objective
    ap.add_argument("--metric", choices=["cost"], default="cost")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=1.0)
    ap.add_argument("--fpr-target", type=float, default=None,
                    help="If set, prefer solutions with BU-FPR <= target (e.g., 0.01 or 0.02).")

    # Decision-function floor (≥ 0 means at/above the IF boundary); small margin avoids borderline flips
    ap.add_argument("--min-if-threshold", type=float, default=0.02,
                    help="Minimum IF threshold in decision_function units (floor at/above 0).")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    ensure_paths([args.val_csv, args.features, args.label_map, args.scaler, args.rf,
                  args.if_complete, args.if_partial])
    os.makedirs(args.models_dir, exist_ok=True)

    # Load data & artefacts
    feats = load_features_list(args.features)
    df = pd.read_csv(args.val_csv)
    if args.label_col not in df.columns:
        print(f"[ERROR] Label column '{args.label_col}' not in validation CSV.", file=sys.stderr); sys.exit(2)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print(f"[ERROR] Validation CSV missing required features (first 8 shown): {missing[:8]}", file=sys.stderr); sys.exit(2)

    Xdf = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_true_raw = df[args.label_col].values
    is_partial = df[args.partial_col].values if args.partial_col in df.columns else np.zeros(len(df), dtype=int)

    scaler = joblib.load(args.scaler)
    rf = joblib.load(args.rf)
    if_complete = joblib.load(args.if_complete)
    if_partial = joblib.load(args.if_partial)

    id2label = invert_label_map(args.label_map)
    Xs = scaler.transform(Xdf)

    # RF predictions & confidences
    if not hasattr(rf, "predict_proba"):
        print("[ERROR] RF model lacks predict_proba; ensure RandomForestClassifier was trained.", file=sys.stderr)
        sys.exit(2)
    proba = rf.predict_proba(Xs)
    rf_classes = list(rf.classes_)
    idx = np.argmax(proba, axis=1)

    def to_str(v):
        if isinstance(v, (int, np.integer)) and int(v) in id2label:
            return id2label[int(v)]
        try:
            return id2label[int(v)]
        except Exception:
            return str(v)

    s_pred = np.array([to_str(rf_classes[i]) for i in idx], dtype=object)
    s_conf = np.max(proba, axis=1)

    # IF decision_function scores (higher = more normal/inlier; boundary at 0.0)
    scores_if = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        model = if_partial if (is_partial[i] == 1) else if_complete
        scores_if[i] = model.decision_function(Xs[i].reshape(1, -1))[0]

    # Ground truth labels as strings
    y_true = np.array([to_str(v) for v in y_true_raw], dtype=object)

    # Candidate thresholds from Benign distributions (per subset)
    q_comp = parse_percentiles(args.q_complete)
    q_part = parse_percentiles(args.q_partial)

    benign_complete = scores_if[(y_true == "Benign") & (is_partial == 0)]
    benign_partial  = scores_if[(y_true == "Benign") & (is_partial == 1)]

    def quantile_candidates(arr, qs, fallback_from_arr):
        if arr.size:
            return sorted(set(float(np.quantile(arr, q)) for q in qs))
        if fallback_from_arr.size:
            return sorted(set(float(np.quantile(fallback_from_arr, q)) for q in qs))
        # final fallback: lowish mid value
        vmin, vmax = float(np.min(scores_if)), float(np.max(scores_if))
        mid = vmin + 0.25 * (vmax - vmin)
        return [mid]

    t_complete_cands = quantile_candidates(benign_complete, q_comp, scores_if[is_partial == 0])
    t_partial_cands  = quantile_candidates(benign_partial,  q_part, scores_if[is_partial == 1])

    # ---- Clamp candidates to be at/above the decision boundary (+ small margin) ----
    floor_ = float(args.min_if_threshold)   # decision_function units; 0.02 by default
    t_complete_cands = sorted({ max(x, floor_) for x in t_complete_cands })
    t_partial_cands  = sorted({ max(x, floor_) for x in t_partial_cands })

    gates = np.arange(args.gate_min, args.gate_max + 1e-12, args.gate_step)

    # --- Search over (gate, t_complete, t_partial)
    best = None
    for g in gates:
        low = (s_conf < g)
        for tc in t_complete_cands:
            for tp in t_partial_cands:
                thr = np.where(is_partial == 1, tp, tc)
                inlier = scores_if >= thr

                # Default: keep RF prediction; override only for low-conf + IF-inlier
                h = s_pred.copy()
                h[low & inlier] = "Benign"

                fpr, fnr = bu_metrics(y_true, h)  # Benign vs Unknown only

                if args.fpr_target is not None:
                    meets = (fpr <= args.fpr_target + 1e-12)
                    # Tie-break: prefer stricter (higher tc/tp) and smaller gate (route less)
                    key = (0 if meets else 1,
                           fnr if meets else (fpr - args.fpr_target) ** 2 + fnr,
                           fpr,
                           -tc, -tp, g)
                    if (best is None) or (key < best["key"]):
                        best = {
                            "key": key, "conf_gate": float(g),
                            "if_complete_threshold": float(tc),
                            "if_partial_threshold":  float(tp),
                            "val_fpr": float(fpr), "val_fnr": float(fnr),
                            "objective": "fpr_target"
                        }
                else:
                    score = float(args.cost_fp * fpr + args.cost_fn * fnr)
                    # Tie-break on equality: prefer higher thresholds (stricter), then lower gate
                    key = (score, -tc, -tp, g)
                    if (best is None) or (key < best["key"]):
                        best = {
                            "key": key, "score": score, "conf_gate": float(g),
                            "if_complete_threshold": float(tc),
                            "if_partial_threshold":  float(tp),
                            "val_fpr": float(fpr), "val_fnr": float(fnr),
                            "objective": "cost_unknown_only"
                        }

    if best is None:
        print("[ERROR] Tuning failed to produce a solution.", file=sys.stderr)
        sys.exit(2)

    # Write minimal thresholds.json (exact keys runtime expects)
    minimal = {
        "conf_gate": best["conf_gate"],
        "if_complete_threshold": best["if_complete_threshold"],
        "if_partial_threshold":  best["if_partial_threshold"],
    }
    with open(args.out, "w") as f:
        json.dump(minimal, f, indent=2)
    print("Wrote calibrated thresholds →", args.out)
    print(json.dumps(minimal, indent=2))

    # Lightweight report for audit/debug
    rep = {
        "objective": best["objective"],
        "val_rows": int(len(df)),
        "conf_gate": best["conf_gate"],
        "if_complete_threshold": best["if_complete_threshold"],
        "if_partial_threshold":  best["if_partial_threshold"],
        "val_fpr_unknown_only": best["val_fpr"],
        "val_fnr_unknown_only": best["val_fnr"],
        "candidates_complete": t_complete_cands[:10],
        "candidates_partial":  t_partial_cands[:10],
        "gate_range": [float(gates.min()) if gates.size else None,
                       float(gates.max()) if gates.size else None],
        "min_if_threshold": floor_,
    }
    rep_path = os.path.join(args.models_dir, "tuning_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2)
    print("Tuning report →", rep_path)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()

