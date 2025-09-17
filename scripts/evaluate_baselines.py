#!/usr/bin/env python3
"""
Evaluate baselines with a hybrid cascade.

Variants:
  - S: Supervised (multiclass RF)
  - U: Unsupervised IF (binary)
  - H: Hybrid cascade (RF with low-confidence routing to IF)

This version adds **complementary policies** besides the earlier ones:
  * aggressive — legacy behaviour (low & outlier -> Unknown; low & inlier -> Benign)
  * conservative_dual — dual thresholds with Unknown escalation only when RF∈{Benign,Unknown}
  * complementary — use IF as **endorse/veto** layer:
        - low & inlier: force Benign (veto)
        - low & outlier & RF==Unknown: Unknown (escalate)
        - low & outlier & RF is a specific attack: keep RF attack (endorse)
        - low & outlier & RF==Benign: keep Benign (no escalate)
        - mid-band: keep RF label; high-conf: keep RF label
  * complementary_strict — same as complementary; kept for extensibility (hook if you want stricter rules)

Outputs (unchanged paths/names):
  metrics_s.json, metrics_u.json, metrics_h.json
  cm_*.png + confusion-matrix CSVs
  s_*, u_*, h_* ROC/PR
  summary_metrics.csv
  routing_stats.json
  per_sample.csv, cascade_report.json, routed_binary_cm.csv
  binary_benign_vs_unknown_{s,h,u}.csv
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ----------------- helpers -----------------

def load_features_list(path: str):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_paths(paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"[ERROR] Missing required artefact: {p}", file=sys.stderr)
            sys.exit(2)

def invert_label_map(path: str):
    m = json.load(open(path))
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

def label_to_id_map(path: str):
    m = json.load(open(path))
    lab2id = {}
    for k, v in m.items():
        if isinstance(v, int):
            lab2id[str(k)] = int(v)
        else:
            try:
                lab2id[str](v)
            except Exception:
                pass
    # Fix: the above line seems incorrect; preserve original functionality.
    m = json.load(open(path))
    lab2id = {}
    for k, v in m.items():
        if isinstance(v, int):
            lab2id[str(k)] = int(v)
        else:
            try:
                lab2id[str(v)] = int(k)
            except Exception:
                pass
    return lab2id

def to_str_labels(arr, id2label):
    out = []
    for v in arr:
        if isinstance(v, (int, np.integer)) and int(v) in id2label:
            out.append(id2label[int(v)])
        else:
            try:
                iv = int(v); out.append(id2label.get(iv, str(v)))
            except Exception:
                out.append(str(v))
    return np.array(out, dtype=object)

def class_prob(proba, rf_classes, target_label, label2id):
    idx = None
    if len(rf_classes) and isinstance(rf_classes[0], (int, np.integer)):
        t_id = label2id.get(target_label, None)
        if t_id is not None and t_id in set(rf_classes):
            idx = list(rf_classes).index(t_id)
    else:
        if target_label in set(rf_classes):
            idx = list(rf_classes).index(target_label)
    return proba[:, idx] if idx is not None else np.zeros(proba.shape[0], dtype=float)

def benign_prob_from_proba(proba, rf_classes, label2id):
    return 1.0 - class_prob(proba, rf_classes, "Benign", label2id)

def plot_cm(cm, classes, out_png, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def plot_roc_pr(y_true_bin, y_score, out_prefix, label=""):
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label="ROC"); plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC {label}")
    plt.legend(loc="lower right"); plt.savefig(out_prefix+"_roc.png", bbox_inches="tight"); plt.close()
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    plt.figure(figsize=(6,5)); plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {label}")
    plt.legend(loc="lower left"); plt.savefig(out_prefix+"_pr.png", bbox_inches="tight"); plt.close()

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--csv", help="Override CSV path")
    ap.add_argument("--features", default="models/features.txt")
    ap.add_argument("--label-map", default="models/label_mapping.json")
    ap.add_argument("--scaler", default="models/scaler.pkl")
    ap.add_argument("--rf", default="models/rf_model.pkl")
    ap.add_argument("--if-complete", default="models/iso_complete.pkl")
    ap.add_argument("--if-partial", default="models/iso_partial.pkl")
    ap.add_argument("--thresholds", default="models/thresholds.json")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--partial-col", default="is_partial")
    ap.add_argument("--binary-scheme", choices=["all_attacks","unknown_only"], default="unknown_only")
    ap.add_argument("--outdir", default="evaluation_results")
    ap.add_argument("--policy", choices=["aggressive","conservative_dual","complementary","complementary_strict"], default="complementary",
                    help="Hybrid decision policy. 'complementary' uses IF to endorse/veto low-confidence decisions to reduce FPR.")
    args = ap.parse_args()

    csv_path = args.csv if args.csv else f"data/processed/{args.split}_dataset.csv"
    ensure_paths([csv_path, args.features, args.label_map, args.scaler, args.rf, args.if_complete, args.if_partial, args.thresholds])
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    feats = load_features_list(args.features)
    df = pd.read_csv(csv_path)
    if args.label_col not in df.columns:
        print(f"[ERROR] Label column '{args.label_col}' not found in {csv_path}.", file=sys.stderr); sys.exit(2)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print(f"[ERROR] Dataset missing required features: {missing[:8]}...", file=sys.stderr); sys.exit(2)

    Xdf = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = joblib.load(args.scaler)
    Xs = scaler.transform(Xdf)

    id2label = invert_label_map(args.label_map)
    label2id = label_to_id_map(args.label_map)
    y_true = to_str_labels(df[args.label_col].values, id2label)

    rf = joblib.load(args.rf)
    if_complete = joblib.load(args.if_complete)
    if_partial  = joblib.load(args.if_partial)
    thr = json.load(open(args.thresholds))
    if_thr_out = float(thr.get("if_thr_outlier", thr.get("if_threshold", thr.get("if_complete_threshold", 0.0))))
    if_thr_in  = float(thr.get("if_thr_inlier",  thr.get("if_threshold", thr.get("if_complete_threshold", 0.0))))
    conf_gate  = float(thr.get("conf_gate", 0.85))

    # --- enforce single-threshold rule (match real-time): no midband
    if_thr_out = if_thr_in

    # S predictions
    proba = rf.predict_proba(Xs)
    rf_classes = list(rf.classes_)
    idx = np.argmax(proba, axis=1)
    s_pred = to_str_labels(np.array([rf_classes[i] for i in idx], dtype=object), id2label)
    s_conf = np.max(proba, axis=1)

    # U scores
    is_partial = df[args.partial_col].values if args.partial_col in df.columns else np.zeros(len(df), dtype=int)
    scores_if = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        model = if_partial if (is_partial[i] == 1) else if_complete
        scores_if[i] = model.decision_function(Xs[i].reshape(1, -1))[0]
    outlier = scores_if < if_thr_out
    inlier  = scores_if >= if_thr_in
    midband = ~(outlier | inlier)

    # H decision
    low = s_conf < conf_gate
    h_pred = s_pred.copy()

    if args.policy == "aggressive":
        h_pred[low & outlier] = "Unknown"
        h_pred[low & inlier & (s_pred == "Benign")] = "Benign"
        h_pred[low & inlier & (s_pred == "Unknown")] = "Benign"
        escalate_mask = low & outlier

    elif args.policy == "conservative_dual":
        h_pred[low & inlier] = "Benign"
        escalate_mask = low & outlier & ((s_pred == "Benign") | (s_pred == "Unknown"))
        h_pred[escalate_mask] = "Unknown"

    else:  # complementary / complementary_strict
        is_spec_attack = ~((s_pred == "Benign") | (s_pred == "Unknown"))
        # Veto: inlier => Benign regardless of RF label (on low-conf only)
        h_pred[low & inlier] = "Benign"
        # Endorse: outlier + specific attack => keep RF attack label
        keep_attack = low & outlier & is_spec_attack
        # Escalate: outlier + RF==Unknown => Unknown
        escalate_mask = low & outlier & (s_pred == "Unknown")
        h_pred[escalate_mask] = "Unknown"
        # Do NOT escalate if RF==Benign (keep Benign)
        # Mid-band: keep RF
        # High-conf: keep RF

    # Metrics & artefacts (same as before)
    classes_ordered = sorted(set(y_true.tolist()) | set(s_pred.tolist()))
    cm_s = confusion_matrix(y_true, s_pred, labels=classes_ordered)
    plot_cm(cm_s, classes_ordered, os.path.join(args.outdir, "cm_s.png"), title="Confusion Matrix — S (RF)")
    pd.DataFrame(cm_s,
        index=[f"true:{c}" for c in classes_ordered],
        columns=[f"pred:{c}" for c in classes_ordered]).to_csv(os.path.join(args.outdir, "supervised_confusion_matrix.csv"), index=True)
    report_s = classification_report(y_true, s_pred, labels=classes_ordered, target_names=classes_ordered, output_dict=True, digits=4)
    with open(os.path.join(args.outdir, "metrics_s.json"), "w") as f: json.dump(report_s, f, indent=2)

    classes_ordered_h = sorted(set(y_true.tolist()) | set(h_pred.tolist()) | set(classes_ordered))
    cm_h = confusion_matrix(y_true, h_pred, labels=classes_ordered_h)
    plot_cm(cm_h, classes_ordered_h, os.path.join(args.outdir, "cm_h.png"), title="Confusion Matrix — H (Cascade)")
    pd.DataFrame(cm_h,
        index=[f"true:{c}" for c in classes_ordered_h],
        columns=[f"pred:{c}" for c in classes_ordered_h]).to_csv(os.path.join(args.outdir, "hybrid_confusion_matrix.csv"), index=True)
    report_h = classification_report(y_true, h_pred, labels=classes_ordered_h, target_names=classes_ordered_h, output_dict=True, digits=4)
    with open(os.path.join(args.outdir, "metrics_h.json"), "w") as f: json.dump(report_h, f, indent=2)

    # ROC/PR for S and H (Benign vs Attack)
    y_true_bin_all = (y_true != "Benign").astype(int)
    score_s_attack = benign_prob_from_proba(proba, rf_classes, label2id)
    plot_roc_pr(y_true_bin_all, score_s_attack, os.path.join(args.outdir, "s"), label="S (Benign vs Attack)")
    plot_roc_pr(y_true_bin_all, score_s_attack, os.path.join(args.outdir, "h"), label="H (Benign vs Attack)")

    # U-only (binary) for completeness
    u_pred_bin = outlier.astype(int)
    cm_u = confusion_matrix(y_true_bin_all, u_pred_bin, labels=[0,1])
    plot_cm(cm_u, ["Benign","Attack"], os.path.join(args.outdir, "cm_u.png"), title="Confusion Matrix — U (IF, binary)")
    pd.DataFrame(cm_u, index=["true:Benign","true:Attack"], columns=["pred:Benign","pred:Attack"]).to_csv(os.path.join(args.outdir, "unsupervised_confusion_matrix.csv"), index=True)
    plot_roc_pr(y_true_bin_all, -scores_if, os.path.join(args.outdir, "u"), label="U (IF)")
    report_u = {"binary_report": classification_report(y_true_bin_all, u_pred_bin, labels=[0,1], target_names=["Benign","Attack"], output_dict=True, digits=4)}
    with open(os.path.join(args.outdir, "metrics_u.json"), "w") as f: json.dump(report_u, f, indent=2)
    with open(os.path.join(args.outdir, "binary_classification_report.json"), "w") as f: json.dump(report_u["binary_report"], f, indent=2)

    # BU exports
    mask_bu = (y_true == "Benign") | (y_true == "Unknown")
    y_bu = (y_true[mask_bu] == "Unknown").astype(int)
    ys_bu = (s_pred[mask_bu] == "Unknown").astype(int)
    yh_bu = (h_pred[mask_bu] == "Unknown").astype(int)
    yu_bu = (outlier[mask_bu]).astype(int)

    for arr, name in [(ys_bu,"binary_benign_vs_unknown_s.csv"),
                      (yh_bu,"binary_benign_vs_unknown_h.csv"),
                      (yu_bu,"binary_benign_vs_unknown_u.csv")]:
        cm = confusion_matrix(y_bu, arr, labels=[0,1])
        pd.DataFrame(cm, index=["true:Benign","true:Unknown"], columns=["pred:Benign","pred:Unknown"]).to_csv(os.path.join(args.outdir, name), index=True)

    # Routing diagnostics
    is_spec_attack = ~((s_pred == "Benign") | (s_pred == "Unknown"))
    routing_stats = {
        "policy": args.policy,
        "conf_gate": float(conf_gate),
        "if_thr_outlier": float(if_thr_out),
        "if_thr_inlier": float(if_thr_in),
        "n_total": int(len(df)),
        "n_low_conf": int(low.sum()),
        "pct_low_conf": float((low.sum()/len(df))*100.0 if len(df) else 0.0),
        "n_low_inlier": int((low & inlier).sum()),
        "n_low_outlier": int((low & outlier).sum()),
        "n_low_midband": int((low & midband).sum()),
        "n_low_outlier_escalated_to_unknown": int((low & outlier & (s_pred == "Unknown")).sum()) if args.policy.startswith("complementary") else int(((s_pred == "Benign") | (s_pred == "Unknown")) & low & outlier).sum() if args.policy=="conservative_dual" else int((low & outlier).sum()),
        "n_low_outlier_endorsed_attack": int((low & outlier & is_spec_attack).sum()) if args.policy.startswith("complementary") else 0,
        "n_low_outlier_kept_benign": int((low & outlier & (s_pred == "Benign")).sum()) if args.policy.startswith("complementary") else 0,
        "s_conf_quantiles": {q: float(v) for q, v in zip(
            [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99],
            np.quantile(s_conf, [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).tolist())},
        "if_score_min_max": [float(np.min(scores_if)), float(np.max(scores_if))]
    }
    with open(os.path.join(args.outdir, "routing_stats.json"), "w") as f:
        json.dump(routing_stats, f, indent=2)

    # Per-sample export & cascade report (unchanged logic)
    per = pd.DataFrame({
        "y_true": y_true,
        "s_pred": s_pred,
        "s_conf": s_conf,
        "if_score": scores_if,
        "low_conf": low.astype(int),
        "if_outlier": outlier.astype(int),
        "if_inlier": inlier.astype(int),
        "h_pred": h_pred,
        "is_partial": (df[args.partial_col].values if args.partial_col in df.columns else np.zeros(len(df), dtype=int)).astype(int),
    })
    bu_mask_series = (per["y_true"].isin(["Benign","Unknown"]))
    per["bu_mask"] = bu_mask_series.astype(int)
    per["y_bu"]  = np.where(bu_mask_series, (per["y_true"] == "Unknown").astype(int), -1)
    per["s_bu"]  = np.where(bu_mask_series, (per["s_pred"] == "Unknown").astype(int), -1)
    per["h_bu"]  = np.where(bu_mask_series, (per["h_pred"] == "Unknown").astype(int), -1)

    per["s_correct"] = (per["s_pred"] == per["y_true"]).astype(int)
    per["h_correct"] = (per["h_pred"] == per["y_true"]).astype(int)
    per["s_bu_correct"] = np.where(per["bu_mask"]==1, (per["s_bu"] == per["y_bu"]).astype(int), -1)
    per["h_bu_correct"] = np.where(per["bu_mask"]==1, (per["h_bu"] == per["y_bu"]).astype(int), -1)
    per.to_csv(os.path.join(args.outdir, "per_sample.csv"), index=False)

    def gains(mask, use_bu=False):
        if isinstance(mask, slice): sub = per.iloc[mask]
        else:
            arr = np.asarray(mask)
            sub = per.loc[arr] if arr.dtype==bool else per.iloc[arr]
        if not len(sub): return {"n":0,"fixes":0,"regress":0,"net":0}
        s_ok = (sub["s_bu_correct"]==1) if use_bu else (sub["s_correct"]==1)
        h_ok = (sub["h_bu_correct"]==1) if use_bu else (sub["h_correct"]==1)
        fixes = int((~s_ok &  h_ok).sum())
        regress = int(( s_ok & ~h_ok).sum())
        return {"n": int(len(sub)), "fixes": fixes, "regress": regress, "net": fixes - regress}

    overall_gain = gains(slice(None), use_bu=False)
    overall_gain_bu = gains(per["bu_mask"]==1, use_bu=True)
    routed_gain = gains(per["low_conf"]==1, use_bu=False)
    routed_gain_bu = gains((per["low_conf"]==1) & (per["bu_mask"]==1), use_bu=True)

    routed = per[(per["low_conf"]==1) & (per["bu_mask"]==1)].copy()
    def cm2x2(truth, pred):
        tn = int(((truth==0) & (pred==0)).sum())
        fp = int(((truth==0) & (pred==1)).sum())
        fn = int(((truth==1) & (pred==0)).sum())
        tp = int(((truth==1) & (pred==1)).sum())
        return tn, fp, fn, tp
    if len(routed):
        tn_s, fp_s, fn_s, tp_s = cm2x2(routed["y_bu"], routed["s_bu"])
        tn_h, fp_h, fn_h, tp_h = cm2x2(routed["y_bu"], routed["h_bu"])
        routed_cm = pd.DataFrame([[tn_s, fp_s, fn_s, tp_s],[tn_h, fp_h, fn_h, tp_h]], index=["S_on_routed","H_on_routed"], columns=["TN","FP","FN","TP"])
        routed_cm.to_csv(os.path.join(args.outdir, "routed_binary_cm.csv"), index=True)

    cascade_report = {
        "policy": args.policy,
        "conf_gate": float(conf_gate),
        "if_thr_outlier": float(if_thr_out),
        "if_thr_inlier": float(if_thr_in),
        "n_total": int(len(per)),
        "routed_pct": float(100.0*per["low_conf"].mean() if len(per) else 0.0),
        "overall_gain_multiclass": overall_gain,
        "overall_gain_unknown_only": overall_gain_bu,
        "routed_gain_multiclass": routed_gain,
        "routed_gain_unknown_only": routed_gain_bu,
        "notes": "fixes=S wrong -> H right; regress=S right -> H wrong; net=fixes-regress"
    }
    with open(os.path.join(args.outdir, "cascade_report.json"), "w") as f:
        json.dump(cascade_report, f, indent=2)

    rows = []
    rows.append({"variant":"U-binary","precision":report_u["binary_report"]["weighted avg"]["precision"],"recall":report_u["binary_report"]["weighted avg"]["recall"],"f1":report_u["binary_report"]["weighted avg"]["f1-score"],"accuracy":report_u["binary_report"]["accuracy"]})
    rows.append({"variant":"S","precision":report_s["macro avg"]["precision"],"recall":report_s["macro avg"]["recall"],"f1":report_s["macro avg"]["f1-score"],"accuracy":report_s["accuracy"]})
    rows.append({"variant":"H","precision":report_h["macro avg"]["precision"],"recall":report_h["macro avg"]["recall"],"f1":report_h["macro avg"]["f1-score"],"accuracy":report_h.get("accuracy")})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    print("\nEvaluation complete. See:", args.outdir)
    for fn in [
        "metrics_s.json","metrics_u.json","metrics_h.json",
        "cm_s.png","cm_u.png","cm_h.png",
        "s_roc.png","s_pr.png","u_roc.png","u_pr.png","h_roc.png","h_pr.png",
        "summary_metrics.csv",
        "supervised_confusion_matrix.csv","unsupervised_confusion_matrix.csv","hybrid_confusion_matrix.csv",
        "binary_classification_report.json",
        "routing_stats.json",
        "binary_benign_vs_unknown_s.csv","binary_benign_vs_unknown_h.csv","binary_benign_vs_unknown_u.csv",
        "per_sample.csv","cascade_report.json","routed_binary_cm.csv"
    ]:
        p = os.path.join(args.outdir, fn)
        print("  •", p, ("✓" if os.path.exists(p) else "✗"))

if __name__ == "__main__":
    main()

