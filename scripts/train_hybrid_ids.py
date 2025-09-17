#!/usr/bin/env python3
# UPDATED: train_hybrid_ids.py
# - RF + dual IF (complete vs partial)
# - EXCLUDES is_partial from model features to match runtime routing logic
# - Scaler fitted on a pandas DataFrame (preserves feature_names_in_)
# - Optional RF probability calibration on validation split

import os
import json
import argparse
import joblib
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

# ------------------------
# Helpers
# ------------------------
def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric (coerce errors), replace inf with NaN,
    drop columns that are entirely NaN, then fill remaining NaNs with 0.0.
    """
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)  # drop all-NaN columns
    df.fillna(0.0, inplace=True)
    return df

def load_csv_or_fail(path, msg_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg_name} not found: {path}")
    return pd.read_csv(path, low_memory=False)

def save_features_list(path, cols):
    with open(path, "w") as f:
        for c in cols:
            f.write(f"{c}\n")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def make_calibrator(prefit_estimator, method):
    """Return CalibratedClassifierCV compatible with both old/new sklearn APIs."""
    try:
        # sklearn >= 1.4
        return CalibratedClassifierCV(estimator=prefit_estimator, method=method, cv="prefit")
    except TypeError:
        # sklearn < 1.4
        return CalibratedClassifierCV(base_estimator=prefit_estimator, method=method, cv="prefit")

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Train RF + dual IF (complete/partial); optionally calibrate RF probabilities on validation split.")
    ap.add_argument("--train-csv", default="data/processed/train_dataset.csv", help="Training CSV with Label and features.")
    ap.add_argument("--val-csv",   default="data/processed/val_dataset.csv",   help="Validation CSV for calibration (used if --calibrate != none).")
    ap.add_argument("--models-dir", default="models", help="Output directory for model artefacts.")
    ap.add_argument("--label-col",  default="Label",  help="Ground-truth column name.")
    ap.add_argument("--partial-col", default="is_partial", help="Column indicating partial flows (0/1).")
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none", help="Post-hoc probability calibration method for RF.")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    TRAIN_PATH = args.train_csv
    VAL_PATH   = args.val_csv
    MODEL_DIR  = args.models_dir
    LABEL_COL  = args.label_col
    PARTIAL_COL = args.partial_col
    RANDOM_STATE = args.random_state

    MODEL_PATH         = f"{MODEL_DIR}/rf_model.pkl"
    SCALER_PATH        = f"{MODEL_DIR}/scaler.pkl"
    FEATURES_PATH      = f"{MODEL_DIR}/features.txt"
    LABEL_MAPPING_PATH = f"{MODEL_DIR}/label_mapping.json"
    ISO_COMPLETE_PATH  = f"{MODEL_DIR}/iso_complete.pkl"
    ISO_PARTIAL_PATH   = f"{MODEL_DIR}/iso_partial.pkl"

    ensure_dir(MODEL_DIR)

    print("Loading training dataset…")
    df_train = load_csv_or_fail(TRAIN_PATH, "Training CSV")
    if LABEL_COL not in df_train.columns:
        raise ValueError(f"'{LABEL_COL}' column not found in the training dataset.")
    if PARTIAL_COL not in df_train.columns:
        raise ValueError(f"'{PARTIAL_COL}' column not found in the training dataset (needed for IF split).")

    df_train.dropna(subset=[LABEL_COL], inplace=True)

    # --- Label encoding (persist mapping string -> int)
    y_str = df_train[LABEL_COL].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)
    print(f"Training rows: {len(df_train)}")
    print("Label distribution (string labels):")
    print(y_str.value_counts())
    print("\nLabel mapping (string → encoded):")
    for i, c in enumerate(classes):
        print(f"  {c} → {i}")

    # --- Feature matrix (EXCLUDE label + is_partial)
    # Keep PARTIAL_COL only for routing/branching; do NOT give it to models.
    feature_candidates = df_train.columns.drop([LABEL_COL, PARTIAL_COL])
    X = df_train[feature_candidates].copy()
    X = to_numeric_df(X)
    y = y[:len(X)]  # align in case of drops
    feature_columns = X.columns.tolist()
    print(f"Using {len(feature_columns)} features (excluded: {PARTIAL_COL})")

    # Persist artefacts for runtime
    save_features_list(FEATURES_PATH, feature_columns)
    with open(LABEL_MAPPING_PATH, "w") as f:
        json.dump({cls: int(idx) for idx, cls in enumerate(classes)}, f, indent=2)

    # --- Scaling on DataFrame to preserve feature_names_in_
    print(f"Pre-scale shape: {X.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit on DataFrame with named columns
    joblib.dump(scaler, SCALER_PATH)

    # --- Class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=y)
    class_weight_dict = {int(k): float(v) for k, v in zip(np.arange(len(classes)), class_weights)}
    print(f"Class weights: {class_weight_dict}")

    # --- Train RandomForest
    print("Training RandomForestClassifier…")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_scaled, y)

    # --- Optional: probability calibration on the validation split
    if args.calibrate != "none":
        print(f"Calibrating RF probabilities on validation split using method: {args.calibrate}")
        df_val = load_csv_or_fail(VAL_PATH, "Validation CSV")
        if LABEL_COL not in df_val.columns:
            raise ValueError(f"'{LABEL_COL}' column not found in the validation dataset.")

        # Use the SAME feature set; exclude PARTIAL_COL here as well
        if set(feature_columns).issubset(df_val.columns):
            Xv = df_val[feature_columns].copy()
        else:
            # fallback: drop label & partial, then align
            drop_cols = [c for c in [LABEL_COL, PARTIAL_COL] if c in df_val.columns]
            Xv = df_val.drop(columns=drop_cols).copy()

        Xv = to_numeric_df(Xv)
        # Add any missing columns and order columns exactly
        missing_cols = [c for c in feature_columns if c not in Xv.columns]
        for c in missing_cols:
            Xv[c] = 0.0
        Xv = Xv[feature_columns]

        yv_str = df_val[LABEL_COL].astype(str).reindex(Xv.index)
        yv = le.transform(yv_str)
        Xv_scaled = scaler.transform(Xv)

        calibrator = make_calibrator(rf, args.calibrate)
        calibrator.fit(Xv_scaled, yv)
        rf = calibrator  # replace with calibrated estimator

    # Persist RF (calibrated if chosen)
    joblib.dump(rf, MODEL_PATH)

    # --- Train IsolationForest models (benign only; split by is_partial)
    print("Training IsolationForest on benign samples…")
    benign_df = df_train[df_train[LABEL_COL] == "Benign"].copy()

    # Build benign feature frames EXCLUDING label & is_partial, then split by is_partial for routing only
    benign_features = benign_df.drop(columns=[LABEL_COL, PARTIAL_COL])
    benign_features = to_numeric_df(benign_features)

    # Align to RF feature set and scale once
    benign_features = benign_features.reindex(columns=feature_columns, fill_value=0.0)
    benign_features_scaled = scaler.transform(benign_features)

    # Masks for complete vs partial based on original column (not in features)
    mask_complete = (benign_df[PARTIAL_COL] == 0)
    mask_partial  = (benign_df[PARTIAL_COL] == 1)

    X_comp_scaled = benign_features_scaled[mask_complete.values]
    X_part_scaled = benign_features_scaled[mask_partial.values]
    print(f"Benign complete/partial shapes (scaled): {X_comp_scaled.shape} / {X_part_scaled.shape}")

    iso_complete = IsolationForest(n_estimators=100, contamination=0.001, random_state=RANDOM_STATE)
    iso_complete.fit(X_comp_scaled)
    joblib.dump(iso_complete, ISO_COMPLETE_PATH)

    iso_partial = IsolationForest(n_estimators=100, contamination=0.001, random_state=RANDOM_STATE)
    iso_partial.fit(X_part_scaled)
    joblib.dump(iso_partial, ISO_PARTIAL_PATH)

    mem = psutil.virtual_memory()
    print(f"\nMemory used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

    print("\nHybrid model training complete:")
    print(f"   • RandomForest:               {MODEL_PATH}")
    print(f"   • IsolationForest (complete): {ISO_COMPLETE_PATH}")
    print(f"   • IsolationForest (partial):  {ISO_PARTIAL_PATH}")
    print(f"   • Scaler:                     {SCALER_PATH}")
    print(f"   • Features:                   {FEATURES_PATH}")
    print(f"   • Label Mapping:              {LABEL_MAPPING_PATH}")
    if args.calibrate != "none":
        print(f"   • RF probabilities calibrated via: {args.calibrate} (validation: {VAL_PATH})")

if __name__ == "__main__":
    main()

