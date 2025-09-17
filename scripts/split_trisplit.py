#!/usr/bin/env python3
"""
split_trisplit.py — Safe tri-split (train/val/test) with robust stratification for small datasets.
Default ratios: 80/10/10. Produces CSVs in data/processed/ without touching existing files.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def infer_strat_col(df, label_col: str, partial_col: str = "is_partial"):
    # Prefer stratifying on (Label, is_partial) if it is present and not too sparse.
    if partial_col in df.columns:
        combo = df[label_col].astype(str) + "|" + df[partial_col].astype(str)
        counts = combo.value_counts()
        # If any bucket has < 5 samples overall, fall back to Label-only stratification (safer for small data).
        if (counts < 5).any():
            return df[label_col].astype(str)
        return combo
    else:
        return df[label_col].astype(str)

def check_min_per_class(y, split_name: str, min_per_class: int = 50):
    vc = pd.Series(y).value_counts()
    too_small = vc[vc < min_per_class]
    if not too_small.empty:
        print(f"[WARN] {split_name} has classes below {min_per_class} samples:\\n{too_small.to_string()}\\n"
              f"       Consider adjusting --val/--test ratios or aggregating rare classes.", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cleaned/cleaned_dataset.csv", help="Input CSV (cleaned)")
    ap.add_argument("--train-out", default="data/processed/train_dataset.csv")
    ap.add_argument("--val-out", default="data/processed/val_dataset.csv")
    ap.add_argument("--test-out", default="data/processed/test_dataset.csv")
    ap.add_argument("--val", type=float, default=0.10, help="Validation fraction (default 0.10)")
    ap.add_argument("--test", type=float, default=0.10, help="Test fraction (default 0.10)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--label-col", default="Label", help="Label column name")
    args = ap.parse_args()

    total_holdout = args.val + args.test
    if not (0 < args.val < 0.5 and 0 < args.test < 0.5 and total_holdout < 0.9):
        print("[ERROR] Please choose sensible --val and --test fractions (e.g., 0.10 0.10).", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)

    df = pd.read_csv(args.input)
    if args.label_col not in df.columns:
        print(f"[ERROR] Label column '{args.label_col}' not found in CSV. Columns: {list(df.columns)[:10]}...", file=sys.stderr)
        sys.exit(2)

    # Build stratification column
    y_strat = infer_strat_col(df, args.label_col)

    # First split train and holdout (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_holdout, random_state=args.seed)
    idx_train, idx_holdout = next(sss1.split(df, y_strat))
    train = df.iloc[idx_train].reset_index(drop=True)
    holdout = df.iloc[idx_holdout].reset_index(drop=True)

    # Now split holdout into val and test
    holdout_y = infer_strat_col(holdout, args.label_col)
    test_fraction_within_holdout = args.test / (args.val + args.test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction_within_holdout, random_state=args.seed)
    idx_val, idx_test = next(sss2.split(holdout, holdout_y))
    val = holdout.iloc[idx_val].reset_index(drop=True)
    test = holdout.iloc[idx_test].reset_index(drop=True)

    # Save
    train.to_csv(args.train_out, index=False)
    val.to_csv(args.val_out, index=False)
    test.to_csv(args.test_out, index=False)

    # Reports
    print(f"\\nTri-split complete (seed={args.seed}):")
    print(f"   • Train: {len(train):>7}  → {args.train_out}")
    print(f"   • Val:   {len(val):>7}  → {args.val_out}")
    print(f"   • Test:  {len(test):>7}  → {args.test_out}")

    print("\\nClass distribution (train):")
    print(train[args.label_col].value_counts())

    print("\\nClass distribution (val):")
    print(val[args.label_col].value_counts())

    print("\\nClass distribution (test):")
    print(test[args.label_col].value_counts())

    # Optional sanity warnings for small datasets
    check_min_per_class(train[args.label_col], "train", min_per_class=50)
    check_min_per_class(val[args.label_col], "val", min_per_class=50)
    check_min_per_class(test[args.label_col], "test", min_per_class=50)

if __name__ == "__main__":
    main()

