#!/usr/bin/env python3
# prepare_dataset.py — strict passthrough cleaner
# - Drops only: src_ip, dst_ip, src_port, dst_port
# - Preserves all other values/formatting exactly (no numeric coercion, no union schema)
# - Uses the first file's columns (minus the 4) as canonical; later files must match or are skipped
# - Keeps rows only if they contain a non-empty Label

import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/cleaned/cleaned_dataset.csv"

DROP_COLS = {"src_ip", "dst_ip", "src_port", "dst_port"}

def read_csv_as_text(path: str) -> pd.DataFrame:
    # Read everything as text; do not treat anything as NA automatically
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False
    )

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        print(f"No CSV files found in {RAW_DIR}")
        return

    total_in = total_out = 0

    # --- Establish canonical columns from the FIRST file (after dropping the 4 IDs)
    first = files[0]
    df0 = read_csv_as_text(first)
    total_in += len(df0)

    # Must have Label
    if "Label" not in df0.columns:
        print(f"{first}: missing 'Label' column. Aborting.")
        return

    # Drop only the 4 ID columns that are present; preserve order of the rest
    canon_cols = [c for c in df0.columns if c not in DROP_COLS]
    # Keep only rows with non-empty Label
    df0 = df0[canon_cols]
    df0 = df0[df0["Label"].astype(str) != ""].copy()

    # Write header and first chunk
    df0.to_csv(OUT_PATH, index=False)
    total_out += len(df0)
    print(f"{first} → wrote {len(df0)} rows, {len(canon_cols)} cols")

    # --- Process the rest; require exact header match (after dropping the 4)
    for path in files[1:]:
        df = read_csv_as_text(path)
        total_in += len(df)

        # Quick checks
        if "Label" not in df.columns:
            print(f"{path}: no 'Label' column → skipped")
            continue

        cols_after_drop = [c for c in df.columns if c not in DROP_COLS]
        if cols_after_drop != canon_cols:
            print(f"{path}: columns don't match canonical schema → skipped")
            print(f"    expected: {canon_cols[:6]} … {canon_cols[-6:]}")
            print(f"    got     : {cols_after_drop[:6]} … {cols_after_drop[-6:]}")
            continue

        # Keep only rows with non-empty Label
        df = df[canon_cols]
        df = df[df["Label"].astype(str) != ""].copy()

        # Append without header
        df.to_csv(OUT_PATH, mode="a", header=False, index=False)
        total_out += len(df)
        print(f"{path} → appended {len(df)} rows")

    print(f"\nCleaned dataset → {OUT_PATH}")
    print(f"Rows in: {total_in:,}  ➜  Rows written: {total_out:,}")
    print(f"Columns: {len(canon_cols)}")
    print(f"Schema: {canon_cols[:10]} … {canon_cols[-10:]}")

if __name__ == "__main__":
    main()

