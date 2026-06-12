"""Clean raw HMDA data into a tidy, model-ready table.

HMDA is real reported data, so it carries real-world mess that we fix here:
  - debt_to_income_ratio arrives as TEXT bands ("30%-<36%", ">60%") mixed with exact ints.
  - income is reported in $thousands.
  - "Exempt" / "NA" / blanks appear in numeric columns.
  - property_value / loan_amount carry huge multifamily/commercial outliers.
  - interest_rate & other post-origination fields ONLY exist for approved loans, so
    they leak the target and are deliberately excluded.

Run:
    python -m loan_predictor.clean --input ../../hmda_multistate_2023.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

# Application-time features only (no post-decision fields that would leak the label).
NUMERIC_RAW = ["loan_amount", "loan_to_value_ratio", "property_value", "income", "loan_term"]
CATEGORICAL = [
    "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "preapproval", "conforming_loan_limit", "total_units",
    "construction_method", "derived_loan_product_type",
]
# DTI bands -> representative midpoint (exact integers 36-49 are kept as-is).
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}


def clean(input_path: str, output_path: str) -> None:
    print(f"Reading {input_path} ...")
    # Read everything as str first; HMDA mixes types and uses Exempt/NA sentinels.
    df = pl.read_csv(input_path, infer_schema_length=0, ignore_errors=True)
    print(f"  raw shape: {df.height:,} rows x {df.width} cols")

    # --- Target: 1 = originated/approved, 3 = denied ---
    df = df.filter(pl.col("action_taken").is_in(["1", "3"]))
    df = df.with_columns(
        (pl.col("action_taken") == "1").cast(pl.Int8).alias("approved")
    )

    # --- Parse DTI bands -> numeric midpoints ---
    dti = pl.col("debt_to_income_ratio")
    dti_expr = pl.when(dti == "<20%").then(15.0)
    for band, val in DTI_BANDS.items():
        dti_expr = dti_expr.when(dti == band).then(val)
    dti_expr = dti_expr.otherwise(dti.cast(pl.Float64, strict=False)).alias("dti")
    df = df.with_columns(dti_expr)

    # --- Coerce numeric features (Exempt/NA -> null) ---
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in NUMERIC_RAW
    ])
    # income reported in $thousands -> dollars
    df = df.with_columns((pl.col("income") * 1000).alias("income"))

    # --- Winsorise heavy-tailed money/ratio columns to 1st-99th percentile ---
    for c in ["loan_amount", "property_value", "income", "loan_to_value_ratio"]:
        lo = df.select(pl.col(c).quantile(0.01)).item()
        hi = df.select(pl.col(c).quantile(0.99)).item()
        if lo is not None and hi is not None:
            df = df.with_columns(pl.col(c).clip(lo, hi).alias(c))

    # --- Engineered feature: loan-to-income ---
    df = df.with_columns(
        pl.when(pl.col("income") > 0)
        .then(pl.col("loan_amount") / pl.col("income"))
        .otherwise(None)
        .alias("loan_to_income")
    )

    keep = ["approved"] + NUMERIC_RAW + ["dti", "loan_to_income"] + CATEGORICAL
    keep = [c for c in keep if c in df.columns]
    out = df.select(keep)

    # Remove exact duplicate rows (same features + label). Nulls are left as-is
    # and imputed later inside the training pipeline.
    before = out.height
    out = out.unique(maintain_order=True)
    print(f"  removed {before - out.height:,} duplicate rows ({out.height:,} remain)")

    bal = out.group_by("approved").len().sort("approved")
    print("  target balance (0=denied, 1=approved):")
    print(bal)
    print(f"  clean shape: {out.height:,} rows x {out.width} cols")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/processed/hmda_clean.parquet")
    args = ap.parse_args()
    clean(args.input, args.output)
