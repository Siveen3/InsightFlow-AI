
from pathlib import Path

import pandas as pd

from src.load_data import load_csv


# 1. Load the dataset
PROJECT_ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df = load_csv(CSV_PATH)


# 2. Find TotalCharges values that are missing or invalid
charges_numeric = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
)

invalid_mask = charges_numeric.isna()

print("\n--- TotalCharges Quality Check ---")
print("Invalid or missing values:", invalid_mask.sum())

print("\nAffected rows:")
print(
    df.loc[
        invalid_mask,
        ["customerID", "tenure", "TotalCharges"],
    ].to_string()
)


# 3. Check whether the invalid values are blank
charges_text = df["TotalCharges"].astype("string").str.strip()

blank_mask = charges_text.isna() | charges_text.eq("")

print("\nBlank or whitespace-only TotalCharges:", blank_mask.sum())

print("\nAffected rows with tenure:")
print(
    df.loc[
        invalid_mask,
        ["customerID", "tenure"],
    ].to_string()
)


# 4. Count exact duplicate rows
exact_duplicates = df.duplicated().sum()

print("\n--- Duplicate Check ---")
print("Exact duplicate rows:", exact_duplicates)


# 5. Count duplicate rows ignoring customerID
df_without_id = df.drop(columns=["customerID"])

duplicates_without_id = df_without_id.duplicated().sum()

print(
    "Duplicate rows ignoring customerID:",
    duplicates_without_id,
)

print("\nRows duplicated when customerID is ignored:")
print(
    df.loc[
        df_without_id.duplicated(keep=False),
    ].sort_values("tenure").to_string()
)
