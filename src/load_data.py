
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return its DataFrame."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    csv_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = load_csv(csv_path)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)

    print("\n--- First 5 Rows ---")
    print(df.head())
