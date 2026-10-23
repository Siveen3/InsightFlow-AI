"""Data profiling for InsightFlow AI.

Rules for every function here: return plain data, never print,
never modify the DataFrame that was passed in.
"""
import re
from collections.abc import Iterable
from typing import Literal

import pandas as pd
FREE_TEXT_COLUMN_NAMES = {
    "comment", "comments", "description", "feedback",
    "review", "message", "notes", "bio",
}

ColumnType = Literal["numeric", "categorical", "identifier", "numeric_as_text", "free_text"]

DEFAULT_PLACEHOLDERS = ("", "?", "N/A", "null")

ID_THRESHOLD = 0.95          # uniqueness ratio that suggests an identifier
MIN_ROWS_FOR_RATIOS = 30     # ratios are unreliable on tiny datasets
MAX_CATEGORY_RATIO = 0.5     # categorical when unique values are not most rows
NUMERIC_TEXT_THRESHOLD = 0.90


# ---------- small helpers (private: the leading underscore) ----------
# this function checks if a pandas Series is of a text type (object or string), which is useful for determining how to handle the data in profiling functions.
def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or isinstance(
        series.dtype, pd.StringDtype
    )
#this function normalizes a list of placeholder strings by stripping whitespace and converting them to lowercase, returning a frozenset for efficient membership testing.

def _normalize_placeholders(placeholders: Iterable[str]) -> frozenset[str]:
    return frozenset(placeholder.strip().lower() for placeholder in placeholders)

#this function creates a boolean mask for a pandas Series, indicating which non-null text cells contain placeholder values (like empty strings or '?'). It returns a Series of booleans where True indicates a placeholder value.
def _placeholder_mask(series: pd.Series, placeholders: frozenset[str]) -> pd.Series:
    """True where a non-null text cell is a placeholder like '' or '?'."""
    cleaned = series.astype("string").str.strip().str.lower()
    return (cleaned.isin(placeholders) & series.notna()).fillna(False)

def _looks_like_free_text(series: pd.Series, name: str) -> bool:
    """Heuristic: column name or values suggest free text."""
    if name.strip().lower() in FREE_TEXT_COLUMN_NAMES:
        return True
    if not _is_text(series):
        return False
    values = series.dropna()
    if len(values) < MIN_ROWS_FOR_RATIOS:
        return False
    unique_count = values.nunique()
    observed_count = len(values)
    ratio = unique_count / observed_count
    return ratio > MAX_CATEGORY_RATIO

#this function calculates the proportion of non-missing text values in a pandas Series that can be successfully converted to numeric values. It first cleans the text data, removes any placeholder values, and then attempts to convert the remaining values to numeric, returning the ratio of successfully converted values to the total number of non-missing values.
def _numeric_ratio(series: pd.Series, placeholders: frozenset[str]) -> float:
    """Share of non-missing text values that parse as numbers."""
    text = series.astype("string").str.strip()
    text = text.mask(_placeholder_mask(series, placeholders))
    values = text.dropna()
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())

# this function checks if a pandas Series contains only integer values. It first checks if the Series has an integer data type. If not, it drops any missing values and checks if all remaining values are integers by verifying that the remainder of each value divided by 1 is zero. It returns True if all values are integers, and False otherwise.
def _is_integer_valued(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True
    values = series.dropna()
    return bool((values % 1 == 0).all())

# this function checks if a column name looks like an identifier (e.g., 'customerID' -> 'customer_id').
def _looks_like_id_name(name: str) -> bool:
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name.strip())
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized).lower()

    return normalized == "id" or normalized.endswith("_id")

# ---------- public API ----------

def find_missing(
    df: pd.DataFrame,
    placeholders: Iterable[str] = DEFAULT_PLACEHOLDERS,
) -> dict[str, dict[str, int | float]]:
    """Per column: real NaNs and hidden placeholder values, reported separately."""
    placeholders = _normalize_placeholders(placeholders)
    report = {}
    n = len(df)
    for column in df.columns:
        series = df[column]
        real = int(series.isna().sum())
        hidden = (
            int(_placeholder_mask(series, placeholders).sum())
            if _is_text(series)
            else 0
        )
        total = real + hidden
        report[str(column)] = {
            "real_missing": real,
            "placeholder_missing": hidden,
            "total_missing": total,
            "missing_percentage": round(total / n * 100, 2) if n else 0.0,
        }
    return report


def find_duplicates(df: pd.DataFrame, ignore_columns: Iterable[object] | None = None,) -> int:
    """Count repeated rows, optionally ignoring columns such as an ID."""
    ignored = (
        {str(column) for column in ignore_columns}
        if ignore_columns is not None
        else set()
    )
    columns = [c for c in df.columns if str(c) not in ignored]
    if not columns:
        return 0
    return int(df[columns].duplicated().sum())


def classify_columns(df: pd.DataFrame,placeholders: Iterable[str] = DEFAULT_PLACEHOLDERS,) -> dict[str, ColumnType]:
    """Map each column to: numeric, categorical, identifier, numeric_as_text."""
    placeholders = _normalize_placeholders(placeholders)
    result = {}

    for column in df.columns:
        series = df[column]
        key = str(column)
        unique_count = series.nunique(dropna=True)
        observed_count = int(series.notna().sum())
        ratio = unique_count / observed_count if observed_count else 0.0
        enough_rows = observed_count >= MIN_ROWS_FOR_RATIOS

        if _looks_like_id_name(key):
            kind = "identifier"
        elif pd.api.types.is_bool_dtype(series):
            kind = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            if enough_rows and ratio >= ID_THRESHOLD and _is_integer_valued(series):
                kind = "identifier"          # e.g. row_number 1..N
            elif ratio <= MAX_CATEGORY_RATIO:
                kind = "categorical"         # e.g. SeniorCitizen 0/1
            else:
                kind = "numeric"
        elif _is_text(series):
            numeric_ratio = _numeric_ratio(series, placeholders)

            if numeric_ratio >= NUMERIC_TEXT_THRESHOLD:
                kind = "numeric_as_text"
            elif _looks_like_free_text(series, key):
                kind = "free_text"
            elif enough_rows and ratio >= ID_THRESHOLD:
                kind = "identifier"
            elif ratio <= MAX_CATEGORY_RATIO:
                kind = "categorical"
            else:
                kind = "free_text"
            if _numeric_ratio(series, placeholders) >= NUMERIC_TEXT_THRESHOLD:
                kind = "numeric_as_text"     # checked BEFORE uniqueness
            elif enough_rows and ratio >= ID_THRESHOLD:
                kind = "identifier"
            elif ratio <= MAX_CATEGORY_RATIO:
                kind = "categorical"
            else:
                kind = "free_text"
        else:
            kind = "categorical"

        result[key] = kind

    return result


def profile_dataframe(
    df: pd.DataFrame,
    placeholders: Iterable[str] = DEFAULT_PLACEHOLDERS,
) -> dict[str, object]:
    placeholders = _normalize_placeholders(placeholders)
    column_types = classify_columns(df, placeholders)
    identifiers = [c for c, k in column_types.items() if k == "identifier"]

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing": find_missing(df, placeholders),
        "duplicates": {
            "exact": find_duplicates(df),
            "ignoring_identifiers": find_duplicates(df, ignore_columns=identifiers),
        },
        "column_types": column_types,
    }