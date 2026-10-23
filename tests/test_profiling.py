import pandas as pd

from src.profiling import classify_columns




def test_high_uniqueness_ticket_numbers_are_identifiers():
    df = pd.DataFrame({
        "ticket_number": [f"TCK-{i:05d}" for i in range(100)]
    })

    assert classify_columns(df)["ticket_number"] == "identifier"


def test_city_with_40_values_is_categorical():
    df = pd.DataFrame({
        "city": [f"City_{i % 40}" for i in range(10_000)]
    })

    assert classify_columns(df)["city"] == "categorical"


def test_numeric_with_many_repeated_values_is_categorical():
    df = pd.DataFrame({
        "rating": list(range(40)) * 2 + [None] * 20
    })

    assert classify_columns(df)["rating"] == "categorical"


def test_numeric_strings_remain_numeric_as_text():
    df = pd.DataFrame({
        "amount": [str(i) for i in range(100)]
    })

    assert classify_columns(df)["amount"] == "numeric_as_text"


def test_boolean_column_is_categorical():
    df = pd.DataFrame({
        "is_active": [True, False] * 50
    })

    assert classify_columns(df)["is_active"] == "categorical"


def test_id_column_name_overrides_other_heuristics():
    df = pd.DataFrame({
        "customer_id": [f"Customer {i}" for i in range(100)]
    })

    assert classify_columns(df)["customer_id"] == "identifier"