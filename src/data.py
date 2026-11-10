import pandas as pd

FEATURE = "funding_total_usd"
TARGET = "status"

def load_dataset():

    data = pd.read_csv("datasets/data.csv")

    data = data.drop_duplicates().copy()

    data[FEATURE] = pd.to_numeric(data[FEATURE], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    if data[[FEATURE, TARGET]].isna().any().any():
        raise ValueError("Invalid values found in the dataset.")

    if (data[FEATURE] < 0).any():
        raise ValueError("Funding cannot be negative.")

    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Status must contain both 0 and 1.")

    return data.reset_index(drop=True)
