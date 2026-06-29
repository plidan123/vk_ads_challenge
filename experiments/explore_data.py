from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.paths import DATA_RAW
except ModuleNotFoundError:
    from paths import DATA_RAW


def read_tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / name, sep="\t")


def parse_list_column(series: pd.Series) -> pd.Series:
    return series.fillna("").map(
        lambda value: [] if value == "" else [int(item) for item in str(value).split(",")]
    )


def print_section(title: str) -> None:
    print(f"\n## {title}")


def describe_frame(name: str, df: pd.DataFrame) -> None:
    print_section(name)
    print(f"shape: {df.shape}")
    print("columns:", ", ".join(df.columns))
    print("missing values:")
    print(df.isna().sum().to_string())


def main() -> None:
    users = read_tsv("users.tsv")
    history = read_tsv("history.tsv")
    validate = read_tsv("validate.tsv")
    answers = read_tsv("validate_answers.tsv")

    describe_frame("users.tsv", users)
    print("unique users:", users["user_id"].nunique())
    print("age describe:")
    print(users["age"].describe().to_string())
    print("sex counts:")
    print(users["sex"].value_counts(dropna=False).sort_index().to_string())
    print("top cities:")
    print(users["city_id"].value_counts().head(10).to_string())

    describe_frame("history.tsv", history)
    print("hour range:", int(history["hour"].min()), int(history["hour"].max()))
    print("unique users:", history["user_id"].nunique())
    print("unique publishers:", history["publisher"].nunique())
    print("cpm describe:")
    print(history["cpm"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print("top publishers by impressions:")
    print(history["publisher"].value_counts().head(20).to_string())
    print("history impressions by hour modulo day:")
    print(history.assign(hour_of_day=history["hour"] % 24)["hour_of_day"].value_counts().sort_index().to_string())

    describe_frame("validate.tsv", validate)
    publisher_lists = parse_list_column(validate["publishers"])
    user_lists = parse_list_column(validate["user_ids"])
    window_lengths = validate["hour_end"] - validate["hour_start"] + 1
    print("validate hour_start/end ranges:")
    print(
        {
            "hour_start_min": int(validate["hour_start"].min()),
            "hour_start_max": int(validate["hour_start"].max()),
            "hour_end_min": int(validate["hour_end"].min()),
            "hour_end_max": int(validate["hour_end"].max()),
        }
    )
    print("window length describe:")
    print(window_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string())
    print("audience_size describe:")
    print(validate["audience_size"].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string())
    print("publisher count describe:")
    print(publisher_lists.map(len).describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string())
    print("audience_size matches user_ids length:", bool((validate["audience_size"] == user_lists.map(len)).all()))
    print("validate cpm describe:")
    print(validate["cpm"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    describe_frame("validate_answers.tsv", answers)
    print("answers describe:")
    print(answers.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).to_string())


if __name__ == "__main__":
    main()
