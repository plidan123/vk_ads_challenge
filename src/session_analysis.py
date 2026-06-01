from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_history() -> pd.DataFrame:
    return pd.read_csv(ROOT / "history.tsv", sep="\t")


def add_sessions(history: pd.DataFrame) -> pd.DataFrame:
    ordered = history.sort_values(["user_id", "hour", "publisher", "cpm"]).copy()
    previous_hour = ordered.groupby("user_id")["hour"].shift(1)  # groupby чтобы у каждого нового пользователя previous hour начинался с Nan
    is_new_session = previous_hour.isna() | ((ordered["hour"] - previous_hour) >= 6)
    ordered["session_id"] = is_new_session.groupby(ordered["user_id"]).cumsum().astype("int32")
    return ordered


def print_describe(name: str, series: pd.Series) -> None:
    print(f"\n## {name}")
    print(series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())


def main() -> None:
    history = read_history()
    history_sessions = add_sessions(history)

    session_stats = (
        history_sessions.groupby(["user_id", "session_id"])
        .agg(
            start_hour=("hour", "min"),
            end_hour=("hour", "max"),
            impressions=("hour", "size"),
            publishers=("publisher", "nunique"),
        )
        .reset_index()
    )
    session_stats["duration_hours"] = session_stats["end_hour"] - session_stats["start_hour"] + 1

    user_stats = (
        session_stats.groupby("user_id")
        .agg(
            sessions=("session_id", "size"),
            impressions=("impressions", "sum"),
            mean_session_impressions=("impressions", "mean"),
            mean_session_duration=("duration_hours", "mean"),
        )
        .reset_index()
    )

    print("history rows:", len(history_sessions))
    print("users with history:", history_sessions["user_id"].nunique())
    print("sessions:", len(session_stats))
    print("mean sessions per active user:", round(user_stats["sessions"].mean(), 4))

    print_describe("sessions per user", user_stats["sessions"])
    print_describe("impressions per user", user_stats["impressions"])
    print_describe("impressions per session", session_stats["impressions"])
    print_describe("session duration hours", session_stats["duration_hours"])
    print_describe("publishers per session", session_stats["publishers"])

    print("\n## top session counts")
    print(user_stats.sort_values("sessions", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
