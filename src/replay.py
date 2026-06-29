from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from src.metrics import get_smoothed_mean_log_accuracy_ratio
    from src.paths import DATA_RAW, PREDICTIONS
except ModuleNotFoundError:
    from metrics import get_smoothed_mean_log_accuracy_ratio
    from paths import DATA_RAW, PREDICTIONS


MONTH_SHIFT_HOURS = 31 * 24
TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]


def add_sessions(history: pd.DataFrame) -> pd.DataFrame:
    ordered = history.sort_values(["user_id", "hour", "publisher", "cpm"]).copy()
    previous_hour = ordered.groupby("user_id")["hour"].shift(1)
    is_new_session = previous_hour.isna() | ((ordered["hour"] - previous_hour) >= 6)
    ordered["session_id"] = is_new_session.groupby(ordered["user_id"]).cumsum().astype("int32")
    return ordered


def parse_int_list(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item}


def session_success_probability(event_probs: pd.Series) -> float:
    return float(1.0 - np.prod(1.0 - event_probs.to_numpy(dtype=float)))


def at_least_probabilities(session_probs: list[float]) -> tuple[float, float, float]:
    distribution = np.array([1.0, 0.0, 0.0, 0.0])
    for probability in session_probs:
        updated = distribution.copy()
        updated[1:] += distribution[:-1] * probability
        updated[:-1] -= distribution[:-1] * probability
        distribution = updated
    return (
        float(distribution[1:].sum()),
        float(distribution[2:].sum()),
        float(distribution[3]),
    )


def replay_row(row: pd.Series, history: pd.DataFrame) -> tuple[float, float, float]:
    publishers = parse_int_list(row["publishers"])
    users = parse_int_list(row["user_ids"])

    candidates = history[
        (history["hour"] >= row["hour_start"])
        & (history["hour"] <= row["hour_end"])
        & (history["publisher"].isin(publishers))
        & (history["user_id"].isin(users))
    ].copy()

    if candidates.empty:
        return 0.0, 0.0, 0.0

    candidates["win_probability"] = np.select(
        [row["cpm"] > candidates["cpm"], row["cpm"] == candidates["cpm"]],
        [1.0, 0.5],
        default=0.0,
    )
    candidates = candidates[candidates["win_probability"] > 0.0]

    if candidates.empty:
        return 0.0, 0.0, 0.0

    user_counts = []
    for user_id, user_events in candidates.groupby("user_id"):
        session_probs = (
            user_events.groupby("session_key")["win_probability"]
            .apply(session_success_probability)
            .tolist()
        )
        user_counts.append(at_least_probabilities(session_probs))

    per_user = np.array(user_counts)
    audience_size = float(row["audience_size"])
    return tuple(per_user.sum(axis=0) / audience_size)


def add_session_key(history: pd.DataFrame) -> pd.DataFrame:
    history = add_sessions(history)
    history["session_key"] = (
        history["user_id"].astype(str) + ":" + history["session_id"].astype(str)
    )
    return history


def shifted_row(row: pd.Series, shift_hours: int) -> pd.Series:
    shifted = row.copy()
    shifted["hour_start"] = row["hour_start"] - shift_hours
    shifted["hour_end"] = row["hour_end"] - shift_hours
    return shifted


def replay_shifted_row(row: pd.Series, history: pd.DataFrame, shift_hours: int) -> tuple[float, float, float]:
    available_history = history[history["hour"] < row["hour_start"]]
    return replay_row(shifted_row(row, shift_hours), available_history)


def build_replay_predictions(
    tasks: pd.DataFrame,
    history: pd.DataFrame,
    shift_hours: int | None = None,
) -> pd.DataFrame:
    if shift_hours is None:
        predictions = tasks.apply(
            lambda row: replay_row(row, history),
            axis=1,
            result_type="expand",
        )
    else:
        predictions = tasks.apply(
            lambda row: replay_shifted_row(row, history, shift_hours),
            axis=1,
            result_type="expand",
        )
    predictions.columns = TARGET_COLUMNS
    return predictions.clip(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay auction predictions.")
    parser.add_argument("--tasks", default=DATA_RAW / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=DATA_RAW / "validate_answers.tsv", type=Path)
    parser.add_argument("--history", default=DATA_RAW / "history.tsv", type=Path)
    parser.add_argument("--mode", choices=["shift", "direct"], default="shift")
    parser.add_argument("--output", default=PREDICTIONS / "shift_replay_predictions.tsv", type=Path)
    args = parser.parse_args()

    history = add_session_key(pd.read_csv(args.history, sep="\t"))
    tasks = pd.read_csv(args.tasks, sep="\t")
    answers = pd.read_csv(args.answers, sep="\t")
    shift_hours = MONTH_SHIFT_HOURS if args.mode == "shift" else None

    predictions = build_replay_predictions(tasks, history, shift_hours=shift_hours)
    predictions.to_csv(args.output, sep="\t", index=False)

    metric = get_smoothed_mean_log_accuracy_ratio(answers, predictions)
    print(f"metric: {metric}")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
