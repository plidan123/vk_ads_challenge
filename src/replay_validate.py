from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from metrics import get_smoothed_mean_log_accuracy_ratio  # noqa: E402
from session_analysis import add_sessions  # noqa: E402


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


def main() -> None:
    history = pd.read_csv(ROOT / "history.tsv", sep="\t")
    validate = pd.read_csv(ROOT / "validate.tsv", sep="\t")
    answers = pd.read_csv(ROOT / "validate_answers.tsv", sep="\t")

    history = add_sessions(history)
    history["session_key"] = (
        history["user_id"].astype(str) + ":" + history["session_id"].astype(str)
    )

    predictions = validate.apply(
        lambda row: replay_row(row, history),
        axis=1,
        result_type="expand",
    )
    predictions.columns = ["at_least_one", "at_least_two", "at_least_three"]
    predictions.to_csv(ROOT / "replay_validate_predictions.tsv", sep="\t", index=False)

    metric = get_smoothed_mean_log_accuracy_ratio(answers, predictions)
    print(f"metric: {metric}")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
