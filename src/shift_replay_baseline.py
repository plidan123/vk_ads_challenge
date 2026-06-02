from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from metrics import get_smoothed_mean_log_accuracy_ratio  # noqa: E402
from replay_validate import replay_row  # noqa: E402
from session_analysis import add_sessions  # noqa: E402


MONTH_SHIFT_HOURS = 31 * 24


def shifted_row(row: pd.Series, shift_hours: int) -> pd.Series:
    shifted = row.copy()
    shifted["hour_start"] = row["hour_start"] - shift_hours
    shifted["hour_end"] = row["hour_end"] - shift_hours
    return shifted


def replay_shifted_row(row: pd.Series, history: pd.DataFrame, shift_hours: int) -> tuple[float, float, float]:
    available_history = history[history["hour"] < row["hour_start"]]
    return replay_row(shifted_row(row, shift_hours), available_history)


def main() -> None:
    history = pd.read_csv(ROOT / "history.tsv", sep="\t")
    validate = pd.read_csv(ROOT / "validate.tsv", sep="\t")
    answers = pd.read_csv(ROOT / "validate_answers.tsv", sep="\t")

    history = add_sessions(history)
    history["session_key"] = (
        history["user_id"].astype(str) + ":" + history["session_id"].astype(str)
    )

    predictions = validate.apply(
        lambda row: replay_shifted_row(row, history, MONTH_SHIFT_HOURS),
        axis=1,
        result_type="expand",
    )
    predictions.columns = ["at_least_one", "at_least_two", "at_least_three"]
    predictions = predictions.clip(0.0, 1.0)
    predictions.to_csv(ROOT / "shift_replay_predictions.tsv", sep="\t", index=False)

    metric = get_smoothed_mean_log_accuracy_ratio(answers, predictions)
    print(f"metric: {metric}")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
