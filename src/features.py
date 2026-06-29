from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from src.paths import DATA_FEATURES, DATA_RAW
    from src.replay import MONTH_SHIFT_HOURS, add_sessions, replay_shifted_row
except ModuleNotFoundError:
    from paths import DATA_FEATURES, DATA_RAW
    from replay import MONTH_SHIFT_HOURS, add_sessions, replay_shifted_row


TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]
EPSILON = 0.005
RANDOM_STATE = 42
RECENT_WINDOWS = [31 * 24]


def parse_int_list(value: str) -> list[int]:
    if pd.isna(value) or value == "":
        return []
    return [int(item) for item in str(value).split(",") if item]


def read_tsv(filename: str | Path) -> pd.DataFrame:
    return pd.read_csv(filename, sep="\t")


def add_user_history_features(users: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    history_extended = history.assign(hour_of_day=history["hour"] % 24)
    user_history = (
        history_extended.groupby("user_id")
        .agg(
            user_impressions=("hour", "size"),
            user_active_hours=("hour", "nunique"),
            user_publishers=("publisher", "nunique"),
            user_cpm_mean=("cpm", "mean"),
            user_cpm_median=("cpm", "median"),
            user_cpm_max=("cpm", "max"),
            user_hour_mean=("hour_of_day", "mean"),
            user_hour_std=("hour_of_day", "std"),
        )
        .reset_index()
    )

    features = users.merge(user_history, on="user_id", how="left")
    history_columns = [column for column in features.columns if column.startswith("user_")]
    features[history_columns] = features[history_columns].fillna(0.0)
    return features.set_index("user_id")


def build_publisher_features(history: pd.DataFrame) -> pd.DataFrame:
    history_extended = history.assign(hour_of_day=history["hour"] % 24)
    publisher_features = (
        history_extended.groupby("publisher")
        .agg(
            pub_impressions=("hour", "size"),
            pub_users=("user_id", "nunique"),
            pub_cpm_mean=("cpm", "mean"),
            pub_cpm_median=("cpm", "median"),
            pub_cpm_p75=("cpm", lambda values: values.quantile(0.75)),
            pub_cpm_p90=("cpm", lambda values: values.quantile(0.90)),
            pub_cpm_p95=("cpm", lambda values: values.quantile(0.95)),
            pub_hour_mean=("hour_of_day", "mean"),
            pub_hour_std=("hour_of_day", "std"),
        )
        .fillna(0.0)
    )
    return publisher_features


def aggregate_numeric(prefix: str, values: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    if values.empty:
        for column in columns:
            result[f"{prefix}_{column}_mean"] = 0.0
            result[f"{prefix}_{column}_median"] = 0.0
            result[f"{prefix}_{column}_max"] = 0.0
        return result

    selected = values[columns]
    for column in columns:
        result[f"{prefix}_{column}_mean"] = float(selected[column].mean())
        result[f"{prefix}_{column}_median"] = float(selected[column].median())
        result[f"{prefix}_{column}_max"] = float(selected[column].max())
    return result


def build_history_index(history: pd.DataFrame, key: str) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for value, group in history.sort_values([key, "hour"]).groupby(key):
        result[int(value)] = {
            "hour": group["hour"].to_numpy(),
            "cpm": group["cpm"].to_numpy(dtype=float),
            "user_id": group["user_id"].to_numpy(),
            "session_key": group["session_key"].to_numpy(),
        }
    return result


def sliced_stats(
    groups: list[dict[str, np.ndarray]],
    start_hour: int,
    end_hour: int,
    cpm: float,
    audience_size: float,
    prefix: str,
    feature_row: dict[str, float],
) -> None:
    impressions = 0
    active_entities = 0
    session_count = 0
    cpm_sum = 0.0
    win_probability_sum = 0.0

    for group in groups:
        hours = group["hour"]
        left = int(np.searchsorted(hours, start_hour, side="left"))
        right = int(np.searchsorted(hours, end_hour, side="left"))
        if right <= left:
            continue

        active_entities += 1
        group_cpms = group["cpm"][left:right]
        impressions += len(group_cpms)
        cpm_sum += float(group_cpms.sum())
        session_count += len(np.unique(group["session_key"][left:right]))
        win_probability_sum += float(
            np.select(
                [cpm > group_cpms, cpm == group_cpms],
                [1.0, 0.5],
                default=0.0,
            ).sum()
        )

    feature_row[f"{prefix}_impressions"] = float(impressions)
    feature_row[f"{prefix}_log_impressions"] = float(np.log1p(impressions))
    feature_row[f"{prefix}_active_entities"] = float(active_entities)
    feature_row[f"{prefix}_active_entity_share"] = active_entities / max(audience_size, 1.0)
    feature_row[f"{prefix}_sessions"] = float(session_count)
    feature_row[f"{prefix}_sessions_per_user"] = session_count / max(audience_size, 1.0)

    if impressions == 0:
        feature_row[f"{prefix}_cpm_mean"] = 0.0
        feature_row[f"{prefix}_win_rate"] = 0.0
        return

    feature_row[f"{prefix}_cpm_mean"] = cpm_sum / impressions
    feature_row[f"{prefix}_win_rate"] = win_probability_sum / impressions


def intersection_sliced_stats(
    publisher_groups: list[dict[str, np.ndarray]],
    user_ids: list[int],
    start_hour: int,
    end_hour: int,
    cpm: float,
    audience_size: float,
    prefix: str,
    feature_row: dict[str, float],
) -> None:
    user_array = np.array(user_ids)
    impressions = 0
    active_users: set[int] = set()
    session_count = 0
    cpm_sum = 0.0
    win_probability_sum = 0.0

    for group in publisher_groups:
        hours = group["hour"]
        left = int(np.searchsorted(hours, start_hour, side="left"))
        right = int(np.searchsorted(hours, end_hour, side="left"))
        if right <= left:
            continue

        group_users = group["user_id"][left:right]
        mask = np.isin(group_users, user_array)
        if not mask.any():
            continue

        group_cpms = group["cpm"][left:right][mask]
        group_sessions = group["session_key"][left:right][mask]
        matched_users = group_users[mask]

        impressions += len(group_cpms)
        active_users.update(int(user_id) for user_id in matched_users)
        session_count += len(np.unique(group_sessions))
        cpm_sum += float(group_cpms.sum())
        win_probability_sum += float(
            np.select(
                [cpm > group_cpms, cpm == group_cpms],
                [1.0, 0.5],
                default=0.0,
            ).sum()
        )

    feature_row[f"{prefix}_impressions"] = float(impressions)
    feature_row[f"{prefix}_log_impressions"] = float(np.log1p(impressions))
    feature_row[f"{prefix}_active_users"] = float(len(active_users))
    feature_row[f"{prefix}_active_user_share"] = len(active_users) / max(audience_size, 1.0)
    feature_row[f"{prefix}_sessions"] = float(session_count)
    feature_row[f"{prefix}_sessions_per_user"] = session_count / max(audience_size, 1.0)

    if impressions == 0:
        feature_row[f"{prefix}_cpm_mean"] = 0.0
        feature_row[f"{prefix}_win_rate"] = 0.0
        return

    feature_row[f"{prefix}_cpm_mean"] = cpm_sum / impressions
    feature_row[f"{prefix}_win_rate"] = win_probability_sum / impressions


def add_recent_history_features(
    feature_row: dict[str, float],
    row: pd.Series,
    publishers: list[int],
    user_ids: list[int],
    user_history_index: dict[int, dict[str, np.ndarray]],
    publisher_history_index: dict[int, dict[str, np.ndarray]],
) -> None:
    audience_size = float(row.audience_size)
    cpm = float(row.cpm)
    user_groups = [user_history_index[user_id] for user_id in user_ids if user_id in user_history_index]
    publisher_groups = [
        publisher_history_index[publisher]
        for publisher in publishers
        if publisher in publisher_history_index
    ]

    for window_hours in RECENT_WINDOWS:
        days = window_hours // 24
        start_hour = int(row.hour_start - window_hours)
        end_hour = int(row.hour_start)

        sliced_stats(
            user_groups,
            start_hour,
            end_hour,
            cpm,
            audience_size,
            f"recent_{days}d_audience",
            feature_row,
        )
        sliced_stats(
            publisher_groups,
            start_hour,
            end_hour,
            cpm,
            audience_size,
            f"recent_{days}d_publishers",
            feature_row,
        )
        intersection_sliced_stats(
            publisher_groups,
            user_ids,
            start_hour,
            end_hour,
            cpm,
            audience_size,
            f"recent_{days}d_intersection",
            feature_row,
        )


def build_campaign_features(
    tasks: pd.DataFrame,
    user_features: pd.DataFrame,
    publisher_features: pd.DataFrame,
    user_history_index: dict[int, dict[str, np.ndarray]],
    publisher_history_index: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    user_columns = [
        "sex",
        "age",
        "city_id",
        "user_impressions",
        "user_active_hours",
        "user_publishers",
        "user_cpm_mean",
        "user_cpm_median",
        "user_cpm_max",
        "user_hour_mean",
        "user_hour_std",
    ]
    publisher_columns = list(publisher_features.columns)
    rows: list[dict[str, float]] = []

    for row in tasks.itertuples(index=False):
        publishers = parse_int_list(row.publishers)
        user_ids = parse_int_list(row.user_ids)
        window_length = row.hour_end - row.hour_start + 1
        selected_users = user_features.reindex(user_ids).fillna(0.0)
        selected_publishers = publisher_features.reindex(publishers).fillna(0.0)

        feature_row: dict[str, float] = {
            "cpm": float(row.cpm),
            "log_cpm": float(np.log1p(row.cpm)),
            "hour_start": float(row.hour_start),
            "hour_end": float(row.hour_end),
            "window_length": float(window_length),
            "log_window_length": float(np.log1p(window_length)),
            "start_hour_of_day": float(row.hour_start % 24),
            "end_hour_of_day": float(row.hour_end % 24),
            "publisher_count": float(len(publishers)),
            "audience_size": float(row.audience_size),
            "log_audience_size": float(np.log1p(row.audience_size)),
        }

        feature_row.update(aggregate_numeric("audience", selected_users, user_columns))
        feature_row.update(aggregate_numeric("publisher", selected_publishers, publisher_columns))
        add_recent_history_features(
            feature_row,
            row,
            publishers,
            user_ids,
            user_history_index,
            publisher_history_index,
        )

        if selected_publishers.empty:
            feature_row["cpm_to_pub_p90"] = 0.0
            feature_row["cpm_to_pub_p95"] = 0.0
        else:
            pub_p90 = float(selected_publishers["pub_cpm_p90"].mean())
            pub_p95 = float(selected_publishers["pub_cpm_p95"].mean())
            feature_row["cpm_to_pub_p90"] = float(row.cpm / (pub_p90 + EPSILON))
            feature_row["cpm_to_pub_p95"] = float(row.cpm / (pub_p95 + EPSILON))

        rows.append(feature_row)

    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_shift_replay_features(tasks: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    replay_features = tasks.apply(
        lambda row: replay_shifted_row(row, history, MONTH_SHIFT_HOURS),
        axis=1,
        result_type="expand",
    )
    replay_features.columns = [
        "shift_replay_at_least_one",
        "shift_replay_at_least_two",
        "shift_replay_at_least_three",
    ]
    return replay_features.reset_index(drop=True)


def add_shift_replay_features(
    features: pd.DataFrame,
    tasks: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    replay_features = build_shift_replay_features(tasks, history)
    return pd.concat(
        [
            features.reset_index(drop=True),
            replay_features,
        ],
        axis=1,
    )


def prepare_features_for_tasks(
    tasks: pd.DataFrame,
    users: pd.DataFrame,
    history: pd.DataFrame,
    include_shift_replay: bool = True,
) -> pd.DataFrame:
    history_sessions = add_sessions(history)
    history_sessions["session_key"] = (
        history_sessions["user_id"].astype(str) + ":" + history_sessions["session_id"].astype(str)
    )
    user_history_index = build_history_index(history_sessions, "user_id")
    publisher_history_index = build_history_index(history_sessions, "publisher")

    user_features = add_user_history_features(users, history)
    publisher_features = build_publisher_features(history)
    features = build_campaign_features(
        tasks,
        user_features,
        publisher_features,
        user_history_index,
        publisher_history_index,
    )
    if include_shift_replay:
        return add_shift_replay_features(features, tasks, history_sessions)
    return features


def read_replay_features(path: Path) -> pd.DataFrame:
    replay_features = read_tsv(path)
    return replay_features.rename(
        columns={
            "at_least_one": "shift_replay_at_least_one",
            "at_least_two": "shift_replay_at_least_two",
            "at_least_three": "shift_replay_at_least_three",
        }
    )


def append_replay_feature_file(features: pd.DataFrame, replay_features_path: Path | None) -> pd.DataFrame:
    if replay_features_path is None:
        return features
    replay_features = read_replay_features(replay_features_path)
    return pd.concat([features.reset_index(drop=True), replay_features.reset_index(drop=True)], axis=1)


def transform_target(y: pd.DataFrame) -> np.ndarray:
    return np.log(y.to_numpy(dtype=float) + EPSILON)


def inverse_transform_target(y_transformed: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(y_transformed) - EPSILON, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build campaign feature table.")
    parser.add_argument("--tasks", default=DATA_RAW / "validate.tsv", type=Path)
    parser.add_argument("--output", default=DATA_FEATURES / "features_validate.tsv", type=Path)
    parser.add_argument("--include-shift-replay", action="store_true")
    args = parser.parse_args()

    users = read_tsv(DATA_RAW / "users.tsv")
    history = read_tsv(DATA_RAW / "history.tsv")
    tasks = read_tsv(args.tasks)
    features = prepare_features_for_tasks(
        tasks,
        users,
        history,
        include_shift_replay=args.include_shift_replay,
    )
    features.to_csv(args.output, sep="\t", index=False)
    print(f"Built features: rows={len(features)}, columns={len(features.columns)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
