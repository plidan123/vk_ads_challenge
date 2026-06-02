from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from metrics import get_smoothed_mean_log_accuracy_ratio  # noqa: E402


TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]
EPSILON = 0.005
RANDOM_STATE = 42


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


def build_campaign_features(
    tasks: pd.DataFrame,
    user_features: pd.DataFrame,
    publisher_features: pd.DataFrame,
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


def make_models() -> dict[str, object]:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def transform_target(y: pd.DataFrame) -> np.ndarray:
    return np.log(y.to_numpy(dtype=float) + EPSILON)


def inverse_transform_target(y_transformed: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(y_transformed) - EPSILON, 0.0, 1.0)


def cross_validate_models(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    folds: int,
) -> tuple[str, pd.DataFrame]:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for name, base_model in make_models().items():
        oof_predictions = np.zeros((len(features), len(TARGET_COLUMNS)), dtype=float)
        for train_index, valid_index in kfold.split(features):
            model = MultiOutputRegressor(base_model)
            model.fit(features.iloc[train_index], transform_target(answers.iloc[train_index]))
            fold_predictions = inverse_transform_target(model.predict(features.iloc[valid_index]))
            oof_predictions[valid_index] = fold_predictions

        prediction_frame = pd.DataFrame(oof_predictions, columns=TARGET_COLUMNS)
        metric = get_smoothed_mean_log_accuracy_ratio(answers, prediction_frame)
        scores.append({"model": name, "metric": metric})

    score_frame = pd.DataFrame(scores).sort_values("metric", kind="stable").reset_index(drop=True)
    return str(score_frame.loc[0, "model"]), score_frame


def fit_predict_best_model(
    model_name: str,
    train_features: pd.DataFrame,
    answers: pd.DataFrame,
    predict_features: pd.DataFrame,
) -> pd.DataFrame:
    model = MultiOutputRegressor(make_models()[model_name])
    model.fit(train_features, transform_target(answers))
    predictions = inverse_transform_target(model.predict(predict_features))
    return pd.DataFrame(predictions, columns=TARGET_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate simple ML baselines.")
    parser.add_argument("--tasks", default=ROOT / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=ROOT / "validate_answers.tsv", type=Path)
    parser.add_argument("--predict-tasks", default=None, type=Path)
    parser.add_argument("--output", default=ROOT / "ml_baseline_predictions.tsv", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    args = parser.parse_args()

    users = read_tsv(ROOT / "users.tsv")
    history = read_tsv(ROOT / "history.tsv")
    train_tasks = read_tsv(args.tasks)
    answers = read_tsv(args.answers)[TARGET_COLUMNS]
    predict_tasks = read_tsv(args.predict_tasks) if args.predict_tasks else train_tasks

    user_features = add_user_history_features(users, history)
    publisher_features = build_publisher_features(history)
    train_features = build_campaign_features(train_tasks, user_features, publisher_features)
    predict_features = build_campaign_features(predict_tasks, user_features, publisher_features)

    best_model, score_frame = cross_validate_models(train_features, answers, args.folds)
    print("Cross-validation metric:")
    print(score_frame.to_string(index=False))
    print(f"\nBest model: {best_model}")

    predictions = fit_predict_best_model(best_model, train_features, answers, predict_features)
    predictions.to_csv(args.output, sep="\t", index=False)
    if args.predict_tasks is None:
        train_metric = get_smoothed_mean_log_accuracy_ratio(answers, predictions)
        print(f"Full-train metric for saved predictions: {train_metric}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
