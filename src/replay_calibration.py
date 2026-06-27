from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from metrics import get_smoothed_mean_log_accuracy_ratio  # noqa: E402
from ml_baseline import (  # noqa: E402
    EPSILON,
    RANDOM_STATE,
    TARGET_COLUMNS,
    append_replay_feature_file,
    prepare_features_for_tasks,
    read_tsv,
    transform_target,
)


REPLAY_COLUMNS = {
    "at_least_one": "shift_replay_at_least_one",
    "at_least_two": "shift_replay_at_least_two",
    "at_least_three": "shift_replay_at_least_three",
}
SHRINK_VALUES = [round(value, 2) for value in np.arange(0.0, 1.01, 0.05)]


def make_calibration_models() -> dict[str, object]:
    models: dict[str, object] = {
        "ridge_50": make_pipeline(StandardScaler(), Ridge(alpha=50.0)),
        "elastic_net": make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.1,
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
        ),
        "hist_gradient_boosting_conservative": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.025,
            max_leaf_nodes=8,
            l2_regularization=0.2,
            random_state=RANDOM_STATE,
        ),
    }

    if CatBoostRegressor is not None:
        models["catboost_conservative"] = CatBoostRegressor(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def get_time_split_indices(tasks: pd.DataFrame, valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    ordered_indices = tasks.sort_values("hour_start").index.to_numpy()
    valid_size = max(1, int(round(len(ordered_indices) * valid_fraction)))
    return ordered_indices[:-valid_size], ordered_indices[-valid_size:]


def get_base_predictions(features: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REPLAY_COLUMNS.values() if column not in features.columns]
    if missing:
        raise ValueError(
            "Replay calibration requires shift replay features. "
            f"Missing columns: {', '.join(missing)}"
        )
    return features[list(REPLAY_COLUMNS.values())].rename(
        columns={source: target for target, source in REPLAY_COLUMNS.items()}
    )


def inverse_log_predictions(log_predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(log_predictions) - EPSILON, 0.0, 1.0)


def fit_predict_residuals(
    base_model: object,
    features: pd.DataFrame,
    answers: pd.DataFrame,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    base_predictions = get_base_predictions(features)
    base_log = transform_target(base_predictions)
    answer_log = transform_target(answers)
    residual_target = answer_log - base_log

    model = MultiOutputRegressor(base_model)
    model.fit(features.iloc[train_index], residual_target[train_index])
    residual_predictions = model.predict(features.iloc[valid_index])

    prediction_frames = {}
    for shrink in SHRINK_VALUES:
        calibrated = inverse_log_predictions(base_log[valid_index] + shrink * residual_predictions)
        prediction_frames[shrink] = pd.DataFrame(calibrated, columns=TARGET_COLUMNS)

    metrics = []
    valid_answers = answers.iloc[valid_index].reset_index(drop=True)
    for shrink, predictions in prediction_frames.items():
        metrics.append(
            {
                "shrink": shrink,
                "metric": get_smoothed_mean_log_accuracy_ratio(valid_answers, predictions),
            }
        )

    return pd.DataFrame(metrics), prediction_frames


def evaluate_time_split(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    tasks: pd.DataFrame,
    valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_index, valid_index = get_time_split_indices(tasks, valid_fraction)
    base_predictions = get_base_predictions(features).iloc[valid_index].reset_index(drop=True)
    valid_answers = answers.iloc[valid_index].reset_index(drop=True)

    predictions_by_key: dict[tuple[str, float], pd.DataFrame] = {
        ("shift_replay", 0.0): base_predictions.copy()
    }
    rows = [
        {
            "model": "shift_replay",
            "shrink": 0.0,
            "metric": get_smoothed_mean_log_accuracy_ratio(valid_answers, base_predictions),
        }
    ]

    for model_name, base_model in make_calibration_models().items():
        shrink_metrics, prediction_frames = fit_predict_residuals(
            base_model,
            features,
            answers,
            train_index,
            valid_index,
        )
        for row in shrink_metrics.to_dict("records"):
            rows.append({"model": model_name, **row})
            predictions_by_key[(model_name, float(row["shrink"]))] = prediction_frames[float(row["shrink"])]

    scores = pd.DataFrame(rows).sort_values("metric", kind="stable")
    best = scores.iloc[0]
    best_predictions = predictions_by_key[(str(best["model"]), float(best["shrink"]))]
    return scores, best_predictions


def evaluate_kfold(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_predictions = get_base_predictions(features)
    predictions_by_key: dict[tuple[str, float], pd.DataFrame] = {
        ("shift_replay", 0.0): base_predictions.copy()
    }
    rows = [
        {
            "model": "shift_replay",
            "shrink": 0.0,
            "metric": get_smoothed_mean_log_accuracy_ratio(answers, base_predictions),
        }
    ]

    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for model_name, base_model in make_calibration_models().items():
        residual_oof = np.zeros((len(features), len(TARGET_COLUMNS)), dtype=float)
        base_log = transform_target(base_predictions)
        answer_log = transform_target(answers)
        residual_target = answer_log - base_log

        for train_index, valid_index in kfold.split(features):
            model = MultiOutputRegressor(base_model)
            model.fit(features.iloc[train_index], residual_target[train_index])
            residual_oof[valid_index] = model.predict(features.iloc[valid_index])

        for shrink in SHRINK_VALUES:
            predictions = pd.DataFrame(
                inverse_log_predictions(base_log + shrink * residual_oof),
                columns=TARGET_COLUMNS,
            )
            rows.append(
                {
                    "model": model_name,
                    "shrink": shrink,
                    "metric": get_smoothed_mean_log_accuracy_ratio(answers, predictions),
                }
            )
            predictions_by_key[(model_name, shrink)] = predictions

    scores = pd.DataFrame(rows).sort_values("metric", kind="stable")
    best = scores.iloc[0]
    best_predictions = predictions_by_key[(str(best["model"]), float(best["shrink"]))]
    return scores, best_predictions


def load_features(args: argparse.Namespace, tasks: pd.DataFrame) -> pd.DataFrame:
    if args.features:
        features = read_tsv(args.features)
    else:
        users = read_tsv(ROOT / "users.tsv")
        history = read_tsv(ROOT / "history.tsv")
        features = prepare_features_for_tasks(
            tasks,
            users,
            history,
            include_shift_replay=args.replay_features is None,
        )
    return append_replay_feature_file(features, args.replay_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate shift replay predictions with residual ML.")
    parser.add_argument("--tasks", default=ROOT / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=ROOT / "validate_answers.tsv", type=Path)
    parser.add_argument("--features", default=None, type=Path)
    parser.add_argument("--replay-features", default=ROOT / "shift_replay_predictions.tsv", type=Path)
    parser.add_argument("--output", default=ROOT / "replay_calibrated_predictions.tsv", type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--validation", choices=["kfold", "time"], default="time")
    parser.add_argument("--time-valid-fraction", default=0.2, type=float)
    args = parser.parse_args()

    tasks = read_tsv(args.tasks)
    answers = read_tsv(args.answers)[TARGET_COLUMNS]
    features = load_features(args, tasks)

    if args.validation == "time":
        scores, predictions = evaluate_time_split(
            features,
            answers,
            tasks,
            args.time_valid_fraction,
        )
        print("Time split residual calibration metric:")
    else:
        scores, predictions = evaluate_kfold(features, answers, args.folds)
        print("K-fold residual calibration metric:")

    print(scores.to_string(index=False))
    predictions.to_csv(args.output, sep="\t", index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
