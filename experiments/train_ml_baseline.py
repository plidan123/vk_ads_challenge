from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.features import (
        RANDOM_STATE,
        TARGET_COLUMNS,
        append_replay_feature_file,
        inverse_transform_target,
        prepare_features_for_tasks,
        read_tsv,
        transform_target,
    )
    from src.metrics import get_smoothed_mean_log_accuracy_ratio
    from src.paths import DATA_RAW, PREDICTIONS
except ModuleNotFoundError:
    from features import (
        RANDOM_STATE,
        TARGET_COLUMNS,
        append_replay_feature_file,
        inverse_transform_target,
        prepare_features_for_tasks,
        read_tsv,
        transform_target,
    )
    from metrics import get_smoothed_mean_log_accuracy_ratio
    from paths import DATA_RAW, PREDICTIONS


def make_ml_models() -> dict[str, object]:
    models: dict[str, object] = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    if CatBoostRegressor is not None:
        models["catboost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def cross_validate_models(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    folds: int,
) -> tuple[str, pd.DataFrame]:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for name, base_model in make_ml_models().items():
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


def evaluate_time_split(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    tasks: pd.DataFrame,
    valid_fraction: float,
) -> tuple[str, pd.DataFrame]:
    ordered_indices = tasks.sort_values("hour_start").index.to_numpy()
    valid_size = max(1, int(round(len(ordered_indices) * valid_fraction)))
    train_index = ordered_indices[:-valid_size]
    valid_index = ordered_indices[-valid_size:]
    scores = []

    print(
        "Time split:"
        f" train rows={len(train_index)}, valid rows={len(valid_index)},"
        f" valid hour_start range={tasks.loc[valid_index, 'hour_start'].min()}"
        f"..{tasks.loc[valid_index, 'hour_start'].max()}"
    )

    for name, base_model in make_ml_models().items():
        model = MultiOutputRegressor(base_model)
        model.fit(features.loc[train_index], transform_target(answers.loc[train_index]))
        predictions = inverse_transform_target(model.predict(features.loc[valid_index]))
        prediction_frame = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
        metric = get_smoothed_mean_log_accuracy_ratio(
            answers.loc[valid_index].reset_index(drop=True),
            prediction_frame,
        )
        scores.append({"model": name, "metric": metric})

    score_frame = pd.DataFrame(scores).sort_values("metric", kind="stable").reset_index(drop=True)
    return str(score_frame.loc[0, "model"]), score_frame


def fit_predict_best_model(
    model_name: str,
    train_features: pd.DataFrame,
    answers: pd.DataFrame,
    predict_features: pd.DataFrame,
) -> pd.DataFrame:
    model = MultiOutputRegressor(make_ml_models()[model_name])
    model.fit(train_features, transform_target(answers))
    predictions = inverse_transform_target(model.predict(predict_features))
    return pd.DataFrame(predictions, columns=TARGET_COLUMNS)


def load_features_for_tasks(tasks: pd.DataFrame, features_path: Path | None) -> pd.DataFrame:
    if features_path:
        return read_tsv(features_path)
    users = read_tsv(DATA_RAW / "users.tsv")
    history = read_tsv(DATA_RAW / "history.tsv")
    return prepare_features_for_tasks(tasks, users, history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ML baseline.")
    parser.add_argument("--tasks", default=DATA_RAW / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=DATA_RAW / "validate_answers.tsv", type=Path)
    parser.add_argument("--predict-tasks", default=None, type=Path)
    parser.add_argument("--output", default=PREDICTIONS / "ml_baseline_predictions.tsv", type=Path)
    parser.add_argument("--features", default=None, type=Path)
    parser.add_argument("--predict-features", default=None, type=Path)
    parser.add_argument("--replay-features", default=None, type=Path)
    parser.add_argument("--predict-replay-features", default=None, type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--validation", choices=["kfold", "time"], default="kfold")
    parser.add_argument("--time-valid-fraction", default=0.2, type=float)
    args = parser.parse_args()

    train_tasks = read_tsv(args.tasks)
    answers = read_tsv(args.answers)[TARGET_COLUMNS]
    predict_tasks = read_tsv(args.predict_tasks) if args.predict_tasks else train_tasks

    train_features = load_features_for_tasks(train_tasks, args.features)
    train_features = append_replay_feature_file(train_features, args.replay_features)

    if args.predict_features:
        predict_features = read_tsv(args.predict_features)
    elif args.predict_tasks:
        predict_features = load_features_for_tasks(predict_tasks, None)
    else:
        predict_features = train_features
    if args.predict_features or args.predict_tasks:
        predict_features = append_replay_feature_file(
            predict_features,
            args.predict_replay_features or args.replay_features,
        )

    if args.validation == "time":
        best_model, score_frame = evaluate_time_split(
            train_features,
            answers,
            train_tasks,
            args.time_valid_fraction,
        )
        print("Time split metric:")
    else:
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
