from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.metrics import get_smoothed_mean_log_accuracy_ratio
    from src.paths import DATA_RAW, PREDICTIONS
    from src.features import (
        RANDOM_STATE,
        TARGET_COLUMNS,
        inverse_transform_target,
        append_replay_feature_file,
        prepare_features_for_tasks,
        read_tsv,
        transform_target,
    )
except ModuleNotFoundError:
    from metrics import get_smoothed_mean_log_accuracy_ratio
    from paths import DATA_RAW, PREDICTIONS
    from features import (
        RANDOM_STATE,
        TARGET_COLUMNS,
        inverse_transform_target,
        append_replay_feature_file,
        prepare_features_for_tasks,
        read_tsv,
        transform_target,
    )


def get_time_split_indices(tasks: pd.DataFrame, valid_fraction: float) -> tuple[pd.Index, pd.Index]:
    ordered_indices = tasks.sort_values("hour_start").index
    valid_size = max(1, int(round(len(ordered_indices) * valid_fraction)))
    return ordered_indices[:-valid_size], ordered_indices[-valid_size:]


def make_catboost(params: dict[str, float | int]) -> CatBoostRegressor:
    return CatBoostRegressor(
        **params,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


def suggest_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "iterations": trial.suggest_int("iterations", 250, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
    }


def evaluate_params(
    params: dict[str, float | int],
    features: pd.DataFrame,
    answers: pd.DataFrame,
    train_index: pd.Index,
    valid_index: pd.Index,
) -> float:
    model = MultiOutputRegressor(make_catboost(params))
    model.fit(features.loc[train_index], transform_target(answers.loc[train_index]))
    predictions = inverse_transform_target(model.predict(features.loc[valid_index]))
    prediction_frame = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    return float(
        get_smoothed_mean_log_accuracy_ratio(
            answers.loc[valid_index].reset_index(drop=True),
            prediction_frame,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CatBoost with Optuna on a time split.")
    parser.add_argument("--tasks", default=DATA_RAW / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=DATA_RAW / "validate_answers.tsv", type=Path)
    parser.add_argument("--predict-tasks", default=None, type=Path)
    parser.add_argument("--output", default=PREDICTIONS / "catboost_optuna_predictions.tsv", type=Path)
    parser.add_argument("--features", default=None, type=Path)
    parser.add_argument("--predict-features", default=None, type=Path)
    parser.add_argument("--replay-features", default=None, type=Path)
    parser.add_argument("--predict-replay-features", default=None, type=Path)
    parser.add_argument(
        "--importance-output",
        default=ROOT / "artifacts" / "experiments" / "optuna_hyperparameter_importance.tsv",
        type=Path,
    )
    parser.add_argument("--trials", default=20, type=int)
    parser.add_argument("--time-valid-fraction", default=0.2, type=float)
    args = parser.parse_args()

    train_tasks = read_tsv(args.tasks)
    answers = read_tsv(args.answers)[TARGET_COLUMNS]
    predict_tasks = read_tsv(args.predict_tasks) if args.predict_tasks else train_tasks

    if args.features:
        train_features = read_tsv(args.features)
    else:
        users = read_tsv(DATA_RAW / "users.tsv")
        history = read_tsv(DATA_RAW / "history.tsv")
        train_features = prepare_features_for_tasks(train_tasks, users, history)
    train_features = append_replay_feature_file(train_features, args.replay_features)

    if args.predict_features:
        predict_features = read_tsv(args.predict_features)
    elif args.predict_tasks:
        users = read_tsv(DATA_RAW / "users.tsv")
        history = read_tsv(DATA_RAW / "history.tsv")
        predict_features = prepare_features_for_tasks(predict_tasks, users, history)
    else:
        predict_features = train_features
    if args.predict_features or args.predict_tasks:
        predict_features = append_replay_feature_file(
            predict_features,
            args.predict_replay_features or args.replay_features,
        )

    train_index, valid_index = get_time_split_indices(train_tasks, args.time_valid_fraction)
    print(
        "Time split:"
        f" train rows={len(train_index)}, valid rows={len(valid_index)},"
        f" valid hour_start range={train_tasks.loc[valid_index, 'hour_start'].min()}"
        f"..{train_tasks.loc[valid_index, 'hour_start'].max()}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        return evaluate_params(params, train_features, answers, train_index, valid_index)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("Best time split metric:", round(study.best_value, 4))
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    importances = optuna.importance.get_param_importances(study)
    importance_frame = pd.DataFrame(
        {
            "hyperparameter": list(importances.keys()),
            "importance": list(importances.values()),
        }
    )
    args.importance_output.parent.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(args.importance_output, sep="\t", index=False)
    print(f"Saved hyperparameter importances to: {args.importance_output}")

    model = MultiOutputRegressor(make_catboost(study.best_params))
    model.fit(train_features, transform_target(answers))
    predictions = pd.DataFrame(
        inverse_transform_target(model.predict(predict_features)),
        columns=TARGET_COLUMNS,
    )
    predictions.to_csv(args.output, sep="\t", index=False)

    if args.predict_tasks is None:
        train_metric = get_smoothed_mean_log_accuracy_ratio(answers, predictions)
        print(f"Full-train metric for saved predictions: {train_metric}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
