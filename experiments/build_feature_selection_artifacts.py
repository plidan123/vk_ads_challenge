from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from features import (  # noqa: E402
    EPSILON,
    RANDOM_STATE,
    TARGET_COLUMNS,
    append_replay_feature_file,
    inverse_transform_target,
    read_tsv,
    transform_target,
)
from metrics import get_smoothed_mean_log_accuracy_ratio  # noqa: E402
from paths import DATA_FEATURES, DATA_RAW, PREDICTIONS  # noqa: E402
from train_replay_calibration import evaluate_time_split as evaluate_calibration_time_split  # noqa: E402


REPLAY_COLUMNS = [
    "shift_replay_at_least_one",
    "shift_replay_at_least_two",
    "shift_replay_at_least_three",
]


def feature_group(feature: str) -> str:
    if feature.startswith("shift_replay_"):
        return "replay"
    if feature.startswith("audience_"):
        return "audience"
    if feature.startswith("publisher_"):
        return "publisher"
    if feature.startswith("recent_"):
        return "recent"
    return "base"


def get_time_split_indices(tasks: pd.DataFrame, valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    ordered_indices = tasks.sort_values("hour_start").index.to_numpy()
    valid_size = max(1, int(round(len(ordered_indices) * valid_fraction)))
    return ordered_indices[:-valid_size], ordered_indices[-valid_size:]


def make_ml_models() -> dict[str, object]:
    models: dict[str, object] = {
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


def make_importance_model() -> object:
    if CatBoostRegressor is not None:
        return CatBoostRegressor(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model_time_split(
    model: object,
    features: pd.DataFrame,
    answers: pd.DataFrame,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> float:
    estimator = MultiOutputRegressor(clone(model))
    estimator.fit(features.iloc[train_index], transform_target(answers.iloc[train_index]))
    predictions = inverse_transform_target(estimator.predict(features.iloc[valid_index]))
    prediction_frame = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    return float(
        get_smoothed_mean_log_accuracy_ratio(
            answers.iloc[valid_index].reset_index(drop=True),
            prediction_frame,
        )
    )


def rank_features(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    train_index: np.ndarray,
) -> list[str]:
    model = MultiOutputRegressor(make_importance_model())
    model.fit(features.iloc[train_index], transform_target(answers.iloc[train_index]))
    importances = np.vstack(
        [estimator.feature_importances_ for estimator in model.estimators_]
    ).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": features.columns, "importance": importances})
        .sort_values("importance", ascending=False, kind="stable")
        ["feature"]
        .tolist()
    )
    return ranking


def build_feature_sets(features: pd.DataFrame, ranked_features: list[str]) -> dict[str, list[str]]:
    groups = {name: [column for column in features.columns if feature_group(column) == name] for name in [
        "base",
        "audience",
        "publisher",
        "recent",
        "replay",
    ]}
    all_features = list(features.columns)
    sets: dict[str, list[str]] = {
        "all": all_features,
        "without_recent": [column for column in all_features if feature_group(column) != "recent"],
        "without_publisher": [column for column in all_features if feature_group(column) != "publisher"],
        "without_audience": [column for column in all_features if feature_group(column) != "audience"],
        "without_base": [column for column in all_features if feature_group(column) != "base"],
        "without_replay": [column for column in all_features if feature_group(column) != "replay"],
        "only_base": groups["base"],
        "only_recent": groups["recent"],
        "only_audience": groups["audience"],
        "only_publisher": groups["publisher"],
        "only_replay": groups["replay"],
    }
    for count in [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80]:
        sets[f"top_{count}"] = ranked_features[:count]
    return sets


def evaluate_feature_sets(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    tasks: pd.DataFrame,
    ranked_features: list[str],
    valid_fraction: float,
) -> pd.DataFrame:
    train_index, valid_index = get_time_split_indices(tasks, valid_fraction)
    model = make_ml_models().get("catboost") or make_ml_models()["hist_gradient_boosting"]
    rows = []
    for name, columns in build_feature_sets(features, ranked_features).items():
        metric = evaluate_model_time_split(
            model,
            features[columns],
            answers,
            train_index,
            valid_index,
        )
        rows.append({"feature_set": name, "feature_count": len(columns), "metric": metric})
        print(f"feature_set={name} count={len(columns)} metric={metric}")
    return pd.DataFrame(rows).sort_values("metric", kind="stable").reset_index(drop=True)


def build_calibration_feature_sets(features: pd.DataFrame, ranked_non_replay: list[str]) -> dict[str, list[str]]:
    groups = {name: [column for column in features.columns if feature_group(column) == name] for name in [
        "base",
        "audience",
        "publisher",
        "recent",
        "replay",
    ]}
    all_features = list(features.columns)
    sets: dict[str, list[str]] = {
        "all": all_features,
        "replay_only": REPLAY_COLUMNS,
        "without_recent": [column for column in all_features if feature_group(column) != "recent"],
        "without_publisher": [column for column in all_features if feature_group(column) != "publisher"],
        "without_audience": [column for column in all_features if feature_group(column) != "audience"],
        "without_base": [column for column in all_features if feature_group(column) != "base"],
        "replay_plus_base": REPLAY_COLUMNS + groups["base"],
        "replay_plus_recent": REPLAY_COLUMNS + groups["recent"],
        "replay_plus_audience": REPLAY_COLUMNS + groups["audience"],
        "replay_plus_publisher": REPLAY_COLUMNS + groups["publisher"],
    }
    for count in [5, 10, 15, 20, 30, 40, 60]:
        top = [column for column in ranked_non_replay if column not in REPLAY_COLUMNS][:count]
        sets[f"replay_plus_top_{count}"] = REPLAY_COLUMNS + top
    return {name: list(dict.fromkeys(columns)) for name, columns in sets.items()}


def evaluate_calibration_feature_sets(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    tasks: pd.DataFrame,
    ranked_features: list[str],
    valid_fraction: float,
) -> pd.DataFrame:
    ranked_non_replay = [column for column in ranked_features if column not in REPLAY_COLUMNS]
    rows = []
    for name, columns in build_calibration_feature_sets(features, ranked_non_replay).items():
        scores, _ = evaluate_calibration_time_split(
            features[columns],
            answers,
            tasks,
            valid_fraction,
        )
        best = scores.iloc[0]
        rows.append(
            {
                "feature_set": name,
                "feature_count": len(columns),
                "model": str(best["model"]),
                "shrink": float(best["shrink"]),
                "metric": float(best["metric"]),
            }
        )
        print(
            f"calibration_set={name} count={len(columns)} "
            f"model={best['model']} shrink={best['shrink']} metric={best['metric']}"
        )
    return pd.DataFrame(rows).sort_values("metric", kind="stable").reset_index(drop=True)


def evaluate_model_metrics(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    tasks: pd.DataFrame,
    selected_features: list[str],
    valid_fraction: float,
) -> pd.DataFrame:
    train_index, valid_index = get_time_split_indices(tasks, valid_fraction)
    rows = []
    for name, model in make_ml_models().items():
        metric = evaluate_model_time_split(
            model,
            features[selected_features],
            answers,
            train_index,
            valid_index,
        )
        rows.append({"model": name, "metric": metric})
        print(f"model={name} metric={metric}")
    return pd.DataFrame(rows).sort_values("metric", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild feature-selection experiment artifacts.")
    parser.add_argument("--features", default=DATA_FEATURES / "features_validate.tsv", type=Path)
    parser.add_argument("--replay-features", default=PREDICTIONS / "shift_replay_predictions.tsv", type=Path)
    parser.add_argument("--tasks", default=DATA_RAW / "validate.tsv", type=Path)
    parser.add_argument("--answers", default=DATA_RAW / "validate_answers.tsv", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "artifacts" / "experiments", type=Path)
    parser.add_argument("--time-valid-fraction", default=0.2, type=float)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tsv(args.tasks)
    answers = read_tsv(args.answers)[TARGET_COLUMNS]
    features = append_replay_feature_file(read_tsv(args.features), args.replay_features)
    train_index, _ = get_time_split_indices(tasks, args.time_valid_fraction)

    ranked_features = rank_features(features, answers, train_index)

    feature_results = evaluate_feature_sets(
        features,
        answers,
        tasks,
        ranked_features,
        args.time_valid_fraction,
    )
    feature_results.to_csv(args.output_dir / "feature_selection_results.tsv", sep="\t", index=False)

    selected_feature_set = str(feature_results.iloc[0]["feature_set"])
    selected_features = build_feature_sets(features, ranked_features)[selected_feature_set]
    (args.output_dir / "selected_features.txt").write_text(
        "\n".join(selected_features) + "\n",
        encoding="utf-8",
    )
    selected_table = pd.DataFrame({"feature": selected_features})
    selected_table["group"] = selected_table["feature"].map(feature_group)
    selected_table.to_csv(args.output_dir / "selected_features_table.tsv", sep="\t", index=False)

    calibration_results = evaluate_calibration_feature_sets(
        features,
        answers,
        tasks,
        ranked_features,
        args.time_valid_fraction,
    )
    calibration_results.to_csv(
        args.output_dir / "calibration_feature_selection_results.tsv",
        sep="\t",
        index=False,
    )

    best_calibration_set = str(calibration_results.iloc[0]["feature_set"])
    selected_calibration_features = build_calibration_feature_sets(
        features,
        [column for column in ranked_features if column not in REPLAY_COLUMNS],
    )[best_calibration_set]
    (args.output_dir / "selected_calibration_features.txt").write_text(
        "\n".join(selected_calibration_features) + "\n",
        encoding="utf-8",
    )

    model_metrics = evaluate_model_metrics(
        features,
        answers,
        tasks,
        selected_features,
        args.time_valid_fraction,
    )
    model_metrics.to_csv(args.output_dir / "model_metrics.tsv", sep="\t", index=False)

    print(f"Saved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
