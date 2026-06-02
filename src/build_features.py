from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from ml_baseline import prepare_features_for_tasks, read_tsv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache campaign features.")
    parser.add_argument("--tasks", default=ROOT / "validate.tsv", type=Path)
    parser.add_argument("--output", default=ROOT / "features_validate.tsv", type=Path)
    parser.add_argument("--include-shift-replay", action="store_true")
    args = parser.parse_args()

    users = read_tsv(ROOT / "users.tsv")
    history = read_tsv(ROOT / "history.tsv")
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
