"""Replay the submitted prediction from saved checkpoints. No retraining."""
from pathlib import Path
import argparse
import pickle
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, ROOT, save_submission
from model import MEMBERS, predict_log

FOLD_COUNT = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE.parent / "checkpoints")
    parser.add_argument("--output", type=Path, default=HERE.parent / "outputs" / "submission.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    X = raw.drop(columns=["id"])
    final_log = np.zeros(len(raw))
    model_count = 0
    for member in MEMBERS:
        directory = args.checkpoint_dir / member["key"]
        if not (directory / "full_model.pkl").exists():
            raise SystemExit(
                f"Missing checkpoints in {directory}. "
                "Run `python src/train.py` first, or unpack the release asset into ./checkpoints.")

        bag_log = np.zeros(len(raw))
        for fold in range(FOLD_COUNT):
            with (directory / f"fold{fold}_builder.pkl").open("rb") as file:
                features = pickle.load(file)
            with (directory / f"fold{fold}_model.pkl").open("rb") as file:
                estimator = pickle.load(file)
            if member["family"] == "xgb":
                estimator.set_params(device="cpu", n_jobs=8)
            bag_log += predict_log(estimator, features.transform(X), member["family"]) / FOLD_COUNT
            model_count += 1

        with (directory / "full_builder.pkl").open("rb") as file:
            features = pickle.load(file)
        with (directory / "full_model.pkl").open("rb") as file:
            estimator = pickle.load(file)
        if member["family"] == "xgb":
            estimator.set_params(device="cpu", n_jobs=8)
        full_log = predict_log(estimator, features.transform(X), member["family"])
        final_log += member["weight"] * (0.5 * bag_log + 0.5 * full_log)
        model_count += 1

    predictions = np.clip(np.expm1(final_log), 0, 1)
    save_submission(args.output, raw.id, predictions)
    print(f"Wrote {len(predictions)} rows using {model_count} fitted models to {args.output}.")


if __name__ == "__main__":
    main()
