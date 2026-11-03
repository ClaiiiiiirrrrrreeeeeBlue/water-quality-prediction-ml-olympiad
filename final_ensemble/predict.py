"""Stage 3 of 3: rebuild the prediction directly from the fitted models.

This is the verification path for stage 2. ``blend.py`` combines the per-member
test predictions that were saved during training; this script reloads all 30
fitted models and recomputes the same thing from scratch, so agreement between
the two is evidence that the saved artefacts and the saved predictions match.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, save_submission
from model import MEMBERS, predict_log

FOLD_COUNT = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--manifest", type=Path, default=ROOT / "blend" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission.csv")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing {args.manifest}. Run `python final_ensemble/blend.py` first.")
    manifest = json.loads(args.manifest.read_text())
    weights = dict(zip(manifest["members"], manifest["weights"]))
    families = {m["key"]: m["family"] for m in MEMBERS}

    raw = pd.read_csv(args.input)
    X = raw.drop(columns=["id"])
    final_log = np.zeros(len(raw))
    model_count = 0

    for key, weight in weights.items():
        if weight < 1e-9:
            continue
        directory = args.checkpoint_dir / key
        if not (directory / "full_model.pkl").exists():
            raise SystemExit(
                f"Missing checkpoints in {directory}. Run `python final_ensemble/train.py` "
                "first, or unpack the release asset into ./checkpoints.")

        if key not in families:
            raise SystemExit(f"Unknown member {key!r}; check final_ensemble/members.json")
        family = families[key]

        bag_log = np.zeros(len(raw))
        for fold in range(FOLD_COUNT):
            with (directory / f"fold{fold}_builder.pkl").open("rb") as file:
                features = pickle.load(file)
            with (directory / f"fold{fold}_model.pkl").open("rb") as file:
                estimator = pickle.load(file)
            if family == "xgb":
                estimator.set_params(device="cpu", n_jobs=8)
            bag_log += predict_log(estimator, features.transform(X), family) / FOLD_COUNT
            model_count += 1

        with (directory / "full_builder.pkl").open("rb") as file:
            features = pickle.load(file)
        with (directory / "full_model.pkl").open("rb") as file:
            estimator = pickle.load(file)
        if family == "xgb":
            estimator.set_params(device="cpu", n_jobs=8)
        full_log = predict_log(estimator, features.transform(X), family)
        final_log += weight * (0.5 * bag_log + 0.5 * full_log)
        model_count += 1

    prediction = np.clip(np.expm1(final_log), 0, 1)
    save_submission(args.output, raw.id, prediction)
    print(f"Wrote {len(prediction)} rows using {model_count} fitted models to {args.output}.")


if __name__ == "__main__":
    main()
