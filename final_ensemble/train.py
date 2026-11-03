"""Stage 1 of 3: train the final ensemble members.

Each member in ``members.json`` is trained on the shared 5-fold split (seed 42),
producing five fold models and one full-data model. Everything is written under
``checkpoints/`` and ``runs/``; nothing is blended here.

Round selection mirrors the original run: the fold models are early-stopped on
their validation fold and the full-data model uses the median best round, unless
``--frozen-rounds`` is used, in which case the recorded counts are applied and
early stopping is skipped. The recorded counts are what produced the submission.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, load_data, rmsle, save_submission
from model import MEMBERS, make_features, fit_model, predict_log, best_rounds

FOLD_COUNT = 5
FOLD_SEED = 42


def train_member(member, train, test, target, folds, checkpoint_dir, runs_dir, frozen):
    family = member["family"]
    X = train.drop(columns=["id", "result"])
    T = test.drop(columns=["id"])

    weights_dir = Path(checkpoint_dir) / member["key"]
    weights_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(runs_dir) / member["key"]
    run_dir.mkdir(parents=True, exist_ok=True)

    oof = np.zeros(len(train))
    bag_log = np.zeros(len(T))
    rounds = []
    for fold, (train_index, valid_index) in enumerate(folds):
        features = make_features(member)
        A = features.fit_transform(X.iloc[train_index], target[train_index])
        B = features.transform(X.iloc[valid_index])
        if frozen:
            count = member["fold_rounds"][fold]
            estimator = fit_model(member, A, target[train_index], rounds=count)
        else:
            estimator = fit_model(member, A, target[train_index], B, target[valid_index])
            count = best_rounds(estimator, family)
        rounds.append(count)
        oof[valid_index] = np.clip(np.expm1(predict_log(estimator, B, family)), 0, 1)
        bag_log += predict_log(estimator, features.transform(T), family) / len(folds)
        for name, value in (("builder", features), ("model", estimator)):
            with (weights_dir / f"fold{fold}_{name}.pkl").open("wb") as file:
                pickle.dump(value, file)
        print(f"  {member['key']} fold {fold + 1}/{len(folds)} rounds={count}", flush=True)

    full_count = member["full_rounds"] if frozen else int(np.median(rounds))
    features = make_features(member)
    A = features.fit_transform(X, target)
    estimator = fit_model(member, A, target, rounds=full_count)
    full_log = predict_log(estimator, features.transform(T), family)
    for name, value in (("builder", features), ("model", estimator)):
        with (weights_dir / f"full_{name}.pkl").open("wb") as file:
            pickle.dump(value, file)

    pd.DataFrame({"id": train.id, "result": train.result, "prediction": oof}).to_csv(
        run_dir / "oof.csv", index=False)
    save_submission(run_dir / "submission_bag.csv", test.id, np.clip(np.expm1(bag_log), 0, 1))
    save_submission(run_dir / "submission_full.csv", test.id, np.clip(np.expm1(full_log), 0, 1))

    score = rmsle(train.result, oof)
    metrics = {
        "member": member["key"], "family": family, "seed": member["seed"],
        "mode": member["mode"], "fold_rounds": rounds, "full_rounds": full_count,
        "oof_rmsle": score, "round_mode": "frozen" if frozen else "early_stopping",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"  {member['key']} full rounds={full_count}  oof_rmsle={score:.6f}", flush=True)
    return score


def run(data_dir=DATA_DIR, checkpoint_dir=ROOT / "checkpoints", runs_dir=ROOT / "runs", frozen=True):
    started = time.perf_counter()
    train, test = load_data(data_dir)
    target = np.log1p(train.result.to_numpy())
    folds = list(KFold(FOLD_COUNT, shuffle=True, random_state=FOLD_SEED).split(train.drop(columns=["id", "result"])))

    scores = {}
    for member in MEMBERS:
        print(f"[{member['key']}] {member['family']} depth={member['depth']} leaves={member['leaves']}", flush=True)
        scores[member["key"]] = train_member(
            member, train, test, target, folds, checkpoint_dir, runs_dir, frozen)

    summary = {"fold_seed": FOLD_SEED, "members": scores,
               "seconds": time.perf_counter() - started}
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    (Path(runs_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nTrained {len(MEMBERS)} members in {summary['seconds']:.1f}s")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--early-stopping", action="store_true",
                        help="re-derive round counts instead of using the recorded ones")
    args = parser.parse_args()
    run(args.data_dir, args.checkpoint_dir, args.runs_dir, frozen=not args.early_stopping)
