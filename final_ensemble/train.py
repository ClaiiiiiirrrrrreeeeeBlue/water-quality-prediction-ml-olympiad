"""Train the five final members and write a submission.

Two ways to choose the number of boosting rounds:

``--frozen-rounds`` (default)
    Use the round counts stored in ``config/members.json``. These are the counts
    that produced the submitted models, so a rerun is deterministic given the
    same data, package versions and CPU.

``--early-stopping``
    Rerun the original selection procedure: early-stop inside every fold and use
    the median best round for the full-data model. This is how the recorded OOF
    was produced, and it is the honest way to re-derive the round counts.
"""
from pathlib import Path
import argparse
import json
import pickle
import sys
import time
import numpy as np
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, ROOT, load_data, rmsle, save_submission
from model import MEMBERS, make_features, fit_model, predict_log, best_rounds

FOLD_SEED = 42
FOLD_COUNT = 5


def train_member(member, X, target, T, folds, checkpoint_dir, frozen):
    directory = Path(checkpoint_dir) / member["key"]
    directory.mkdir(parents=True, exist_ok=True)
    family = member["family"]
    oof_log = np.zeros(len(X))
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
        oof_log[valid_index] = predict_log(estimator, B, family)
        bag_log += predict_log(estimator, features.transform(T), family) / len(folds)
        for name, value in (("builder", features), ("model", estimator)):
            with (directory / f"fold{fold}_{name}.pkl").open("wb") as file:
                pickle.dump(value, file)
        print(f"{member['key']} fold {fold + 1} rounds={count}", flush=True)

    if frozen:
        full_count = member["full_rounds"]
    else:
        full_count = int(np.median(rounds))
    features = make_features(member)
    A = features.fit_transform(X, target)
    estimator = fit_model(member, A, target, rounds=full_count)
    full_log = predict_log(estimator, features.transform(T), family)
    for name, value in (("builder", features), ("model", estimator)):
        with (directory / f"full_{name}.pkl").open("wb") as file:
            pickle.dump(value, file)
    print(f"{member['key']} full rounds={full_count}", flush=True)
    return oof_log, 0.5 * bag_log + 0.5 * full_log, rounds, full_count


def run(data_dir=DATA_DIR, output_dir=HERE.parent / "outputs",
        checkpoint_dir=HERE.parent / "checkpoints", frozen=True):
    started = time.perf_counter()
    train, test = load_data(data_dir)
    X = train.drop(columns=["id", "result"])
    T = test.drop(columns=["id"])
    target = np.log1p(train.result.to_numpy())
    folds = list(KFold(FOLD_COUNT, shuffle=True, random_state=FOLD_SEED).split(X))

    oof_log = np.zeros(len(train))
    test_log = np.zeros(len(test))
    rounds_used = {}
    for member in MEMBERS:
        member_oof, member_test, fold_rounds, full_rounds = train_member(
            member, X, target, T, folds, checkpoint_dir, frozen)
        oof_log += member["weight"] * member_oof
        test_log += member["weight"] * member_test
        rounds_used[member["key"]] = {"folds": fold_rounds, "full": full_rounds}

    oof = np.clip(np.expm1(oof_log), 0, 1)
    predictions = np.clip(np.expm1(test_log), 0, 1)
    score = rmsle(train.result, oof)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_submission(output_dir / "submission.csv", test.id, predictions)
    save_submission(output_dir / "oof.csv", train.id, oof)
    metrics = {
        "oof_rmsle": score,
        "round_mode": "frozen" if frozen else "early_stopping",
        "fold_seed": FOLD_SEED,
        "rounds": rounds_used,
        "weights_source": "config/members.json",
        "seconds": time.perf_counter() - started,
        "note": "Frozen weights were selected on this same 5-fold split, "
                "so the OOF score is a development estimate, not a blind test.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"OOF RMSLE: {score:.6f}  ({metrics['seconds']:.1f}s)")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "outputs")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE.parent / "checkpoints")
    parser.add_argument("--early-stopping", action="store_true",
                        help="re-derive round counts instead of using the frozen ones")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.checkpoint_dir, frozen=not args.early_stopping)
