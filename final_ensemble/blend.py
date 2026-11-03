"""Stage 2 of 3: select blend weights and combine the members.

Two modes:

``--frozen`` (default)
    Apply the weights recorded in ``members.json``. These are the weights behind
    the submitted file.

``--refit``
    Re-solve the constrained least-squares problem on the OOF predictions of the
    members that are actually present. Useful for a new ensemble, but it will not
    reproduce the frozen weights exactly: those were solved over a larger pool of
    candidate members, and the zero-weight ones were then dropped.

The blend is computed in ``log1p`` space:
``expm1(sum_m w_m * 0.5 * (bag_m + full_m))``, clipped to ``[0, 1]``.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, load_data, rmsle, save_submission
from model import MEMBERS

PENALTY = 1e-5


def weights_fit(P, y, penalty=PENALTY):
    """Convex weights minimising OOF MSE in log space, with a weak uniform prior."""
    n = P.shape[1]
    prior = np.full(n, 1 / n)

    def objective(w):
        return np.mean((P @ w - y) ** 2) + penalty * np.sum((w - prior) ** 2)

    def gradient(w):
        return 2 * (P.T @ (P @ w - y)) / len(y) + 2 * penalty * (w - prior)

    result = minimize(objective, prior, jac=gradient, method="SLSQP",
                      bounds=[(0, 1)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones(n)}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def load_runs(runs_dir, train, test):
    names, P, B, F = [], [], [], []
    for member in MEMBERS:
        directory = Path(runs_dir) / member["key"]
        if not (directory / "oof.csv").exists():
            raise SystemExit(f"Missing {directory}/oof.csv. Run `python final_ensemble/train.py` first.")
        oof = pd.read_csv(directory / "oof.csv")
        bag = pd.read_csv(directory / "submission_bag.csv")
        full = pd.read_csv(directory / "submission_full.csv")
        if not oof.id.equals(train.id):
            raise SystemExit(f"{directory}/oof.csv rows are not aligned with train.csv")
        if not bag.id.equals(test.id) or not full.id.equals(test.id):
            raise SystemExit(f"{directory} test predictions are not aligned with test.csv")
        names.append(member["key"])
        P.append(np.log1p(oof.prediction.to_numpy()))
        B.append(np.log1p(bag.result.to_numpy()))
        F.append(np.log1p(full.result.to_numpy()))
    return names, np.array(P).T, np.array(B).T, np.array(F).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "blend")
    parser.add_argument("--refit", action="store_true",
                        help="re-solve the weights instead of using the frozen ones")
    args = parser.parse_args()

    train, test = load_data(args.data_dir)
    y = np.log1p(train.result.to_numpy())
    names, P, B, F = load_runs(args.runs_dir, train, test)

    frozen = np.array([m["weight"] for m in MEMBERS])
    if args.refit:
        weights = weights_fit(P, y)
        mode = "refit"
    else:
        weights = frozen
        mode = "frozen"

    bag_log = B @ weights
    full_log = F @ weights
    mixed = (bag_log + full_log) / 2
    oof_log = P @ weights

    oof = np.clip(np.expm1(oof_log), 0, 1)
    prediction = np.clip(np.expm1(mixed), 0, 1)
    score = rmsle(train.result, oof)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_submission(output_dir / "submission.csv", test.id, prediction)
    pd.DataFrame({"id": train.id, "result": train.result, "prediction": oof}).to_csv(
        output_dir / "oof.csv", index=False)

    per_member = {name: rmsle(train.result, np.clip(np.expm1(P[:, k]), 0, 1))
                  for k, name in enumerate(names)}
    manifest = {
        "mode": mode,
        "members": names,
        "weights": [float(w) for w in weights],
        "weights_frozen": [float(w) for w in frozen],
        "member_oof_rmsle": per_member,
        "oof_rmsle": score,
        "prediction_rule": "clip(expm1(sum(w * 0.5 * (bag + full))), 0, 1)",
        "note": "Weights were selected on this same OOF split, so oof_rmsle is a "
                "development estimate. Bag and full predictions are averaged in log space.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    table = pd.DataFrame({"member": names, "weight": weights,
                          "oof_rmsle": [per_member[n] for n in names]})
    print(table.to_string(index=False))
    print(f"\nweight sum      : {weights.sum():.9f}")
    print(f"equal weights   : {rmsle(train.result, np.clip(np.expm1(P.mean(1)), 0, 1)):.9f}")
    print(f"blend OOF RMSLE : {score:.9f}  ({mode} weights)")
    print(f"bag vs full corr: {np.corrcoef(bag_log, full_log)[0, 1]:.6f}")
    if args.refit:
        print(f"max |weight - frozen| : {np.max(np.abs(weights - frozen)):.6f}")


if __name__ == "__main__":
    main()
