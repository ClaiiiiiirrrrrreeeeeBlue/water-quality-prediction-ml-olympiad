"""Check the shipped submission and reference artefacts against recorded values."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, ROOT, rmsle


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    reference = json.loads((ROOT / "reference" / "metrics.json").read_text())
    submission = ROOT / "submission" / "final_submission.csv"
    frame = pd.read_csv(submission)
    failures = []

    digest = sha256(submission)
    print(f"submission sha256   : {digest}")
    if digest != reference["submission_sha256"]:
        failures.append("submission sha256 does not match the recorded file")

    print(f"submission rows     : {len(frame)} (expected {reference['rows']})")
    if len(frame) != reference["rows"]:
        failures.append("unexpected row count")

    if not np.isfinite(frame.result).all():
        failures.append("submission has non-finite values")
    if not frame.result.between(0, 1).all():
        failures.append("submission values fall outside [0, 1]")
    print(f"submission range    : {frame.result.min():.6f} .. {frame.result.max():.6f}")

    test_path = Path(args.data_dir) / "test.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        if frame.id.equals(test.id):
            print("id order            : matches data/test.csv")
        else:
            failures.append("id order does not match data/test.csv")
    else:
        print("id order            : skipped (data/test.csv not found)")

    oof_path = ROOT / "reference" / "oof.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        score = rmsle(oof.result, oof.prediction)
        print(f"reference OOF RMSLE : {score:.9f} (recorded {reference['oof_rmsle']:.9f})")
        if abs(score - reference["oof_rmsle"]) > 1e-9:
            failures.append("reference OOF RMSLE does not reproduce")

    if failures:
        for reason in failures:
            print(f"FAIL: {reason}")
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
