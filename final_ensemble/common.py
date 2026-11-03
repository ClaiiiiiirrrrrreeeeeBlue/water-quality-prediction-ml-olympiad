"""Shared I/O and metric helpers for the entry points."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_data(directory=DATA_DIR):
    """Read the competition CSVs. The files are downloaded separately, not tracked."""
    directory = Path(directory)
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    missing = {"id", "result"} - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    if "id" not in test.columns:
        raise ValueError("test.csv is missing the id column")
    return train, test


def rmsle(y_true, y_pred):
    """Root mean squared log error; the competition target is non-negative."""
    a = np.log1p(np.asarray(y_true, dtype=float))
    b = np.log1p(np.maximum(np.asarray(y_pred, dtype=float), 0.0))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def save_submission(path, ids, values):
    frame = pd.DataFrame({"id": ids, "result": np.asarray(values, dtype=float)})
    if not np.isfinite(frame.result).all():
        raise ValueError("submission contains NaN or infinite values")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
