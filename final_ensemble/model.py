"""Fixed model definitions for the five final members.

There is no search space here: every hyperparameter is the value that produced the
submitted prediction. Weights and round counts live in ``config/members.json``.
"""
from pathlib import Path
import json
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from ensemble_features import EnsembleFeatureBuilder

HERE = Path(__file__).resolve().parent
MEMBERS = json.loads((HERE / "members.json").read_text())


def make_features(member):
    return EnsembleFeatureBuilder(member["mode"], smooth=10,
                                  seed=member["seed"], dropout=0.2, pair_smooth=3)


def prepare(frame, family):
    """CatBoost needs plain strings; XGBoost and LightGBM take the pandas categories."""
    if family != "cat":
        return frame
    frame = frame.copy()
    for column in frame.select_dtypes(include="category"):
        frame[column] = frame[column].astype(object).fillna("__UNKNOWN__").astype(str)
    return frame


def fit_model(member, frame, target, valid_frame=None, valid_target=None, rounds=None):
    family, seed = member["family"], member["seed"]
    count = 3500 if rounds is None else rounds
    if family == "xgb":
        estimator = XGBRegressor(n_estimators=count, learning_rate=.025,
            max_depth=member["depth"], min_child_weight=10, subsample=.85,
            colsample_bytree=.9, reg_lambda=10, reg_alpha=.1,
            max_cat_to_onehot=4, max_cat_threshold=64, tree_method="hist",
            enable_categorical=True, objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=150 if valid_frame is not None else None,
            n_jobs=8, random_state=seed, device="cpu")
        estimator.fit(frame, target,
            eval_set=[(valid_frame, valid_target)] if valid_frame is not None else None,
            verbose=False)
    elif family == "cat":
        frame = prepare(frame, family)
        estimator = CatBoostRegressor(iterations=count, depth=member["depth"],
            learning_rate=.03, loss_function="RMSE", l2_leaf_reg=5,
            random_seed=seed, thread_count=8, verbose=False,
            allow_writing_files=False, random_strength=.5,
            early_stopping_rounds=150 if valid_frame is not None else None)
        estimator.fit(frame, target,
            cat_features=frame.select_dtypes(include="object").columns.tolist(),
            eval_set=(prepare(valid_frame, family), valid_target) if valid_frame is not None else None,
            use_best_model=valid_frame is not None, verbose=False)
    else:
        estimator = LGBMRegressor(n_estimators=count, learning_rate=.025,
            num_leaves=member["leaves"], min_child_samples=20,
            reg_lambda=5, reg_alpha=.1, colsample_bytree=.9,
            subsample=.9, subsample_freq=1, n_jobs=8, random_state=seed,
            verbosity=-1, deterministic=True, force_col_wise=True)
        estimator.fit(frame, target,
            eval_set=[(valid_frame, valid_target)] if valid_frame is not None else None,
            callbacks=[early_stopping(150, verbose=False), log_evaluation(0)]
            if valid_frame is not None else [])
    return estimator


def predict_log(estimator, frame, family):
    """Predict, apply the [0, 1] clip in original space, then return log1p."""
    frame = prepare(frame, family)
    if family == "cat":
        values = estimator.predict(frame, thread_count=8)
    else:
        values = estimator.predict(frame)
    return np.log1p(np.clip(np.expm1(values), 0, 1))


def best_rounds(estimator, family):
    if family == "xgb":
        return estimator.best_iteration + 1
    if family == "cat":
        return estimator.get_best_iteration() + 1
    return estimator.best_iteration_
