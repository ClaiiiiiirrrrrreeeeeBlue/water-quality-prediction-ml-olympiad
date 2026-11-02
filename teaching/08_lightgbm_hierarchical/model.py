from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from ensemble_features import EnsembleFeatureBuilder

def make_features():
    return EnsembleFeatureBuilder('te_residual_more_hier', smooth=10,
        seed=42, dropout=0.2, pair_smooth=3)

def fit_model(frame, target, valid_frame=None, valid_target=None, rounds=None):
    estimator = LGBMRegressor(n_estimators=rounds if rounds is not None else 3500,
        learning_rate=0.025, num_leaves=31, min_child_samples=20,
        reg_lambda=5, reg_alpha=0.1, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, n_jobs=8, random_state=42,
        verbosity=-1, deterministic=True, force_col_wise=True)
    estimator.fit(frame, target,
        eval_set=[(valid_frame, valid_target)] if valid_frame is not None else None,
        callbacks=[early_stopping(150, verbose=False), log_evaluation(0)]
        if valid_frame is not None else [])
    return estimator

def predict_model(estimator, frame):
    return estimator.predict(frame)

def best_rounds(estimator):
    return estimator.best_iteration_
