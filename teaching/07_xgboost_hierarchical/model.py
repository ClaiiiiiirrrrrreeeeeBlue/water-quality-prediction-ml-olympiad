from xgboost import XGBRegressor
from ensemble_features import EnsembleFeatureBuilder

def make_features():
    return EnsembleFeatureBuilder('te_residual_more_hier', smooth=10,
        seed=42, dropout=0.2, pair_smooth=3)

def fit_model(frame, target, valid_frame=None, valid_target=None, rounds=None):
    # 在 log1p 目标上最小化 RMSE；外层仍按原始目标的 RMSLE 报告。
    estimator = XGBRegressor(n_estimators=rounds if rounds is not None else 3500,
        learning_rate=0.025, max_depth=5, min_child_weight=10,
        subsample=0.85, colsample_bytree=0.9, reg_lambda=10, reg_alpha=0.1,
        max_cat_to_onehot=4, max_cat_threshold=64, tree_method='hist',
        enable_categorical=True, objective='reg:squarederror', eval_metric='rmse',
        early_stopping_rounds=150 if valid_frame is not None else None,
        n_jobs=8, random_state=42, device='cpu')
    estimator.fit(frame, target,
        eval_set=[(valid_frame, valid_target)] if valid_frame is not None else None,
        verbose=False)
    return estimator

def predict_model(estimator, frame):
    return estimator.predict(frame)

def best_rounds(estimator):
    return estimator.best_iteration + 1
