from catboost import CatBoostRegressor
from ensemble_features import EnsembleFeatureBuilder

def make_features():
    return EnsembleFeatureBuilder('te_residual_more_hier', smooth=10,
        seed=42, dropout=0.2, pair_smooth=3)

def prepare_categories(frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include='category'):
        frame[column] = frame[column].astype(object).fillna('__UNKNOWN__').astype(str)
    return frame

def fit_model(frame, target, valid_frame=None, valid_target=None, rounds=None):
    frame = prepare_categories(frame)
    estimator = CatBoostRegressor(iterations=rounds if rounds is not None else 3500,
        depth=6, learning_rate=0.03, loss_function='RMSE', l2_leaf_reg=5,
        random_seed=42, thread_count=8, verbose=False, allow_writing_files=False,
        random_strength=0.5, early_stopping_rounds=150 if valid_frame is not None else None)
    categories = frame.select_dtypes(include='object').columns.tolist()
    validation = (prepare_categories(valid_frame), valid_target) if valid_frame is not None else None
    estimator.fit(frame, target, cat_features=categories,
        eval_set=validation, use_best_model=valid_frame is not None, verbose=False)
    return estimator

def predict_model(estimator, frame):
    return estimator.predict(prepare_categories(frame), thread_count=8)

def best_rounds(estimator):
    return estimator.get_best_iteration() + 1
