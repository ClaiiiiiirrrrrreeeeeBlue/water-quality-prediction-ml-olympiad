from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

class IdentityFeatures:
    """本课不额外产生特征；保留统一的 fit/transform 课堂接口。"""
    def fit_transform(self, frame, target):
        return frame.copy()
    def transform(self, frame):
        return frame.copy()

def make_features():
    return IdentityFeatures()

def fit_model(frame, target, valid_frame=None, valid_target=None, rounds=None):
    categories = [c for c in frame if c.startswith('category') or c == 'unit']
    numbers = [c for c in frame if c not in categories]
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numbers),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categories)
    ], sparse_threshold=1.0)
    forest = RandomForestRegressor(n_estimators=200, max_features=0.5,
        min_samples_leaf=2, random_state=42, n_jobs=4)
    estimator = make_pipeline(preprocessing, forest)
    estimator.fit(frame, target)
    return estimator

def predict_model(estimator, frame):
    return estimator.predict(frame)

def best_rounds(estimator):
    return 200
